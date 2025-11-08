from __future__ import annotations

from airflow.decorators import dag, task
from datetime import datetime
import os
import shutil
import sys
import uuid
from pathlib import Path
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.db_utils import ensure_prediction_schema, get_db_connection
from scripts.simple_xlsx import dataframe_to_xlsx


def _default_data_root() -> Path:
    """Return the most sensible data root for both local and Airflow runs."""
    repo_data = Path(__file__).resolve().parents[1] / "data"
    if repo_data.exists():
        return repo_data

    airflow_home = Path(os.getenv("AIRFLOW_HOME", "/opt/airflow"))
    return airflow_home / "data"


DATA_ROOT = Path(os.getenv("AIRFLOW_DATA_HOME", _default_data_root()))
RAW_DATA_FOLDER = DATA_ROOT / "raw-data"
GOOD_DATA_FOLDER = DATA_ROOT / "good-data"
INGESTED_ARCHIVE_FOLDER = DATA_ROOT / "ingested-archive"


def _default_source_csv() -> Path:
    candidate_paths = [
        DATA_ROOT / "movies.csv",
        DATA_ROOT / "movies.csv.gz",
        Path(__file__).resolve().parents[1] / "movies.csv",
        Path(__file__).resolve().parents[1] / "movies.csv.gz",
        Path(os.getenv("AIRFLOW_HOME", "/opt/airflow")) / "movies.csv",
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    return candidate_paths[0]


SOURCE_CSV_PATH = Path(os.getenv("MOVIES_SOURCE_CSV", str(_default_source_csv())))

@dag(
    dag_id="movie_data_ingestion_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["data_pipeline", "ingestion"]
)
def data_ingestion_dag():
    
    RAW_DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    GOOD_DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    INGESTED_ARCHIVE_FOLDER.mkdir(parents=True, exist_ok=True)

    @task
    def split_movies_csv(chunk_size: int = 20) -> list[str]:
        """Split the source movies.csv into Excel files with ``chunk_size`` rows each.

        The ingestion DAG now produces ``.xlsx`` files so that the review step in the
        process can work with Excel spreadsheets directly.  Each generated file is
        written to the raw-data folder and contains at most ``chunk_size`` rows from
        the source dataset.
        """
        if not SOURCE_CSV_PATH.exists():
            print(f"Source CSV not found at {SOURCE_CSV_PATH}. Skipping.")
            return []
        try:
            if SOURCE_CSV_PATH.suffix == ".gz":
                df = pd.read_csv(SOURCE_CSV_PATH, compression="gzip")
            else:
                df = pd.read_csv(SOURCE_CSV_PATH)
        except Exception as e:
            print(f"Error reading source CSV: {e}")
            return []

        if df.empty:
            print("Source CSV is empty. Nothing to split.")
            return []

        created_files: list[str] = []
        file_tag = uuid.uuid4().hex[:8]
        total_rows = len(df)
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            chunk = df.iloc[start:end]
            fname = f"movies_chunk_{file_tag}_{start//chunk_size + 1}.xlsx"
            out_path = RAW_DATA_FOLDER / fname
            try:
                chunk.to_excel(out_path, index=False)
            except ValueError as exc:
                # pandas raises ValueError when the optional Excel dependency is missing.
                raise RuntimeError(
                    "Failed to write Excel file. Install the 'openpyxl' dependency in the Airflow environment."
                ) from exc
            created_files.append(str(out_path))
        print(f"Created {len(created_files)} files in raw-data from {total_rows} rows.")
        return created_files

    @task
    def move_files_to_good(raw_file_paths: list[str]):
        """Copy each file from raw-data to good-data and archive originals."""
        if not raw_file_paths:
            print("No files to move. Skipping.")
            return []
        moved = []
        for raw_file_path in raw_file_paths:
            raw_path = Path(raw_file_path)
            file_name = raw_path.name
            good_file_path = GOOD_DATA_FOLDER / file_name
            archive_file_path = INGESTED_ARCHIVE_FOLDER / file_name
            shutil.copy(raw_path, good_file_path)
            shutil.move(raw_path, archive_file_path)
            print(f"Copied to good: {good_file_path} and archived: {archive_file_path}")
            moved.append(str(good_file_path))
        return moved

    @task
    def record_data_quality_metrics(processed_files: list[str]):
        """Compute and persist data-quality metrics for each ingested file."""

        if not processed_files:
            print("No processed files available for quality metrics.")
            return []

        ensure_prediction_schema()

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                for file_path in processed_files:
                    file_path_obj = Path(file_path)
                    start_time = datetime.now()

                    if not file_path_obj.exists():
                        print(f"File {file_path_obj} no longer exists. Skipping metrics computation.")
                        continue

                    if file_path_obj.suffix.lower() == ".xlsx":
                        df = pd.read_excel(file_path_obj)
                    else:
                        df = pd.read_csv(file_path_obj)

                    total_rows = len(df)

                    required_columns = {"overview", "vote_average", "vote_count"}
                    missing_required_columns = required_columns.difference(df.columns)
                    missing_required_feature_count = len(missing_required_columns)

                    overview_series = df.get("overview")
                    if overview_series is not None:
                        missing_overview = overview_series.fillna("").astype(str).str.strip().eq("").sum()
                    else:
                        missing_overview = total_rows

                    vote_average_series = df.get("vote_average")
                    numeric_vote_average = None
                    if vote_average_series is not None:
                        numeric_vote_average = pd.to_numeric(vote_average_series, errors="coerce")
                        missing_vote_average = numeric_vote_average.isna().sum()
                        zero_vote_average = (numeric_vote_average.fillna(0) == 0).sum()
                    else:
                        missing_vote_average = total_rows
                        zero_vote_average = total_rows

                    vote_count_series = df.get("vote_count")
                    numeric_vote_count = None
                    if vote_count_series is not None:
                        numeric_vote_count = pd.to_numeric(vote_count_series, errors="coerce")
                        missing_vote_count = numeric_vote_count.isna().sum()
                        negative_vote_count = (numeric_vote_count.fillna(0) < 0).sum()
                    else:
                        missing_vote_count = total_rows
                        negative_vote_count = total_rows

                    if "id" in df.columns:
                        duplicate_id_count = df.duplicated(subset=["id"], keep=False).sum()
                    else:
                        duplicate_id_count = 0

                    invalid_mask = pd.Series(False, index=df.index)
                    if overview_series is not None:
                        invalid_mask |= overview_series.fillna("").astype(str).str.strip().eq("")
                    if vote_average_series is not None and numeric_vote_average is not None:
                        invalid_mask |= numeric_vote_average.isna()
                    else:
                        invalid_mask |= True
                    if vote_count_series is not None and numeric_vote_count is not None:
                        invalid_mask |= numeric_vote_count.isna()
                        invalid_mask |= numeric_vote_count.fillna(0) < 0
                    else:
                        invalid_mask |= True

                    invalid_rows = int(invalid_mask.sum())
                    valid_rows = max(total_rows - invalid_rows, 0)

                    processing_seconds = (datetime.now() - start_time).total_seconds()

                    cur.execute(
                        """
                        INSERT INTO ingestion_events (
                            file_name,
                            ingestion_time,
                            total_rows,
                            valid_rows,
                            invalid_rows,
                            missing_required_feature_count,
                            missing_overview_count,
                            missing_vote_average_count,
                            missing_vote_count,
                            negative_vote_count,
                            zero_vote_average_count,
                            duplicate_id_count,
                            processing_seconds
                        ) VALUES (%s, CURRENT_TIMESTAMP, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            file_path_obj.name,
                            total_rows,
                            valid_rows,
                            invalid_rows,
                            missing_required_feature_count,
                            int(missing_overview),
                            int(missing_vote_average),
                            int(missing_vote_count),
                            int(negative_vote_count),
                            int(zero_vote_average),
                            int(duplicate_id_count),
                            processing_seconds,
                        ),
                    )
                    event_id = cur.fetchone()[0]
                    print(
                        "Recorded data quality metrics for %s (event id=%s, invalid_rows=%s)."
                        % (file_path_obj.name, event_id, invalid_rows)
                    )
            conn.commit()
        finally:
            conn.close()

        return processed_files


    created_files = split_movies_csv()
    moved_files = move_files_to_good(raw_file_paths=created_files)
    record_data_quality_metrics(processed_files=moved_files)


data_ingestion_dag()
