from airflow.decorators import dag, task
from datetime import datetime
import os
import shutil
import uuid
from pathlib import Path
import pandas as pd


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


    created_files = split_movies_csv()
    move_files_to_good(raw_file_paths=created_files)


data_ingestion_dag()
