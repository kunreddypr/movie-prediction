from airflow.decorators import dag, task
from datetime import datetime
import os
import shutil
import uuid
import pandas as pd


BASE_DATA_FOLDER = os.path.join(os.getcwd(), 'data')
RAW_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER, 'raw-data')
GOOD_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER, 'good-data')
INGESTED_ARCHIVE_FOLDER = os.path.join(BASE_DATA_FOLDER, 'ingested-archive')
SOURCE_CSV_PATH = os.getenv('MOVIES_SOURCE_CSV', os.path.join(BASE_DATA_FOLDER, 'movies.csv'))

@dag(
    dag_id="movie_data_ingestion_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["data_pipeline", "ingestion"]
)
def data_ingestion_dag():
    
    os.makedirs(RAW_DATA_FOLDER, exist_ok=True)
    os.makedirs(GOOD_DATA_FOLDER, exist_ok=True)
    os.makedirs(INGESTED_ARCHIVE_FOLDER, exist_ok=True)

    @task
    def split_movies_csv(chunk_size: int = 20) -> list[str]:
        """Split the source movies.csv into multiple CSVs with chunk_size rows each in raw-data."""
        if not os.path.exists(SOURCE_CSV_PATH):
            print(f"Source CSV not found at {SOURCE_CSV_PATH}. Skipping.")
            return []
        try:
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
            fname = f"movies_chunk_{file_tag}_{start//chunk_size + 1}.csv"
            out_path = os.path.join(RAW_DATA_FOLDER, fname)
            chunk.to_csv(out_path, index=False)
            created_files.append(out_path)
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
            file_name = os.path.basename(raw_file_path)
            good_file_path = os.path.join(GOOD_DATA_FOLDER, file_name)
            archive_file_path = os.path.join(INGESTED_ARCHIVE_FOLDER, file_name)
            shutil.copy(raw_file_path, good_file_path)
            shutil.move(raw_file_path, archive_file_path)
            print(f"Copied to good: {good_file_path} and archived: {archive_file_path}")
            moved.append(good_file_path)
        return moved


    created_files = split_movies_csv()
    move_files_to_good(raw_file_paths=created_files)


data_ingestion_dag()
