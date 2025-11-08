from __future__ import annotations

from airflow.decorators import dag, task
from datetime import datetime
import os
import shutil
import sys
from pathlib import Path
from typing import List
import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.db_utils import get_db_connection
from scripts.simple_xlsx import read_xlsx


def _default_data_root() -> Path:
    repo_data = Path(__file__).resolve().parents[1] / "data"
    if repo_data.exists():
        return repo_data

    airflow_home = Path(os.getenv("AIRFLOW_HOME", "/opt/airflow"))
    return airflow_home / "data"


DATA_ROOT = Path(os.getenv("AIRFLOW_DATA_HOME", _default_data_root()))
GOOD_DATA_FOLDER = DATA_ROOT / "good-data"
PROCESSED_DATA_FOLDER = DATA_ROOT / "processed-data"


FASTAPI_URL = "http://api:8000/predict"

@dag(
    dag_id="movie_prediction_job",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["prediction", "api"]
)
def prediction_dag():
    
    GOOD_DATA_FOLDER.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

    @task
    def check_for_new_data() -> list[str]:
        """Checks for new files in good-data and returns only the next file to process."""

        patterns = ('*.csv', '*.xlsx')
        discovered: List[Path] = []
        for pattern in patterns:
            discovered.extend(GOOD_DATA_FOLDER.glob(pattern))

        if not discovered:
            print("No new files found in good-data. Skipping downstream tasks.")
            return []

        # Process files deterministically by modification time, then name.
        discovered.sort(key=lambda p: (p.stat().st_mtime, p.name))
        next_file = discovered[0]
        print(
            f"Selected {next_file.name} for processing. {len(discovered) - 1} other files remain in the queue."
        )
        return [str(next_file)]

    @task
    def make_predictions(file_paths: list):
       
        if not file_paths:
            print("No files to process. Task completed.")
            return

        conn = get_db_connection()
        cur = conn.cursor()
        total_predictions = 0

        for file_path in file_paths:
            file_path_obj = Path(file_path)
            file_name = file_path_obj.name
            print(f"Processing file: {file_name}")
            
            try:
                if file_path_obj.suffix.lower() == '.xlsx':
                    df = pd.read_excel(file_path_obj)
                else:
                    df = pd.read_csv(file_path_obj)
                for index, row in df.iterrows():
                    payload = {
                        "overview": row.get('overview', ''),
                        "vote_average": float(row.get('vote_average', 0.0)),
                        "vote_count": int(row.get('vote_count', 0))
                    }
                    
          
                    response = requests.post(FASTAPI_URL, json=payload, timeout=10)
                    response.raise_for_status()
                    prediction_result = response.json().get('predicted_popularity')

     
                    insert_query = """
                    INSERT INTO prediction_history 
                    (overview, vote_average, vote_count, predicted_popularity)
                    VALUES (%s, %s, %s, %s);
                    """
                    cur.execute(insert_query, (
                        row.get('overview'), 
                        row.get('vote_average'), 
                        row.get('vote_count'), 
                        prediction_result
                    ))
                    total_predictions += 1
                

                shutil.move(file_path_obj, PROCESSED_DATA_FOLDER / file_name)
                print(f"Successfully processed and archived file: {file_name}")

            except requests.exceptions.RequestException as e:
                print(f"API Error processing {file_name}. The file will not be moved. Error: {e}")
            except Exception as e:
                print(f"A general error occurred while processing {file_name}. The file will not be moved. Error: {e}")

        conn.commit()
        cur.close()
        conn.close()
        print(f"Total new predictions stored in DB: {total_predictions}")


    new_files_list = check_for_new_data()
    make_predictions(file_paths=new_files_list)


prediction_dag()
