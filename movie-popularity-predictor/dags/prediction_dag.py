from __future__ import annotations

from airflow.decorators import dag, task
from datetime import datetime
import os
import shutil
import pandas as pd
import requests


from scripts.db_utils import get_db_connection


DATA_FOLDER = "/opt/airflow/data"
GOOD_DATA_FOLDER = os.path.join(DATA_FOLDER, 'good-data')
PROCESSED_DATA_FOLDER = os.path.join(DATA_FOLDER, 'processed-data')


FASTAPI_URL = "http://api:8000/predict"

@dag(
    dag_id="movie_prediction_job",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["prediction", "api"]
)
def prediction_dag():
    
    os.makedirs(GOOD_DATA_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)

    @task
    def check_for_new_data() -> list[str]:
        """Checks for new files in good-data. Returns list of file paths."""
        new_files = [f for f in os.listdir(GOOD_DATA_FOLDER) if f.endswith('.csv')]
        
        if not new_files:
            print("No new files found in good-data. Skipping downstream tasks.")
            return [] 

        file_paths = [os.path.join(GOOD_DATA_FOLDER, f) for f in new_files]
        print(f"Found {len(file_paths)} new files for prediction.")
        return file_paths

    @task
    def make_predictions(file_paths: list):
       
        if not file_paths:
            print("No files to process. Task completed.")
            return

        conn = get_db_connection()
        cur = conn.cursor()
        total_predictions = 0

        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            print(f"Processing file: {file_name}")
            
            try:
                df = pd.read_csv(file_path)
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
                

                shutil.move(file_path, os.path.join(PROCESSED_DATA_FOLDER, file_name))
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
