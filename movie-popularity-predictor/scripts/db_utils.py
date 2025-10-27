# scripts/db_utils.py

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """Establishes and returns a connection to the predictions Postgres database.
    Prefers PREDICTIONS_* env vars, falls back to POSTGRES_*.
    """
    host = os.getenv("PREDICTIONS_HOST", os.getenv("POSTGRES_HOST", "postgres_db"))
    port = int(os.getenv("PREDICTIONS_PORT", os.getenv("POSTGRES_PORT", 5432)))
    database = os.getenv("PREDICTIONS_DB", os.getenv("POSTGRES_DB", "airflow"))
    user = os.getenv("PREDICTIONS_USER", os.getenv("POSTGRES_USER", "airflow"))
    password = os.getenv("PREDICTIONS_PASSWORD", os.getenv("POSTGRES_PASSWORD", "airflow"))
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
    )
    return conn

def create_predictions_table():
    """Ensures the prediction_history table exists (used by the app and DAG)."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id SERIAL PRIMARY KEY,
            overview TEXT,
            vote_average REAL,
            vote_count INTEGER,
            predicted_popularity REAL,
            prediction_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_table_query)
        conn.commit()
        print("Table 'prediction_history' ensured/created successfully.")
        cur.close()
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL or creating table:", error)
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    create_predictions_table()
    print("Database setup script executed.")
