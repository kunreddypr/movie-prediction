# scripts/db_utils.py

import os
from typing import Mapping, Optional

import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

load_dotenv()


def get_db_connection():
    """Establish and return a connection to the predictions Postgres database."""

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


def _ensure_column(cur: psycopg2.extensions.cursor, table: str, column: str, definition: str) -> None:
    """Add a column to ``table`` when it does not already exist."""

    cur.execute(
        sql.SQL("""
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = %s
              AND column_name = %s
        """),
        (table, column),
    )
    if cur.fetchone():
        return
    cur.execute(sql.SQL("ALTER TABLE {table} ADD COLUMN {column} {definition}" ).format(
        table=sql.Identifier(table),
        column=sql.Identifier(column),
        definition=sql.SQL(definition),
    ))


def ensure_prediction_schema(conn: Optional[psycopg2.extensions.connection] = None) -> None:
    """Create the database objects required for monitoring if needed."""

    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_events (
                    id SERIAL PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    ingestion_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    total_rows INTEGER NOT NULL,
                    valid_rows INTEGER NOT NULL,
                    invalid_rows INTEGER NOT NULL,
                    missing_required_feature_count INTEGER NOT NULL,
                    missing_overview_count INTEGER NOT NULL,
                    missing_vote_average_count INTEGER NOT NULL,
                    missing_vote_count INTEGER NOT NULL,
                    negative_vote_count INTEGER NOT NULL,
                    zero_vote_average_count INTEGER NOT NULL,
                    duplicate_id_count INTEGER NOT NULL,
                    processing_seconds DOUBLE PRECISION
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_runs (
                    id SERIAL PRIMARY KEY,
                    source_file TEXT,
                    ingestion_event_id INTEGER,
                    total_predictions INTEGER NOT NULL,
                    zero_prediction_count INTEGER NOT NULL,
                    low_prediction_count INTEGER NOT NULL,
                    high_prediction_count INTEGER NOT NULL,
                    average_prediction DOUBLE PRECISION,
                    std_prediction DOUBLE PRECISION,
                    average_vote_average DOUBLE PRECISION,
                    average_vote_count DOUBLE PRECISION,
                    run_started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    run_finished_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS model_feature_baselines (
                    feature_name TEXT PRIMARY KEY,
                    baseline_mean DOUBLE PRECISION,
                    baseline_std DOUBLE PRECISION,
                    baseline_p95 DOUBLE PRECISION,
                    reference_sample_size INTEGER,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id SERIAL PRIMARY KEY,
                    overview TEXT,
                    vote_average REAL,
                    vote_count INTEGER,
                    predicted_popularity REAL,
                    prediction_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    source_file TEXT,
                    ingestion_event_id INTEGER
                );
                """
            )

            _ensure_column(cur, "prediction_history", "source_file", "TEXT")
            _ensure_column(cur, "prediction_history", "ingestion_event_id", "INTEGER")

        conn.commit()
    finally:
        if close_conn:
            conn.close()


def store_feature_baselines(statistics_by_feature: Mapping[str, Mapping[str, float]]) -> None:
    """Persist training-time baseline statistics for drift monitoring."""

    conn = get_db_connection()
    try:
        ensure_prediction_schema(conn)
        with conn.cursor() as cur:
            for feature_name, stats in statistics_by_feature.items():
                cur.execute(
                    """
                    INSERT INTO model_feature_baselines (
                        feature_name,
                        baseline_mean,
                        baseline_std,
                        baseline_p95,
                        reference_sample_size,
                        updated_at
                    ) VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (feature_name)
                    DO UPDATE SET
                        baseline_mean = EXCLUDED.baseline_mean,
                        baseline_std = EXCLUDED.baseline_std,
                        baseline_p95 = EXCLUDED.baseline_p95,
                        reference_sample_size = EXCLUDED.reference_sample_size,
                        updated_at = CURRENT_TIMESTAMP;
                    """,
                    (
                        feature_name,
                        float(stats.get("mean", 0.0)),
                        float(stats.get("std", 0.0)),
                        float(stats.get("p95", 0.0)),
                        int(stats.get("count", 0)),
                    ),
                )
        conn.commit()
    finally:
        conn.close()


if __name__ == '__main__':
    ensure_prediction_schema()
    print("Database setup script executed.")
