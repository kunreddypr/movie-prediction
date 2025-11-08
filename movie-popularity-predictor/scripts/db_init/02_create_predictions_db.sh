#!/usr/bin/env bash
set -euo pipefail

# This script runs only on first-time initialization of the Postgres volume.
# It creates a dedicated 'predictions' database and the monitoring tables used by Grafana.

echo "Ensuring predictions database exists..."
DB_EXISTS=$(psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "SELECT 1 FROM pg_database WHERE datname='predictions'")
if [[ "$DB_EXISTS" != "1" ]]; then
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "CREATE DATABASE predictions;"
  echo "Created database 'predictions'."
else
  echo "Database 'predictions' already exists."
fi

echo "Ensuring monitoring schema exists in 'predictions'..."
psql -U "$POSTGRES_USER" -d predictions -v ON_ERROR_STOP=1 <<'SQL'
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

CREATE TABLE IF NOT EXISTS model_feature_baselines (
    feature_name TEXT PRIMARY KEY,
    baseline_mean DOUBLE PRECISION,
    baseline_std DOUBLE PRECISION,
    baseline_p95 DOUBLE PRECISION,
    reference_sample_size INTEGER,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

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

ALTER TABLE prediction_history ADD COLUMN IF NOT EXISTS source_file TEXT;
ALTER TABLE prediction_history ADD COLUMN IF NOT EXISTS ingestion_event_id INTEGER;
SQL

echo "Predictions database initialization complete."

