#!/usr/bin/env bash
set -euo pipefail

# This script runs only on first-time initialization of the Postgres volume.
# It creates a dedicated 'predictions' database and the prediction_history table.

echo "Ensuring predictions database exists..."
DB_EXISTS=$(psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tAc "SELECT 1 FROM pg_database WHERE datname='predictions'")
if [[ "$DB_EXISTS" != "1" ]]; then
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "CREATE DATABASE predictions;"
  echo "Created database 'predictions'."
else
  echo "Database 'predictions' already exists."
fi

echo "Ensuring prediction_history table exists in 'predictions'..."
psql -U "$POSTGRES_USER" -d predictions -v ON_ERROR_STOP=1 <<'SQL'
CREATE TABLE IF NOT EXISTS prediction_history (
    id SERIAL PRIMARY KEY,
    overview TEXT,
    vote_average REAL,
    vote_count INTEGER,
    predicted_popularity REAL,
    prediction_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
SQL

echo "Predictions database initialization complete."

