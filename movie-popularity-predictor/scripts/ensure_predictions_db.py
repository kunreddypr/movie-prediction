import os
import psycopg2


def env(name: str, default: str | None = None):
    val = os.getenv(name)
    return val if val is not None else default


def connect(dbname: str):
    host = env("PREDICTIONS_HOST", env("POSTGRES_HOST", "127.0.0.1"))
    port = int(env("PREDICTIONS_PORT", env("POSTGRES_PORT", "5432")))
    user = env("PREDICTIONS_USER", env("POSTGRES_USER", "airflow"))
    password = env("PREDICTIONS_PASSWORD", env("POSTGRES_PASSWORD", "airflow"))
    return psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)


def ensure_db_and_table():
    server_db = env("POSTGRES_DB", "postgres")
    predictions_db = env("PREDICTIONS_DB", "predictions")

    # 1) Ensure DB exists
    with connect(server_db) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (predictions_db,))
            exists = cur.fetchone() is not None
            if not exists:
                cur.execute(f"CREATE DATABASE {predictions_db};")
                print(f"Created database '{predictions_db}'.")
            else:
                print(f"Database '{predictions_db}' already exists.")

    # 2) Ensure table exists
    with connect(predictions_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id SERIAL PRIMARY KEY,
                    overview TEXT,
                    vote_average REAL,
                    vote_count INTEGER,
                    predicted_popularity REAL,
                    prediction_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()
            print("Ensured table 'prediction_history' exists.")


if __name__ == "__main__":
    ensure_db_and_table()
    print("Done.")

