import os
import sys
import argparse
import psycopg2


def env_or(name: str, default: str):
    return os.getenv(name, default)


def get_args():
    parser = argparse.ArgumentParser(description="Check predictions database and table status.")
    parser.add_argument("--host", default=env_or("PREDICTIONS_HOST", env_or("POSTGRES_HOST", "127.0.0.1")))
    parser.add_argument("--port", type=int, default=int(env_or("PREDICTIONS_PORT", env_or("POSTGRES_PORT", "5432"))))
    parser.add_argument("--db", default=env_or("PREDICTIONS_DB", "predictions"))
    parser.add_argument("--user", default=env_or("PREDICTIONS_USER", env_or("POSTGRES_USER", "airflow")))
    parser.add_argument("--password", default=env_or("PREDICTIONS_PASSWORD", env_or("POSTGRES_PASSWORD", "airflow")))
    parser.add_argument("--server-db", default=env_or("POSTGRES_DB", "postgres"), help="Database to connect for server-level checks")
    return parser.parse_args()


def connect(dbname: str, host: str, port: int, user: str, password: str):
    return psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)


def main() -> int:
    args = get_args()
    ok = True

    # 1) Check that the predictions database exists
    try:
        with connect(args.server_db, args.host, args.port, args.user, args.password) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (args.db,))
                exists = cur.fetchone() is not None
                print(f"Database '{args.db}': {'FOUND' if exists else 'MISSING'}")
                if not exists:
                    ok = False
    except Exception as e:
        print(f"Error checking databases on server DB '{args.server_db}': {e}")
        return 2

    # 2) If DB exists, check table and row count
    if ok:
        try:
            with connect(args.db, args.host, args.port, args.user, args.password) as conn:
                with conn.cursor() as cur:
                    # Check table existence
                    cur.execute("SELECT to_regclass('public.prediction_history')")
                    tbl = cur.fetchone()[0]
                    if not tbl:
                        print("Table 'prediction_history': MISSING")
                        ok = False
                    else:
                        print("Table 'prediction_history': FOUND")
                        cur.execute("SELECT COUNT(*) FROM prediction_history")
                        count = cur.fetchone()[0]
                        print(f"Rows in prediction_history: {count}")
        except Exception as e:
            print(f"Error checking table in DB '{args.db}': {e}")
            return 3

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

