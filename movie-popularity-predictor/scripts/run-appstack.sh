#!/usr/bin/env bash
set -euo pipefail

cd /app/app

uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Give the API a brief moment to start to avoid initial timeouts
sleep 1

streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
WEB_PID=$!

# If either process exits, stop the other
wait -n "$API_PID" "$WEB_PID"
kill -TERM "$API_PID" "$WEB_PID" 2>/dev/null || true
wait || true

