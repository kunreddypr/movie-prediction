"""
Lightweight smoke tests for the FastAPI app.

Run locally (outside Docker):
  $ python -m pip install -r requirements.txt
  $ python app/test_api_smoke.py

Or, if the API container is running, you can switch BASE_URL below
to the container host URL and run this as a simple client.
"""

from pathlib import Path
import sys
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    # Prefer in-process test to avoid network; falls back to HTTP if import fails
    from fastapi.testclient import TestClient  # type: ignore
    from app.api import app  # type: ignore
    _inprocess = True
except Exception:
    _inprocess = False

import json
import os
import requests


def _http_predict(base_url: str, payload: Dict[str, Any]) -> requests.Response:
    return requests.post(f"{base_url}/predict", json=payload, timeout=10)


def run_smoke_tests() -> int:
    payload = {
        "overview": "A brave hero saves the world from impending doom.",
        "vote_average": 7.8,
        "vote_count": 3200,
    }

    if _inprocess:
        client = TestClient(app)
        r = client.get("/")
        assert r.status_code == 200 and r.json().get("status"), "Root endpoint failed"

        r = client.post("/predict", json=payload)
        assert r.status_code == 200, f"Predict failed: {r.text}"
        body = r.json()
        assert "predicted_popularity" in body, "Missing predicted_popularity"
        print("In-process API smoke test passed.")
        return 0

    base = os.getenv("API_URL", "http://127.0.0.1:8000")
    r = requests.get(f"{base}/", timeout=5)
    if r.status_code != 200:
        print(f"Root endpoint unhealthy: {r.status_code} {r.text}")
        return 1
    r = _http_predict(base, payload)
    if r.status_code != 200:
        print(f"Predict failed: {r.status_code} {r.text}")
        return 1
    print("HTTP API smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_smoke_tests())

