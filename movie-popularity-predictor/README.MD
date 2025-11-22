Movie Popularity Predictor
This project is a complete MLOps pipeline for predicting the popularity of movies. It includes data ingestion and processing orchestrated by Apache Airflow, a machine learning model training service, a REST API for serving predictions, and a web-based UI built with Streamlit for interacting with the model. The entire application is containerized using Docker for easy setup and deployment.

Table of Contents
Project Overview

Architecture

Folder Structure

Technology Stack

Prerequisites

Setup and Installation

How to Run

Accessing the Services

Project Overview
The goal of this project is to provide an end-to-end solution for a machine learning system. The key components are:

Data Ingestion: An Airflow DAG (data_ingestion_dag.py) fetches raw movie data and stores it.

Model Training: A script (train_model.py) preprocesses the data and trains a Random Forest model to predict movie popularity.

Prediction API: A FastAPI application (api.py) exposes an endpoint to get predictions from the trained model.

User Interface: A Streamlit application (streamlit_app.py) provides a user-friendly interface to input movie details and view the predicted popularity.

Architecture
The application is composed of several microservices that are managed by Docker Compose.

postgres_db: A PostgreSQL database that serves as the backend for Apache Airflow.

airflow-webserver & airflow-scheduler: The core components of Apache Airflow for orchestrating data pipelines (DAGs).

app-init: A one-time service that runs the train_model.py script to ensure a model is available at startup.

api: A FastAPI server that loads the trained model and serves predictions over a REST API.

webapp: A Streamlit service that provides the interactive user interface.

Folder Structure
The project is organized into the following key directories:

MOVIE-POPULARITY-PREDICTOR/
│
├── .env                # Environment variables for Docker configuration (passwords, etc.)
├── app/                # Source code for the application (API, UI, model training)
│   ├── api.py          # FastAPI application
│   ├── streamlit_app.py# Streamlit web application
│   ├── train_model.py  # Model training script
│   └── ...
│
├── dags/               # Contains the Airflow DAGs for data pipelines
│   ├── data_ingestion_dag.py
│   └── prediction_dag.py
│
├── data/               # Directory for storing raw, processed, and archived data
│
├── scripts/            # Utility and initialization scripts
│   └── db_init/
│       └── init.sql    # SQL script to initialize the database
│
├── Dockerfile.airflow  # Dockerfile for building the Airflow services
├── Dockerfile.app      # Dockerfile for building the application services
├── docker-compose.yml  # Defines and configures all the project's services
└── requirements.txt    # Python dependencies

Technology Stack
Orchestration: Apache Airflow

Backend & API: FastAPI

Frontend: Streamlit

Database: PostgreSQL

Containerization: Docker & Docker Compose

Machine Learning: Scikit-learn, Pandas

Prerequisites
Docker

Docker Compose (usually included with Docker Desktop)

Setup and Installation
Before running the application, you must configure the environment variables.

If you are working inside GitHub Codespaces or a local Python environment, install
the application dependencies and build the model artefacts once:

```
python -m pip install -r requirements.txt
python -m app.train_model
```

The training script downloads the necessary NLTK resources when internet access
is available and falls back to an embedded stop-word list when it is not, so the
workflow is deterministic even in restricted environments.

Clone the repository:

git clone <your-repository-url>
cd MOVIE-POPULARITY-PREDICTOR

Create the environment file:
Create a file named .env in the root of the project directory. This file will hold all the necessary credentials and configurations.

Populate the .env file:
Copy and paste the following content into your .env file. You can change the credentials if you wish, but these defaults will work out of the box.

# .env file

# PostgreSQL Credentials for both Airflow and the Apps
POSTGRES_USER=airflow
POSTGRES_PASSWORD=airflow
POSTGRES_DB=airflow

# Set the Airflow User ID to avoid permission issues
AIRFLOW_UID=50000

(For Linux/macOS users) Set Directory Permissions:
To prevent permission errors with the mounted log volumes, create the necessary directories and set their ownership before the first run.

mkdir -p ./logs ./plugins
sudo chown -R 50000:50000 ./logs ./plugins ./dags

How to Run
Once the setup is complete, you can start the entire application stack with a single command.

Build and start the services:
Run the following command from the root of the project directory. The --build flag ensures that the Docker images are built from your Dockerfiles.

docker compose up --build

Quick commands:
- Slim app image + combined API/Streamlit runner: `docker compose -f docker-compose.min.yml up --build appstack`
- Minimal app stack with separate API and UI: `docker compose up --build postgres_db api webapp`
- Full environment including Airflow: `docker compose --profile airflow --profile train up --build`

To stop the application:
Press Ctrl + C in the terminal, and then run:

docker-compose down

Accessing the Services
Once the containers are running, you can access the different parts of the application:

Apache Airflow UI: http://localhost:8080

(The default login is airflow / airflow)

FastAPI (API Docs): http://localhost:8000/docs

Streamlit Web App: http://localhost:8501

Monitoring and Validation
- Grafana dashboards are pre-provisioned from grafana/dashboards and become available at http://localhost:3000 (login admin / admin). The ingestion-data and model-drift dashboards read directly from the ingestion_events and prediction_runs tables so the graphs update automatically as your Airflow DAGs run.
- To populate the dashboards with live data, trigger the movie_data_ingestion_pipeline DAG to write ingestion statistics, then trigger movie_prediction_job to push prediction-level statistics. Both DAGs write to the shared Postgres instance used by Grafana.
- Great Expectations-style quality checks are implemented inside the data_ingestion_dag.py task record_data_quality_metrics, which validates required fields, numeric ranges, and duplicates before saving summary counts. Failures stay in the database for Grafana to visualize; the DAG does not move files out of the good-data folder so they remain available for future audits.

Grafana Monitoring Dashboards: http://localhost:3000 (default login admin / admin)

- **Ingested Data Quality Monitoring** – tracks invalid samples, missing required fields and ingestion issue categories in real time.
- **Model Drift and Prediction Health** – monitors serving drift against the training baselines and highlights anomalous prediction behaviour.

Additional Notes

Codespaces / Local Development without Docker
- After installing the dependencies you can launch the API directly with
  `uvicorn app.api:app --reload`.
- Streamlit can be started with `streamlit run app/streamlit_app.py` (ensure the
  API is reachable via the `API_URL` environment variable or defaults to
  `http://127.0.0.1:8000`).
- The FastAPI service retrains the model automatically if the pickle files are
  missing, so a clean checkout is immediately usable.

GitHub Codespaces Quickstart
- Create a Codespace with at least 4 vCPUs so Docker has enough resources.
- In the Codespace terminal, switch to the project directory and export an
  Airflow UID to avoid permission issues:
  ```bash
  cd movie-prediction/movie-popularity-predictor
  export AIRFLOW_UID=50000
  ```
- Create the log and plugin folders with the right ownership:
  ```bash
  mkdir -p logs plugins dags
  sudo chown -R "${AIRFLOW_UID}:${AIRFLOW_UID}" logs plugins dags
  ```
- Build and start the complete stack (PostgreSQL, Airflow, API, Streamlit):
  ```bash
  docker compose --profile airflow --profile train up --build
  ```
- Once the containers are healthy you can trigger the ingestion DAG from the
  Airflow UI (http://127.0.0.1:8080) or from the terminal:
  ```bash
  docker compose exec airflow-webserver airflow dags trigger data_ingestion
  ```
- Trigger the prediction DAG in the same way when a single ready file should be
  scored:
  ```bash
  docker compose exec airflow-webserver airflow dags trigger prediction
  ```
- To inspect stored predictions in PostgreSQL from Codespaces:
  ```bash
  docker compose exec postgres_db psql "$POSTGRES_DB" "$POSTGRES_USER"
  ```
- When finished, shut everything down and clean the containers:
  ```bash
  docker compose down
  ```

Data Volume Persistence
- The Airflow services mount the host `./data` directory to `/opt/airflow/data` so that raw/good/processed files persist across container rebuilds and are easy to inspect.
- Paths used by DAGs:
  - Raw: `data/raw-data`
  - Good: `data/good-data`
  - Processed: `data/processed-data`

Configurable Environment Variables
- Training (app/train_model.py):
  - `SOURCE_DATA_PATH` (default: `movies.csv`)
  - `MODEL_PATH` (default: `random_forest_model.pkl`)
  - `TFIDF_PATH` (default: `tfidf_vectorizer.pkl`)
  - `TEST_SIZE` (default: `0.2`)
  - `RANDOM_STATE` (default: `42`)
  - `MAX_FEATURES` (default: `800`)
  - `N_ESTIMATORS` (default: `8`)
- API (app/api.py):
  - `MODEL_PATH` (default: `random_forest_model.pkl`)
  - `TFIDF_PATH` (default: `tfidf_vectorizer.pkl`)
- Database env for services (compose):
  - `POSTGRES_HOST`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_PORT`

Healthchecks
- The `api` and `webapp` services include HTTP healthchecks. The webapp waits until the API is healthy before starting, improving reliability during startup.

API Smoke Test
- A lightweight smoke test is available at `app/test_api_smoke.py`.
- Run in-process (no Docker):
  - `python -m pip install -r requirements.txt`
  - `python app/test_api_smoke.py`
- Or, when the API is running:
  - `API_URL=http://127.0.0.1:8000 python app/test_api_smoke.py`
  - The smoke test adjusts `sys.path` automatically when executed from within
    the `app/` directory, so running it directly now works in Codespaces and
    other non-Docker environments.

Quick Airflow Commands
- Trigger ingestion DAG (splits movies.csv into 20-row chunks and moves to good-data):
  - `docker-compose exec airflow-webserver airflow dags trigger movie_data_ingestion_pipeline`
- Trigger prediction DAG (calls API and stores predictions in DB):
  - `docker-compose exec airflow-webserver airflow dags trigger movie_prediction_job`

Predictions DB Check Script
- Run from host (defaults assume Docker port mapping 5432:5432):
  - `PREDICTIONS_HOST=127.0.0.1 PREDICTIONS_DB=predictions PREDICTIONS_USER=airflow PREDICTIONS_PASSWORD=airflow python scripts/check_predictions_db.py`
- Run inside a container (webapp, scheduler, or webserver):
  - `docker-compose exec webapp python /app/scripts/check_predictions_db.py`

Minimal 2–3 Container Setup
- If your laptop struggles with the full stack (Airflow + multiple services), use the minimal compose file which runs only Postgres and a single container that hosts both the API and Streamlit.

Steps:
- Ensure your `.env` contains:
  - `POSTGRES_USER=airflow`
  - `POSTGRES_PASSWORD=airflow`
  - `POSTGRES_DB=airflow`
- Start the minimal stack:
  - `docker compose -f docker-compose.min.yml up --build`
- Access:
  - Streamlit UI: `http://localhost:8501`
  - API docs (optional): `http://localhost:8000/docs`

Notes:
- The minimal stack still uses Postgres and will initialize a `predictions` database via `scripts/db_init` on first run.
- To verify the DB from the minimal stack:
  - `docker compose -f docker-compose.min.yml exec appstack python /app/scripts/check_predictions_db.py`

--command to create airflow user while running in the docker 
docker exec -it movie-popularity-predictor-airflow-scheduler-1 airflow users create --username admin --firstname kunreddy --lastname pramod --role Admin --email reddypramod162@gmail.com --password admin
