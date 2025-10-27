# Use a slim, stable Python base image
# Aligned Python version with the Airflow Dockerfile for consistency
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install utilities used by healthchecks and diagnostics
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the app-specific requirements file
COPY requirements-app.txt .

# FIX: Use 'python -m pip' to ensure the correct pip is used,
# which resolves the ModuleNotFoundError for nltk.
RUN python -m pip install --no-cache-dir -r requirements-app.txt \
    && python -m nltk.downloader wordnet stopwords

# Copy the rest of the application code into the container
COPY ./app ./app
COPY ./scripts ./scripts

# Expose the ports for the API and the webapp
EXPOSE 8000 8501

# The command to run the application will be provided by docker-compose.yml

