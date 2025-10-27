# Multi-stage build that keeps the runtime layer minimal
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

FROM base AS deps

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader --dir ${NLTK_DATA} wordnet stopwords

FROM deps AS artifacts

COPY app /app/app
RUN python -m app.train_model \
    && gzip -9 -c /app/app/movies.csv > /app/app/movies.csv.gz \
    && rm -f /app/app/movies.csv

FROM base AS runtime

# Runtime dependencies that are not included in python:3.11-slim but required by scikit-learn wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=deps /usr/local /usr/local
COPY --from=artifacts /app/app /app/app
RUN mkdir -p /app/scripts
COPY scripts/run-appstack.sh /app/scripts/run-appstack.sh

RUN chmod +x /app/scripts/run-appstack.sh

EXPOSE 8000 8501

# The command is supplied by docker compose so we only provide a working directory
WORKDIR /app/app
