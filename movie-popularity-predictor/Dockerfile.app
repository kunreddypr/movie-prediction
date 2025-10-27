# Multi-stage build to keep the runtime image small
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

FROM base AS deps

# Build-essential is installed only in this stage so compiled wheels can be built
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader --dir ${NLTK_DATA} wordnet stopwords

FROM base AS runtime

COPY --from=deps /usr/local /usr/local

COPY ./app ./app
COPY ./scripts ./scripts

RUN python -m app.train_model

EXPOSE 8000 8501

# The service command is defined in docker-compose
