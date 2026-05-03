FROM python:3.13-slim

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    curl \
    git \
    libgomp1 \
    make \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.0
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app
ENV PYTHONPATH=/app

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --without dev

COPY . .

RUN mkdir -p predictions/history data/raw data/processed models/artifacts

EXPOSE 8000 8501