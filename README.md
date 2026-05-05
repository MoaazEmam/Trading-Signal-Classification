# Trading Signal Classification

**CMPS344 Applied Data Science — Spring 2026**

A full ML pipeline that classifies daily stock movements as **Buy / Hold / Sell** for 491 large-cap equities. The project covers the complete data-science lifecycle: multi-source ingestion, cleaning, feature engineering, model training, MLflow experiment tracking, a FastAPI serving layer, a Streamlit dashboard, and deployment to Azure Container Apps via GitHub Actions CI/CD.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Team](8)
- [Prerequisites](#prerequisites)
- [Local Setup](#local-setup)
- [Environment Variables](#environment-variables)
- [Running the Pipeline](#running-the-pipeline)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [MLflow Tracking](#mlflow-tracking)
- [Docker](#docker)
- [Deployment](#deployment)
- [CI/CD](#cicd)

---

## Project Structure

```
Trading-Signal-Classification/
├── .github/workflows/
│   ├── ci.yml               # Lint + test on every push / PR
│   └── deploy.yml           # Build, push image, deploy to Azure
├── app/
│   ├── streamlit_app.py     # Main EDA dashboard page
│   ├── loaders.py           # Cached data loaders for Streamlit
│   └── pages/               # Multi-page Streamlit app
│       ├── 1_Feature_Explorer.py
│       ├── 2_MI_Importance.py
│       ├── 3_Class_Conditional.py
│       ├── 4_Correlation.py
│       ├── 5_Drift.py
│       ├── 6_Predictions.py
│       └── 7_Model_Performance.py   # MLflow runs + backtest results
├── data/
│   ├── raw/                 # Downloaded raw data (gitignored)
│   ├── processed/           # Cleaned / split / transformed data (gitignored)
│   └── samples/             # Small committed CSVs used only in tests
├── docker/
│   └── entrypoint.sh        # Container startup (routes SERVICE env var)
├── models/
│   ├── *.pkl                # Trained model files
│   └── artifacts/           # JSON results, confusion matrices, PKL artifacts
├── notebooks/               # Exploratory notebooks (ingestion, training, EDA …)
├── predictions/             # Latest and historical daily predictions (JSON/CSV)
├── reports/                 # Cleaning log, validation report
├── scheduler/
│   └── entrypoint.sh        # Cron job: runs prediction pipeline nightly
├── scripts/
│   └── run_backtest.py      # Standalone backtest runner
├── src/
│   ├── config.py            # Centralised settings (loaded from .env)
│   ├── data/
│   │   ├── ingestion.py     # Kaggle + Yahoo Finance + FRED ingestion
│   │   ├── validation.py    # Great Expectations data quality checks
│   │   ├── cleaning.py      # Cleaner class (accuracy / consistency / completeness)
│   │   ├── labeling.py      # Triple-barrier labeling (Buy / Hold / Sell)
│   │   ├── preprocessing.py # Orchestrates cleaning → labeling → splitting
│   │   └── splitting.py     # Temporal 80/20 train-test split
│   ├── eda/                 # EDA loaders, stats helpers, Plotly plots
│   ├── features/
│   │   ├── engineering.py   # Technical indicators (RSI, MACD, Bollinger …)
│   │   ├── transform.py     # Winsorisation + StandardScaler pipeline
│   │   ├── pipeline.py      # sklearn Pipeline builder
│   │   └── features_selection/  # MI + RFECV feature selection
│   ├── models/
│   │   ├── trainer.py       # GridSearch / RandomSearch + 5-model training
│   │   ├── evaluate.py      # Full metric evaluation (MCC, F1, AUC-PR, business)
│   │   ├── run_evaluation.py # Eval + MLflow logging (clears previous runs first)
│   │   ├── ml_flow.py       # MLflow logging from training_results.json
│   │   └── mlflow_helpers.py
│   ├── pipelines/
│   │   ├── train.py         # End-to-end training pipeline
│   │   └── predict.py       # Daily prediction pipeline
│   ├── backtesting/
│   │   ├── engine.py        # Simulated trading engine
│   │   └── metrics.py       # Sharpe, win rate, profit factor …
│   └── serving/
│       └── api.py           # FastAPI inference + backtest endpoints
├── tests/
│   ├── conftest.py
│   └── unit/                # pytest unit tests
├── mlflow.db                # MLflow SQLite tracking database (baked into image)
├── Makefile                 # Pipeline automation targets
├── pyproject.toml           # Poetry dependency manifest
├── poetry.lock
├── Dockerfile.dev           # Development image
├── Dockerfile.prod          # Production image (pulls models from Azure)
├── docker-compose.dev.yml   # Local dev stack (api + streamlit + mlflow + scheduler)
├── docker-compose.prod.yml  # Local prod-image testing stack
└── .env.example             # Template for required environment variables
```

---

## Team

| Name | Student ID | Email |
|------|-----------|-------|
| Moaaz Emam | — | moaaz.emam06@eng-st.cu.edu.eg |

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.13.x | `pyproject.toml` pins `>=3.13,<3.14` |
| Poetry | ≥ 1.8 | `pip install poetry` |
| Git | any | — |
| Kaggle account | — | needed for `make ingest` |
| FRED account | — | needed for `make ingest` |
| Docker + Compose | ≥ 24 | only for container workflow |

---

## Local Setup

### 1. Clone

```bash
git clone https://github.com/MoaazEmam/Trading-Signal-Classification.git
cd Trading-Signal-Classification
```

### 2. Install dependencies

```bash
poetry install
```

### 3. Configure environment variables

```bash
cp .env.example .env
# edit .env with your Kaggle + FRED credentials
```

### 4. Install pre-commit hooks

```bash
poetry run pre-commit install --hook-type pre-commit --hook-type pre-push
```

Two hooks are registered:
- **pre-commit** — `ruff` lint + format check on every commit
- **pre-push** — unit test suite before any push

### 5. Verify

```bash
poetry run pytest tests/unit/ -q
```

All unit tests should pass against the sample data in `data/samples/`.

---

## Environment Variables

Copy `.env.example` to `.env`:

```dotenv
# Kaggle — required for `make ingest`
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_API_TOKEN=your_kaggle_api_key

# FRED — required for macroeconomic features
FRED_API_KEY=your_fred_api_key

# MLflow — use localhost for local runs; set to http://mlflow:5000 in Docker
MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# FastAPI base URL used by the Streamlit dashboard
API_URL=http://localhost:8000
```

---

## Running the Pipeline

Individual stages or the full pipeline can be run through `make`:

```bash
# Download raw data from Kaggle + Yahoo Finance + FRED → data/raw/
make ingest

# Validate raw data with Great Expectations
make validate

# Clean, label, and split data → data/processed/
make clean-data   # cleaning only
make label        # labeling only (requires cleaned data)
make split        # temporal split (requires labeled data)

# Feature engineering + transformation + selection
make engineer
make transform
make select

# Train all 6 models and save results JSON
make train

# Evaluate all models and log runs to MLflow
make evaluate

# Run daily prediction pipeline
make predict

# Full pipeline end-to-end
make full-pipeline
```

---

## Streamlit Dashboard

```bash
# Local (no Docker)
make app
# then open http://localhost:8501

# Docker dev stack
docker compose -f docker-compose.dev.yml up --build
# Streamlit → http://localhost:8501
# MLflow UI  → http://localhost:5000
# FastAPI    → http://localhost:8000/docs
```

Dashboard pages:

| Page | Description |
|------|-------------|
| EDA Overview | Train/test timeline, class proportions per company and over time |
| Feature Explorer | Distribution of every engineered feature |
| MI & Importance | Mutual information ranking + model feature importances |
| Class Conditional | Per-class feature distributions |
| Correlation Lab | Pearson correlation heatmap + cluster analysis |
| Drift | Feature distribution shift between train and test |
| Daily Predictions | Latest and historical Buy/Hold/Sell signals |
| Model Performance | MLflow run browser, backtest dashboard, on-demand backtest trigger |

---

## Testing

```bash
# All tests with coverage report (must stay ≥ 60%)
make test

# Unit tests only (fast, no external dependencies)
make test-unit

# Integration tests
make test-integration
```

Coverage is measured on `src/` and reported to the terminal and `coverage.xml`. Files that require live external services (`ingestion.py`, `api.py`, `config.py`, `__init__.py`) are excluded from coverage.

---

## Code Quality

```bash
make lint       # ruff check with auto-fix
make format     # black formatting
make clean      # remove .pyc files and coverage artifacts
```

The project uses **ruff** for linting and import sorting, and **black** for formatting (88-char line length). Notebooks are excluded from linting.

---

## MLflow Tracking

All 6 model runs are persisted in `mlflow.db` and visible in the Streamlit **Model Performance → MLflow Runs** tab.

### Local (without Docker)

```bash
# Start the MLflow UI
make mlflow-server
# Open http://localhost:5000

# Re-log all runs from training_results.json (no retraining)
make mlflow-log

# Re-evaluate all models and log fresh runs
make evaluate
```

### Docker dev stack

The `docker-compose.dev.yml` stack includes an MLflow service that mounts `./mlflow.db`. The Streamlit service connects to it via `MLFLOW_TRACKING_URI=http://mlflow:5000`.

```bash
docker compose -f docker-compose.dev.yml up --build
# MLflow UI → http://localhost:5000
```

---

## Docker

### Development stack

```bash
docker compose -f docker-compose.dev.yml up --build
```

Services started:
- **api** — FastAPI on port 8000
- **streamlit** — Streamlit on port 8501
- **mlflow** — MLflow UI on port 5000
- **scheduler** — nightly cron prediction job

### Production stack (local prod-image testing)

```bash
# Requires ACR_LOGIN_SERVER and IMAGE_TAG env vars
docker compose -f docker-compose.prod.yml up
```

---

## Deployment

The app is deployed to **Azure Container Apps** as three separate container apps sharing the same Docker image:

| Container App | `SERVICE` env var | Port |
|--------------|------------------|------|
| `trading-signal-api` | `api` | 8000 |
| `trading-signal-streamlit` | `streamlit` | 8501 |
| `trading-signal-scheduler` | `scheduler` | — |

The Docker image is built with `Dockerfile.prod` which:
1. Installs all dependencies via Poetry
2. Downloads trained model `.pkl` files from Azure Blob Storage
3. Downloads processed data files from an Azure Files share
4. Copies `mlflow.db` directly into the image so the Streamlit MLflow tab is populated

Models and the `mlruns/` artifact directory are stored in Azure Blob Storage (`tradingmodels` account). `mlflow.db` is baked into the image at build time via `COPY . .` (it is not excluded from `.dockerignore`).

Live URLs:
- **API**: `https://trading-signal-api.redgrass-d6b7e126.southafricanorth.azurecontainerapps.io`
- **Streamlit**: deployed alongside the API on Azure Container Apps

---

## CI/CD

GitHub Actions workflows are in `.github/workflows/`:

| Workflow | Trigger | Steps |
|---------|---------|-------|
| `ci.yml` | Push / PR to `main` or `dev` | Lint (ruff), format check (black), unit tests, full test suite with coverage gate (≥ 60%) |
| `deploy.yml` | Push to `main` | Build Docker image, push to Azure Container Registry, upload MLflow artifacts to Azure Blob, deploy all three container apps |
