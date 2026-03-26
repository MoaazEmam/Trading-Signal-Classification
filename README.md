# Trading-Signal-Classification

A machine-learning pipeline for classifying trading signals into buy/sell/hold. The project covers the full ML lifecycle: data ingestion from Kaggle, validation, preprocessing, feature engineering, model training, and evaluation — with MLflow experiment tracking, CI via GitHub Actions, and a serving layer via FastAPI.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Local Setup](#local-setup)
- [Environment Variables](#environment-variables)
- [Running the Pipeline](#running-the-pipeline)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [MLflow Tracking](#mlflow-tracking)

---

## Project Structure

```
cyberattack-detection/
├── .github/workflows/       # GitHub Actions CI config
├── data/
│   ├── raw/                 # Downloaded raw data (gitignored)
│   ├── processed/           # Final model-ready data (gitignored)
│   └── samples/             # Tiny committed CSVs used only in tests
├── notebooks/               # Exploratory & phase notebooks (not used in pipeline)
├── reports/figures/         # Generated plots and evaluation figures
├── src/
│   ├── config.py            # Centralised settings (loaded from .env)
│   ├── data/
│   │   ├── ingestion.py     # Downloads dataset from Kaggle
│   │   ├── validation.py    # Great Expectations data checks
│   │   └── preprocessing.py # Cleaning, encoding, train/test split
│   ├── features/
│   │   ├── engineering.py   # Feature construction
│   │   └── selection.py     # Feature selection helpers
│   ├── models/
│   │   ├── train.py         # Model training + MLflow logging
│   │   └── evaluate.py      # Metrics, confusion matrix, reports
│   └── serving/
│       └── api.py           # FastAPI inference endpoint
└── tests/
    ├── unit/                # Fast, isolated unit tests
    └── integration/         # End-to-end pipeline tests
```

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.13.x | [python.org](https://www.python.org/downloads/) |
| Poetry | ≥ 1.8 | `pip install poetry` |
| Git | any | [git-scm.com](https://git-scm.com/) |
| Kaggle account | — | [kaggle](https://www.kaggle.com/) — needed for data ingestion |
| FRED account | — | [fred](https://fred.stlouisfed.org/) — needed for data ingestion |

> **Python version note:** `pyproject.toml` pins `python = ">=3.13,<3.14"`.

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/MoaazEmam/Cyber-Attack-Classification.git
cd Cyber-Attack-Classification
```

### 2. Install dependencies

```bash
poetry install
```

This creates a virtual environment and installs all production and dev dependencies (pytest, ruff, black, ipykernel, etc.).

### 3. Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env` with your real values (see [Environment Variables](#environment-variables) below). This file is gitignored and must never be committed.

### 4. Install pre-commit hooks

```bash
poetry run pre-commit install --hook-type pre-commit --hook-type pre-push
```

This sets up two hooks:
- **pre-commit**: runs `ruff` linting and formatting on every commit
- **pre-push**: runs the unit test suite before any push to remote

### 5. Verify the setup

```bash
poetry run pytest tests/unit/ -q
```

All unit tests should pass against the sample data in `data/samples/`.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values:

```dotenv
# Kaggle credentials — required for `make ingest`
# Get these from https://www.kaggle.com/settings → API → Create New Token
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# MLflow tracking server — use localhost for local runs
MLFLOW_TRACKING_URI=http://localhost:5000
```

> **Kaggle API key:** Download `kaggle.json` from your Kaggle account settings. The values inside map directly to `KAGGLE_USERNAME` and `KAGGLE_KEY`.

---

## Running the Pipeline

Each stage can be run individually or chained. All commands use the Poetry virtual environment automatically.

```bash
# Download raw data from Kaggle → data/raw/
make ingest

# Validate raw data with Great Expectations
make validate

# Run both ingestion + validation (Phase 2 shortcut)
make phase2

# Clean, encode, and split data → data/processed/
make preprocess

# Build features → updates data/processed/
make engineer

# Train model and log run to MLflow
make train

# Evaluate model and write reports to reports/figures/
make evaluate
```

To run the full pipeline end-to-end:

```bash
make phase2 && make preprocess && make engineer && make train && make evaluate
```

---

## Testing

```bash
# All tests with coverage report (must stay above 60%)
make test

# Unit tests only (fast, no external deps)
make test-unit

# Integration tests only
make test-integration
```

Coverage is measured on `src/` and reported to the terminal and `coverage.xml`. The pipeline excludes `ingestion.py`, `api.py`, `config.py`, and `__init__.py` files from coverage since they require live external services.

---

## Code Quality

```bash
# Check for lint errors
make lint

# Auto-fix formatting
make format

# Clean up .pyc files and coverage artifacts
make clean
```

The project uses **ruff** for linting and import sorting, and **black** for formatting (both set to 88-char line length). Notebooks in `notebooks/` are excluded from linting.

---

## MLflow Tracking

Start a local MLflow server before training:

```bash
poetry run mlflow ui --port 5000
```

Then open [http://localhost:5000](http://localhost:5000) in your browser. All `make train` and `make evaluate` runs will log parameters, metrics, and artifacts there automatically (controlled by `MLFLOW_TRACKING_URI` in `.env`).
