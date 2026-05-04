"""
src/models/ml_flow.py
=====================
MLflow experiment tracking wrapper for Trading Signal Classification.

Reads pre-computed results from training_results.json and logs them to MLflow.
Does NOT re-train or re-evaluate — all metrics come from the JSON.

Per-run logging
---------------
Hyperparameters : model_name, all tuned params
Metrics         : train_accuracy, test_accuracy, overfit_gap, train_time_s,
                  cv_accuracy (when available)
Artifacts       : trained model .pkl

Post-all-runs
-------------
Artifacts       : model_comparison.csv

Usage
-----
    poetry run python -m src.models.ml_flow
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src.config import settings
from src.models.trainer import ARTIFACT_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_NAME = "Trading-Signal-Classification"


def _safe(value: Any) -> float | None:
    """Return None for NaN/null, otherwise the float value."""
    if value is None:
        return None
    try:
        return None if math.isnan(float(value)) else float(value)
    except (TypeError, ValueError):
        return None


def run_mlflow_tracking() -> None:
    """
    Logs one MLflow run per model using metrics already stored in
    training_results.json. No data files or re-training required.
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    desired_artifact_loc = settings.mlflow_artifact_root or str(
        _PROJECT_ROOT / "mlruns"
    )
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=desired_artifact_loc
        )
    elif exp.artifact_location != desired_artifact_loc:
        # MLflow has no public API to update artifact_location; patch the DB directly
        db_path = settings.mlflow_tracking_uri.replace("sqlite:///", "")
        if not Path(db_path).is_absolute():
            db_path = str(_PROJECT_ROOT / db_path)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE experiments SET artifact_location=? WHERE experiment_id=?",
                (desired_artifact_loc, exp.experiment_id),
            )
        logger.info(
            "Patched artifact_location from %s to %s",
            exp.artifact_location,
            desired_artifact_loc,
        )
    mlflow.set_experiment(EXPERIMENT_NAME)

    results_path = ARTIFACT_DIR / "training_results.json"
    with open(results_path) as f:
        training_results: dict[str, Any] = json.load(f)["models"]

    sep = "=" * 60
    logger.info(sep)
    logger.info("MLflow TRACKING START — %d models", len(training_results))
    logger.info("Experiment: %s", EXPERIMENT_NAME)
    logger.info(sep)

    rows = []

    for name, tr in training_results.items():
        logger.info("--- %s ---", name.upper())

        raw_path = Path(tr["artifact_path"])
        model_path = raw_path if raw_path.is_absolute() else _PROJECT_ROOT / raw_path
        if not model_path.exists():
            model_path = MODELS_DIR / f"{name}.pkl"

        train_acc = _safe(tr.get("train_accuracy"))
        test_acc = _safe(tr.get("test_accuracy"))
        cv_acc = _safe(tr.get("cv_accuracy"))
        train_time = _safe(tr.get("train_time_s"))

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            for param, value in (tr.get("best_params") or {}).items():
                mlflow.log_param(param, value)

            metrics: dict[str, float] = {}
            if train_acc is not None:
                metrics["train_accuracy"] = train_acc
            if test_acc is not None:
                metrics["test_accuracy"] = test_acc
            if train_acc is not None and test_acc is not None:
                metrics["overfit_gap"] = train_acc - test_acc
            if cv_acc is not None:
                metrics["cv_accuracy"] = cv_acc
            if train_time is not None:
                metrics["train_time_s"] = train_time

            if metrics:
                mlflow.log_metrics(metrics)

            if model_path.exists():
                mlflow.log_artifact(str(model_path), artifact_path="models")
            else:
                logger.warning(
                    "%s: .pkl not found at %s — skipping artifact", name, model_path
                )

        rows.append(
            {
                "model": name,
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "cv_accuracy": cv_acc,
                "overfit_gap": (
                    (train_acc - test_acc) if (train_acc and test_acc) else None
                ),
                "train_time_s": train_time,
            }
        )
        logger.info("MLflow run complete: %s", name)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(sep)
    comp = pd.DataFrame(rows).set_index("model")
    logger.info("\nModel Comparison:\n%s", comp.to_string())

    comp_path = ARTIFACT_DIR / "model_comparison.csv"
    comp.to_csv(comp_path)

    with mlflow.start_run(run_name="summary"):
        mlflow.log_artifact(str(comp_path), artifact_path="reports")
        best = comp["test_accuracy"].idxmax()
        mlflow.log_param("best_model_by_test_accuracy", best)
        logger.info("Best by test accuracy: %s", best)

    logger.info(sep)
    logger.info("MLflow TRACKING COMPLETE")
    logger.info(sep)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    run_mlflow_tracking()
