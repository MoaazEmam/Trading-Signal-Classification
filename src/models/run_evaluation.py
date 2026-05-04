"""
src/models/run_evaluation.py
============================
Pure evaluation and MLflow logging — no retraining.

Loads each trained model from models/*.pkl, runs full evaluation using
evaluate_model() on the processed train/test splits, then logs all metrics,
parameters, and artifacts to MLflow.

Each run() call clears previous runs from the experiment first so the UI
always shows only the latest evaluation results.

Usage
-----
    poetry run python -m src.models.run_evaluation
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd

from src.backtesting.engine import run_backtest, run_baseline_backtest
from src.config import settings
from src.models.evaluate import (
    build_comparison_table,
    evaluate_model,
)
from src.models.mlflow_helpers import log_all_metrics, prepare_features
from src.models.trainer import ARTIFACT_DIR, COMPANY_COL, MODELS_DIR, TARGET_COL

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "Trading-Signal-Classification"

MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "adaboost",
    "random_forest",
    "lightgbm",
    "voting_classifier",
]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _log_params(name: str, tr: dict[str, Any]) -> None:
    mlflow.log_param("model_name", name)
    for param, value in (tr.get("best_params") or {}).items():
        mlflow.log_param(param, value)


def _align_features(name: str, model: Any, x: pd.DataFrame) -> pd.DataFrame | None:
    if not hasattr(model, "feature_names_in_"):
        return x
    required = list(model.feature_names_in_)
    missing = [c for c in required if c not in x.columns]
    if missing:
        logger.warning(
            "%s: %d features missing from data — skipping. First missing: %s",
            name,
            len(missing),
            missing[:5],
        )
        return None
    return x[required]


def _save_csv(df: pd.DataFrame, filename: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / filename
    df.to_csv(path)
    return path


def _clear_previous_runs(experiment_name: str) -> None:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return

    # delete all runs from the tracking store (all states, not just active)
    existing = client.search_runs(
        [experiment.experiment_id],
        max_results=1000,
        run_view_type=mlflow.entities.ViewType.ALL,
    )
    for run in existing:
        client.delete_run(run.info.run_id)

    # remove artifact directories from mlruns/{experiment_id}/
    mlruns_exp_dir = _PROJECT_ROOT / "mlruns" / experiment.experiment_id
    if mlruns_exp_dir.exists():
        for item in mlruns_exp_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)

    if existing:
        logger.info(
            "Cleared %d previous run(s) from '%s'", len(existing), experiment_name
        )


def _save_best_model_summary(
    best_model: str,
    comparison_table: pd.DataFrame,
    all_results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    best_eval = next((r for r in all_results if r["model_name"] == best_model), {})

    summary = {
        "model_name": best_model,
        "model_path": f"models/{best_model}.pkl",
        "test_f1_macro": float(best_eval.get("test_weighted_f1", 0.0)),
        "test_accuracy": float(best_eval.get("test_accuracy", 0.0)),
        "test_precision_macro": float(best_eval.get("test_weighted_precision", 0.0)),
        "test_recall_macro": float(best_eval.get("test_weighted_recall", 0.0)),
        "test_mcc": float(best_eval.get("test_mcc", 0.0)),
        "selected_at": pd.Timestamp.utcnow().isoformat() + "Z",
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


def run_evaluation(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    _clear_previous_runs(EXPERIMENT_NAME)

    results_path = ARTIFACT_DIR / "training_results.json"
    training_results: dict[str, Any] = {}
    if results_path.exists():
        with open(results_path) as f:
            training_results = json.load(f).get("models", {})

    x_train, y_train = prepare_features(train_df, TARGET_COL, COMPANY_COL)
    x_test, y_test = prepare_features(test_df, TARGET_COL, COMPANY_COL)

    all_results: list[dict[str, Any]] = []

    sep = "=" * 60
    logger.info(sep)
    logger.info(
        "EVALUATION START — %d train rows | %d test rows", len(x_train), len(x_test)
    )
    logger.info("Experiment: %s", EXPERIMENT_NAME)
    logger.info(sep)

    # --- Naive baseline (must run before model loop) ---
    logger.info(sep)
    logger.info("Running naive Buy-everything baseline for comparison")
    try:
        baseline_results = run_baseline_backtest()
        baseline_metrics = baseline_results["metrics"]
        logger.info(
            "Baseline — return=%.2f%% sharpe=%.3f win_rate=%.2f%% trades=%d",
            baseline_metrics["total_return_pct"],
            baseline_metrics["sharpe_ratio"] or 0.0,
            (baseline_metrics["win_rate"] or 0.0) * 100,
            baseline_results["total_trades"],
        )
        with mlflow.start_run(run_name="naive_buy_baseline"):
            mlflow.log_param("model_name", "naive_buy_baseline")
            finite_baseline = {
                f"backtest_{k}": v
                for k, v in baseline_metrics.items()
                if isinstance(v, float) and math.isfinite(v)
            }
            if finite_baseline:
                mlflow.log_metrics(finite_baseline)
    except Exception as exc:
        logger.warning("Baseline backtest failed — skipping: %s", exc)
    logger.info(sep)

    for name in MODEL_NAMES:
        model_path = MODELS_DIR / f"{name}.pkl"
        if not model_path.exists():
            logger.warning("%s: .pkl not found at %s — skipping", name, model_path)
            continue

        logger.info("--- %s ---", name.upper())
        model = joblib.load(model_path)
        tr = training_results.get(name, {})

        cv_acc = tr.get("cv_accuracy")
        if cv_acc is not None:
            try:
                cv_acc = None if math.isnan(float(cv_acc)) else float(cv_acc)
            except (TypeError, ValueError):
                cv_acc = None

        x_train_m = _align_features(name, model, x_train)
        x_test_m = _align_features(name, model, x_test)
        if x_train_m is None:
            continue

        with mlflow.start_run(run_name=name):
            eval_result = evaluate_model(
                name=name,
                model=model,
                x_train=x_train_m,
                y_train=y_train,
                x_test=x_test_m,
                y_test=y_test,
            )

            _log_params(name, tr)
            log_all_metrics(cv_acc, eval_result)

            try:
                backtest_results = run_backtest(model=model, model_name=name)
                backtest_metrics = backtest_results["metrics"]
                finite_metrics = {
                    f"backtest_{k}": v
                    for k, v in backtest_metrics.items()
                    if isinstance(v, float) and math.isfinite(v)
                }
                if finite_metrics:
                    mlflow.log_metrics(finite_metrics)
            except Exception as exc:
                logger.warning("Backtest failed for %s — skipping: %s", name, exc)

            mlflow.log_artifact(str(model_path), artifact_path="models")

            cm_df = pd.DataFrame(
                eval_result["confusion_matrix"],
                index=["Hold", "Buy", "Sell"],
                columns=["Pred Hold", "Pred Buy", "Pred Sell"],
            )
            cm_path = _save_csv(cm_df, f"confusion_matrix_{name}.csv")
            mlflow.log_artifact(str(cm_path), artifact_path="confusion_matrices")

        all_results.append(eval_result)
        logger.info("MLflow run complete: %s", name)

    if not all_results:
        logger.error("No models evaluated — check .pkl files exist in %s", MODELS_DIR)
        return

    logger.info(sep)
    logger.info("POST-EVALUATION SUMMARY")
    logger.info(sep)

    comparison_table = build_comparison_table(all_results)
    logger.info("\nModel Comparison:\n%s", comparison_table.to_string())
    comp_path = _save_csv(comparison_table, "model_comparison.csv")

    best_model = str(comparison_table["MCC"].idxmax())
    best_model_path = ARTIFACT_DIR / "best_model.json"
    _save_best_model_summary(best_model, comparison_table, all_results, best_model_path)

    mlflow.end_run()
    with mlflow.start_run(run_name="summary"):
        mlflow.log_artifact(str(best_model_path), artifact_path="reports")
        mlflow.log_artifact(str(comp_path), artifact_path="reports")
        mlflow.log_param("best_model", best_model)
        for model_name, row in comparison_table.iterrows():
            mlflow.log_metric(f"mcc_{model_name}", row["MCC"])

    logger.info("Best model by MCC: %s", best_model)
    logger.info(sep)
    logger.info("EVALUATION COMPLETE")
    logger.info(sep)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    processed_dir = _PROJECT_ROOT / "data" / "processed"
    train_df = pd.read_csv(processed_dir / "train_val_transformed.csv")
    test_df = pd.read_csv(processed_dir / "test_transformed.csv")
    run_evaluation(train_df=train_df, test_df=test_df)
