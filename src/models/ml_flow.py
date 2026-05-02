"""
src/models/ml_flow.py
=====================
MLflow experiment tracking wrapper for Trading Signal Classification.

Loads pre-trained models and results saved by trainer.py — does NOT re-train.

Per-run logging
---------------
Hyperparameters : model_name, all tuned params, cv_strategy, n_cv_splits
Standard metrics: accuracy, weighted_f1, weighted_precision, weighted_recall
                  macro_f1, macro_precision, macro_recall
                  micro_f1, micro_precision, micro_recall
MCC             : matthews correlation coefficient (robust to imbalance)
Business metrics: business_signal_precision, business_hold_recall
Overfitting     : train_accuracy, overfit_gap
CV              : cv_accuracy (best fold mean), per-fold scores as params
Artifacts       : trained model .pkl, confusion matrix CSV, PR curve CSVs

Post-all-runs
-------------
Artifacts       : model_comparison.csv, paired_ttest_results.csv

Usage
-----
    poetry run python -m src.models.ml_flow
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.models.evaluate import (
    build_comparison_table,
    evaluate_model,
    paired_ttest_model_comparison,
)
from src.models.mlflow_helpers import log_all_metrics, prepare_features
from src.models.trainer import (
    ARTIFACT_DIR,
    COMPANY_COL,
    MODELS_DIR,
    N_CV_SPLITS,
    RANDOM_STATE,
    TARGET_COL,
)

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "Trading-Signal-Classification"
PR_CURVE_DIR = ARTIFACT_DIR / "pr_curves"


def _fold_scores_from_model(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
) -> list[float]:
    """
    Computes per-fold CV accuracy using the model's already-fitted hyperparameters.

    sklearn clones the model (same params, unfitted) and fits it on each fold —
    no hyperparameter search is repeated. Used when fold_scores were not saved
    to training_results.json (e.g. models loaded from Google Drive).
    """
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    scores = cross_val_score(model, x, y, cv=tscv, scoring="accuracy", n_jobs=-1)
    logger.info("CV fold scores (post-hoc): %s", np.round(scores, 4))
    return scores.tolist()


def _save_csv(df: pd.DataFrame, filename: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / filename
    df.to_csv(path)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_mlflow_tracking(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Logs MLflow runs for all models already trained and saved by trainer.py.

    Loads each model from its .pkl path and per-fold CV scores from
    training_results.json — no hyperparameter search is repeated.

    Per-model steps
    ---------------
    1. Load pre-trained model from MODELS_DIR.
    2. Evaluate on train + test (all Lecture 08 metrics + business metrics).
    3. Open MLflow run -> log params, metrics, model artifact, confusion matrix,
       PR curve CSVs.

    After all models
    ----------------
    4. Summary run: model comparison table + Paired T-Test table as artifacts.
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PR_CURVE_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(EXPERIMENT_NAME)

    x_train, y_train = prepare_features(train_df, TARGET_COL, COMPANY_COL)
    x_test, y_test = prepare_features(test_df, TARGET_COL, COMPANY_COL)

    results_path = ARTIFACT_DIR / "training_results.json"
    with open(results_path) as f:
        training_results: dict[str, Any] = json.load(f)["models"]

    all_results: list[dict[str, Any]] = []
    cv_fold_scores: dict[str, list[float]] = {}

    sep = "=" * 60
    logger.info(sep)
    logger.info("MLflow TRACKING START — %d rows | %d features", *x_train.shape)
    logger.info("Experiment: %s", EXPERIMENT_NAME)
    logger.info(sep)

    for name, tr in training_results.items():
        logger.info("--- %s ---", name.upper())

        # artifact_path may point to a different machine; fall back to local MODELS_DIR
        model_path = Path(tr["artifact_path"])
        if not model_path.exists():
            model_path = MODELS_DIR / f"{name}.pkl"
            logger.info("%s: artifact_path not found on this machine, loading from %s", name, model_path)
        model = joblib.load(model_path)

        # cv_accuracy is NaN (not null) when HPO failed or wasn't saved — treat as missing
        cv_acc = tr.get("cv_accuracy")
        if isinstance(cv_acc, float) and np.isnan(cv_acc):
            cv_acc = None

        with mlflow.start_run(run_name=name):
            eval_result = evaluate_model(
                name=name,
                model=model,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                pr_output_dir=PR_CURVE_DIR,
            )

            # log hyperparameters
            mlflow.log_param("model_name", name)
            mlflow.log_param("random_state", RANDOM_STATE)
            for param, value in tr["best_params"].items():
                mlflow.log_param(param, value)

            fold_scores: list[float] = tr.get("fold_scores") or []
            if not fold_scores and name != "voting_classifier":
                logger.info("%s: fold_scores missing — computing via cross_val_score", name)
                fold_scores = _fold_scores_from_model(model, x_train, y_train)
            if fold_scores:
                mlflow.log_param("cv_strategy", "TimeSeriesSplit")
                mlflow.log_param("n_cv_splits", N_CV_SPLITS)
                for i, score in enumerate(fold_scores):
                    mlflow.log_param(f"cv_fold_{i}_score", round(score, 4))
                cv_fold_scores[name] = fold_scores

            log_all_metrics(cv_acc, eval_result)

            mlflow.log_artifact(str(model_path), artifact_path="models")

            cm_df = pd.DataFrame(
                eval_result["confusion_matrix"],
                index=["Hold", "Buy", "Sell"],
                columns=["Pred Hold", "Pred Buy", "Pred Sell"],
            )
            cm_path = ARTIFACT_DIR / f"confusion_matrix_{name}.csv"
            cm_df.to_csv(cm_path)
            mlflow.log_artifact(str(cm_path), artifact_path="confusion_matrices")

            for pr_path in eval_result.get("pr_curve_paths", []):
                mlflow.log_artifact(pr_path, artifact_path="pr_curves")

            all_results.append(eval_result)
            logger.info("MLflow run complete: %s", name)

    # ── Summary: comparison table + Paired T-Test ─────────────────────────────
    logger.info(sep)
    logger.info("POST-TRAINING ANALYSIS")
    logger.info(sep)

    comparison_table = build_comparison_table(all_results)
    logger.info("\nModel Comparison:\n%s", comparison_table.to_string())

    comp_path = _save_csv(comparison_table, "model_comparison.csv")

    with mlflow.start_run(run_name="summary_artifacts"):
        mlflow.log_artifact(str(comp_path), artifact_path="reports")

        if cv_fold_scores:
            ttest_table = paired_ttest_model_comparison(cv_fold_scores)
            logger.info("\nPaired T-Test Results:\n%s", ttest_table.to_string(index=False))
            ttest_path = _save_csv(ttest_table, "paired_ttest_results.csv")
            mlflow.log_artifact(str(ttest_path), artifact_path="reports")

        best_by_acc = comparison_table["Test Acc"].idxmax()
        best_by_mcc = comparison_table["MCC"].idxmax()
        best_by_f1 = comparison_table["Weighted F1"].idxmax()

        mlflow.log_param("best_model_by_test_accuracy", best_by_acc)
        mlflow.log_param("best_model_by_mcc", best_by_mcc)
        mlflow.log_param("best_model_by_weighted_f1", best_by_f1)

        logger.info("Best by Test Acc : %s", best_by_acc)
        logger.info("Best by MCC      : %s", best_by_mcc)
        logger.info("Best by W-F1     : %s", best_by_f1)

    logger.info(sep)
    logger.info("MLflow TRACKING COMPLETE")
    logger.info(sep)

    return all_results


# ══════════════════════════════════════════════════════════════════════════════
# Standalone entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    processed_dir = _PROJECT_ROOT / "data" / "processed"
    train_s = pd.read_csv(processed_dir / "train_val_selected.csv")
    test_s = pd.read_csv(processed_dir / "test_selected.csv")
    run_mlflow_tracking(train_df=train_s, test_df=test_s)
