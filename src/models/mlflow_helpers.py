"""
src/models/mlflow_helpers.py
=============================
Helpers for ml_flow.py.
"""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd


def prepare_features(
    df: pd.DataFrame,
    target_col: str,
    company_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [c for c in [target_col, company_col] if c in df.columns]
    return df.drop(columns=drop_cols), df[target_col]


def log_all_metrics(cv_accuracy: float | None, eval_result: dict[str, Any]) -> None:
    """
    Logs every metric from evaluate_model() to the active MLflow run.

    Minimum required by project spec (Section 5.7):
      - 2 standard metrics  -> accuracy, weighted_f1
      - 2 business metrics  -> business_signal_precision, business_hold_recall

    Additional metrics logged (Lecture 08 coverage):
      - MCC             : robust to class imbalance
      - macro_*         : equal weight per class (detects class neglect)
      - micro_*         : aggregated counts
      - weighted_*      : weighted by class support (primary)
      - train_accuracy  : overfitting detection
      - overfit_gap     : train_acc - test_acc
      - cv_accuracy     : best cross-validation mean score
    """
    # ── 2 required standard metrics ───────────────────────────────────────────
    mlflow.log_metric("accuracy", eval_result["test_accuracy"])
    mlflow.log_metric("weighted_f1", eval_result["test_weighted_f1"])

    # ── 2 required business metrics ───────────────────────────────────────────
    mlflow.log_metric(
        "business_signal_precision",
        eval_result["test_business_signal_precision"],
    )
    mlflow.log_metric(
        "business_hold_recall",
        eval_result["test_business_hold_recall"],
    )

    # ── Additional standard metrics (Lecture 08 — all averaging strategies) ──
    mlflow.log_metric("weighted_precision", eval_result["test_weighted_precision"])
    mlflow.log_metric("weighted_recall", eval_result["test_weighted_recall"])

    mlflow.log_metric("macro_f1", eval_result["test_macro_f1"])
    mlflow.log_metric("macro_precision", eval_result["test_macro_precision"])
    mlflow.log_metric("macro_recall", eval_result["test_macro_recall"])

    mlflow.log_metric("micro_f1", eval_result["test_micro_f1"])
    mlflow.log_metric("micro_precision", eval_result["test_micro_precision"])
    mlflow.log_metric("micro_recall", eval_result["test_micro_recall"])

    # ── MCC (Lecture 08 — robust to imbalance) ────────────────────────────────
    mlflow.log_metric("mcc", eval_result["test_mcc"])

    # ── Overfitting ───────────────────────────────────────────────────────────
    mlflow.log_metric("train_accuracy", eval_result["train_accuracy"])
    mlflow.log_metric("overfit_gap", eval_result["overfit_gap"])

    # ── CV score ──────────────────────────────────────────────────────────────
    if cv_accuracy is not None:
        mlflow.log_metric("cv_accuracy", cv_accuracy)
