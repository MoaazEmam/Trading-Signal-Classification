"""
src/models/mlflow_helpers.py
=============================
Helpers for run_evaluation.py.
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
    Logs all evaluation metrics to the active MLflow run.

    Standard  : train_accuracy, test_accuracy, overfit_gap,
                weighted_f1, weighted_precision, weighted_recall, mcc
    Business  : business_signal_precision, business_hold_recall
    AUC-PR    : auc_pr_hold, auc_pr_buy, auc_pr_sell (when available)
    CV        : cv_accuracy (when available)
    """
    mlflow.log_metric("train_accuracy", eval_result["train_accuracy"])
    mlflow.log_metric("test_accuracy", eval_result["test_accuracy"])
    mlflow.log_metric("overfit_gap", eval_result["overfit_gap"])

    mlflow.log_metric("weighted_f1", eval_result["test_weighted_f1"])
    mlflow.log_metric("weighted_precision", eval_result["test_weighted_precision"])
    mlflow.log_metric("weighted_recall", eval_result["test_weighted_recall"])

    mlflow.log_metric("mcc", eval_result["test_mcc"])

    mlflow.log_metric(
        "business_signal_precision", eval_result["test_business_signal_precision"]
    )
    mlflow.log_metric(
        "business_hold_recall", eval_result["test_business_hold_recall"]
    )

    for class_name, auc_val in eval_result.get("auc_pr", {}).items():
        mlflow.log_metric(f"auc_pr_{class_name.lower()}", auc_val)

    if cv_accuracy is not None:
        mlflow.log_metric("cv_accuracy", cv_accuracy)
