"""
src/models/evaluate.py
======================
Evaluation module for Trading Signal Classification.

Metrics
-------
Accuracy              : baseline sanity check
Weighted F1/P/R       : single averaging strategy weighted by class support
MCC                   : primary metric — robust to class imbalance
AUC-PR per class      : area under precision-recall curve (one-vs-rest)
Business metrics      : signal_precision (Buy+Sell), hold_recall
Confusion matrix      : raw matrix + per-class TP/TN/FP/FN

Classes : 0 = Hold | 1 = Buy | 2 = Sell
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.models.evaluate_helpers import (
    ALL_LABELS,
    compute_auc_pr,
    compute_metrics,
    confusion_matrix_breakdown,
)

logger = logging.getLogger(__name__)

HOLD = 0
BUY = 1
SELL = 2


def evaluate_model(
    name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Full evaluation of a trained model on both train and test splits.

    Returns a flat dict with:
      train_{metric}              for every metric in compute_metrics()
      test_{metric}               for every metric in compute_metrics()
      overfit_gap                 train_accuracy - test_accuracy
      auc_pr                      {Hold/Buy/Sell: float} — only when predict_proba available
      confusion_matrix_breakdown  per-class TP/TN/FP/FN (test split)
      classification_report       sklearn text report (test split)
      confusion_matrix            raw matrix as list[list[int]] (test split)
    """
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_metrics = {
        f"train_{k}": v for k, v in compute_metrics(y_train, y_train_pred).items()
    }
    test_metrics = {
        f"test_{k}": v for k, v in compute_metrics(y_test, y_test_pred).items()
    }

    overfit_gap = round(
        train_metrics["train_accuracy"] - test_metrics["test_accuracy"], 4
    )

    cm_breakdown = confusion_matrix_breakdown(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_test_pred, labels=ALL_LABELS)

    auc_pr: dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(x_test)
            auc_pr = compute_auc_pr(y_test, y_prob)
        except Exception as exc:
            logger.warning("AUC-PR computation failed for %s: %s", name, exc)

    logger.info("── %s ──", name.upper())
    logger.info(
        "Overfitting | train_acc=%.4f | test_acc=%.4f | gap=%.4f",
        train_metrics["train_accuracy"],
        test_metrics["test_accuracy"],
        overfit_gap,
    )
    logger.info("MCC (test) = %.4f", test_metrics["test_mcc"])
    logger.info("Weighted F1 (test) = %.4f", test_metrics["test_weighted_f1"])
    logger.info(
        "Signal Precision (test) = %.4f", test_metrics["test_business_signal_precision"]
    )
    logger.info("Hold Recall (test) = %.4f", test_metrics["test_business_hold_recall"])
    logger.info("Classification Report (test):\n%s", report)
    logger.info("Confusion Matrix (test):\n%s", cm)

    return {
        "model_name": name,
        **train_metrics,
        **test_metrics,
        "overfit_gap": overfit_gap,
        "auc_pr": auc_pr,
        "confusion_matrix_breakdown": cm_breakdown,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def build_comparison_table(all_results: list[dict]) -> pd.DataFrame:
    """
    Human-readable comparison DataFrame indexed by model name.

    Columns: Train Acc, Test Acc, Overfit Gap, Weighted F1, MCC,
             Signal Precision (B), Hold Recall (B)
    """
    rows = []
    for r in all_results:
        rows.append(
            {
                "Model": r["model_name"],
                "Train Acc": r["train_accuracy"],
                "Test Acc": r["test_accuracy"],
                "Overfit Gap": r["overfit_gap"],
                "Weighted F1": r["test_weighted_f1"],
                "MCC": r["test_mcc"],
                "Signal Precision (B)": r["test_business_signal_precision"],
                "Hold Recall (B)": r["test_business_hold_recall"],
            }
        )
    return pd.DataFrame(rows).set_index("Model").round(4)
