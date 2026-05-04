"""
src/models/evaluate_helpers.py
================================
Internal helpers for evaluate.py — not part of the public API.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

HOLD = 0
BUY = 1
SELL = 2
ALL_LABELS = [HOLD, BUY, SELL]
LABEL_NAMES = {HOLD: "Hold", BUY: "Buy", SELL: "Sell"}


def confusion_matrix_breakdown(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, dict[str, int]]:
    """Per-class TP, TN, FP, FN via one-vs-rest decomposition."""
    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
    breakdown: dict[str, dict[str, int]] = {}

    for i, label in enumerate(ALL_LABELS):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(cm.sum() - tp - fp - fn)
        breakdown[LABEL_NAMES[label]] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

    return breakdown


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Core metrics: accuracy, weighted F1/precision/recall, MCC, business metrics.
    """
    metrics: dict[str, float] = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    metrics["weighted_f1"] = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    metrics["weighted_precision"] = float(
        precision_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    metrics["weighted_recall"] = float(
        recall_score(y_true, y_pred, average="weighted", zero_division=0)
    )

    present_labels = sorted(y_true.unique().tolist())
    label_to_idx = {lbl: i for i, lbl in enumerate(present_labels)}

    per_class_precision = precision_score(
        y_true, y_pred, average=None, labels=present_labels, zero_division=0
    )
    per_class_recall = recall_score(
        y_true, y_pred, average=None, labels=present_labels, zero_division=0
    )

    actionable = [lbl for lbl in [BUY, SELL] if lbl in label_to_idx]
    metrics["business_signal_precision"] = (
        float(np.mean([per_class_precision[label_to_idx[lbl]] for lbl in actionable]))
        if actionable
        else 0.0
    )
    metrics["business_hold_recall"] = (
        float(per_class_recall[label_to_idx[HOLD]]) if HOLD in label_to_idx else 0.0
    )

    return metrics


def compute_auc_pr(
    y_true: pd.Series,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """AUC-PR scalar per class (one-vs-rest)."""
    result: dict[str, float] = {}

    for i, label in enumerate(ALL_LABELS):
        if y_prob.shape[1] <= i:
            continue
        binary_true = (y_true == label).astype(int)
        precision, recall, _ = precision_recall_curve(binary_true, y_prob[:, i])
        result[LABEL_NAMES[label]] = float(auc(recall, precision))

    return result
