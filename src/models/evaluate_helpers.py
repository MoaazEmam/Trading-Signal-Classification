"""
src/models/evaluate_helpers.py
================================
Internal helpers for evaluate.py — not part of the public API.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    confusion_matrix,
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
    """Per-class TP, TN, FP, FN via one-vs-rest decomposition (Lecture 08)."""
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
    All Lecture 08 metrics plus business-specific metrics.

    Standard: accuracy, macro/micro/weighted precision/recall/f1, MCC.
    Business: signal precision (Buy+Sell), hold recall.
    """
    metrics: dict[str, float] = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    for avg in ("macro", "micro", "weighted"):
        metrics[f"{avg}_precision"] = float(
            precision_score(y_true, y_pred, average=avg, zero_division=0)
        )
        metrics[f"{avg}_recall"] = float(
            recall_score(y_true, y_pred, average=avg, zero_division=0)
        )
        metrics[f"{avg}_f1"] = float(
            f1_score(y_true, y_pred, average=avg, zero_division=0)
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
        float(per_class_recall[label_to_idx[HOLD]])
        if HOLD in label_to_idx
        else 0.0
    )

    return metrics


def compute_pr_curves(
    y_true: pd.Series,
    y_prob: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-class Precision-Recall curves (one-vs-rest) for threshold analysis (Lecture 08)."""
    pr_curves: dict[str, dict[str, np.ndarray]] = {}

    for i, label in enumerate(ALL_LABELS):
        if y_prob.shape[1] <= i:
            continue
        binary_true = (y_true == label).astype(int)
        precision, recall, thresholds = precision_recall_curve(binary_true, y_prob[:, i])
        pr_curves[LABEL_NAMES[label]] = {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
        }

    return pr_curves


def save_pr_curves(
    pr_curves: dict[str, dict[str, np.ndarray]],
    model_name: str,
    output_dir: Path,
) -> list[Path]:
    """Saves each per-class PR curve as a CSV for MLflow artifact logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for class_name, curve_data in pr_curves.items():
        n = min(
            len(curve_data["precision"]),
            len(curve_data["recall"]),
            len(curve_data["thresholds"]) + 1,
        )
        df = pd.DataFrame(
            {
                "precision": curve_data["precision"][:n],
                "recall": curve_data["recall"][:n],
                "threshold": list(curve_data["thresholds"][: n - 1]) + [1.0],
            }
        )
        path = output_dir / f"pr_curve_{model_name}_{class_name.lower()}.csv"
        df.to_csv(path, index=False)
        saved.append(path)
        logger.info("PR curve saved: %s", path)

    return saved
