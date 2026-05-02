"""
src/models/evaluate.py
======================
Evaluation module for Trading Signal Classification.

Covers everything from Lecture 07 (Statistical Significance Tests) and
Lecture 08 (Evaluation Metrics) as applied to this project.

Lecture 08 metrics implemented
--------------------------------
Confusion Matrix   : TP / TN / FP / FN per class (one-vs-rest)
Accuracy           : (TP + TN) / Total  — reported but not primary metric
                     (may be misleading if classes are imbalanced)
Precision          : TP / (TP + FP)     — quality of positive predictions
Recall             : TP / (TP + FN)     — ability to find all positives
F1-Score           : harmonic mean of Precision and Recall
MCC                : Matthews Correlation Coefficient — robust to imbalance
Averaging          : Macro (equal class weight), Micro (aggregated counts),
                     Weighted (by class support) — all three reported
PR Curve           : Precision-Recall curve saved per model (threshold analysis)

Lecture 07 statistical test
------------------------------
Paired T-Test      : on per-fold CV accuracy scores across model pairs.
                     H0: mu_diff = 0 (no real difference between models).
                     If p-value > 0.05 -> difference may be a fluke.

Business metrics (trading domain)
------------------------------------
Signal Precision   : mean Precision on Buy (1) + Sell (2) classes only.
                     False Buy/Sell = a real trade executed = financial loss.
Hold Recall        : Recall on Hold (0) class.
                     Missed Hold = unnecessary trade = transaction cost.

Classes : 0 = Hold | 1 = Buy | 2 = Sell
"""

from __future__ import annotations

import logging
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix

from src.models.evaluate_helpers import (
    ALL_LABELS,
    compute_metrics,
    compute_pr_curves,
    confusion_matrix_breakdown,
    save_pr_curves,
)

logger = logging.getLogger(__name__)

HOLD = 0
BUY = 1
SELL = 2
ALPHA = 0.05


# ══════════════════════════════════════════════════════════════════════════════
# Full model evaluation (train + test)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    pr_output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Full evaluation of a trained model on both train and test splits.

    Returns a flat dict with:
      train_{metric}              for every metric in compute_metrics()
      test_{metric}               for every metric in compute_metrics()
      overfit_gap                 train_accuracy - test_accuracy
      confusion_matrix_breakdown  per-class TP/TN/FP/FN (test split)
      classification_report       sklearn text report (test split)
      confusion_matrix            raw matrix as list[list[int]] (test split)
      pr_curve_paths              list of saved PR curve CSV paths

    Logging both splits detects overfitting: a large overfit_gap signals
    the model memorised training data rather than generalising.
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

    logger.info("── %s ──", name.upper())
    logger.info(
        "Overfitting | train_acc=%.4f | test_acc=%.4f | gap=%.4f",
        train_metrics["train_accuracy"],
        test_metrics["test_accuracy"],
        overfit_gap,
    )
    logger.info("MCC (test) = %.4f", test_metrics["test_mcc"])
    logger.info("Weighted F1 (test) = %.4f", test_metrics["test_weighted_f1"])
    logger.info("Signal Precision (test) = %.4f", test_metrics["test_business_signal_precision"])
    logger.info("Hold Recall (test) = %.4f", test_metrics["test_business_hold_recall"])
    logger.info("Classification Report (test):\n%s", report)
    logger.info("Confusion Matrix (test):\n%s", cm)
    logger.info("Per-class TP/TN/FP/FN: %s", cm_breakdown)

    pr_curve_paths: list[Path] = []
    if pr_output_dir is not None and hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(x_test)
            pr_curves = compute_pr_curves(y_test, y_prob)
            pr_curve_paths = save_pr_curves(pr_curves, name, pr_output_dir)
        except Exception as exc:
            logger.warning("PR curve generation failed for %s: %s", name, exc)

    return {
        "model_name": name,
        **train_metrics,
        **test_metrics,
        "overfit_gap": overfit_gap,
        "confusion_matrix_breakdown": cm_breakdown,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "pr_curve_paths": [str(p) for p in pr_curve_paths],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Comparison table
# ══════════════════════════════════════════════════════════════════════════════

def build_comparison_table(all_results: list[dict]) -> pd.DataFrame:
    """
    Builds a human-readable comparison DataFrame from evaluate_model outputs.

    Column order:
      Overfitting check | Standard (weighted) | MCC | Macro F1 | Business
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
                "Weighted Precision": r["test_weighted_precision"],
                "Weighted Recall": r["test_weighted_recall"],
                "Macro F1": r["test_macro_f1"],
                "Macro Precision": r["test_macro_precision"],
                "Macro Recall": r["test_macro_recall"],
                "MCC": r["test_mcc"],
                "Signal Precision (B)": r["test_business_signal_precision"],
                "Hold Recall (B)": r["test_business_hold_recall"],
            }
        )
    return pd.DataFrame(rows).set_index("Model").round(4)


# ══════════════════════════════════════════════════════════════════════════════
# Paired T-Test — statistical model comparison (Lecture 07)
# ══════════════════════════════════════════════════════════════════════════════

def paired_ttest_model_comparison(
    cv_scores: dict[str, list[float]],
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """
    Pairwise Paired T-Tests on per-fold CV scores (Lecture 07).

    H0: mu_diff = 0 (no real performance difference).
    If p-value > 0.05 the observed better performance may be a fluke.
    """
    rows = []

    for model_a, model_b in combinations(cv_scores.keys(), 2):
        scores_a = np.array(cv_scores[model_a], dtype=float)
        scores_b = np.array(cv_scores[model_b], dtype=float)

        if len(scores_a) != len(scores_b):
            logger.warning(
                "Skipping %s vs %s: unequal fold counts (%d vs %d)",
                model_a, model_b, len(scores_a), len(scores_b),
            )
            continue

        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
        n = len(scores_a)
        df_deg = n - 1
        t_critical = float(stats.t.ppf(1 - alpha / 2, df_deg))

        mean_diff = float(np.mean(scores_a - scores_b))
        significant = bool(p_value < alpha)

        better = (
            model_a if mean_diff > 0
            else model_b if mean_diff < 0
            else "Tie"
        )

        rows.append(
            {
                "Model A": model_a,
                "Model B": model_b,
                "Mean A": round(float(np.mean(scores_a)), 4),
                "Mean B": round(float(np.mean(scores_b)), 4),
                "Mean Diff (A-B)": round(mean_diff, 4),
                "t-statistic": round(float(t_stat), 4),
                "t-critical (a=0.05)": round(t_critical, 4),
                "p-value": round(float(p_value), 6),
                "Significant": "Yes" if significant else "No (fluke)",
                "Better Model": better,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("p-value")
        .reset_index(drop=True)
    )
