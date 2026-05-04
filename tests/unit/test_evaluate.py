"""
Unit tests for src/models/evaluate.py and src/models/evaluate_helpers.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.models.evaluate import (
    build_comparison_table,
    evaluate_model,
)
from src.models.evaluate_helpers import (
    compute_metrics,
    confusion_matrix_breakdown,
)

N_SAMPLES = 90


@pytest.fixture()
def y_true() -> pd.Series:
    return pd.Series([0, 1, 2] * (N_SAMPLES // 3))


@pytest.fixture()
def y_pred_perfect(y_true: pd.Series) -> pd.Series:
    return y_true.copy()


@pytest.fixture()
def y_pred_random() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.integers(0, 3, N_SAMPLES))


@pytest.fixture()
def x_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.standard_normal((N_SAMPLES, 5)),
        columns=[f"f{i}" for i in range(5)],
    )


@pytest.fixture()
def fitted_dummy(x_df: pd.DataFrame, y_true: pd.Series) -> DummyClassifier:
    model = DummyClassifier(strategy="prior", random_state=0)
    model.fit(x_df, y_true)
    return model


@pytest.fixture()
def sample_eval_result() -> dict:
    return {
        "model_name": "dummy",
        "train_accuracy": 0.9,
        "test_accuracy": 0.7,
        "overfit_gap": 0.2,
        "test_weighted_f1": 0.68,
        "test_weighted_precision": 0.70,
        "test_weighted_recall": 0.70,
        "test_mcc": 0.55,
        "test_business_signal_precision": 0.65,
        "test_business_hold_recall": 0.72,
        "confusion_matrix": [[10, 2, 1], [1, 12, 0], [0, 1, 11]],
        "confusion_matrix_breakdown": {},
        "classification_report": "",
        "auc_pr": {"Hold": 0.80, "Buy": 0.75, "Sell": 0.70},
    }


class TestComputeMetrics:
    def test_returns_required_keys(self, y_true, y_pred_perfect):
        result = compute_metrics(y_true, y_pred_perfect)
        for key in (
            "accuracy",
            "weighted_f1",
            "weighted_precision",
            "weighted_recall",
            "mcc",
            "business_signal_precision",
            "business_hold_recall",
        ):
            assert key in result

    def test_no_macro_or_micro_keys(self, y_true, y_pred_random):
        result = compute_metrics(y_true, y_pred_random)
        for key in result:
            assert not key.startswith("macro_")
            assert not key.startswith("micro_")

    def test_perfect_accuracy_is_one(self, y_true, y_pred_perfect):
        result = compute_metrics(y_true, y_pred_perfect)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_mcc_in_valid_range(self, y_true, y_pred_random):
        result = compute_metrics(y_true, y_pred_random)
        assert -1.0 <= result["mcc"] <= 1.0

    def test_business_signal_precision_in_range(self, y_true, y_pred_random):
        result = compute_metrics(y_true, y_pred_random)
        assert 0.0 <= result["business_signal_precision"] <= 1.0

    def test_business_hold_recall_in_range(self, y_true, y_pred_random):
        result = compute_metrics(y_true, y_pred_random)
        assert 0.0 <= result["business_hold_recall"] <= 1.0

    def test_all_hold_predictions_give_hold_recall_one(self):
        y_true = pd.Series([0, 0, 0, 1, 2])
        y_pred = pd.Series([0, 0, 0, 0, 0])
        result = compute_metrics(y_true, y_pred)
        assert result["business_hold_recall"] == pytest.approx(1.0)


class TestConfusionMatrixBreakdown:
    def test_returns_breakdown_for_all_classes(self, y_true, y_pred_perfect):
        result = confusion_matrix_breakdown(y_true, y_pred_perfect)
        assert set(result.keys()) == {"Hold", "Buy", "Sell"}

    def test_each_class_has_tp_tn_fp_fn(self, y_true, y_pred_perfect):
        result = confusion_matrix_breakdown(y_true, y_pred_perfect)
        for cls in result.values():
            assert set(cls.keys()) == {"TP", "TN", "FP", "FN"}

    def test_perfect_predictions_have_no_fp(self, y_true, y_pred_perfect):
        result = confusion_matrix_breakdown(y_true, y_pred_perfect)
        for cls in result.values():
            assert cls["FP"] == 0

    def test_perfect_predictions_have_no_fn(self, y_true, y_pred_perfect):
        result = confusion_matrix_breakdown(y_true, y_pred_perfect)
        for cls in result.values():
            assert cls["FN"] == 0

    def test_tp_plus_fp_plus_fn_plus_tn_equals_total(self, y_true, y_pred_random):
        result = confusion_matrix_breakdown(y_true, y_pred_random)
        n = len(y_true)
        for cls in result.values():
            assert cls["TP"] + cls["TN"] + cls["FP"] + cls["FN"] == n


class TestEvaluateModel:
    def test_returns_required_keys(self, fitted_dummy, x_df, y_true):
        result = evaluate_model("dummy", fitted_dummy, x_df, y_true, x_df, y_true)
        for key in (
            "model_name",
            "train_accuracy",
            "test_accuracy",
            "overfit_gap",
            "confusion_matrix",
            "auc_pr",
        ):
            assert key in result

    def test_model_name_stored_in_result(self, fitted_dummy, x_df, y_true):
        result = evaluate_model("my_model", fitted_dummy, x_df, y_true, x_df, y_true)
        assert result["model_name"] == "my_model"

    def test_overfit_gap_equals_train_minus_test(self, fitted_dummy, x_df, y_true):
        result = evaluate_model("dummy", fitted_dummy, x_df, y_true, x_df, y_true)
        expected = round(result["train_accuracy"] - result["test_accuracy"], 4)
        assert result["overfit_gap"] == pytest.approx(expected)

    def test_auc_pr_populated_for_probabilistic_model(self, fitted_dummy, x_df, y_true):
        result = evaluate_model("dummy", fitted_dummy, x_df, y_true, x_df, y_true)
        assert len(result["auc_pr"]) > 0

    def test_auc_pr_values_in_range(self, fitted_dummy, x_df, y_true):
        result = evaluate_model("dummy", fitted_dummy, x_df, y_true, x_df, y_true)
        for val in result["auc_pr"].values():
            assert 0.0 <= val <= 1.0

    def test_confusion_matrix_is_3x3(self, fitted_dummy, x_df, y_true):
        result = evaluate_model("dummy", fitted_dummy, x_df, y_true, x_df, y_true)
        cm = result["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

    def test_accuracies_between_zero_and_one(self, fitted_dummy, x_df, y_true):
        result = evaluate_model("dummy", fitted_dummy, x_df, y_true, x_df, y_true)
        assert 0.0 <= result["train_accuracy"] <= 1.0
        assert 0.0 <= result["test_accuracy"] <= 1.0


class TestBuildComparisonTable:
    def test_returns_dataframe(self, sample_eval_result):
        table = build_comparison_table([sample_eval_result])
        assert isinstance(table, pd.DataFrame)

    def test_indexed_by_model_name(self, sample_eval_result):
        table = build_comparison_table([sample_eval_result])
        assert table.index[0] == "dummy"

    def test_has_expected_columns(self, sample_eval_result):
        table = build_comparison_table([sample_eval_result])
        for col in (
            "Train Acc",
            "Test Acc",
            "Overfit Gap",
            "Weighted F1",
            "MCC",
            "Signal Precision (B)",
            "Hold Recall (B)",
        ):
            assert col in table.columns

    def test_no_macro_or_micro_columns(self, sample_eval_result):
        table = build_comparison_table([sample_eval_result])
        for col in table.columns:
            assert "macro" not in col.lower()
            assert "micro" not in col.lower()

    def test_multiple_models_produces_correct_row_count(self, sample_eval_result):
        r2 = {**sample_eval_result, "model_name": "other"}
        table = build_comparison_table([sample_eval_result, r2])
        assert len(table) == 2

    def test_values_rounded_to_four_decimals(self, sample_eval_result):
        table = build_comparison_table([sample_eval_result])
        val = table.loc["dummy", "Test Acc"]
        assert val == round(val, 4)
