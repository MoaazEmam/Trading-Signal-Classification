"""Unit tests for src/models/mlflow_helpers.py"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from src.models.mlflow_helpers import log_all_metrics, prepare_features

# ---------------------------------------------------------------------------
# prepare_features
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [4.0, 5.0, 6.0],
            "Company": ["AAPL", "MSFT", "AAPL"],
            "label": [0, 1, 2],
        }
    )


class TestPrepareFeatures:
    def test_drops_target_col(self, sample_train_df):
        X, y = prepare_features(sample_train_df, "label", "Company")
        assert "label" not in X.columns

    def test_drops_company_col(self, sample_train_df):
        X, y = prepare_features(sample_train_df, "label", "Company")
        assert "Company" not in X.columns

    def test_y_is_target_series(self, sample_train_df):
        X, y = prepare_features(sample_train_df, "label", "Company")
        pd.testing.assert_series_equal(y, sample_train_df["label"])

    def test_x_has_correct_columns(self, sample_train_df):
        X, y = prepare_features(sample_train_df, "label", "Company")
        assert set(X.columns) == {"feature_a", "feature_b"}

    def test_x_row_count_preserved(self, sample_train_df):
        X, y = prepare_features(sample_train_df, "label", "Company")
        assert len(X) == len(sample_train_df)

    def test_missing_company_col_not_dropped(self, sample_train_df):
        df_no_company = sample_train_df.drop(columns=["Company"])
        X, y = prepare_features(df_no_company, "label", "Company")
        assert "feature_a" in X.columns

    def test_missing_target_col_raises(self, sample_train_df):
        with pytest.raises(KeyError):
            prepare_features(sample_train_df, "nonexistent_label", "Company")

    def test_handles_df_with_only_target_and_company(self):
        df = pd.DataFrame({"label": [0, 1, 2], "Company": ["A", "B", "A"]})
        X, y = prepare_features(df, "label", "Company")
        assert X.empty
        assert len(y) == 3


# ---------------------------------------------------------------------------
# log_all_metrics
# ---------------------------------------------------------------------------


def _make_eval_result(**overrides) -> dict:
    base = {
        "train_accuracy": 0.75,
        "test_accuracy": 0.70,
        "overfit_gap": 0.05,
        "test_weighted_f1": 0.68,
        "test_weighted_precision": 0.70,
        "test_weighted_recall": 0.68,
        "test_mcc": 0.45,
        "test_business_signal_precision": 0.65,
        "test_business_hold_recall": 0.80,
        "auc_pr": {"Hold": 0.82, "Buy": 0.61, "Sell": 0.55},
    }
    base.update(overrides)
    return base


class TestLogAllMetrics:
    def test_logs_train_accuracy(self):
        with patch("mlflow.log_metric") as mock_log:
            log_all_metrics(None, _make_eval_result())
            calls = {c.args[0]: c.args[1] for c in mock_log.call_args_list}
            assert "train_accuracy" in calls
            assert calls["train_accuracy"] == pytest.approx(0.75)

    def test_logs_test_accuracy(self):
        with patch("mlflow.log_metric") as mock_log:
            log_all_metrics(None, _make_eval_result())
            calls = {c.args[0]: c.args[1] for c in mock_log.call_args_list}
            assert "test_accuracy" in calls

    def test_logs_overfit_gap(self):
        with patch("mlflow.log_metric") as mock_log:
            log_all_metrics(None, _make_eval_result())
            calls = {c.args[0] for c in mock_log.call_args_list}
            assert "overfit_gap" in calls

    def test_logs_weighted_metrics(self):
        with patch("mlflow.log_metric") as mock_log:
            log_all_metrics(None, _make_eval_result())
            calls = {c.args[0] for c in mock_log.call_args_list}
            assert "weighted_f1" in calls
            assert "weighted_precision" in calls
            assert "weighted_recall" in calls

    def test_logs_mcc(self):
        with patch("mlflow.log_metric") as mock_log:
            log_all_metrics(None, _make_eval_result())
            calls = {c.args[0] for c in mock_log.call_args_list}
            assert "mcc" in calls

    def test_logs_business_metrics(self):
        with patch("mlflow.log_metric") as mock_log:
            log_all_metrics(None, _make_eval_result())
            calls = {c.args[0] for c in mock_log.call_args_list}
            assert "business_signal_precision" in calls
            assert "business_hold_recall" in calls

    def test_logs_auc_pr_per_class(self):
        with patch("mlflow.log_metric") as mock_log:
            log_all_metrics(None, _make_eval_result())
            calls = {c.args[0] for c in mock_log.call_args_list}
            assert "auc_pr_hold" in calls
            assert "auc_pr_buy" in calls
            assert "auc_pr_sell" in calls

    def test_logs_cv_accuracy_when_provided(self):
        with patch("mlflow.log_metric") as mock_log:
            log_all_metrics(0.72, _make_eval_result())
            calls = {c.args[0]: c.args[1] for c in mock_log.call_args_list}
            assert "cv_accuracy" in calls
            assert calls["cv_accuracy"] == pytest.approx(0.72)

    def test_skips_cv_accuracy_when_none(self):
        with patch("mlflow.log_metric") as mock_log:
            log_all_metrics(None, _make_eval_result())
            calls = {c.args[0] for c in mock_log.call_args_list}
            assert "cv_accuracy" not in calls

    def test_handles_empty_auc_pr(self):
        with patch("mlflow.log_metric") as mock_log:
            eval_result = _make_eval_result(auc_pr={})
            log_all_metrics(None, eval_result)
            calls = {c.args[0] for c in mock_log.call_args_list}
            assert "auc_pr_hold" not in calls
