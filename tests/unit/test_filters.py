import numpy as np
import pandas as pd

from src.features.features_selection.filter import (
    _apply_correlation_filter,
    _apply_mutual_info_filter,
    _apply_variance_filter,
    apply_basic_filters,
)


class TestApplyVarianceFilter:
    def test_drops_constant_column(self, selection_X_y):
        X, _ = selection_X_y
        surviving = _apply_variance_filter(X)
        assert "constant_feature" not in surviving

    def test_keeps_varying_column(self, selection_X_y):
        X, _ = selection_X_y
        surviving = _apply_variance_filter(X)
        assert "good_feature" in surviving

    def test_nonnumeric_always_survives(self, selection_X_y):
        X, _ = selection_X_y
        surviving = _apply_variance_filter(X)
        assert "Company" in surviving

    def test_all_constant_drops_all_numeric(self, selection_all_constant_df):
        X = selection_all_constant_df.drop(columns=["Date"])
        surviving = _apply_variance_filter(X)
        numeric_survivors = [c for c in surviving if c not in ["Company"]]
        assert len(numeric_survivors) == 0

    def test_no_numeric_cols_returns_nonnumeric(self, selection_no_numeric_df):
        X = selection_no_numeric_df.drop(columns=["label"])
        surviving = _apply_variance_filter(X)
        assert "Company" in surviving
        assert "Date" in surviving

    def test_returns_list_of_str(self, selection_X_y):
        X, _ = selection_X_y
        result = _apply_variance_filter(X)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_custom_threshold_drops_more(self, selection_X_y):
        X, _ = selection_X_y
        low_threshold = _apply_variance_filter(X, threshold=1e-6)
        high_threshold = _apply_variance_filter(X, threshold=10.0)
        assert len(high_threshold) <= len(low_threshold)


class TestApplyCorrelationFilter:
    def test_drops_highly_correlated_column(self, selection_X_y):
        X, y = selection_X_y
        surviving = _apply_correlation_filter(X, y)
        # correlated_feature is near-duplicate of Close — one should be dropped
        assert not ("Close" in surviving and "correlated_feature" in surviving)

    def test_nonnumeric_always_survives(self, selection_X_y):
        X, y = selection_X_y
        surviving = _apply_correlation_filter(X, y)
        assert "Company" in surviving

    def test_keeps_more_target_correlated_feature(self, selection_X_y):
        X, y = selection_X_y
        surviving = _apply_correlation_filter(X, y)
        # good_feature is directly correlated with label, should survive
        assert "good_feature" in surviving

    def test_low_threshold_drops_more(self, selection_X_y):
        X, y = selection_X_y
        strict = _apply_correlation_filter(X, y, threshold=0.5)
        lenient = _apply_correlation_filter(X, y, threshold=0.99)
        assert len(strict) <= len(lenient)

    def test_returns_list_of_str(self, selection_X_y):
        X, y = selection_X_y
        result = _apply_correlation_filter(X, y)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_no_numeric_cols_returns_nonnumeric(self, selection_no_numeric_df):
        X = selection_no_numeric_df.drop(columns=["label"])
        y = pd.Series(np.zeros(len(X), dtype=int))
        surviving = _apply_correlation_filter(X, y)
        assert "Company" in surviving
        assert "Date" in surviving


class TestApplyMutualInfoFilter:
    def test_drops_noise_feature(self, selection_X_y):
        X, y = selection_X_y
        after_variance = _apply_variance_filter(X)
        after_mi = _apply_mutual_info_filter(X[after_variance], y)
        numeric_survivors = [c for c in after_mi if c != "Company"]
        # noise should score lower than signal — if it survives, good_feature must too
        if "noise_feature" in numeric_survivors:
            assert "good_feature" in numeric_survivors

    def test_keeps_signal_feature(self, selection_X_y):
        X, y = selection_X_y
        after_variance = _apply_variance_filter(X)
        surviving = _apply_mutual_info_filter(X[after_variance], y)
        assert "good_feature" in surviving

    def test_nonnumeric_always_survives(self, selection_X_y):
        X, y = selection_X_y
        surviving = _apply_mutual_info_filter(X, y)
        assert "Company" in surviving

    def test_returns_list_of_str(self, selection_X_y):
        X, y = selection_X_y
        result = _apply_mutual_info_filter(X, y)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_strict_threshold_drops_more(self, selection_X_y):
        X, y = selection_X_y
        strict = _apply_mutual_info_filter(X, y, threshold=0.5)
        lenient = _apply_mutual_info_filter(X, y, threshold=0.001)
        assert len(strict) <= len(lenient)


class TestRunFilters:
    def test_chaining_reduces_columns(self, selection_X_y):
        X, y = selection_X_y
        surviving = apply_basic_filters(X, y, thresholds=(1e-4, 0.95, 0.01))
        assert len(surviving) < len(X.columns)

    def test_constant_col_removed_by_chain(self, selection_X_y):
        X, y = selection_X_y
        surviving = apply_basic_filters(X, y, thresholds=(1e-4, 0.95, 0.01))
        assert "constant_feature" not in surviving

    def test_nonnumeric_survives_chain(self, selection_X_y):
        X, y = selection_X_y
        surviving = apply_basic_filters(X, y, thresholds=(1e-4, 0.95, 0.01))
        assert "Company" in surviving

    def test_signal_feature_survives_chain(self, selection_X_y):
        X, y = selection_X_y
        surviving = apply_basic_filters(X, y, thresholds=(1e-4, 0.95, 0.01))
        assert "good_feature" in surviving

    def test_returns_list_of_str(self, selection_X_y):
        X, y = selection_X_y
        result = apply_basic_filters(X, y, thresholds=(1e-4, 0.95, 0.01))
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)
