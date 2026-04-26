import pandas as pd

from src.features.features_selection.importance import (
    apply_importance_filter,
    get_importance_scores,
)


class TestGetImportanceScores:
    def test_returns_series_indexed_by_feature_names(self, selection_X_y):
        X, y = selection_X_y
        numeric_X = X.select_dtypes(include="number")
        scores = get_importance_scores(numeric_X, y)
        assert isinstance(scores, pd.Series)
        assert set(scores.index).issubset(set(numeric_X.columns))

    def test_scores_are_nonnegative(self, selection_X_y):
        X, y = selection_X_y
        numeric_X = X.select_dtypes(include="number")
        scores = get_importance_scores(numeric_X, y)
        assert (scores >= 0).all()

    def test_sorted_descending(self, selection_X_y):
        X, y = selection_X_y
        numeric_X = X.select_dtypes(include="number")
        scores = get_importance_scores(numeric_X, y)
        assert list(scores) == sorted(scores, reverse=True)

    def test_signal_feature_scores_higher_than_noise(self, selection_X_y):
        X, y = selection_X_y
        numeric_X = X.select_dtypes(include="number")
        scores = get_importance_scores(numeric_X, y)
        assert scores["good_feature"] > scores["noise_feature"]


class TestApplyImportanceFilter:
    def test_drops_noise_feature(self, selection_X_y):
        X, y = selection_X_y
        surviving = apply_importance_filter(X, y, threshold=0.01)
        assert "noise_feature" not in surviving

    def test_keeps_signal_feature(self, selection_X_y):
        X, y = selection_X_y
        surviving = apply_importance_filter(X, y, threshold=0.01)
        assert "good_feature" in surviving

    def test_nonnumeric_always_survives(self, selection_X_y):
        X, y = selection_X_y
        surviving = apply_importance_filter(X, y)
        assert "Company" in surviving

    def test_strict_threshold_drops_more(self, selection_X_y):
        X, y = selection_X_y
        strict = apply_importance_filter(X, y, threshold=0.5)
        lenient = apply_importance_filter(X, y, threshold=0.001)
        assert len(strict) <= len(lenient)

    def test_returns_list_of_str(self, selection_X_y):
        X, y = selection_X_y
        result = apply_importance_filter(X, y)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)
