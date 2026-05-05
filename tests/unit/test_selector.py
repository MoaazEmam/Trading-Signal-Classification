"""Unit tests for src/features/features_selection/selector.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.features_selection.selector import FeatureSelector


@pytest.fixture()
def xy_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(7)
    n = 300
    signal = rng.normal(0, 1, n)
    X = pd.DataFrame(
        {
            "good_a": signal + rng.normal(0, 0.1, n),
            "good_b": -signal + rng.normal(0, 0.1, n),
            "noise_c": rng.normal(0, 1, n),
            "constant_d": np.ones(n),
        }
    )
    y = pd.Series((signal > 0).astype(int), name="label")
    return X, y


class TestFeatureSelectorFit:
    def test_fit_returns_self(self, xy_data):
        X, y = xy_data
        selector = FeatureSelector()
        result = selector.fit(X, y)
        assert result is selector

    def test_fit_sets_selected_features(self, xy_data):
        X, y = xy_data
        selector = FeatureSelector()
        selector.fit(X, y)
        assert hasattr(selector, "selected_features_")

    def test_fit_sets_importance_scores(self, xy_data):
        X, y = xy_data
        selector = FeatureSelector()
        selector.fit(X, y)
        assert hasattr(selector, "importance_scores_")

    def test_fit_sets_n_features_in(self, xy_data):
        X, y = xy_data
        selector = FeatureSelector()
        selector.fit(X, y)
        assert selector.n_features_in_ == X.shape[1]

    def test_fit_sets_fitted_at(self, xy_data):
        X, y = xy_data
        selector = FeatureSelector()
        selector.fit(X, y)
        assert hasattr(selector, "fitted_at_")

    def test_fit_drops_constant_feature(self, xy_data):
        X, y = xy_data
        selector = FeatureSelector(filter_thresholds=(1e-4, 0.95, 0.01))
        selector.fit(X, y)
        assert "constant_d" not in selector.selected_features_


class TestFeatureSelectorTransform:
    def test_transform_returns_dataframe(self, xy_data):
        X, y = xy_data
        selector = FeatureSelector()
        selector.fit(X, y)
        out = selector.transform(X)
        assert isinstance(out, pd.DataFrame)

    def test_transform_only_returns_selected_cols(self, xy_data):
        X, y = xy_data
        selector = FeatureSelector()
        selector.fit(X, y)
        out = selector.transform(X)
        assert set(out.columns).issubset(set(X.columns))

    def test_transform_preserves_row_count(self, xy_data):
        X, y = xy_data
        selector = FeatureSelector()
        selector.fit(X, y)
        out = selector.transform(X)
        assert len(out) == len(X)

    def test_transform_before_fit_raises(self, xy_data):
        X, _ = xy_data
        selector = FeatureSelector()
        with pytest.raises(RuntimeError, match="must be fit"):
            selector.transform(X)

    def test_transform_raises_if_column_missing(self, xy_data):
        X, y = xy_data
        selector = FeatureSelector()
        selector.fit(X, y)
        with pytest.raises(ValueError, match="missing from input"):
            selector.transform(X.drop(columns=["good_a"]))


class TestFeatureSelectorFitTransform:
    def test_fit_transform_equivalent_to_fit_then_transform(self, xy_data):
        X, y = xy_data
        s1 = FeatureSelector()
        out1 = s1.fit_transform(X, y)

        s2 = FeatureSelector()
        s2.fit(X, y)
        out2 = s2.transform(X)

        assert list(out1.columns) == list(out2.columns)


class TestFeatureSelectorSaveLoad:
    def test_save_creates_file(self, xy_data, tmp_path):
        X, y = xy_data
        selector = FeatureSelector()
        selector.fit(X, y)
        path = tmp_path / "selector.pkl"
        selector.save(path)
        assert path.exists()

    def test_load_returns_feature_selector(self, xy_data, tmp_path):
        X, y = xy_data
        selector = FeatureSelector()
        selector.fit(X, y)
        path = tmp_path / "selector.pkl"
        selector.save(path)
        loaded = FeatureSelector.load(path)
        assert isinstance(loaded, FeatureSelector)

    def test_loaded_selector_has_same_features(self, xy_data, tmp_path):
        X, y = xy_data
        selector = FeatureSelector()
        selector.fit(X, y)
        path = tmp_path / "selector.pkl"
        selector.save(path)
        loaded = FeatureSelector.load(path)
        assert loaded.selected_features_ == selector.selected_features_

    def test_load_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FeatureSelector.load(tmp_path / "nonexistent.pkl")

    def test_load_raises_if_wrong_type(self, tmp_path):
        import joblib

        joblib.dump({"not": "a selector"}, tmp_path / "wrong.pkl")
        with pytest.raises(TypeError):
            FeatureSelector.load(tmp_path / "wrong.pkl")


class TestFeatureSelectorGetFeatureNames:
    def test_get_feature_names_out_returns_array(self, xy_data):
        X, y = xy_data
        selector = FeatureSelector()
        selector.fit(X, y)
        names = selector.get_feature_names_out()
        assert len(names) > 0

    def test_get_feature_names_before_fit_raises(self):
        selector = FeatureSelector()
        with pytest.raises(RuntimeError):
            selector.get_feature_names_out()
