"""Unit tests for src/eda/stats.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eda.stats import (
    class_conditional_stats,
    correlation_clusters,
    feature_summary,
    mutual_info_table,
)


@pytest.fixture()
def numeric_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "feat_a": rng.normal(0, 1, n),
            "feat_b": rng.normal(5, 2, n),
            "feat_c": rng.uniform(0, 1, n),
            "str_col": ["x"] * n,
        }
    )


@pytest.fixture()
def label_series(numeric_df) -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(rng.choice([0, 1, 2], size=len(numeric_df)), name="label")


# ---------------------------------------------------------------------------
# feature_summary
# ---------------------------------------------------------------------------


class TestFeatureSummary:
    def test_returns_dataframe(self, numeric_df):
        result = feature_summary(numeric_df)
        assert isinstance(result, pd.DataFrame)

    def test_only_numeric_features_in_index(self, numeric_df):
        result = feature_summary(numeric_df)
        assert "str_col" not in result.index
        assert "feat_a" in result.index

    def test_expected_columns_present(self, numeric_df):
        result = feature_summary(numeric_df)
        for col in [
            "count",
            "mean",
            "std",
            "min",
            "p25",
            "p50",
            "p75",
            "max",
            "skew",
            "kurtosis",
            "n_unique",
        ]:
            assert col in result.columns, f"missing column: {col}"

    def test_index_name_is_feature(self, numeric_df):
        result = feature_summary(numeric_df)
        assert result.index.name == "feature"

    def test_count_equals_n_rows(self, numeric_df):
        result = feature_summary(numeric_df)
        assert (result["count"] == len(numeric_df)).all()

    def test_min_leq_p50_leq_max(self, numeric_df):
        result = feature_summary(numeric_df)
        assert (result["min"] <= result["p50"]).all()
        assert (result["p50"] <= result["max"]).all()

    def test_all_numeric_input(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        result = feature_summary(df)
        assert len(result) == 2

    def test_empty_numeric_returns_empty(self):
        df = pd.DataFrame({"s": ["a", "b"]})
        result = feature_summary(df)
        assert result.empty


# ---------------------------------------------------------------------------
# class_conditional_stats
# ---------------------------------------------------------------------------


class TestClassConditionalStats:
    def test_returns_dataframe(self, numeric_df, label_series):
        result = class_conditional_stats(numeric_df, label_series)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, numeric_df, label_series):
        result = class_conditional_stats(numeric_df, label_series)
        for col in ["feature", "class", "mean", "median", "std", "count"]:
            assert col in result.columns

    def test_one_row_per_feature_per_class(self, numeric_df, label_series):
        result = class_conditional_stats(numeric_df, label_series)
        n_classes = label_series.nunique()
        n_numeric_features = numeric_df.select_dtypes("number").shape[1]
        assert len(result) == n_classes * n_numeric_features

    def test_only_numeric_features(self, numeric_df, label_series):
        result = class_conditional_stats(numeric_df, label_series)
        assert "str_col" not in result["feature"].values

    def test_all_classes_present(self, numeric_df, label_series):
        result = class_conditional_stats(numeric_df, label_series)
        assert set(result["class"].unique()) == set(label_series.unique())

    def test_count_sums_to_total(self, numeric_df, label_series):
        result = class_conditional_stats(numeric_df, label_series)
        for feature in result["feature"].unique():
            total = result[result["feature"] == feature]["count"].sum()
            assert total == len(numeric_df)

    def test_binary_labels(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        y = pd.Series([0, 0, 1, 1])
        result = class_conditional_stats(df, y)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# mutual_info_table
# ---------------------------------------------------------------------------


class TestMutualInfoTable:
    def test_returns_dataframe(self, numeric_df, label_series):
        result = mutual_info_table(numeric_df, label_series)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, numeric_df, label_series):
        result = mutual_info_table(numeric_df, label_series)
        for col in ["feature", "mutual_info", "f_stat", "f_pvalue"]:
            assert col in result.columns

    def test_one_row_per_numeric_feature(self, numeric_df, label_series):
        result = mutual_info_table(numeric_df, label_series)
        n_numeric = numeric_df.select_dtypes("number").shape[1]
        assert len(result) == n_numeric

    def test_sorted_by_mutual_info_descending(self, numeric_df, label_series):
        result = mutual_info_table(numeric_df, label_series)
        assert result["mutual_info"].is_monotonic_decreasing

    def test_mutual_info_non_negative(self, numeric_df, label_series):
        result = mutual_info_table(numeric_df, label_series)
        assert (result["mutual_info"] >= 0).all()

    def test_f_pvalue_between_0_and_1(self, numeric_df, label_series):
        result = mutual_info_table(numeric_df, label_series)
        assert (result["f_pvalue"] >= 0).all()
        assert (result["f_pvalue"] <= 1).all()

    def test_subsampling_returns_same_columns(self, numeric_df, label_series):
        result = mutual_info_table(numeric_df, label_series, n_samples=50)
        for col in ["feature", "mutual_info", "f_stat", "f_pvalue"]:
            assert col in result.columns

    def test_no_subsampling_when_n_samples_exceeds_rows(self, numeric_df, label_series):
        result = mutual_info_table(numeric_df, label_series, n_samples=10000)
        assert len(result) == numeric_df.select_dtypes("number").shape[1]


# ---------------------------------------------------------------------------
# correlation_clusters
# ---------------------------------------------------------------------------


class TestCorrelationClusters:
    def test_returns_dataframe(self, numeric_df):
        result = correlation_clusters(numeric_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, numeric_df):
        result = correlation_clusters(numeric_df)
        for col in ["feature", "cluster_id", "n_in_cluster"]:
            assert col in result.columns

    def test_one_row_per_numeric_feature(self, numeric_df):
        result = correlation_clusters(numeric_df)
        n_numeric = numeric_df.select_dtypes("number").shape[1]
        assert len(result) == n_numeric

    def test_all_features_assigned_cluster(self, numeric_df):
        result = correlation_clusters(numeric_df)
        assert result["cluster_id"].notna().all()

    def test_perfectly_correlated_features_same_cluster(self):
        n = 100
        x = np.linspace(0, 1, n)
        df = pd.DataFrame(
            {"a": x, "b": x * 2 + 1, "c": np.random.default_rng(99).normal(0, 1, n)}
        )
        result = correlation_clusters(df, corr_threshold=0.99)
        a_cluster = result.loc[result["feature"] == "a", "cluster_id"].iloc[0]
        b_cluster = result.loc[result["feature"] == "b", "cluster_id"].iloc[0]
        assert a_cluster == b_cluster

    def test_n_in_cluster_consistent(self, numeric_df):
        result = correlation_clusters(numeric_df)
        for cluster_id, group in result.groupby("cluster_id"):
            assert (group["n_in_cluster"] == len(group)).all()
