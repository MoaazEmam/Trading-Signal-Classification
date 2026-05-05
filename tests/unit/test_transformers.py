"""Unit tests for src/features/transformers.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.transformers import ColumnDropper, GlobalWinsorizer, GroupedWinsorizer


@pytest.fixture()
def grouped_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 100
    companies = ["AAPL"] * 50 + ["MSFT"] * 50
    volumes = list(rng.integers(1_000, 10_000, 50)) + list(
        rng.integers(5_000, 50_000, 50)
    )
    volumes[-1] = 500_000  # extreme outlier for MSFT
    return pd.DataFrame(
        {"Company": companies, "Volume": volumes, "Close": rng.normal(100, 10, n)}
    )


@pytest.fixture()
def global_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 200
    data = rng.normal(0, 1, n)
    data[0] = 100.0  # extreme upper outlier
    data[1] = -100.0  # extreme lower outlier
    return pd.DataFrame({"return_1d": data, "log_return": rng.normal(0, 0.01, n)})


# ---------------------------------------------------------------------------
# GroupedWinsorizer
# ---------------------------------------------------------------------------


class TestGroupedWinsorizerFit:
    def test_fit_sets_upper_caps(self, grouped_df):
        w = GroupedWinsorizer(group_col="Company", cols=["Volume"], q=0.99)
        w.fit(grouped_df)
        assert "Volume" in w.upper_caps_
        assert "AAPL" in w.upper_caps_["Volume"]
        assert "MSFT" in w.upper_caps_["Volume"]

    def test_fit_sets_global_upper(self, grouped_df):
        w = GroupedWinsorizer(group_col="Company", cols=["Volume"], q=0.99)
        w.fit(grouped_df)
        assert "Volume" in w.global_upper_

    def test_fit_stores_feature_names(self, grouped_df):
        w = GroupedWinsorizer(group_col="Company", cols=["Volume"], q=0.99)
        w.fit(grouped_df)
        assert "Volume" in w.feature_names_in_

    def test_fit_lower_q_sets_lower_caps(self, grouped_df):
        w = GroupedWinsorizer(
            group_col="Company", cols=["Volume"], q=0.99, lower_q=0.01
        )
        w.fit(grouped_df)
        assert hasattr(w, "lower_caps_")
        assert "Volume" in w.lower_caps_

    def test_fit_raises_for_non_dataframe(self):
        w = GroupedWinsorizer(group_col="Company", cols=["Volume"])
        with pytest.raises(TypeError):
            w.fit([[1, 2], [3, 4]])

    def test_fit_raises_if_group_col_missing(self, grouped_df):
        w = GroupedWinsorizer(group_col="NonExistent", cols=["Volume"])
        with pytest.raises(ValueError, match="group_col"):
            w.fit(grouped_df)

    def test_fit_raises_if_cols_missing(self, grouped_df):
        w = GroupedWinsorizer(group_col="Company", cols=["MissingCol"])
        with pytest.raises(ValueError, match="Columns not found"):
            w.fit(grouped_df)

    def test_fit_raises_if_q_out_of_range(self, grouped_df):
        w = GroupedWinsorizer(group_col="Company", cols=["Volume"], q=1.5)
        with pytest.raises(ValueError, match="`q` must be in"):
            w.fit(grouped_df)

    def test_fit_raises_if_lower_q_exceeds_q(self, grouped_df):
        w = GroupedWinsorizer(group_col="Company", cols=["Volume"], q=0.5, lower_q=0.6)
        with pytest.raises(ValueError, match="`lower_q` must be in"):
            w.fit(grouped_df)

    def test_fit_raises_invalid_unseen_policy(self, grouped_df):
        w = GroupedWinsorizer(
            group_col="Company", cols=["Volume"], unseen_group_policy="invalid"
        )
        with pytest.raises(ValueError, match="unseen_group_policy"):
            w.fit(grouped_df)


class TestGroupedWinsorizerTransform:
    def test_clips_extreme_upper_values(self, grouped_df):
        w = GroupedWinsorizer(group_col="Company", cols=["Volume"], q=0.99)
        w.fit(grouped_df)
        out = w.transform(grouped_df)
        msft_cap = float(w.upper_caps_["Volume"]["MSFT"])
        assert out.loc[out["Company"] == "MSFT", "Volume"].max() <= msft_cap + 1

    def test_preserves_shape(self, grouped_df):
        w = GroupedWinsorizer(group_col="Company", cols=["Volume"], q=0.99)
        w.fit(grouped_df)
        out = w.transform(grouped_df)
        assert out.shape == grouped_df.shape

    def test_non_target_columns_unchanged(self, grouped_df):
        w = GroupedWinsorizer(group_col="Company", cols=["Volume"], q=0.99)
        w.fit(grouped_df)
        out = w.transform(grouped_df)
        pd.testing.assert_series_equal(out["Close"], grouped_df["Close"])

    def test_transform_before_fit_raises(self, grouped_df):
        w = GroupedWinsorizer(group_col="Company", cols=["Volume"], q=0.99)
        with pytest.raises(RuntimeError, match="must be fit"):
            w.transform(grouped_df)

    def test_unseen_group_uses_global_cap(self, grouped_df):
        w = GroupedWinsorizer(
            group_col="Company", cols=["Volume"], q=0.99, unseen_group_policy="global"
        )
        w.fit(grouped_df)
        new_row = pd.DataFrame(
            {"Company": ["TSLA"], "Volume": [999_999_999], "Close": [100.0]}
        )
        out = w.transform(new_row)
        assert out["Volume"].iloc[0] <= w.global_upper_["Volume"] + 1

    def test_unseen_group_passthrough_policy(self, grouped_df):
        w = GroupedWinsorizer(
            group_col="Company",
            cols=["Volume"],
            q=0.99,
            unseen_group_policy="passthrough",
        )
        w.fit(grouped_df)
        extreme_val = 999_999_999
        new_row = pd.DataFrame(
            {"Company": ["TSLA"], "Volume": [extreme_val], "Close": [100.0]}
        )
        out = w.transform(new_row)
        assert out["Volume"].iloc[0] == extreme_val

    def test_lower_q_clips_low_values(self, grouped_df):
        w = GroupedWinsorizer(
            group_col="Company", cols=["Volume"], q=0.99, lower_q=0.01
        )
        w.fit(grouped_df)
        out = w.transform(grouped_df)
        for company in ["AAPL", "MSFT"]:
            lower_cap = float(w.lower_caps_["Volume"][company])
            assert out.loc[out["Company"] == company, "Volume"].min() >= lower_cap - 1

    def test_returns_dataframe(self, grouped_df):
        w = GroupedWinsorizer(group_col="Company", cols=["Volume"], q=0.99)
        w.fit(grouped_df)
        out = w.transform(grouped_df)
        assert isinstance(out, pd.DataFrame)

    def test_get_feature_names_out(self, grouped_df):
        w = GroupedWinsorizer(group_col="Company", cols=["Volume"], q=0.99)
        w.fit(grouped_df)
        names = w.get_feature_names_out()
        assert "Volume" in names


# ---------------------------------------------------------------------------
# GlobalWinsorizer
# ---------------------------------------------------------------------------


class TestGlobalWinsorizerFit:
    def test_fit_sets_upper_and_lower_caps(self, global_df):
        w = GlobalWinsorizer(cols=["return_1d"], upper_q=0.99, lower_q=0.01)
        w.fit(global_df)
        assert "return_1d" in w.upper_caps_
        assert "return_1d" in w.lower_caps_

    def test_upper_cap_greater_than_lower_cap(self, global_df):
        w = GlobalWinsorizer(cols=["return_1d"], upper_q=0.99, lower_q=0.01)
        w.fit(global_df)
        assert w.upper_caps_["return_1d"] > w.lower_caps_["return_1d"]

    def test_fit_stores_feature_names(self, global_df):
        w = GlobalWinsorizer(cols=["return_1d"], upper_q=0.99, lower_q=0.01)
        w.fit(global_df)
        assert "return_1d" in w.feature_names_in_

    def test_raises_for_non_dataframe(self):
        w = GlobalWinsorizer(cols=["x"])
        with pytest.raises(TypeError):
            w.fit(np.array([[1, 2], [3, 4]]))

    def test_raises_if_cols_none(self, global_df):
        w = GlobalWinsorizer(cols=None)
        with pytest.raises(ValueError, match="`cols` must be specified"):
            w.fit(global_df)

    def test_raises_if_col_missing(self, global_df):
        w = GlobalWinsorizer(cols=["nonexistent"])
        with pytest.raises(ValueError, match="Columns not found"):
            w.fit(global_df)

    def test_raises_if_quantile_order_invalid(self, global_df):
        w = GlobalWinsorizer(cols=["return_1d"], upper_q=0.01, lower_q=0.99)
        with pytest.raises(ValueError, match="`lower_q` must be in"):
            w.fit(global_df)


class TestGlobalWinsorizerTransform:
    def test_clips_extreme_upper(self, global_df):
        w = GlobalWinsorizer(cols=["return_1d"], upper_q=0.99, lower_q=0.01)
        w.fit(global_df)
        out = w.transform(global_df)
        assert out["return_1d"].max() <= w.upper_caps_["return_1d"]

    def test_clips_extreme_lower(self, global_df):
        w = GlobalWinsorizer(cols=["return_1d"], upper_q=0.99, lower_q=0.01)
        w.fit(global_df)
        out = w.transform(global_df)
        assert out["return_1d"].min() >= w.lower_caps_["return_1d"]

    def test_preserves_shape(self, global_df):
        w = GlobalWinsorizer(cols=["return_1d"], upper_q=0.99, lower_q=0.01)
        w.fit(global_df)
        assert w.transform(global_df).shape == global_df.shape

    def test_untargeted_columns_unchanged(self, global_df):
        w = GlobalWinsorizer(cols=["return_1d"], upper_q=0.99, lower_q=0.01)
        w.fit(global_df)
        out = w.transform(global_df)
        pd.testing.assert_series_equal(out["log_return"], global_df["log_return"])

    def test_transform_before_fit_raises(self, global_df):
        w = GlobalWinsorizer(cols=["return_1d"])
        with pytest.raises(RuntimeError, match="must be fit"):
            w.transform(global_df)

    def test_handles_integer_dtype(self):
        df = pd.DataFrame({"vol": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]})
        w = GlobalWinsorizer(cols=["vol"], upper_q=0.9, lower_q=0.1)
        w.fit(df)
        out = w.transform(df)
        assert out["vol"].dtype == float

    def test_missing_col_in_transform_is_skipped(self, global_df):
        w = GlobalWinsorizer(cols=["return_1d"], upper_q=0.99, lower_q=0.01)
        w.fit(global_df)
        df_no_col = global_df.drop(columns=["return_1d"])
        out = w.transform(df_no_col)
        assert "log_return" in out.columns

    def test_get_feature_names_out_with_input(self, global_df):
        w = GlobalWinsorizer(cols=["return_1d"], upper_q=0.99, lower_q=0.01)
        w.fit(global_df)
        names = w.get_feature_names_out(input_features=["return_1d", "log_return"])
        assert list(names) == ["return_1d", "log_return"]


# ---------------------------------------------------------------------------
# ColumnDropper
# ---------------------------------------------------------------------------


class TestColumnDropper:
    @pytest.fixture()
    def base_df(self) -> pd.DataFrame:
        return pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})

    def test_drops_specified_columns(self, base_df):
        cd = ColumnDropper(cols=["a", "b"])
        cd.fit(base_df)
        out = cd.transform(base_df)
        assert "a" not in out.columns
        assert "b" not in out.columns

    def test_keeps_unspecified_columns(self, base_df):
        cd = ColumnDropper(cols=["a"])
        cd.fit(base_df)
        out = cd.transform(base_df)
        assert "b" in out.columns
        assert "c" in out.columns

    def test_silently_skips_absent_cols(self, base_df):
        cd = ColumnDropper(cols=["a", "nonexistent"])
        cd.fit(base_df)
        out = cd.transform(base_df)
        assert "a" not in out.columns
        assert out.shape[1] == 3

    def test_empty_drop_list_keeps_all_cols(self, base_df):
        cd = ColumnDropper(cols=[])
        cd.fit(base_df)
        out = cd.transform(base_df)
        assert set(out.columns) == set(base_df.columns)

    def test_drop_all_cols(self, base_df):
        cd = ColumnDropper(cols=["a", "b", "c", "d"])
        cd.fit(base_df)
        out = cd.transform(base_df)
        assert out.empty or out.shape[1] == 0

    def test_get_feature_names_out_excludes_dropped(self, base_df):
        cd = ColumnDropper(cols=["a", "b"])
        cd.fit(base_df)
        names = cd.get_feature_names_out()
        assert "a" not in names
        assert "b" not in names
        assert "c" in names

    def test_get_feature_names_out_with_input_features(self, base_df):
        cd = ColumnDropper(cols=["a"])
        cd.fit(base_df)
        names = cd.get_feature_names_out(input_features=["a", "b", "c"])
        assert list(names) == ["b", "c"]

    def test_fit_stores_feature_names_in(self, base_df):
        cd = ColumnDropper(cols=["a"])
        cd.fit(base_df)
        assert "a" in cd.feature_names_in_
        assert "b" in cd.feature_names_in_

    def test_fit_only_drops_present_cols(self, base_df):
        cd = ColumnDropper(cols=["a", "z"])
        cd.fit(base_df)
        assert cd.cols_to_drop_ == ["a"]
