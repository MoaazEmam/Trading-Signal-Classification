"""
Unit tests for src/data/ingestion.py

Pure helper functions are tested with synthetic fixtures.
Network-dependent functions (run_ingestion, _fetch_*) are not tested here —
those belong in integration tests.

Where a sample_df exists, we also run smoke tests against real-shaped data
to catch issues that only appear with actual values (e.g. timezone edge cases,
column name mismatches after a Kaggle schema change).
"""

import pandas as pd

from src.data.ingestion import (
    _get_date_limits,
    _merge_on_date,
    _normalize_date,
)
from src.utils import _save_to_csv


class TestNormalizeDate:
    def test_removes_timezone(self):
        df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01 15:30:00+05:00"])})
        result = _normalize_date(df)
        assert result.dt.tz is None

    def test_strips_time_component(self):
        df = pd.DataFrame({"Date": pd.to_datetime(["2020-06-15 09:45:00"])})
        result = _normalize_date(df)
        assert result.iloc[0] == pd.Timestamp("2020-06-15")

    def test_already_normalized_is_unchanged(self):
        df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01"])})
        result = _normalize_date(df)
        assert result.iloc[0] == pd.Timestamp("2020-01-01")

    def test_custom_column_name(self):
        df = pd.DataFrame({"ts": pd.to_datetime(["2021-03-10 12:00:00"])})
        result = _normalize_date(df, date_col="ts")
        assert result.iloc[0] == pd.Timestamp("2021-03-10")

    def test_on_sample_df_dates_have_no_timezone(self, sample_df):
        """Real data: Date column must be tz-naive after ingestion."""
        result = _normalize_date(sample_df)
        assert result.dt.tz is None


class TestMergeOnDate:
    def test_columns_from_both_dfs_present(self, raw_kaggle_df, raw_fred_df):
        merged = _merge_on_date(raw_kaggle_df, raw_fred_df)
        for col in ["Close", "vix", "fed_funds_rate"]:
            assert col in merged.columns

    def test_row_count_equals_left_df(self, raw_kaggle_df, raw_fred_df):
        merged = _merge_on_date(raw_kaggle_df, raw_fred_df)
        assert len(merged) == len(raw_kaggle_df)

    def test_missing_dates_produce_nan(self, raw_kaggle_df):
        empty_fred = pd.DataFrame(
            {"Date": pd.Series(dtype="datetime64[ns]"), "vix": []}
        )
        merged = _merge_on_date(raw_kaggle_df, empty_fred)
        assert merged["vix"].isna().all()  # type: ignore

    def test_correct_value_joined(self):
        left = pd.DataFrame({"Date": pd.to_datetime(["2020-01-02"]), "Close": [100.0]})
        right = pd.DataFrame({"Date": pd.to_datetime(["2020-01-02"]), "vix": [20.0]})
        merged = _merge_on_date(left, right)
        assert merged.loc[0, "vix"] == 20.0

    def test_no_duplicate_rows_on_clean_input(self, raw_kaggle_df, raw_fred_df):
        merged = _merge_on_date(raw_kaggle_df, raw_fred_df)
        assert merged.duplicated(subset=["Date", "Company"]).sum() == 0

    def test_on_sample_df_expected_columns_present(self, sample_df):
        """Real data: all expected columns must survive the merge."""
        expected = {
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "vix",
            "fed_funds_rate",
            "treasury_10y",
            "sp500_level",
            "fear_greed_score",
            "fear_greed_label",
        }
        assert expected.issubset(set(sample_df.columns))


class TestGetDateLimits:
    def test_returns_correct_min_max(self, raw_kaggle_df):
        start, end = _get_date_limits(raw_kaggle_df)
        assert start == raw_kaggle_df["Date"].min()
        assert end == raw_kaggle_df["Date"].max()

    def test_single_row_df(self):
        df = pd.DataFrame({"Date": pd.to_datetime(["2021-05-10"])})
        start, end = _get_date_limits(df)
        assert start == end == pd.Timestamp("2021-05-10")

    def test_start_before_end(self, raw_kaggle_df):
        start, end = _get_date_limits(raw_kaggle_df)
        assert start < end

    def test_on_sample_df_start_before_end(self, sample_df):
        start, end = _get_date_limits(sample_df)
        assert start < end


class TestSaveToCsv:
    def test_creates_parent_directory(self, tmp_path, raw_kaggle_df):
        target = tmp_path / "nested" / "dir" / "out.csv"
        _save_to_csv(raw_kaggle_df, target)
        assert target.exists()

    def test_saved_csv_has_correct_shape(self, tmp_path, raw_kaggle_df):
        target = tmp_path / "out.csv"
        _save_to_csv(raw_kaggle_df, target)
        loaded = pd.read_csv(target)
        assert loaded.shape == raw_kaggle_df.shape

    def test_saved_csv_has_correct_columns(self, tmp_path, raw_kaggle_df):
        target = tmp_path / "out.csv"
        _save_to_csv(raw_kaggle_df, target)
        loaded = pd.read_csv(target)
        assert set(loaded.columns) == set(raw_kaggle_df.columns)
