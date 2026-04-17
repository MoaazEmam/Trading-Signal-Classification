"""
Unit tests for src/data/cleaning.py

Tests the Cleaner class methods for data quality validation and correction.
Covers accuracy (invalid prices), consistency (dtypes), completeness (missing values),
uniqueness (duplicates), and timeliness (stale data) aspects.
"""

import pandas as pd

from src.data.cleaning import Cleaner


class TestDropInvalidPrices:
    """Test invalid price detection and removal."""

    def test_drops_rows_where_high_less_than_low(self, cleaning_high_low_invalid_df):
        cleaner = Cleaner(cleaning_high_low_invalid_df)
        cleaner.drop_invalid_prices()

        # Should have dropped 2 rows
        assert len(cleaner.df) == len(cleaning_high_low_invalid_df) - 2
        # No rows should have High < Low
        assert (cleaner.df["High"] < cleaner.df["Low"]).sum() == 0

    def test_drops_rows_where_close_greater_than_high(self, cleaning_close_outside_range_df):
        cleaner = Cleaner(cleaning_close_outside_range_df)
        cleaner.drop_invalid_prices()

        # Should have dropped rows where Close > High
        assert (cleaner.df["Close"] > cleaner.df["High"]).sum() == 0

    def test_drops_rows_where_close_less_than_low(self, cleaning_close_outside_range_df):
        cleaner = Cleaner(cleaning_close_outside_range_df)
        cleaner.drop_invalid_prices()

        # Should have dropped rows where Close < Low
        assert (cleaner.df["Close"] < cleaner.df["Low"]).sum() == 0

    def test_drops_rows_with_negative_prices(self, cleaning_negative_prices_df):
        cleaner = Cleaner(cleaning_negative_prices_df)
        cleaner.drop_invalid_prices()

        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            assert (cleaner.df[col] <= 0).sum() == 0

    def test_logs_dropped_rows(self, cleaning_high_low_invalid_df):
        cleaner = Cleaner(cleaning_high_low_invalid_df)
        cleaner.drop_invalid_prices()

        assert len(cleaner.log) > 0
        assert any("drop_invalid_prices" in entry for entry in cleaner.log)

    def test_quarantines_dropped_rows(self, cleaning_high_low_invalid_df):
        cleaner = Cleaner(cleaning_high_low_invalid_df)
        cleaner.drop_invalid_prices()

        # Quarantine should have entries
        assert len(cleaner.quarantine) > 0

    def test_no_rows_dropped_when_all_valid(self, cleaning_base_df):
        cleaner = Cleaner(cleaning_base_df)
        original_len = len(cleaner.df)
        cleaner.drop_invalid_prices()

        # No rows should be dropped
        assert len(cleaner.df) == original_len

    def test_handles_zero_prices(self, cleaning_base_df):
        df = cleaning_base_df.copy()
        df.loc[0, "Open"] = 0.0
        cleaner = Cleaner(df)
        cleaner.drop_invalid_prices()

        assert (cleaner.df["Open"] == 0.0).sum() == 0


class TestFixDtypes:
    """Test dtype fixing and data standardization."""

    def test_replaces_implicit_missing_with_nan(self, cleaning_implicit_missing_df):
        cleaner = Cleaner(cleaning_implicit_missing_df)
        cleaner.fix_dtypes()

        # Should have NaN instead of "N/A", "null", etc.
        assert cleaner.df["vix"].isna().any()
        assert cleaner.df["fed_funds_rate"].isna().any()

    def test_converts_numeric_columns_to_numeric(self, cleaning_wrong_dtypes_df):
        cleaner = Cleaner(cleaning_wrong_dtypes_df)
        cleaner.fix_dtypes()

        # Close and vix should be numeric
        assert pd.api.types.is_numeric_dtype(cleaner.df["Close"])
        assert pd.api.types.is_numeric_dtype(cleaner.df["vix"])

    def test_parses_date_column(self, cleaning_wrong_dtypes_df):
        cleaner = Cleaner(cleaning_wrong_dtypes_df)
        cleaner.fix_dtypes()

        assert pd.api.types.is_datetime64_any_dtype(cleaner.df["Date"])

    def test_standardizes_string_columns_to_title_case(self, cleaning_string_case_df):
        cleaner = Cleaner(cleaning_string_case_df)
        cleaner.fix_dtypes()

        # Check Company column is title-cased
        companies = cleaner.df["Company"].dropna().unique()
        for company in companies:
            assert company == company.title()

    def test_strips_whitespace_from_strings(self, cleaning_string_whitespace_df):
        cleaner = Cleaner(cleaning_string_whitespace_df)
        cleaner.fix_dtypes()

        # Check no whitespace at edges
        for col in ["Company", "fear_greed_label", "label"]:
            if col in cleaner.df.columns:
                non_null = cleaner.df[col].dropna()
                for val in non_null:
                    assert val == val.strip()

    def test_logs_dtype_conversions(self, cleaning_wrong_dtypes_df):
        cleaner = Cleaner(cleaning_wrong_dtypes_df)
        cleaner.fix_dtypes()

        assert len(cleaner.log) > 0
        assert any("fix_dtypes" in entry for entry in cleaner.log)

    def test_handles_unparseable_numeric_values(self, cleaning_base_df):
        df = cleaning_base_df.copy()
        df.loc[0, "vix"] = "abc"  # Unparseable
        cleaner = Cleaner(df)
        cleaner.fix_dtypes()

        # Should convert unparseable to NaN
        assert cleaner.df["vix"].isna().any()

    def test_handles_unparseable_dates(self, cleaning_base_df):
        df = cleaning_base_df.copy()
        df.loc[0, "Date"] = "invalid-date"
        cleaner = Cleaner(df)
        cleaner.fix_dtypes()

        # Should convert unparseable to NaT
        assert cleaner.df["Date"].isna().any()

    def test_converts_none_to_na_in_strings(self, cleaning_base_df):
        df = cleaning_base_df.copy()
        df.loc[0, "Company"] = "None"
        cleaner = Cleaner(df)
        cleaner.fix_dtypes()

        # "None" should be converted to NA
        assert cleaner.df.loc[0, "Company"] is pd.NA or pd.isna(cleaner.df.loc[0, "Company"])

    def test_df_copy_not_modified(self, cleaning_base_df):
        original_len = len(cleaning_base_df)
        cleaner = Cleaner(cleaning_base_df)
        cleaner.fix_dtypes()

        # Original df should not be modified
        assert len(cleaning_base_df) == original_len


class TestDropMissing:
    """Test missing value handling."""

    def test_drops_rows_missing_critical_date(self, cleaning_missing_critical_df):
        cleaner = Cleaner(cleaning_missing_critical_df)
        cleaner.drop_missing()

        # Should drop row with missing Date
        assert cleaner.df["Date"].isna().sum() == 0

    def test_drops_rows_missing_critical_company(self, cleaning_missing_critical_df):
        cleaner = Cleaner(cleaning_missing_critical_df)
        cleaner.drop_missing()

        # Should drop row with missing Company
        assert cleaner.df["Company"].isna().sum() == 0

    def test_drops_rows_missing_critical_close(self, cleaning_missing_critical_df):
        cleaner = Cleaner(cleaning_missing_critical_df)
        cleaner.drop_missing()

        # Should drop row with missing Close
        assert cleaner.df["Close"].isna().sum() == 0

    def test_drops_rows_missing_critical_label(self, cleaning_missing_critical_df):
        cleaner = Cleaner(cleaning_missing_critical_df)
        cleaner.drop_missing()

        # Should drop row with missing label
        assert cleaner.df["label"].isna().sum() == 0

    def test_drops_low_pct_missing_noncritical(self, cleaning_missing_noncritical_low_pct_df):
        cleaner = Cleaner(cleaning_missing_noncritical_low_pct_df)
        cleaner.drop_missing()

        # Should drop rows with <5% missing vix and fed_funds_rate
        assert len(cleaner.df) < len(cleaning_missing_noncritical_low_pct_df)

    def test_keeps_high_pct_missing_noncritical(self, cleaning_missing_noncritical_high_pct_df):
        cleaner = Cleaner(cleaning_missing_noncritical_high_pct_df)
        cleaner.drop_missing()

        # Should NOT drop rows when >=5% missing
        assert cleaner.df["treasury_10y"].isna().sum() > 0

    def test_logs_dropped_missing(self, cleaning_missing_critical_df):
        cleaner = Cleaner(cleaning_missing_critical_df)
        cleaner.drop_missing()

        assert len(cleaner.log) > 0
        assert any("drop_missing" in entry for entry in cleaner.log)

    def test_quarantines_missing_rows(self, cleaning_missing_critical_df):
        cleaner = Cleaner(cleaning_missing_critical_df)
        cleaner.drop_missing()

        assert len(cleaner.quarantine) > 0

    def test_no_rows_dropped_when_no_missing(self, cleaning_base_df):
        cleaner = Cleaner(cleaning_base_df)
        original_len = len(cleaner.df)
        cleaner.drop_missing()

        assert len(cleaner.df) == original_len


class TestDropDuplicates:
    """Test duplicate detection and removal."""

    def test_drops_full_duplicates(self, cleaning_full_duplicates_df):
        cleaner = Cleaner(cleaning_full_duplicates_df)
        cleaner.drop_duplicates()

        # Should remove all duplicate rows
        assert cleaner.df.duplicated().sum() == 0

    def test_drops_date_company_duplicates(self, cleaning_date_company_duplicates_df):
        cleaner = Cleaner(cleaning_date_company_duplicates_df)
        cleaner.drop_duplicates()

        # Should have only first occurrence of each (Date, Company) pair
        assert cleaner.df.duplicated(subset=["Date", "Company"]).sum() == 0

    def test_logs_dropped_full_duplicates(self, cleaning_full_duplicates_df):
        cleaner = Cleaner(cleaning_full_duplicates_df)
        cleaner.drop_duplicates()

        assert any("drop_duplicates" in entry for entry in cleaner.log)

    def test_logs_dropped_date_company_duplicates(self, cleaning_date_company_duplicates_df):
        cleaner = Cleaner(cleaning_date_company_duplicates_df)
        cleaner.drop_duplicates()

        assert any("drop_duplicates" in entry and "Date, Company" in entry for entry in cleaner.log)

    def test_quarantines_duplicates(self, cleaning_full_duplicates_df):
        cleaner = Cleaner(cleaning_full_duplicates_df)
        cleaner.drop_duplicates()

        assert len(cleaner.quarantine) > 0

    def test_keeps_first_occurrence(self, cleaning_full_duplicates_df):
        cleaner = Cleaner(cleaning_full_duplicates_df)
        original_first = cleaner.df.iloc[0].copy()
        cleaner.drop_duplicates()

        # First row should still exist
        assert (cleaner.df.iloc[0] == original_first).all()

    def test_no_rows_dropped_when_no_duplicates(self, cleaning_base_df):
        cleaner = Cleaner(cleaning_base_df)
        original_len = len(cleaner.df)
        cleaner.drop_duplicates()

        assert len(cleaner.df) == original_len


class TestDropStaleRows:
    """Test stale data detection and removal."""

    def test_drops_flat_prices_with_zero_volume(self, cleaning_stale_flat_df):
        cleaner = Cleaner(cleaning_stale_flat_df)
        cleaner.drop_stale_rows()

        # Should drop flat price + 0 volume rows
        assert len(cleaner.df) < len(cleaning_stale_flat_df)

    def test_drops_identical_to_yesterday_with_zero_volume(self, cleaning_stale_identical_yesterday_df):
        cleaner = Cleaner(cleaning_stale_identical_yesterday_df)
        cleaner.drop_stale_rows()

        # Should drop rows identical to previous day with 0 volume
        assert len(cleaner.df) < len(cleaning_stale_identical_yesterday_df)

    def test_logs_dropped_stale_rows(self, cleaning_stale_flat_df):
        cleaner = Cleaner(cleaning_stale_flat_df)
        cleaner.drop_stale_rows()

        assert any("drop_stale_rows" in entry for entry in cleaner.log)

    def test_quarantines_stale_rows(self, cleaning_stale_flat_df):
        cleaner = Cleaner(cleaning_stale_flat_df)
        cleaner.drop_stale_rows()

        assert len(cleaner.quarantine) > 0

    def test_keeps_valid_zero_volume_rows(self, cleaning_base_df):
        df = cleaning_base_df.copy()
        # Zero volume but prices are different (not flat)
        df.loc[0, "Volume"] = 0
        df.loc[0, "Close"] = 99.0

        cleaner = Cleaner(df)
        original_len = len(cleaner.df)
        cleaner.drop_stale_rows()

        # Should not drop rows with 0 volume but different prices
        assert len(cleaner.df) == original_len

    def test_keeps_flat_prices_with_volume(self, cleaning_base_df):
        df = cleaning_base_df.copy()
        # Flat prices but non-zero volume
        df.loc[0, "Open"] = 100.0
        df.loc[0, "High"] = 100.0
        df.loc[0, "Low"] = 100.0
        df.loc[0, "Close"] = 100.0
        df.loc[0, "Volume"] = 1_000_000

        cleaner = Cleaner(df)
        original_len = len(cleaner.df)
        cleaner.drop_stale_rows()

        # Should not drop rows with volume
        assert len(cleaner.df) == original_len


class TestSaveQuarantine:
    """Test quarantine file saving."""

    def test_creates_quarantine_file(self, cleaning_full_duplicates_df, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PWD", str(tmp_path))

        cleaner = Cleaner(cleaning_full_duplicates_df)
        cleaner.drop_duplicates()
        cleaner._save_quarantine()

        quarantine_path = tmp_path / "data" / "processed" / "quarantine.csv"
        assert quarantine_path.exists()

    def test_quarantine_file_contains_rejected_rows(self, cleaning_full_duplicates_df, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        cleaner = Cleaner(cleaning_full_duplicates_df)
        cleaner.drop_duplicates()
        cleaner._save_quarantine()

        quarantine_path = tmp_path / "data" / "processed" / "quarantine.csv"
        quarantine_df = pd.read_csv(quarantine_path)

        assert len(quarantine_df) > 0

    def test_logs_quarantine_save(self, cleaning_full_duplicates_df, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        cleaner = Cleaner(cleaning_full_duplicates_df)
        cleaner.drop_duplicates()
        cleaner._save_quarantine()

        assert any("_save_quarantine" in entry for entry in cleaner.log)

    def test_handles_empty_quarantine(self, cleaning_base_df, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        cleaner = Cleaner(cleaning_base_df)
        cleaner._save_quarantine()

        # Should not create file if quarantine is empty
        quarantine_path = tmp_path / "data" / "processed" / "quarantine.csv"
        assert not quarantine_path.exists()

    def test_removes_duplicates_in_quarantine(self, cleaning_full_duplicates_df, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        cleaner = Cleaner(cleaning_full_duplicates_df)
        cleaner.drop_duplicates()
        cleaner._save_quarantine()

        quarantine_path = tmp_path / "data" / "processed" / "quarantine.csv"
        quarantine_df = pd.read_csv(quarantine_path)

        # Quarantine file should have unique rows
        assert quarantine_df.duplicated().sum() == 0


class TestRunAll:
    """Test full cleaning pipeline."""

    def test_run_all_returns_dataframe(self, cleaning_base_df):
        cleaner = Cleaner(cleaning_base_df)
        result = cleaner.run_all()

        assert isinstance(result, pd.DataFrame)

    def test_run_all_executes_all_steps(self, cleaning_high_low_invalid_df):
        cleaner = Cleaner(cleaning_high_low_invalid_df)
        cleaner.run_all()

        # Should have logs from multiple steps
        assert len(cleaner.log) >= 5  # At least fix_dtypes, drop_missing, drop_duplicates, etc.

    def test_run_all_no_high_less_than_low(self, cleaning_high_low_invalid_df):
        cleaner = Cleaner(cleaning_high_low_invalid_df)
        result = cleaner.run_all()

        assert (result["High"] < result["Low"]).sum() == 0

    def test_run_all_no_close_outside_range(self, cleaning_close_outside_range_df):
        cleaner = Cleaner(cleaning_close_outside_range_df)
        result = cleaner.run_all()

        assert ((result["Close"] > result["High"]) | (result["Close"] < result["Low"])).sum() == 0

    def test_run_all_no_negative_prices(self, cleaning_negative_prices_df):
        cleaner = Cleaner(cleaning_negative_prices_df)
        result = cleaner.run_all()

        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            assert (result[col] <= 0).sum() == 0

    def test_run_all_no_critical_missing(self, cleaning_missing_critical_df):
        cleaner = Cleaner(cleaning_missing_critical_df)
        result = cleaner.run_all()

        critical_cols = ["Date", "Company", "Close", "label"]
        for col in critical_cols:
            assert result[col].isna().sum() == 0

    def test_run_all_no_full_duplicates(self, cleaning_full_duplicates_df):
        cleaner = Cleaner(cleaning_full_duplicates_df)
        result = cleaner.run_all()

        assert result.duplicated().sum() == 0

    def test_run_all_no_date_company_duplicates(self, cleaning_date_company_duplicates_df):
        cleaner = Cleaner(cleaning_date_company_duplicates_df)
        result = cleaner.run_all()

        assert result.duplicated(subset=["Date", "Company"]).sum() == 0

    def test_run_all_correct_dtypes(self, cleaning_wrong_dtypes_df):
        cleaner = Cleaner(cleaning_wrong_dtypes_df)
        result = cleaner.run_all()

        assert pd.api.types.is_datetime64_any_dtype(result["Date"])
        assert pd.api.types.is_numeric_dtype(result["Close"])
        assert pd.api.types.is_numeric_dtype(result["vix"])

    def test_run_all_preserves_valid_data(self, cleaning_base_df):
        original_len = len(cleaning_base_df)
        cleaner = Cleaner(cleaning_base_df)
        result = cleaner.run_all()

        # Should preserve all valid rows
        assert len(result) == original_len

    def test_run_all_creates_log_entries(self, cleaning_base_df):
        cleaner = Cleaner(cleaning_base_df)
        cleaner.run_all()

        assert len(cleaner.log) > 0

    def test_run_all_with_complex_dirty_data(self):
        """Integration test with multiple issues."""
        dates = pd.bdate_range(start="2020-01-02", periods=20)
        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": [100.0] * 20,
                "High": [105.0] * 20,
                "Low": [95.0] * 20,
                "Close": [102.0] * 20,
                "Volume": [1_000_000] * 20,
                "Dividends": [0.0] * 20,
                "Stock Splits": [0.0] * 20,
                "Company": ["AAPL"] * 20,
                "vix": ["20.0"] * 20,  # Wrong dtype
                "fed_funds_rate": ["1.0"] * 20,  # Wrong dtype
                "treasury_10y": [2.0] * 20,
                "sp500_level": [3200.0] * 20,
                "fear_greed_score": [50] * 20,
                "fear_greed_label": ["neutral"] * 20,  # Wrong case
                "label": ["Hold"] * 20,
            }
        )

        # Add issues
        df.loc[0, "High"] = 90.0  # High < Low
        df.loc[1, "Close"] = 999.0  # Close > High
        df.loc[2, "vix"] = "N/A"  # Missing value

        cleaner = Cleaner(df)
        result = cleaner.run_all()

        # All issues should be resolved
        assert (result["High"] < result["Low"]).sum() == 0
        assert ((result["Close"] > result["High"]) | (result["Close"] < result["Low"])).sum() == 0
        assert pd.api.types.is_numeric_dtype(result["vix"])


class TestCleanerState:
    """Test Cleaner class state management."""

    def test_preserves_original_dataframe(self, cleaning_base_df):
        original_copy = cleaning_base_df.copy()
        cleaner = Cleaner(cleaning_base_df)
        cleaner.drop_missing()

        # Original passed df should be unchanged
        pd.testing.assert_frame_equal(cleaning_base_df, original_copy)

    def test_accumulates_log_entries(self, cleaning_base_df):
        cleaner = Cleaner(cleaning_base_df)
        initial_log_len = len(cleaner.log)

        cleaner.fix_dtypes()
        after_fix_dtypes = len(cleaner.log)
        assert after_fix_dtypes > initial_log_len

        cleaner.drop_missing()
        after_drop_missing = len(cleaner.log)
        assert after_drop_missing >= after_fix_dtypes

    def test_accumulates_quarantine(self, cleaning_full_duplicates_df):
        cleaner = Cleaner(cleaning_full_duplicates_df)
        initial_quarantine_len = len(cleaner.quarantine)

        cleaner.drop_duplicates()
        after_drop_duplicates = len(cleaner.quarantine)
        assert after_drop_duplicates > initial_quarantine_len

    def test_initial_state(self, cleaning_base_df):
        cleaner = Cleaner(cleaning_base_df)

        assert len(cleaner.log) == 0
        assert len(cleaner.quarantine) == 0
        assert len(cleaner.df) == len(cleaning_base_df)
