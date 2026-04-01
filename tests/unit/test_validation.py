import numpy as np
import pandas as pd
import pytest

from src.data.validation_helper import Validator


def _issues_containing(validator: Validator, keyword: str) -> list:
    return [i for i in validator.issues if keyword.lower() in i.lower()]


class TestCheckMissing:
    def test_no_issues_on_clean_df(self, clean_validation_df):
        v = Validator(clean_validation_df)
        v.check_missing()
        assert _issues_containing(v, "missing") == []

    def test_flags_missing_column(self, missing_values_df):
        v = Validator(missing_values_df)
        v.check_missing()
        assert len(_issues_containing(v, "missing")) > 0

    def test_high_severity_above_20_pct(self, missing_values_df):
        v = Validator(missing_values_df)
        v.check_missing()
        assert any("HIGH" in i for i in v.issues)

    def test_low_severity_for_small_missing(self, clean_validation_df):
        df = clean_validation_df.copy()
        df.loc[df.index[0], "treasury_10y"] = np.nan
        v = Validator(df)
        v.check_missing()
        assert any("LOW" in i for i in v.issues)


class TestCheckDtypes:
    def test_no_issues_on_clean_df(self, clean_validation_df):
        v = Validator(clean_validation_df)
        v.check_dtypes()
        assert _issues_containing(v, "dtype") == []

    def test_flags_numeric_column_as_string(self, wrong_dtypes_df):
        v = Validator(wrong_dtypes_df)
        v.check_dtypes()
        assert len(_issues_containing(v, "dtype")) > 0


class TestCheckDuplicates:
    def test_no_issues_on_clean_df(self, clean_validation_df):
        v = Validator(clean_validation_df)
        v.check_duplicates()
        assert _issues_containing(v, "duplicate") == []

    def test_flags_fully_duplicate_rows(self, duplicate_rows_df):
        v = Validator(duplicate_rows_df)
        v.check_duplicates()
        assert len(_issues_containing(v, "duplicate")) > 0

    def test_flags_duplicate_date_company_pairs(self, clean_validation_df):
        dup = clean_validation_df.iloc[:1].copy()
        df = pd.concat([clean_validation_df, dup], ignore_index=True)
        v = Validator(df)
        v.check_duplicates()
        assert len(_issues_containing(v, "duplicate")) > 0


class TestCheckClassDistribution:
    def test_no_issues_on_balanced_labels(self, clean_validation_df):
        df = clean_validation_df.copy()
        df["label"] = (["Buy", "Sell", "Hold"] * (len(df) // 3 + 1))[: len(df)]
        v = Validator(df)
        v.check_class_distribution()
        assert _issues_containing(v, "balance") == []

    def test_flags_imbalanced_labels(self, imbalanced_labels_df):
        v = Validator(imbalanced_labels_df)
        v.check_class_distribution()
        assert len(_issues_containing(v, "balance")) > 0


class TestCheckCompanyRows:
    def test_no_issues_on_consistent_coverage(self, clean_validation_df):
        v = Validator(clean_validation_df)
        v.check_company_rows()
        assert _issues_containing(v, "date coverage") == []

    def test_flags_inconsistent_coverage(self, clean_validation_df):
        drop_date = clean_validation_df["Date"].iloc[0]
        df = clean_validation_df[
            ~((clean_validation_df["Company"] == "AAPL") & (clean_validation_df["Date"] == drop_date))
        ]
        v = Validator(df)
        v.check_company_rows()
        assert len(_issues_containing(v, "date coverage")) > 0


class TestCheckDateGaps:
    def test_no_issues_on_continuous_dates(self, clean_validation_df):
        v = Validator(clean_validation_df)
        v.check_date_gaps()
        assert _issues_containing(v, "date gaps") == []

    def test_flags_large_date_gap(self, clean_validation_df):
        mid = clean_validation_df["Date"].median()
        df = clean_validation_df[
            (clean_validation_df["Date"] < mid - pd.Timedelta(days=15))
            | (clean_validation_df["Date"] > mid + pd.Timedelta(days=15))
        ]
        v = Validator(df)
        v.check_date_gaps()
        assert len(_issues_containing(v, "date gaps")) > 0


class TestCheckOutliers:
    def test_no_issues_on_uniform_data(self, clean_validation_df):
        v = Validator(clean_validation_df)
        v.check_outliers()
        assert _issues_containing(v, "outlier") == []

    def test_flags_extreme_outliers(self, clean_validation_df):
        df = clean_validation_df.copy()
        n = int(len(df) * 0.10)
        df.loc[df.index[:n], "Volume"] = 999_999_999
        v = Validator(df)
        v.check_outliers()
        assert len(_issues_containing(v, "outlier")) > 0


class TestCheckPriceSpikes:
    def test_no_issues_on_clean_df(self, clean_validation_df):
        v = Validator(clean_validation_df)
        v.check_price_spikes()
        assert _issues_containing(v, "spike") == []

    def test_no_flag_when_split_present(self, clean_validation_df):
        df = clean_validation_df.copy().sort_values(["Company", "Date"]).reset_index(drop=True)
        idx = df[df["Company"] == "AAPL"].index[10]
        df.loc[idx, "Close"] = 999.0
        df.loc[idx, "Stock Splits"] = 2.0  # split recorded — should NOT flag
        v = Validator(df)
        v.check_price_spikes()
        assert _issues_containing(v, "spike") == []


class TestCheckSanity:
    def test_no_issues_on_clean_df(self, clean_validation_df):
        v = Validator(clean_validation_df)
        v.check_sanity()
        assert v.issues == []

    def test_flags_close_outside_high_low(self, sanity_fail_df):
        v = Validator(sanity_fail_df)
        v.check_sanity()
        assert len(_issues_containing(v, "close")) > 0

    def test_flags_non_positive_price(self, clean_validation_df):
        df = clean_validation_df.copy()
        df.loc[df.index[0], "Open"] = -1.0
        v = Validator(df)
        v.check_sanity()
        assert len(_issues_containing(v, "open")) > 0

    def test_flags_high_less_than_low(self, clean_validation_df):
        df = clean_validation_df.copy()
        df.loc[df.index[0], "High"] = 50.0
        v = Validator(df)
        v.check_sanity()
        assert len(_issues_containing(v, "high")) > 0

    def test_flags_zero_volume(self, clean_validation_df):
        df = clean_validation_df.copy()
        df.loc[df.index[:5], "Volume"] = 0
        v = Validator(df)
        v.check_sanity()
        assert len(_issues_containing(v, "volume")) > 0

    def test_flags_fear_greed_mismatch(self, clean_validation_df):
        df = clean_validation_df.copy()
        df.loc[df.index[:5], "fear_greed_score"] = 90
        df.loc[df.index[:5], "fear_greed_label"] = "Fear"
        v = Validator(df)
        v.check_sanity()
        assert len(_issues_containing(v, "fear")) > 0


class TestCheckStaleData:
    def test_no_issues_on_clean_df(self, clean_validation_df):
        v = Validator(clean_validation_df)
        v.check_stale_data()
        assert _issues_containing(v, "stale") == []

    def test_flags_stale_rows(self, stale_data_df):
        v = Validator(stale_data_df)
        v.check_stale_data()
        assert len(_issues_containing(v, "stale")) > 0


class TestCheckLabelConsistency:
    def test_no_issues_on_clean_df(self, clean_validation_df):
        v = Validator(clean_validation_df)
        v.check_label_consistency()
        assert _issues_containing(v, "consistency") == []

    def test_flags_conflicting_labels(self, clean_validation_df):
        dup = clean_validation_df.iloc[:1].copy()
        dup["label"] = "Sell"
        df = pd.concat([clean_validation_df, dup], ignore_index=True)
        v = Validator(df)
        v.check_label_consistency()
        assert len(_issues_containing(v, "consistency")) > 0


class TestRunAll:
    def test_returns_list(self, clean_validation_df):
        assert isinstance(Validator(clean_validation_df).run_all(), list)

    def test_clean_df_has_no_high_severity_issues(self, clean_validation_df):
        issues = Validator(clean_validation_df).run_all()
        assert not any("HIGH" in i for i in issues)

    def test_dirty_df_produces_issues(self, missing_values_df):
        assert len(Validator(missing_values_df).run_all()) > 0


class TestOnSampleDf:
    def test_validator_runs_without_exception(self, sample_df):
        """
        If sample has no label column (generated before labeling ran),
        check_class_distribution will KeyError. Re-run save_sample() after labeling.
        """
        # Run only checks that don't require the label column if it's missing
        v = Validator(sample_df)
        if "label" not in sample_df.columns:
            pytest.skip("Sample has no label column — re-run save_sample() after labeling.py")
        issues = v.run_all()
        assert isinstance(issues, list)

    def test_no_fully_duplicate_rows_in_sample(self, sample_df):
        v = Validator(sample_df)
        v.check_duplicates()
        fully_dup = [i for i in v.issues if "fully duplicate" in i.lower()]
        assert fully_dup == [], f"Duplicate rows in sample: {fully_dup}"

    def test_no_corrupt_ohlc_rows_in_sample(self, sample_df):
        """
        1 corrupt row is acceptable (known data quality issue in source).
        Flag only if more than 1 row is corrupt.
        """
        v = Validator(sample_df)
        v.check_sanity()
        close_issues = [i for i in v.issues if "close is outside" in i.lower()]
        # Extract count from issue string e.g. "Sanity:1 rows where..."
        if close_issues:
            count = int(close_issues[0].split(":")[1].split()[0])
            assert count <= 1, f"Too many corrupt OHLC rows: {close_issues}"

    def test_expected_columns_present_in_sample(self, sample_df):
        expected_without_label = {
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
        missing = expected_without_label - set(sample_df.columns)
        assert not missing, f"Columns missing from sample: {missing}"

    def test_date_column_is_datetime(self, sample_df):
        assert pd.api.types.is_datetime64_any_dtype(sample_df["Date"])
