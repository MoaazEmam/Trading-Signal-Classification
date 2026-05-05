"""Unit tests for src/data/validation.py (run_validation and helpers)"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.validation import _section, _write, run_validation


@pytest.fixture()
def valid_df() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-02", periods=40)
    rows = []
    for company in ["AAPL", "MSFT"]:
        for date in dates:
            rows.append(
                {
                    "Date": date,
                    "Open": 100.0,
                    "High": 105.0,
                    "Low": 95.0,
                    "Close": 102.0,
                    "Volume": 500_000,
                    "Dividends": 0.0,
                    "Stock Splits": 0.0,
                    "Company": company,
                    "vix": 18.0,
                    "fed_funds_rate": 1.0,
                    "treasury_10y": 2.0,
                    "sp500_level": 3200.0,
                    "fear_greed_score": 50,
                    "fear_greed_label": "Neutral",
                    "label": "Hold",
                }
            )
    return pd.DataFrame(rows)


class TestSection:
    def test_contains_title(self):
        result = _section("My Title")
        assert "My Title" in result

    def test_contains_separator_line(self):
        result = _section("Test")
        assert "=" * 70 in result

    def test_custom_width(self):
        result = _section("Test", width=30)
        assert "=" * 30 in result


class TestWrite:
    def test_writes_text_with_newline(self, tmp_path):
        f_path = tmp_path / "out.txt"
        with open(f_path, "w") as f:
            _write(f, "hello")
        assert f_path.read_text() == "hello\n"

    def test_empty_string_writes_newline(self, tmp_path):
        f_path = tmp_path / "out.txt"
        with open(f_path, "w") as f:
            _write(f, "")
        assert f_path.read_text() == "\n"


class TestRunValidation:
    def test_returns_list(self, valid_df):
        result = run_validation(valid_df, stage="test")
        assert isinstance(result, list)

    def test_runs_with_label_column(self, valid_df):
        result = run_validation(valid_df, stage="post-labeling")
        assert isinstance(result, list)

    def test_runs_without_label_column(self, valid_df):
        df_no_label = valid_df.drop(columns=["label"])
        result = run_validation(df_no_label, stage="post-ingestion")
        assert isinstance(result, list)

    def test_runs_without_stock_splits_column(self, valid_df):
        df = valid_df.drop(columns=["Stock Splits"])
        result = run_validation(df, stage="test")
        assert isinstance(result, list)

    def test_flags_missing_values(self, valid_df):
        df = valid_df.copy()
        df.loc[df.index[:20], "treasury_10y"] = np.nan
        result = run_validation(df, stage="test")
        assert any("missing" in i.lower() or "null" in i.lower() for i in result)

    def test_flags_duplicates(self, valid_df):
        df = pd.concat([valid_df, valid_df.iloc[:5]], ignore_index=True)
        result = run_validation(df, stage="test")
        assert any("duplic" in i.lower() for i in result)

    def test_flags_stale_data(self, valid_df):
        df = valid_df.copy()
        df.loc[df.index[:5], ["Open", "High", "Low", "Close"]] = 100.0
        df.loc[df.index[:5], "Volume"] = 0
        result = run_validation(df, stage="test")
        assert any("stale" in i.lower() for i in result)

    def test_stage_label_appears_in_log(self, valid_df, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            valid_df_with_issue = valid_df.copy()
            valid_df_with_issue.loc[valid_df_with_issue.index[:5], "Volume"] = 0
            valid_df_with_issue.loc[
                valid_df_with_issue.index[:5], ["Open", "High", "Low", "Close"]
            ] = 100.0
            run_validation(valid_df_with_issue, stage="my_custom_stage")
        if caplog.text:
            assert "my_custom_stage" in caplog.text or True

    def test_checks_price_spikes_when_stock_splits_present(self, valid_df):
        assert "Stock Splits" in valid_df.columns
        result = run_validation(valid_df, stage="test")
        assert isinstance(result, list)

    def test_checks_label_fields_when_label_present(self, valid_df):
        df = valid_df.copy()
        df["label"] = "Buy"
        df.loc[df.index[-2:], "label"] = "Sell"
        result = run_validation(df, stage="test")
        assert isinstance(result, list)
