"""Unit tests for helper functions in src/backtesting/engine.py"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.backtesting.engine import _serialize_results, validate_alignment

# ---------------------------------------------------------------------------
# _serialize_results
# ---------------------------------------------------------------------------


class TestSerializeResults:
    def test_finite_float_passes_through(self):
        result = _serialize_results({"value": 3.14})
        assert result["value"] == 3.14

    def test_inf_converted_to_string(self):
        result = _serialize_results({"pf": float("inf")})
        assert result["pf"] == "inf"

    def test_neg_inf_converted_to_string(self):
        result = _serialize_results({"sharpe": float("-inf")})
        assert result["sharpe"] == "-inf"

    def test_nan_converted_to_string(self):
        result = _serialize_results({"win_rate": float("nan")})
        assert result["win_rate"] == "nan"

    def test_nested_dict_is_walked(self):
        result = _serialize_results({"metrics": {"sharpe": float("inf"), "ret": 5.0}})
        assert result["metrics"]["sharpe"] == "inf"
        assert result["metrics"]["ret"] == 5.0

    def test_list_is_walked(self):
        result = _serialize_results({"vals": [1.0, float("inf"), float("nan")]})
        assert result["vals"][0] == 1.0
        assert result["vals"][1] == "inf"
        assert result["vals"][2] == "nan"

    def test_integer_passes_through(self):
        result = _serialize_results({"count": 42})
        assert result["count"] == 42

    def test_string_passes_through(self):
        result = _serialize_results({"model": "lightgbm"})
        assert result["model"] == "lightgbm"

    def test_datetime_converted_to_isoformat(self):
        dt = datetime(2024, 1, 15, 12, 30, 0)
        result = _serialize_results({"ts": dt})
        assert result["ts"] == dt.isoformat()

    def test_none_passes_through(self):
        result = _serialize_results({"val": None})
        assert result["val"] is None

    def test_deeply_nested_structure(self):
        data = {
            "level1": {
                "level2": [
                    {"value": float("inf")},
                    {"value": 1.0},
                ]
            }
        }
        result = _serialize_results(data)
        assert result["level1"]["level2"][0]["value"] == "inf"
        assert result["level1"]["level2"][1]["value"] == 1.0

    def test_empty_dict_is_handled(self):
        result = _serialize_results({})
        assert result == {}

    def test_empty_list_is_handled(self):
        result = _serialize_results({"trades": []})
        assert result["trades"] == []


# ---------------------------------------------------------------------------
# validate_alignment
# ---------------------------------------------------------------------------


class TestValidateAlignment:
    def test_no_warning_when_fully_aligned(self, caplog):
        import logging

        companies = ["AAPL", "MSFT"]
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        mi = pd.MultiIndex.from_product([companies, dates], names=["Company", "Date"])

        test_df = pd.DataFrame(index=mi, data={"feature": range(len(mi))})
        prices_df = pd.DataFrame(index=mi, data={"Close": range(len(mi))})

        with caplog.at_level(logging.INFO):
            validate_alignment(test_df, prices_df)

        assert "matched" in caplog.text

    def test_warning_when_missing_pairs(self, caplog):
        import logging

        companies = ["AAPL", "MSFT"]
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        mi_full = pd.MultiIndex.from_product(
            [companies, dates], names=["Company", "Date"]
        )
        mi_partial = pd.MultiIndex.from_arrays(
            [["AAPL"], [pd.Timestamp("2024-01-02")]], names=["Company", "Date"]
        )

        test_df = pd.DataFrame(index=mi_full, data={"feature": range(len(mi_full))})
        prices_df = pd.DataFrame(index=mi_partial, data={"Close": [100.0]})

        with caplog.at_level(logging.WARNING):
            validate_alignment(test_df, prices_df)

        assert "missing" in caplog.text.lower() or "pairs" in caplog.text.lower()

    def test_logs_total_and_matched_count(self, caplog):
        import logging

        companies = ["AAPL", "MSFT"]
        dates = pd.to_datetime(["2024-01-02"])
        mi = pd.MultiIndex.from_product([companies, dates], names=["Company", "Date"])

        test_df = pd.DataFrame(index=mi, data={"f": [1, 2]})
        prices_df = pd.DataFrame(index=mi, data={"Close": [100.0, 200.0]})

        with caplog.at_level(logging.INFO):
            validate_alignment(test_df, prices_df)

        combined = " ".join(caplog.messages)
        assert "2" in combined  # total pairs logged

    def test_fully_missing_prices(self, caplog):
        import logging

        mi_test = pd.MultiIndex.from_arrays(
            [["AAPL", "MSFT"], pd.to_datetime(["2024-01-02", "2024-01-02"])],
            names=["Company", "Date"],
        )
        mi_prices = pd.MultiIndex.from_arrays(
            [["GOOG"], pd.to_datetime(["2024-01-02"])],
            names=["Company", "Date"],
        )
        test_df = pd.DataFrame(index=mi_test, data={"f": [1, 2]})
        prices_df = pd.DataFrame(index=mi_prices, data={"Close": [300.0]})

        with caplog.at_level(logging.WARNING):
            validate_alignment(test_df, prices_df)

        assert len(caplog.records) > 0
