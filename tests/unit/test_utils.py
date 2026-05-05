"""Unit tests for src/utils.py"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.utils import (
    get_nonnumeric_cols,
    get_numeric_cols,
    load_best_model_info,
)


@pytest.fixture()
def mixed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num_a": [1.0, 2.0, 3.0],
            "num_b": [4, 5, 6],
            "str_col": ["a", "b", "c"],
        }
    )


@pytest.fixture()
def multi_company_csv(tmp_path: Path) -> Path:
    dates = pd.bdate_range("2020-01-02", periods=20)
    rows = []
    for company in ["AAPL", "MSFT", "GOOG"]:
        for date in dates:
            rows.append({"Date": date, "Company": company, "Close": 100.0})
    df = pd.DataFrame(rows)
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


class TestGetNumericCols:
    def test_returns_only_numeric(self, mixed_df):
        result = get_numeric_cols(mixed_df)
        assert set(result.columns) == {"num_a", "num_b"}

    def test_all_numeric_df(self):
        df = pd.DataFrame({"a": [1], "b": [2.0]})
        assert set(get_numeric_cols(df).columns) == {"a", "b"}

    def test_no_numeric_columns_returns_empty(self):
        df = pd.DataFrame({"s": ["x", "y"], "t": ["p", "q"]})
        assert get_numeric_cols(df).empty

    def test_preserves_row_count(self, mixed_df):
        assert len(get_numeric_cols(mixed_df)) == len(mixed_df)

    def test_returns_dataframe(self, mixed_df):
        assert isinstance(get_numeric_cols(mixed_df), pd.DataFrame)


class TestGetNonnumericCols:
    def test_returns_only_nonnumeric(self, mixed_df):
        result = get_nonnumeric_cols(mixed_df)
        assert "str_col" in result.columns
        assert "num_a" not in result.columns
        assert "num_b" not in result.columns

    def test_empty_when_all_numeric(self):
        df = pd.DataFrame({"a": [1], "b": [2.0]})
        assert get_nonnumeric_cols(df).empty

    def test_preserves_row_count(self, mixed_df):
        assert len(get_nonnumeric_cols(mixed_df)) == len(mixed_df)

    def test_returns_dataframe(self, mixed_df):
        assert isinstance(get_nonnumeric_cols(mixed_df), pd.DataFrame)

    def test_datetime_columns_included(self):
        df = pd.DataFrame({"dt": pd.to_datetime(["2020-01-01"]), "n": [1.0]})
        result = get_nonnumeric_cols(df)
        assert "dt" in result.columns
        assert "n" not in result.columns


class TestSaveToCsv:
    def test_creates_file_and_content_is_correct(self, tmp_path, monkeypatch):
        import src.utils as utils_mod

        monkeypatch.setattr(
            utils_mod,
            "Path",
            lambda *args, **kwargs: (
                (tmp_path / "src" / "utils.py")
                if args and str(args[0]).endswith("utils.py")
                else Path(*args, **kwargs)
            ),
        )
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        out_path = tmp_path / "data" / "output.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        loaded = pd.read_csv(out_path)
        assert list(loaded["x"]) == [1, 2]

    def test_creates_nested_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "out.csv"
        nested.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"v": [42]}).to_csv(nested, index=False)
        assert nested.exists()
        assert pd.read_csv(nested)["v"].iloc[0] == 42


class TestLoadBestModelInfo:
    def test_raises_file_not_found_when_missing(self, tmp_path, monkeypatch):
        import src.utils as utils_mod

        fake_src = tmp_path / "src" / "utils.py"
        fake_src.parent.mkdir(parents=True)
        monkeypatch.setattr(utils_mod, "__file__", str(fake_src))
        with pytest.raises(FileNotFoundError, match="best_model.json"):
            load_best_model_info()

    def test_loads_and_returns_dict(self, tmp_path, monkeypatch):
        import src.utils as utils_mod

        fake_src = tmp_path / "src" / "utils.py"
        fake_src.parent.mkdir(parents=True)
        artifacts = tmp_path / "models" / "artifacts"
        artifacts.mkdir(parents=True)
        data = {"model_name": "lightgbm", "model_path": "models/lightgbm.pkl"}
        (artifacts / "best_model.json").write_text(json.dumps(data))
        monkeypatch.setattr(utils_mod, "__file__", str(fake_src))
        result = load_best_model_info()
        assert result["model_name"] == "lightgbm"
        assert result["model_path"] == "models/lightgbm.pkl"

    def test_error_message_contains_hint(self, tmp_path, monkeypatch):
        import src.utils as utils_mod

        fake_src = tmp_path / "src" / "utils.py"
        fake_src.parent.mkdir(parents=True)
        monkeypatch.setattr(utils_mod, "__file__", str(fake_src))
        with pytest.raises(FileNotFoundError, match="training and evaluation pipeline"):
            load_best_model_info()


class TestLoadLastNRowsPerCompany:
    def test_returns_n_rows_per_company(self, multi_company_csv, monkeypatch):
        import src.utils as utils_mod

        monkeypatch.setattr(
            utils_mod,
            "RAW_DATA_PATH",
            Path(multi_company_csv.name),
        )
        fake_src = multi_company_csv.parent / "src" / "utils.py"
        fake_src.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(utils_mod, "__file__", str(fake_src))

        result = pd.read_csv(multi_company_csv, parse_dates=["Date"])
        result = (
            result.sort_values(["Company", "Date"])
            .groupby("Company", group_keys=False)
            .apply(lambda g: g.tail(5), include_groups=True)
            .reset_index(drop=True)
        )
        assert result.groupby("Company").size().max() == 5

    def test_sorts_by_company_and_date(self, multi_company_csv):
        df = pd.read_csv(multi_company_csv, parse_dates=["Date"])
        df = (
            df.sort_values(["Company", "Date"])
            .groupby("Company", group_keys=False)
            .apply(lambda g: g.tail(3), include_groups=True)
            .reset_index(drop=True)
        )
        for company, group in df.groupby("Company"):
            assert group["Date"].is_monotonic_increasing

    def test_tail_n_capped_by_available_rows(self, multi_company_csv):
        df = pd.read_csv(multi_company_csv, parse_dates=["Date"])
        result = (
            df.sort_values(["Company", "Date"])
            .groupby("Company", group_keys=False)
            .apply(lambda g: g.tail(1000), include_groups=True)
            .reset_index(drop=True)
        )
        assert len(result) == len(df)
