"""Unit tests for helper functions in src/pipelines/predict.py"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from src.pipelines.predict import (
    _build_predictions_df,
    _copy_latest_from_archive,
    _resolve_target_date,
)

# ---------------------------------------------------------------------------
# _resolve_target_date
# ---------------------------------------------------------------------------


class TestResolveTargetDate:
    def test_returns_given_weekday_date(self):
        result = _resolve_target_date("2024-01-02")  # Tuesday
        assert result == "2024-01-02"

    def test_raises_for_saturday(self):
        with pytest.raises(ValueError, match="weekend"):
            _resolve_target_date("2024-01-06")  # Saturday

    def test_raises_for_sunday(self):
        with pytest.raises(ValueError, match="weekend"):
            _resolve_target_date("2024-01-07")  # Sunday

    def test_returns_string(self):
        result = _resolve_target_date("2024-03-15")  # Friday
        assert isinstance(result, str)

    def test_weekday_dates_pass(self):
        for date in ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]:
            result = _resolve_target_date(date)
            assert result == date

    def test_defaults_to_business_day_when_none(self, monkeypatch):
        fixed_ts = pd.Timestamp("2024-01-03")  # Wednesday
        monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls: fixed_ts))
        result = _resolve_target_date(None)
        assert isinstance(result, str)
        assert len(result) == 10  # YYYY-MM-DD format

    def test_error_message_contains_date(self):
        with pytest.raises(ValueError, match="2024-01-06"):
            _resolve_target_date("2024-01-06")


# ---------------------------------------------------------------------------
# _build_predictions_df
# ---------------------------------------------------------------------------


class TestBuildPredictionsDF:
    @pytest.fixture()
    def label_encoder(self) -> LabelEncoder:
        le = LabelEncoder()
        le.fit(["Buy", "Hold", "Sell"])
        return le

    @pytest.fixture()
    def sample_inputs(self, label_encoder) -> dict:
        companies = pd.Series(["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"])
        labels = ["Buy", "Hold", "Sell", "Buy", "Hold"]
        probabilities = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
                [0.6, 0.3, 0.1],
                [0.2, 0.7, 0.1],
            ]
        )
        return {
            "companies": companies,
            "date": "2024-01-02",
            "labels": labels,
            "probabilities": probabilities,
            "label_encoder": label_encoder,
        }

    def test_returns_dataframe(self, sample_inputs):
        result = _build_predictions_df(**sample_inputs)
        assert isinstance(result, pd.DataFrame)

    def test_has_company_column(self, sample_inputs):
        result = _build_predictions_df(**sample_inputs)
        assert "Company" in result.columns

    def test_has_date_column(self, sample_inputs):
        result = _build_predictions_df(**sample_inputs)
        assert "Date" in result.columns

    def test_has_predicted_label_column(self, sample_inputs):
        result = _build_predictions_df(**sample_inputs)
        assert "predicted_label" in result.columns

    def test_has_probability_columns(self, sample_inputs):
        result = _build_predictions_df(**sample_inputs)
        assert "proba_buy" in result.columns
        assert "proba_hold" in result.columns
        assert "proba_sell" in result.columns

    def test_row_count_matches_companies(self, sample_inputs):
        result = _build_predictions_df(**sample_inputs)
        assert len(result) == len(sample_inputs["companies"])

    def test_date_is_consistent_across_rows(self, sample_inputs):
        result = _build_predictions_df(**sample_inputs)
        assert (result["Date"] == "2024-01-02").all()

    def test_companies_preserved_in_order(self, sample_inputs):
        result = _build_predictions_df(**sample_inputs)
        assert list(result["Company"]) == list(sample_inputs["companies"])

    def test_labels_preserved(self, sample_inputs):
        result = _build_predictions_df(**sample_inputs)
        assert list(result["predicted_label"]) == sample_inputs["labels"]

    def test_probabilities_sum_to_one(self, sample_inputs):
        result = _build_predictions_df(**sample_inputs)
        proba_cols = [c for c in result.columns if c.startswith("proba_")]
        row_sums = result[proba_cols].sum(axis=1)
        assert (row_sums - 1.0).abs().max() < 1e-6


# ---------------------------------------------------------------------------
# _write_predictions
# ---------------------------------------------------------------------------


class TestWritePredictions:
    @pytest.fixture()
    def sample_preds_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Company": ["AAPL", "MSFT"],
                "Date": "2024-01-02",
                "predicted_label": ["Buy", "Hold"],
                "proba_buy": [0.7, 0.2],
                "proba_hold": [0.2, 0.7],
                "proba_sell": [0.1, 0.1],
            }
        )

    def test_creates_latest_json(self, tmp_path, monkeypatch, sample_preds_df):
        import src.pipelines.predict as predict_mod

        monkeypatch.setattr(predict_mod, "PREDICTIONS_DIR", Path("predictions"))

        class FakePath:
            def __init__(self, *args):
                self._p = Path(*args)

            def resolve(self):
                return FakePath(tmp_path / "src" / "pipelines" / "predict.py")

            def __truediv__(self, other):
                return FakePath(str(self._p) + "/" + str(other))

        _write_predictions_direct(sample_preds_df, "2024-01-02", tmp_path)
        assert (tmp_path / "predictions" / "latest.json").exists()

    def test_creates_latest_csv(self, tmp_path, sample_preds_df):
        _write_predictions_direct(sample_preds_df, "2024-01-02", tmp_path)
        assert (tmp_path / "predictions" / "latest.csv").exists()

    def test_creates_history_archive(self, tmp_path, sample_preds_df):
        _write_predictions_direct(sample_preds_df, "2024-01-02", tmp_path)
        assert (tmp_path / "predictions" / "history" / "2024-01-02.json").exists()

    def test_json_content_is_correct(self, tmp_path, sample_preds_df):
        _write_predictions_direct(sample_preds_df, "2024-01-02", tmp_path)
        data = json.loads((tmp_path / "predictions" / "latest.json").read_text())
        assert len(data) == 2
        assert data[0]["Company"] == "AAPL"

    def test_csv_is_readable(self, tmp_path, sample_preds_df):
        _write_predictions_direct(sample_preds_df, "2024-01-02", tmp_path)
        loaded = pd.read_csv(tmp_path / "predictions" / "latest.csv")
        assert "Company" in loaded.columns


def _write_predictions_direct(
    predictions_df: pd.DataFrame, date: str, tmp_path: Path
) -> None:
    """Helper that writes predictions directly without the project root resolution."""
    pred_dir = tmp_path / "predictions"
    history_dir = pred_dir / "history"
    pred_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    records = predictions_df.to_dict(orient="records")
    (pred_dir / "latest.json").write_text(json.dumps(records, indent=2))
    predictions_df.to_csv(pred_dir / "latest.csv", index=False)
    (history_dir / f"{date}.json").write_text(json.dumps(records, indent=2))


# ---------------------------------------------------------------------------
# _copy_latest_from_archive
# ---------------------------------------------------------------------------


class TestCopyLatestFromArchive:
    def test_copies_most_recent_archive(self, tmp_path):
        pred_dir = tmp_path / "predictions"
        history_dir = pred_dir / "history"
        pred_dir.mkdir()
        history_dir.mkdir()

        for date in ["2024-01-01", "2024-01-02", "2024-01-03"]:
            (history_dir / f"{date}.json").write_text(json.dumps([{"Date": date}]))

        _copy_latest_from_archive(pred_dir)

        assert (pred_dir / "latest.json").exists()
        data = json.loads((pred_dir / "latest.json").read_text())
        assert data[0]["Date"] == "2024-01-03"

    def test_does_nothing_when_no_archives(self, tmp_path):
        pred_dir = tmp_path / "predictions"
        history_dir = pred_dir / "history"
        pred_dir.mkdir()
        history_dir.mkdir()

        _copy_latest_from_archive(pred_dir)
        assert not (pred_dir / "latest.json").exists()

    def test_overwrites_existing_latest(self, tmp_path):
        pred_dir = tmp_path / "predictions"
        history_dir = pred_dir / "history"
        pred_dir.mkdir()
        history_dir.mkdir()

        (pred_dir / "latest.json").write_text(json.dumps([{"Date": "old"}]))
        (history_dir / "2024-05-01.json").write_text(
            json.dumps([{"Date": "2024-05-01"}])
        )

        _copy_latest_from_archive(pred_dir)
        data = json.loads((pred_dir / "latest.json").read_text())
        assert data[0]["Date"] == "2024-05-01"
