"""Unit tests for src/serving/api.py"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.serving.api import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def predictions_dir(tmp_path: Path) -> Path:
    d = tmp_path / "predictions"
    d.mkdir()
    return d


@pytest.fixture()
def history_dir(predictions_dir: Path) -> Path:
    h = predictions_dir / "history"
    h.mkdir()
    return h


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_ok_status(self, client):
        response = client.get("/health")
        assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /predictions/latest
# ---------------------------------------------------------------------------


class TestGetLatestPredictions:
    def test_404_when_no_predictions(self, client, monkeypatch, tmp_path):
        import src.serving.api as api_mod

        monkeypatch.setattr(api_mod, "PREDICTIONS_DIR", tmp_path / "predictions")
        response = client.get("/predictions/latest")
        assert response.status_code == 404

    def test_returns_latest_json_when_exists(
        self, client, monkeypatch, predictions_dir
    ):
        import src.serving.api as api_mod

        monkeypatch.setattr(api_mod, "PREDICTIONS_DIR", predictions_dir)
        predictions = [
            {"Company": "AAPL", "Date": "2024-01-02", "predicted_label": "Buy"}
        ]
        (predictions_dir / "latest.json").write_text(json.dumps(predictions))

        response = client.get("/predictions/latest")
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "latest"
        assert data["count"] == 1
        assert data["predictions"][0]["Company"] == "AAPL"

    def test_returns_date_from_first_prediction(
        self, client, monkeypatch, predictions_dir
    ):
        import src.serving.api as api_mod

        monkeypatch.setattr(api_mod, "PREDICTIONS_DIR", predictions_dir)
        predictions = [
            {"Company": "AAPL", "Date": "2024-01-02", "predicted_label": "Buy"}
        ]
        (predictions_dir / "latest.json").write_text(json.dumps(predictions))

        response = client.get("/predictions/latest")
        assert response.json()["date"] == "2024-01-02"

    def test_falls_back_to_archive_when_latest_missing(
        self, client, monkeypatch, predictions_dir, history_dir
    ):
        import src.serving.api as api_mod

        monkeypatch.setattr(api_mod, "PREDICTIONS_DIR", predictions_dir)
        archive_data = [
            {"Company": "MSFT", "Date": "2024-01-01", "predicted_label": "Hold"}
        ]
        (history_dir / "2024-01-01.json").write_text(json.dumps(archive_data))

        response = client.get("/predictions/latest")
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "archive"
        assert data["predictions"][0]["Company"] == "MSFT"

    def test_404_when_no_latest_and_no_archives(
        self, client, monkeypatch, predictions_dir, history_dir
    ):
        import src.serving.api as api_mod

        monkeypatch.setattr(api_mod, "PREDICTIONS_DIR", predictions_dir)
        response = client.get("/predictions/latest")
        assert response.status_code == 404

    def test_latest_empty_predictions_unknown_date(
        self, client, monkeypatch, predictions_dir
    ):
        import src.serving.api as api_mod

        monkeypatch.setattr(api_mod, "PREDICTIONS_DIR", predictions_dir)
        (predictions_dir / "latest.json").write_text(json.dumps([]))

        response = client.get("/predictions/latest")
        assert response.status_code == 200
        assert response.json()["date"] == "unknown"

    def test_archive_serves_most_recent_when_multiple(
        self, client, monkeypatch, predictions_dir, history_dir
    ):
        import src.serving.api as api_mod

        monkeypatch.setattr(api_mod, "PREDICTIONS_DIR", predictions_dir)
        for date in ["2024-01-01", "2024-01-02", "2024-01-03"]:
            (history_dir / f"{date}.json").write_text(json.dumps([{"Date": date}]))

        response = client.get("/predictions/latest")
        assert response.json()["date"] == "2024-01-03"


# ---------------------------------------------------------------------------
# /predictions/history/{date}
# ---------------------------------------------------------------------------


class TestGetPredictionsByDate:
    def test_404_for_missing_date(self, client, monkeypatch, tmp_path):
        import src.serving.api as api_mod

        monkeypatch.setattr(api_mod, "PREDICTIONS_DIR", tmp_path / "predictions")
        response = client.get("/predictions/history/2024-01-15")
        assert response.status_code == 404

    def test_returns_data_for_existing_date(
        self, client, monkeypatch, predictions_dir, history_dir
    ):
        import src.serving.api as api_mod

        monkeypatch.setattr(api_mod, "PREDICTIONS_DIR", predictions_dir)
        data = [{"Company": "GOOG", "Date": "2024-02-05", "predicted_label": "Sell"}]
        (history_dir / "2024-02-05.json").write_text(json.dumps(data))

        response = client.get("/predictions/history/2024-02-05")
        assert response.status_code == 200
        resp_data = response.json()
        assert resp_data["date"] == "2024-02-05"
        assert resp_data["predictions"][0]["Company"] == "GOOG"
        assert resp_data["source"] == "archive"

    def test_count_matches_predictions_length(
        self, client, monkeypatch, predictions_dir, history_dir
    ):
        import src.serving.api as api_mod

        monkeypatch.setattr(api_mod, "PREDICTIONS_DIR", predictions_dir)
        data = [{"Company": f"CO{i}", "Date": "2024-03-01"} for i in range(5)]
        (history_dir / "2024-03-01.json").write_text(json.dumps(data))

        response = client.get("/predictions/history/2024-03-01")
        assert response.json()["count"] == 5


# ---------------------------------------------------------------------------
# /predictions/dates
# ---------------------------------------------------------------------------


class TestListAvailableDates:
    def test_returns_empty_when_no_history(self, client, monkeypatch, tmp_path):
        import src.serving.api as api_mod

        monkeypatch.setattr(api_mod, "PREDICTIONS_DIR", tmp_path / "predictions")
        response = client.get("/predictions/dates")
        assert response.status_code == 200
        assert response.json()["dates"] == []

    def test_returns_all_available_dates(
        self, client, monkeypatch, predictions_dir, history_dir
    ):
        import src.serving.api as api_mod

        monkeypatch.setattr(api_mod, "PREDICTIONS_DIR", predictions_dir)
        for date in ["2024-01-03", "2024-01-01", "2024-01-02"]:
            (history_dir / f"{date}.json").write_text("[]")

        response = client.get("/predictions/dates")
        data = response.json()
        assert set(data["dates"]) == {"2024-01-01", "2024-01-02", "2024-01-03"}
        assert data["count"] == 3

    def test_dates_sorted_descending(
        self, client, monkeypatch, predictions_dir, history_dir
    ):
        import src.serving.api as api_mod

        monkeypatch.setattr(api_mod, "PREDICTIONS_DIR", predictions_dir)
        for date in ["2024-01-01", "2024-01-03", "2024-01-02"]:
            (history_dir / f"{date}.json").write_text("[]")

        response = client.get("/predictions/dates")
        dates = response.json()["dates"]
        assert dates == sorted(dates, reverse=True)

    def test_returns_empty_when_history_dir_missing(
        self, client, monkeypatch, predictions_dir
    ):
        import src.serving.api as api_mod

        monkeypatch.setattr(api_mod, "PREDICTIONS_DIR", predictions_dir)
        response = client.get("/predictions/dates")
        assert response.json()["dates"] == []


# ---------------------------------------------------------------------------
# /backtest/results
# ---------------------------------------------------------------------------


class TestGetBacktestResults:
    def test_404_when_no_results_file(self, client, monkeypatch, tmp_path):
        import src.serving.api as api_mod

        monkeypatch.setattr(
            api_mod, "BACKTEST_RESULTS_PATH", tmp_path / "backtest_results.json"
        )
        response = client.get("/backtest/results")
        assert response.status_code == 404

    def test_returns_results_when_file_exists(self, client, monkeypatch, tmp_path):
        import src.serving.api as api_mod

        results_path = tmp_path / "backtest_results.json"
        results = {"naive_buy_baseline": {"total_return_pct": 5.0}}
        results_path.write_text(json.dumps(results))
        monkeypatch.setattr(api_mod, "BACKTEST_RESULTS_PATH", results_path)

        response = client.get("/backtest/results")
        assert response.status_code == 200
        assert "naive_buy_baseline" in response.json()
