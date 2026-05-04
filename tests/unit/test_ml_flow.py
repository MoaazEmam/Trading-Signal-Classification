"""
Unit tests for src/models/ml_flow.py and src/models/run_evaluation.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.models.ml_flow import _safe, run_mlflow_tracking
from src.models.run_evaluation import _save_csv


@pytest.fixture()
def artifact_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models" / "artifacts"
    d.mkdir(parents=True)
    return d


@pytest.fixture()
def models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture()
def dummy_model(models_dir: Path) -> Path:
    model = DummyClassifier(strategy="prior", random_state=0)
    path = models_dir / "logistic_regression.pkl"
    joblib.dump(model, path)
    return path


@pytest.fixture()
def training_json(artifact_dir: Path, dummy_model: Path) -> Path:
    data = {
        "models": {
            "logistic_regression": {
                "best_params": {},
                "cv_accuracy": None,
                "train_accuracy": 0.5,
                "test_accuracy": 0.4,
                "train_time_s": 10.0,
                "artifact_path": str(dummy_model),
            }
        }
    }
    path = artifact_dir / "training_results.json"
    path.write_text(json.dumps(data))
    return path


class TestSafe:
    def test_returns_none_for_none(self):
        assert _safe(None) is None

    def test_returns_none_for_nan(self):
        assert _safe(float("nan")) is None

    def test_returns_float_for_valid_value(self):
        assert _safe(0.5) == pytest.approx(0.5)

    def test_returns_float_for_zero(self):
        assert _safe(0.0) == pytest.approx(0.0)

    def test_returns_float_for_integer(self):
        assert _safe(1) == pytest.approx(1.0)

    def test_returns_none_for_invalid_string(self):
        assert _safe("not_a_number") is None

    def test_nan_check_via_math(self):
        result = _safe(float("nan"))
        assert result is None or not math.isnan(result)


class TestRunMlflowTracking:
    def test_creates_experiment_with_correct_name(
        self, tmp_path, monkeypatch, training_json, artifact_dir, models_dir
    ):
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        monkeypatch.setattr("src.models.ml_flow.ARTIFACT_DIR", artifact_dir)
        monkeypatch.setattr("src.models.ml_flow.MODELS_DIR", models_dir)
        run_mlflow_tracking()
        client = mlflow.tracking.MlflowClient()
        names = [e.name for e in client.search_experiments()]
        assert "Trading-Signal-Classification" in names

    def test_creates_run_per_model(
        self, tmp_path, monkeypatch, training_json, artifact_dir, models_dir
    ):
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        monkeypatch.setattr("src.models.ml_flow.ARTIFACT_DIR", artifact_dir)
        monkeypatch.setattr("src.models.ml_flow.MODELS_DIR", models_dir)
        run_mlflow_tracking()
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("Trading-Signal-Classification")
        runs = client.search_runs(experiment.experiment_id)
        run_names = [r.data.tags.get("mlflow.runName") for r in runs]
        assert "logistic_regression" in run_names

    def test_skips_missing_pkl_without_crash(
        self, tmp_path, monkeypatch, artifact_dir, models_dir
    ):
        data = {
            "models": {
                "ghost_model": {
                    "best_params": {},
                    "cv_accuracy": None,
                    "train_accuracy": 0.5,
                    "test_accuracy": 0.4,
                    "train_time_s": 1.0,
                    "artifact_path": str(models_dir / "ghost_model.pkl"),
                }
            }
        }
        (artifact_dir / "training_results.json").write_text(json.dumps(data))
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        monkeypatch.setattr("src.models.ml_flow.ARTIFACT_DIR", artifact_dir)
        monkeypatch.setattr("src.models.ml_flow.MODELS_DIR", models_dir)
        run_mlflow_tracking()

    def test_logs_train_and_test_accuracy(
        self, tmp_path, monkeypatch, training_json, artifact_dir, models_dir
    ):
        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        monkeypatch.setattr("src.models.ml_flow.ARTIFACT_DIR", artifact_dir)
        monkeypatch.setattr("src.models.ml_flow.MODELS_DIR", models_dir)
        run_mlflow_tracking()
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("Trading-Signal-Classification")
        runs = client.search_runs(experiment.experiment_id)
        model_run = next(
            r
            for r in runs
            if r.data.tags.get("mlflow.runName") == "logistic_regression"
        )
        assert "train_accuracy" in model_run.data.metrics
        assert "test_accuracy" in model_run.data.metrics


class TestSaveCsv:
    def test_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.models.run_evaluation.ARTIFACT_DIR", tmp_path)
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = _save_csv(df, "test_output.csv")
        assert path.exists()

    def test_file_is_readable_as_csv(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.models.run_evaluation.ARTIFACT_DIR", tmp_path)
        df = pd.DataFrame({"x": [10, 20]})
        path = _save_csv(df, "readable.csv")
        loaded = pd.read_csv(path)
        assert list(loaded["x"]) == [10, 20]
