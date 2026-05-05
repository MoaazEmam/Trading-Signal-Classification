"""Unit tests for helper functions in src/models/trainer.py"""

from __future__ import annotations

import json

import joblib
import pytest
from sklearn.dummy import DummyClassifier

from src.models.trainer import (
    _base_estimators,
    _build_voting,
    _log_summary,
    _param_grids,
    _save_results,
    load_model,
)

# ---------------------------------------------------------------------------
# _param_grids
# ---------------------------------------------------------------------------


class TestParamGrids:
    EXPECTED_MODELS = {
        "logistic_regression",
        "decision_tree",
        "adaboost",
        "random_forest",
        "lightgbm",
    }

    def test_returns_dict(self):
        assert isinstance(_param_grids(), dict)

    def test_contains_all_model_names(self):
        grids = _param_grids()
        assert self.EXPECTED_MODELS.issubset(set(grids.keys()))

    def test_all_values_are_dicts(self):
        for name, grid in _param_grids().items():
            assert isinstance(grid, dict), f"{name} grid should be a dict"

    def test_all_param_values_are_lists(self):
        for name, grid in _param_grids().items():
            for param, values in grid.items():
                assert isinstance(values, list), f"{name}.{param} should be a list"

    def test_logistic_regression_has_C(self):
        assert "C" in _param_grids()["logistic_regression"]

    def test_decision_tree_has_max_depth(self):
        assert "max_depth" in _param_grids()["decision_tree"]

    def test_lightgbm_has_n_estimators(self):
        assert "n_estimators" in _param_grids()["lightgbm"]

    def test_random_forest_has_max_depth(self):
        assert "max_depth" in _param_grids()["random_forest"]

    def test_adaboost_has_learning_rate(self):
        assert "learning_rate" in _param_grids()["adaboost"]


# ---------------------------------------------------------------------------
# _base_estimators
# ---------------------------------------------------------------------------


class TestBaseEstimators:
    def test_returns_dict(self):
        assert isinstance(_base_estimators(), dict)

    def test_all_expected_models_present(self):
        estimators = _base_estimators()
        for name in [
            "logistic_regression",
            "decision_tree",
            "adaboost",
            "random_forest",
            "lightgbm",
        ]:
            assert name in estimators

    def test_all_estimators_have_fit_method(self):
        for name, est in _base_estimators().items():
            assert hasattr(est, "fit"), f"{name} must have fit()"

    def test_all_estimators_have_predict_method(self):
        for name, est in _base_estimators().items():
            assert hasattr(est, "predict"), f"{name} must have predict()"

    def test_lightgbm_is_correct_class(self):
        from lightgbm import LGBMClassifier

        assert isinstance(_base_estimators()["lightgbm"], LGBMClassifier)

    def test_logistic_regression_uses_saga_solver(self):
        lr = _base_estimators()["logistic_regression"]
        assert lr.solver == "saga"


# ---------------------------------------------------------------------------
# _build_voting
# ---------------------------------------------------------------------------


class TestBuildVoting:
    @pytest.fixture()
    def tuned_estimators(self):
        dummy = DummyClassifier()
        return {
            "lightgbm": dummy,
            "random_forest": dummy,
            "adaboost": dummy,
        }

    def test_returns_voting_classifier(self, tuned_estimators):
        from sklearn.ensemble import VotingClassifier

        voting = _build_voting(tuned_estimators)
        assert isinstance(voting, VotingClassifier)

    def test_uses_soft_voting(self, tuned_estimators):
        voting = _build_voting(tuned_estimators)
        assert voting.voting == "soft"

    def test_includes_lightgbm(self, tuned_estimators):
        voting = _build_voting(tuned_estimators)
        estimator_names = [name for name, _ in voting.estimators]
        assert "lightgbm" in estimator_names

    def test_includes_random_forest(self, tuned_estimators):
        voting = _build_voting(tuned_estimators)
        estimator_names = [name for name, _ in voting.estimators]
        assert "random_forest" in estimator_names

    def test_includes_adaboost(self, tuned_estimators):
        voting = _build_voting(tuned_estimators)
        estimator_names = [name for name, _ in voting.estimators]
        assert "adaboost" in estimator_names


# ---------------------------------------------------------------------------
# _save_results
# ---------------------------------------------------------------------------


class TestSaveResults:
    @pytest.fixture()
    def sample_results(self) -> dict:
        return {
            "logistic_regression": {
                "best_params": {"C": 0.1},
                "cv_accuracy": 0.65,
                "train_accuracy": 0.70,
                "test_accuracy": 0.63,
                "train_time_s": 5.2,
                "artifact_path": "models/logistic_regression.pkl",
            }
        }

    def test_creates_json_file(self, tmp_path, monkeypatch, sample_results):
        import src.models.trainer as trainer_mod

        monkeypatch.setattr(trainer_mod, "ARTIFACT_DIR", tmp_path)
        _save_results(sample_results)
        assert (tmp_path / "training_results.json").exists()

    def test_json_has_generated_at(self, tmp_path, monkeypatch, sample_results):
        import src.models.trainer as trainer_mod

        monkeypatch.setattr(trainer_mod, "ARTIFACT_DIR", tmp_path)
        _save_results(sample_results)
        data = json.loads((tmp_path / "training_results.json").read_text())
        assert "generated_at" in data

    def test_json_has_models_key(self, tmp_path, monkeypatch, sample_results):
        import src.models.trainer as trainer_mod

        monkeypatch.setattr(trainer_mod, "ARTIFACT_DIR", tmp_path)
        _save_results(sample_results)
        data = json.loads((tmp_path / "training_results.json").read_text())
        assert "models" in data

    def test_json_contains_model_name(self, tmp_path, monkeypatch, sample_results):
        import src.models.trainer as trainer_mod

        monkeypatch.setattr(trainer_mod, "ARTIFACT_DIR", tmp_path)
        _save_results(sample_results)
        data = json.loads((tmp_path / "training_results.json").read_text())
        assert "logistic_regression" in data["models"]

    def test_json_preserves_accuracy_values(
        self, tmp_path, monkeypatch, sample_results
    ):
        import src.models.trainer as trainer_mod

        monkeypatch.setattr(trainer_mod, "ARTIFACT_DIR", tmp_path)
        _save_results(sample_results)
        data = json.loads((tmp_path / "training_results.json").read_text())
        lr = data["models"]["logistic_regression"]
        assert lr["train_accuracy"] == pytest.approx(0.70)
        assert lr["test_accuracy"] == pytest.approx(0.63)


# ---------------------------------------------------------------------------
# _log_summary
# ---------------------------------------------------------------------------


class TestLogSummary:
    def test_does_not_raise(self):
        results = {
            "lightgbm": {
                "cv_accuracy": 0.70,
                "train_accuracy": 0.75,
                "test_accuracy": 0.68,
                "train_time_s": 12.5,
            },
            "random_forest": {
                "cv_accuracy": None,
                "train_accuracy": 0.73,
                "test_accuracy": 0.66,
                "train_time_s": 9.0,
            },
        }
        _log_summary(results)

    def test_handles_none_cv_accuracy(self):
        results = {
            "voting_classifier": {
                "cv_accuracy": None,
                "train_accuracy": 0.77,
                "test_accuracy": 0.69,
                "train_time_s": 3.0,
            }
        }
        _log_summary(results)


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_raises_file_not_found_when_missing(self, tmp_path, monkeypatch):
        import src.models.trainer as trainer_mod

        monkeypatch.setattr(trainer_mod, "MODELS_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="ghost_model"):
            load_model("ghost_model")

    def test_loads_existing_model(self, tmp_path, monkeypatch):
        import src.models.trainer as trainer_mod

        monkeypatch.setattr(trainer_mod, "MODELS_DIR", tmp_path)
        model = DummyClassifier()
        joblib.dump(model, tmp_path / "my_model.pkl")
        loaded = load_model("my_model")
        assert isinstance(loaded, DummyClassifier)

    def test_error_message_contains_model_name(self, tmp_path, monkeypatch):
        import src.models.trainer as trainer_mod

        monkeypatch.setattr(trainer_mod, "MODELS_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="lightgbm"):
            load_model("lightgbm")
