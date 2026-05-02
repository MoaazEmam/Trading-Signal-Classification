import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = _PROJECT_ROOT / "models"
ARTIFACT_DIR = _PROJECT_ROOT / "models" / "artifacts"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "label"
COMPANY_COL = "Company"
RANDOM_STATE = 42
N_CV_SPLITS = 3


def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [c for c in [TARGET_COL, COMPANY_COL] if c in df.columns]
    return df.drop(columns=drop_cols), df[TARGET_COL]


def _param_grids() -> dict[str, dict]:
    return {
        "logistic_regression": {
            "C": [0.01, 0.05, 0.1, 0.5],
            "l1_ratio": [0.1, 0.5, 0.9],
        },
        "decision_tree": {
            "max_depth": [4, 6, 8],
            "min_samples_leaf": [100, 200, 400],
        },
        "adaboost": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
        },
        "random_forest": {
            "max_depth": [8, 10, 15],
            "min_samples_leaf": [50, 100, 200],
            "max_samples": [0.6, 0.7, 0.8],
        },
        "lightgbm": {
            "n_estimators": [500, 1000],
            "num_leaves": [31, 48, 63],
            "learning_rate": [0.01, 0.02, 0.05],
            "min_child_samples": [100, 200, 300],
            "reg_lambda": [0.5, 1.0, 2.0],
        },
    }


def _base_estimators() -> dict[str, Any]:
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    return {
        "logistic_regression": LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.5,
            C=0.1,
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=200,
            min_samples_split=400,
            max_features="sqrt",
            random_state=RANDOM_STATE,
        ),
        "adaboost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=100,
                random_state=RANDOM_STATE,
            ),
            n_estimators=200,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=100,
            min_samples_split=200,
            max_features="sqrt",
            max_samples=0.7,
            n_jobs=4,
            random_state=RANDOM_STATE,
        ),
        "lightgbm": LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=1000,
            learning_rate=0.02,
            num_leaves=48,
            max_depth=7,
            min_child_samples=200,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.2,
            reg_lambda=1.0,
            min_split_gain=0.01,
            n_jobs=4,
            random_state=RANDOM_STATE,
            verbosity=-1,
        ),
    }


def _tune(
    name: str,
    estimator: Any,
    param_grid: dict,
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[Any, dict, float]:
    from sklearn.model_selection import (
        GridSearchCV,
        RandomizedSearchCV,
        TimeSeriesSplit,
    )

    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    n_combinations = int(np.prod([len(v) for v in param_grid.values()]))

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }

    common_kwargs = dict(
        cv=tscv,
        scoring=scoring,
        refit="f1_macro",
        return_train_score=True,
        verbose=1,
    )

    n_iter = min(n_combinations, 30)

    if n_combinations > 12:
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            n_jobs=2,
            random_state=RANDOM_STATE,
            **common_kwargs,
        )
    else:
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            n_jobs=2,
            **common_kwargs,
        )

    logger.info(
        "%s: running %s (%d combinations, n_iter=%d) ...",
        name,
        type(search).__name__,
        n_combinations,
        n_iter if isinstance(search, RandomizedSearchCV) else n_combinations,
    )
    search.fit(x, y)
    cv_results = search.cv_results_
    best_idx = search.best_index_
    best_train_f1 = cv_results["mean_train_f1_macro"][best_idx]
    best_val_f1 = cv_results["mean_test_f1_macro"][best_idx]
    best_val_acc = cv_results["mean_test_accuracy"][best_idx]
    overfit_gap = best_train_f1 - best_val_f1

    logger.info(
        "%s | val acc %.4f | val f1_macro %.4f | train f1_macro %.4f | overfit gap %.4f | params %s",
        name,
        best_val_acc,
        best_val_f1,
        best_train_f1,
        overfit_gap,
        search.best_params_,
    )
    if overfit_gap > 0.15:
        logger.warning(
            "%s: large overfit gap (%.4f) — consider tightening regularization params",
            name,
            overfit_gap,
        )

    return search.best_estimator_, search.best_params_, float(best_val_acc)


def _build_voting(tuned_estimators: dict[str, Any]) -> Any:
    from sklearn.ensemble import VotingClassifier

    return VotingClassifier(
        estimators=[
            ("lightgbm", tuned_estimators["lightgbm"]),
            ("random_forest", tuned_estimators["random_forest"]),
            ("adaboost", tuned_estimators["adaboost"]),
        ],
        voting="soft",
        n_jobs=-1,
    )


def train_all(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    x_train, y_train = _prepare(train_df)
    x_test, y_test = _prepare(test_df)

    base_estimators = _base_estimators()
    grids = _param_grids()
    results: dict[str, dict[str, Any]] = {}
    tuned_estimators: dict[str, Any] = {}

    sep = "=" * 60
    logger.info(sep)
    logger.info("TRAINING START — %d rows, %d features", *x_train.shape)
    logger.info(sep)

    for name, estimator in base_estimators.items():
        logger.info("--- %s ---", name.upper())
        t0 = time.perf_counter()

        best_model, best_params, best_cv_acc = _tune(
            name, estimator, grids[name], x_train, y_train
        )

        train_acc = float((best_model.predict(x_train) == y_train).mean())
        test_acc = float((best_model.predict(x_test) == y_test).mean())
        elapsed = round(time.perf_counter() - t0, 2)

        logger.info(
            "%s | train acc %.4f | test acc %.4f | %.1fs",
            name,
            train_acc,
            test_acc,
            elapsed,
        )

        model_path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(best_model, model_path)

        tuned_estimators[name] = best_model
        results[name] = {
            "best_params": best_params,
            "cv_accuracy": best_cv_acc,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_time_s": elapsed,
            "artifact_path": model_path.relative_to(_PROJECT_ROOT).as_posix(),
        }

    logger.info("--- VOTING CLASSIFIER ---")
    t0 = time.perf_counter()
    voting = _build_voting(tuned_estimators)
    voting.fit(x_train, y_train)
    elapsed = round(time.perf_counter() - t0, 2)

    voting_train_acc = float((voting.predict(x_train) == y_train).mean())
    voting_test_acc = float((voting.predict(x_test) == y_test).mean())
    voting_path = MODELS_DIR / "voting_classifier.pkl"
    joblib.dump(voting, voting_path)

    logger.info(
        "voting_classifier | train acc %.4f | test acc %.4f | %.1fs",
        voting_train_acc,
        voting_test_acc,
        elapsed,
    )

    results["voting_classifier"] = {
        "best_params": {
            "estimators": ["lightgbm", "random_forest", "adaboost"],
            "voting": "soft",
        },
        "cv_accuracy": None,
        "train_accuracy": voting_train_acc,
        "test_accuracy": voting_test_acc,
        "train_time_s": elapsed,
        "artifact_path": voting_path.relative_to(_PROJECT_ROOT).as_posix(),
    }

    _save_results(results)

    logger.info(sep)
    logger.info("TRAINING COMPLETE")
    _log_summary(results)
    logger.info(sep)

    return results


def _save_results(results: dict[str, dict[str, Any]]) -> None:
    out_path = ARTIFACT_DIR / "training_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "models": results},
            f,
            indent=2,
            default=str,
        )
    logger.info("Training results saved to %s", out_path)


def _log_summary(results: dict[str, dict[str, Any]]) -> None:
    rows = [
        {
            "model": name,
            "cv_accuracy": (
                f"{r['cv_accuracy']:.4f}" if r["cv_accuracy"] is not None else "—"
            ),
            "train_accuracy": f"{r['train_accuracy']:.4f}",
            "test_accuracy": f"{r['test_accuracy']:.4f}",
            "train_time_s": r["train_time_s"],
        }
        for name, r in results.items()
    ]
    logger.info("\n%s", pd.DataFrame(rows).set_index("model").to_string())


def load_model(name: str) -> Any:
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No artifact found for '{name}' at {path}")
    return joblib.load(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    processed_dir = _PROJECT_ROOT / "data" / "processed"
    train_df = pd.read_csv(processed_dir / "train_val_selected.csv")
    test_df = pd.read_csv(processed_dir / "test_selected.csv")
    train_all(train_df=train_df, test_df=test_df)
