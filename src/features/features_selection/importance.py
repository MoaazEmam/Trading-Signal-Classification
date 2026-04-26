import lightgbm as lgb
import pandas as pd

from src.data.splitting import get_time_series_cv
from src.utils import get_nonnumeric_cols, get_numeric_cols


def _train_selector_model(x: pd.DataFrame, y: pd.Series) -> lgb.Booster:
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "num_leaves": 31,
        "n_estimators": 200,
        "learning_rate": 0.05,
        "verbosity": -1,
        "random_state": 42,
    }
    dataset = lgb.Dataset(x, label=y)
    return lgb.train(params, dataset, num_boost_round=200)


def _extract_importances(booster: lgb.Booster, feature_names: list[str]) -> pd.Series:
    scores = booster.feature_importance(importance_type="gain")
    return pd.Series(scores, index=feature_names).sort_values(ascending=False)


def get_importance_scores(
    x: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> pd.Series:
    """
    trains the lightweight gbm 5 times using time-aware splits to avoid leakage
    returns the average importance scores of each feature across the 5 attempts
    """
    numeric_cols = get_numeric_cols(x)
    numeric_col_names: list[str] = numeric_cols.columns.tolist()
    x_numeric = numeric_cols.fillna(0).reset_index(drop=True)
    y = y.reset_index(drop=True)

    tscv = get_time_series_cv(n_splits=n_splits)
    all_scores = []
    for train_idx, _ in tscv.split(x_numeric):
        X_fold = x_numeric.iloc[train_idx]
        y_fold = y.iloc[train_idx]
        booster = _train_selector_model(X_fold, y_fold)
        scores = _extract_importances(booster, numeric_col_names)
        all_scores.append(scores)

    averaged = pd.concat(all_scores, axis=1).mean(axis=1).sort_values(ascending=False)
    return averaged


def apply_importance_filter(
    x: pd.DataFrame, y: pd.Series, threshold: float = 0.01
) -> list[str]:
    numeric_col_names: list[str] = get_numeric_cols(x).columns.tolist()
    nonnumeric_col_names: list[str] = get_nonnumeric_cols(x).columns.tolist()

    scores = get_importance_scores(x, y)
    min_score = threshold * float(scores.max())
    surviving_numeric = [
        c for c in numeric_col_names if scores.get(c, 0.0) >= min_score
    ]

    return surviving_numeric + nonnumeric_col_names
