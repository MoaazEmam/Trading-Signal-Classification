from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.features.features_selection.filter import apply_basic_filters
from src.features.features_selection.importance import (
    apply_importance_filter,
    get_importance_scores,
)
from src.utils import get_nonnumeric_cols

logger = logging.getLogger(__name__)


class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        filter_thresholds: tuple[float, float, float] = (1e-4, 0.95, 0.01),
        importance_threshold: float = 0.01,
        min_features: int = 10,
        max_features: int = 50,
    ) -> None:
        self.filter_thresholds = filter_thresholds
        self.importance_threshold = importance_threshold
        self.min_features = min_features
        self.max_features = max_features

    def fit(self, x: pd.DataFrame, y: pd.Series) -> FeatureSelector:
        logger.info("FeatureSelector.fit: starting — %d columns in", x.shape[1])

        nonnumeric_cols: list[str] = get_nonnumeric_cols(x).columns.tolist()

        after_filters = apply_basic_filters(x, y, self.filter_thresholds)
        logger.info("after filters: %d columns survive", len(after_filters))

        after_importance = apply_importance_filter(
            x[after_filters], y, self.importance_threshold
        )
        numeric_survivors = [c for c in after_importance if c not in nonnumeric_cols]
        logger.info(
            "after importance threshold: %d numeric features survive",
            len(numeric_survivors),
        )

        self.importance_scores_ = get_importance_scores(x[after_filters], y)

        numeric_survivors = self._apply_bounds(numeric_survivors)
        logger.info(
            "after bounds [%d, %d]: %d numeric features selected",
            self.min_features,
            self.max_features,
            len(numeric_survivors),
        )

        self.selected_features_ = numeric_survivors + nonnumeric_cols
        self.n_features_in_ = x.shape[1]
        self.n_features_selected_ = len(self.selected_features_)
        self.fitted_at_ = datetime.utcnow().isoformat()
        self.feature_names_in_ = np.asarray(x.columns)

        logger.info(
            "FeatureSelector.fit: done — %d → %d features",
            self.n_features_in_,
            self.n_features_selected_,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        missing = [c for c in self.selected_features_ if c not in X.columns]
        if missing:
            raise ValueError(
                f"Columns present at fit time are missing from input: {missing}"
            )
        return X[self.selected_features_]

    def fit_transform(
        self, x: pd.DataFrame, y: pd.Series | None = None, **fit_params
    ) -> pd.DataFrame:
        return self.fit(x, y).transform(x)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("FeatureSelector saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> FeatureSelector:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No selector artifact found at {path}")
        selector = joblib.load(path)
        if not isinstance(selector, cls):
            raise TypeError(f"Loaded object is not a FeatureSelector: {type(selector)}")
        logger.info("FeatureSelector loaded from %s", path)
        return selector

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self.selected_features_)

    def _apply_bounds(self, numeric_survivors: list[str]) -> list[str]:
        scores = self.importance_scores_

        if len(numeric_survivors) < self.min_features:
            logger.warning(
                "only %d features survived threshold — relaxing to top %d by importance",
                len(numeric_survivors),
                self.min_features,
            )
            numeric_survivors = [c for c in scores.index if c in scores.index][
                : self.min_features
            ]

        elif len(numeric_survivors) > self.max_features:
            logger.warning(
                "%d features survived threshold — clipping to top %d by importance",
                len(numeric_survivors),
                self.max_features,
            )
            numeric_survivors = [c for c in scores.index if c in numeric_survivors][
                : self.max_features
            ]

        return numeric_survivors

    def _check_fitted(self) -> None:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError(
                "FeatureSelector must be fit before transform() or save()."
            )
