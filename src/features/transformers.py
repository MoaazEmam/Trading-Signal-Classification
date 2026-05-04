"""
Custom sklearn-compatible transformers for the feature pipeline.

Each transformer follows the fit/transform contract so it can be composed
inside a sklearn Pipeline or ColumnTransformer and persisted with joblib.
State is learned only in fit(); transform() is a pure function of that state
plus the input DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GroupedWinsorizer(BaseEstimator, TransformerMixin):
    """
    Caps specified columns at a per-group quantile learned during fit.

    Motivation
    ----------
    Volume has extreme right-tail outliers that differ in magnitude across
    companies (a mega-cap's normal volume dwarfs a small-cap's spikes).
    A global percentile cap either over-clips small-caps or under-clips
    mega-caps. Per-company caps preserve each ticker's own distribution.

    This is stateful and belongs in the feature pipeline, not in stateless
    cleaning: the caps must be learned on train_val only and reused on test
    and live inference data to prevent leakage.

    Parameters
    ----------
    group_col : str
        Column used to partition rows before computing the quantile.
    cols : list[str]
        Numeric columns to cap.
    q : float
        Upper quantile in (0, 1). Default 0.99.
    lower_q : float | None
        Optional lower quantile. If None, only the upper tail is capped.
    unseen_group_policy : {"global", "passthrough"}
        Behaviour when transform() sees a group not present at fit time.
        "global"      - apply the global (across-all-groups) cap from fit.
        "passthrough" - leave those rows untouched.

    Attributes (set by fit)
    -----------------------
    upper_caps_   : dict[str, pd.Series]   # col -> Series indexed by group -> cap
    global_upper_ : dict[str, float]       # col -> global cap
    lower_caps_   : dict[str, pd.Series]   # only if lower_q is set
    global_lower_ : dict[str, float]       # only if lower_q is set
    feature_names_in_ : np.ndarray
    """

    def __init__(
        self,
        group_col: str = "Company",
        cols: list[str] | None = None,
        q: float = 0.99,
        lower_q: float | None = None,
        unseen_group_policy: str = "global",
    ) -> None:
        self.group_col = group_col
        self.cols = cols
        self.q = q
        self.lower_q = lower_q
        self.unseen_group_policy = unseen_group_policy

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> GroupedWinsorizer:
        self._validate_input(X)
        cols = self._resolve_cols(X)

        grouped = X.groupby(self.group_col, observed=True)

        self.upper_caps_ = {col: grouped[col].quantile(self.q) for col in cols}
        self.global_upper_ = {col: float(X[col].quantile(self.q)) for col in cols}

        if self.lower_q is not None:
            self.lower_caps_ = {
                col: grouped[col].quantile(self.lower_q) for col in cols
            }
            self.global_lower_ = {
                col: float(X[col].quantile(self.lower_q)) for col in cols
            }

        self.feature_names_in_ = np.asarray(X.columns)
        self._fitted_cols_ = cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "upper_caps_"):
            raise RuntimeError("GroupedWinsorizer must be fit before transform().")
        self._validate_input(X)

        out = X.copy()
        groups = out[self.group_col]

        for col in self._fitted_cols_:
            if pd.api.types.is_integer_dtype(out[col]):
                out[col] = out[col].astype(float)
            upper = groups.map(self.upper_caps_[col])
            if self.unseen_group_policy == "global":
                upper = upper.fillna(self.global_upper_[col])
            mask = upper.notna()
            out.loc[mask, col] = np.minimum(out.loc[mask, col], upper[mask])

            if self.lower_q is not None:
                lower = groups.map(self.lower_caps_[col])
                if self.unseen_group_policy == "global":
                    lower = lower.fillna(self.global_lower_[col])
                mask = lower.notna()
                out.loc[mask, col] = np.maximum(out.loc[mask, col], lower[mask])

        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return (
            np.asarray(input_features)
            if input_features is not None
            else self.feature_names_in_
        )

    def _resolve_cols(self, X: pd.DataFrame) -> list[str]:
        if self.cols is None:
            raise ValueError(
                "`cols` must be specified (list of numeric columns to cap)."
            )
        missing = [c for c in self.cols if c not in X.columns]
        if missing:
            raise ValueError(f"Columns not found in input: {missing}")
        return list(self.cols)

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("GroupedWinsorizer expects a pandas DataFrame.")
        if self.group_col not in X.columns:
            raise ValueError(f"group_col '{self.group_col}' not in input columns.")
        if not 0 < self.q < 1:
            raise ValueError("`q` must be in (0, 1).")
        if self.lower_q is not None and not 0 < self.lower_q < self.q:
            raise ValueError("`lower_q` must be in (0, q).")
        if self.unseen_group_policy not in {"global", "passthrough"}:
            raise ValueError("unseen_group_policy must be 'global' or 'passthrough'.")


class GlobalWinsorizer(BaseEstimator, TransformerMixin):
    """
    Caps specified columns at global quantiles learned during fit.

    Used for features where cross-company comparison is already meaningful
    (returns, MACD normalized to price, candle body sizes, etc.) so
    per-company grouping is not appropriate. The caps are computed once
    across all rows in the training set and applied identically to every
    row at transform time.

    Parameters
    ----------
    cols : list[str]
        Numeric columns to cap.
    upper_q : float
        Upper quantile in (0, 1). Default 0.99.
    lower_q : float
        Lower quantile in (0, 1). Default 0.01.

    Attributes (set by fit)
    -----------------------
    upper_caps_ : dict[str, float]
    lower_caps_ : dict[str, float]
    feature_names_in_ : np.ndarray
    """

    def __init__(
        self,
        cols: list[str] | None = None,
        upper_q: float = 0.99,
        lower_q: float = 0.01,
    ) -> None:
        self.cols = cols
        self.upper_q = upper_q
        self.lower_q = lower_q

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> GlobalWinsorizer:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("GlobalWinsorizer expects a pandas DataFrame.")
        if self.cols is None:
            raise ValueError("`cols` must be specified.")
        missing = [c for c in self.cols if c not in X.columns]
        if missing:
            raise ValueError(f"Columns not found in input: {missing}")
        if not 0 < self.lower_q < self.upper_q < 1:
            raise ValueError(
                "`lower_q` must be in (0, upper_q) and `upper_q` in (lower_q, 1)."
            )

        self.upper_caps_ = {
            col: float(X[col].quantile(self.upper_q)) for col in self.cols
        }
        self.lower_caps_ = {
            col: float(X[col].quantile(self.lower_q)) for col in self.cols
        }
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "upper_caps_"):
            raise RuntimeError("GlobalWinsorizer must be fit before transform().")
        out = X.copy()
        for col in self.cols:
            if col not in out.columns:
                continue
            if pd.api.types.is_integer_dtype(out[col]):
                out[col] = out[col].astype(float)
            out[col] = out[col].clip(
                lower=self.lower_caps_[col],
                upper=self.upper_caps_[col],
            )
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return (
            np.asarray(input_features)
            if input_features is not None
            else self.feature_names_in_
        )


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Drops a list of columns at transform time, silently skipping any that
    are absent. Fit stores only the intersection of requested cols and actual
    columns so downstream get_feature_names_out is always accurate.
    """

    def __init__(self, cols: list[str]) -> None:
        self.cols = cols

    def fit(self, X: pd.DataFrame, y=None) -> ColumnDropper:
        self.cols_to_drop_ = [c for c in self.cols if c in X.columns]
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.cols_to_drop_)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        cols = input_features if input_features is not None else self.feature_names_in_
        return np.asarray([c for c in cols if c not in self.cols_to_drop_])
