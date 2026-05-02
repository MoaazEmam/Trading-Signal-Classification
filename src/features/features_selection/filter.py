import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.utils import get_nonnumeric_cols, get_numeric_cols


def _apply_variance_filter(x: pd.DataFrame, threshold: float = 1e-4) -> list[str]:
    """
    remove columns with very low variance, essentially not changing enough to be useful
    """
    numeric_cols = get_numeric_cols(x)
    nonnumeric_cols = get_nonnumeric_cols(x)
    variances = numeric_cols.var()
    surviving_numeric = variances[variances >= threshold].index.tolist()
    return surviving_numeric + nonnumeric_cols.columns.tolist()


def _apply_correlation_filter(
    x: pd.DataFrame, y: pd.Series, threshold: float = 0.95
) -> list[str]:
    """
    find features with high correlation, keep the one with the highest correlation to label
    """
    numeric_cols = get_numeric_cols(x)
    nonnumeric_cols = get_nonnumeric_cols(x)

    corr_matrix = numeric_cols.corr()
    corr_with_target = numeric_cols.corrwith(y)

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = set()
    for col in upper_triangle.columns:
        # skip column if its already marked to drop
        if col in to_drop:
            continue
        correlated_with = upper_triangle.index[
            upper_triangle[col].abs() > threshold
        ].tolist()
        for other in correlated_with:
            if other in to_drop:
                continue
            if abs(corr_with_target[col]) >= abs(corr_with_target[other]):
                to_drop.add(other)
            else:
                to_drop.add(col)
                break

    surviving_numeric = [str(c) for c in numeric_cols if c not in to_drop]
    return surviving_numeric + nonnumeric_cols.columns.tolist()


def _apply_mutual_info_filter(
    x: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.01,
    top_n: int | None = None,
    top_pct: float | None = None,
) -> list[str]:
    """
    mutual info essentially quantifies how much knowing one column reduces uncertainty about the label
    drop the ones that do not help with the label
    """
    numeric_cols = get_numeric_cols(x)
    numeric_col_names = numeric_cols.columns.tolist()
    nonnumeric_cols = get_nonnumeric_cols(x)

    mi_scores = mutual_info_classif(numeric_cols.fillna(0), y, random_state=42)
    mi_series = (pd.Series(mi_scores, index=numeric_col_names)).sort_values(
        ascending=False
    )
    if top_n is not None:
        surviving_numeric = mi_series.head(top_n).index.tolist()
    elif top_pct is not None:
        k = max(1, int(len(mi_series) * top_pct))
        surviving_numeric = mi_series.head(k).index.tolist()
    else:
        min_score = threshold * float(mi_series.max()) if mi_series.max() > 0 else 0.0
        surviving_numeric = mi_series[mi_series >= min_score].index.tolist()
    return surviving_numeric + nonnumeric_cols.columns.tolist()

    # min_score = threshold * float(mi_series.max())
    # surviving_numeric = [str(c) for c in numeric_col_names if mi_series[c] >= min_score]
    #
    # return surviving_numeric + nonnumeric_cols.columns.tolist()


def apply_basic_filters(
    x: pd.DataFrame,
    y: pd.Series,
    thresholds: tuple[float, float, float],
    mi_top_n: int | None = None,
    mi_top_pct: float | None = None,
) -> list[str]:
    variance_threshold, correlation_threshold, mi_threshold = thresholds
    after_variance_list = _apply_variance_filter(x, variance_threshold)
    after_correlation_list = _apply_correlation_filter(
        x[after_variance_list], y, correlation_threshold
    )
    after_mi_filter = _apply_mutual_info_filter(
        x[after_correlation_list], y, mi_threshold, mi_top_n, mi_top_pct
    )
    return after_mi_filter
