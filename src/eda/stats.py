

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import f_classif, mutual_info_classif


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include="number")


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per numeric feature with summary stats.

    Columns: count, mean, std, min, p25, p50, p75, max, skew, kurtosis,
    n_unique. `n_unique` surfaces low-cardinality numeric features
    (e.g. sma_cross flags, month, vol_regime) that act categorical.
    """
    num = _numeric(df)
    summary = pd.DataFrame(
        {
            "count": num.count(),
            "mean": num.mean(),
            "std": num.std(),
            "min": num.min(),
            "p25": num.quantile(0.25),
            "p50": num.median(),
            "p75": num.quantile(0.75),
            "max": num.max(),
            "skew": num.skew(),
            "kurtosis": num.kurt(),
            "n_unique": num.nunique(),
        }
    )
    summary.index.name = "feature"
    return summary


def class_conditional_stats(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Per-class mean / median / std / count for each numeric feature.

    Returns long-format with columns: feature, class, mean, median, std, count.
    """
    num = _numeric(X)
    parts = []
    for cls in pd.unique(y):
        sub = num.loc[y == cls]
        block = pd.DataFrame(
            {
                "mean": sub.mean(),
                "median": sub.median(),
                "std": sub.std(),
                "count": sub.count(),
            }
        )
        block.insert(0, "class", cls)
        block.index.name = "feature"
        parts.append(block.reset_index())
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["feature", "class"])
        .reset_index(drop=True)
    )


def mutual_info_table(
    X: pd.DataFrame,
    y: pd.Series,
    n_samples: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Rank features by mutual information with y and by ANOVA F-statistic.

    Returns DataFrame sorted by mutual_info descending, with columns:
    feature, mutual_info, f_stat, f_pvalue.

    For very large datasets (>200k rows) pass `n_samples` to subsample —
    MI estimation is the bottleneck.
    """
    num = _numeric(X)
    if n_samples is not None and n_samples < len(num):
        idx = num.sample(n=n_samples, random_state=random_state).index
        num = num.loc[idx]
        y_used = y.loc[idx]
    else:
        y_used = y

    mi = mutual_info_classif(num, y_used, random_state=random_state)
    f_stat, p_value = f_classif(num, y_used)

    return (
        pd.DataFrame(
            {
                "feature": num.columns,
                "mutual_info": mi,
                "f_stat": f_stat,
                "f_pvalue": p_value,
            }
        )
        .sort_values("mutual_info", ascending=False)
        .reset_index(drop=True)
    )


def correlation_clusters(
    X: pd.DataFrame,
    corr_threshold: float = 0.7,
    method: str = "average",
) -> pd.DataFrame:
    """Hierarchical clustering of numeric features by |Pearson correlation|.

    Features with |corr| >= corr_threshold get grouped into the same cluster.
    Distance metric is 1 - |corr|; linkage method defaults to "average".

    Returns DataFrame sorted by cluster_id then feature, with columns:
    feature, cluster_id, n_in_cluster.
    """
    num = _numeric(X)
    corr = num.corr().abs()
    dist = 1.0 - corr
    np.fill_diagonal(dist.values, 0.0)
    condensed = squareform(dist.values, checks=False)
    z = linkage(condensed, method=method)
    cluster_ids = fcluster(z, t=1.0 - corr_threshold, criterion="distance")

    out = pd.DataFrame({"feature": num.columns, "cluster_id": cluster_ids})
    out["n_in_cluster"] = out.groupby("cluster_id")["feature"].transform("count")
    return out.sort_values(["cluster_id", "feature"]).reset_index(drop=True)
