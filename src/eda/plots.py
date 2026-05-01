

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

LABEL_COL = "label"
DATE_COL = "Date"
COMPANY_COL = "Company"

CLASS_ORDER = ["Buy", "Hold", "Sell"]
CLASS_COLORS = {"Buy": "#2ca02c", "Hold": "#7f7f7f", "Sell": "#d62728"}


def _ordered_classes(values: Sequence) -> list[str]:
    seen = [c for c in CLASS_ORDER if c in set(values)]
    extras = [c for c in pd.unique(values) if c not in seen]
    return seen + list(extras)




def train_test_timeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = DATE_COL,
) -> go.Figure:
    """Horizontal-bar visualization of train and test date ranges."""
    rows = pd.DataFrame(
        [
            {
                "split": "train_val",
                "start": train_df[date_col].min(),
                "end": train_df[date_col].max(),
            },
            {
                "split": "test",
                "start": test_df[date_col].min(),
                "end": test_df[date_col].max(),
            },
        ]
    )
    rows["days"] = (rows["end"] - rows["start"]).dt.days

    fig = px.timeline(
        rows,
        x_start="start",
        x_end="end",
        y="split",
        color="split",
        custom_data=["days"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>%{base|%Y-%m-%d} → %{x|%Y-%m-%d}"
            "<br>%{customdata[0]} days<extra></extra>"
        )
    )
    fig.update_layout(
        title="Train / Test time coverage",
        xaxis_title="Date",
        yaxis_title="",
        showlegend=False,
        height=260,
    )
    return fig


def class_balance_over_time(
    df: pd.DataFrame,
    freq: str = "ME",
    label_col: str = LABEL_COL,
    date_col: str = DATE_COL,
) -> go.Figure:
    """Stacked-area chart of class proportions resampled to `freq` (default
    month-end). Shows whether one class dominates certain regimes."""
    g = (
        df.groupby([pd.Grouper(key=date_col, freq=freq), label_col])
        .size()
        .unstack(fill_value=0)
    )
    g = g.div(g.sum(axis=1), axis=0)
    classes = _ordered_classes(g.columns)

    fig = go.Figure()
    for cls in classes:
        if cls not in g.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=g.index,
                y=g[cls],
                name=cls,
                stackgroup="one",
                line=dict(color=CLASS_COLORS.get(cls, None), width=0),
                hovertemplate="%{x|%Y-%m}: %{y:.1%}<extra>" + cls + "</extra>",
            )
        )
    fig.update_layout(
        title=f"Class proportions over time (resampled '{freq}')",
        xaxis_title="Period",
        yaxis_title="Proportion",
        yaxis_tickformat=".0%",
        height=380,
    )
    return fig


def class_proportions_by_company(
    df: pd.DataFrame,
    label_col: str = LABEL_COL,
    company_col: str = COMPANY_COL,
) -> go.Figure:
    """Stacked-bar of class % per company."""
    g = (
        df.groupby([company_col, label_col])
        .size()
        .unstack(fill_value=0)
    )
    g = g.div(g.sum(axis=1), axis=0)
    classes = _ordered_classes(g.columns)

    fig = go.Figure()
    for cls in classes:
        if cls not in g.columns:
            continue
        fig.add_trace(
            go.Bar(
                x=g.index,
                y=g[cls],
                name=cls,
                marker_color=CLASS_COLORS.get(cls, None),
                hovertemplate="%{x}: %{y:.1%}<extra>" + cls + "</extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        title="Class proportions per company",
        xaxis_title="Company",
        yaxis_title="Proportion",
        yaxis_tickformat=".0%",
        height=380,
    )
    return fig




def class_conditional_box(
    df: pd.DataFrame,
    feature: str,
    label_col: str = LABEL_COL,
    sample: int | None = 50_000,
    use_violin: bool = False,
) -> go.Figure:
    """Distribution of `feature` split by class. Box by default; violin if
    requested. Subsamples large frames to keep the figure responsive."""
    data = df[[feature, label_col]].dropna()
    if sample is not None and len(data) > sample:
        data = data.sample(n=sample, random_state=0)

    classes = _ordered_classes(data[label_col].unique())
    fig = go.Figure()
    for cls in classes:
        sub = data.loc[data[label_col] == cls, feature]
        color = CLASS_COLORS.get(cls, None)
        if use_violin:
            fig.add_trace(
                go.Violin(y=sub, name=cls, marker_color=color, box_visible=True)
            )
        else:
            fig.add_trace(
                go.Box(y=sub, name=cls, marker_color=color, boxmean=True)
            )
    fig.update_layout(
        title=f"{feature} by class",
        yaxis_title=feature,
        xaxis_title="class",
        showlegend=False,
        height=420,
    )
    return fig


def mi_ranking_bar(
    mi_table: pd.DataFrame,
    top_n: int | None = None,
) -> go.Figure:
    """Horizontal bar chart of features ranked by mutual information.

    Expects the output of `stats.mutual_info_table` (columns: feature,
    mutual_info, f_stat, f_pvalue).
    """
    data = mi_table.copy()
    if top_n is not None:
        data = data.head(top_n)
    data = data.iloc[::-1]  # so highest MI is at the top of the bar chart

    fig = go.Figure(
        go.Bar(
            x=data["mutual_info"],
            y=data["feature"],
            orientation="h",
            customdata=np.stack([data["f_stat"], data["f_pvalue"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "MI: %{x:.4f}<br>"
                "F: %{customdata[0]:.2f}<br>"
                "p: %{customdata[1]:.2e}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Feature mutual information with target",
        xaxis_title="Mutual information",
        yaxis_title="",
        height=max(400, 18 * len(data)),
    )
    return fig





def correlation_heatmap(
    df: pd.DataFrame,
    cluster_order: bool = True,
    method: str = "average",
) -> go.Figure:
    """Pearson correlation heatmap. If `cluster_order`, reorder rows/cols by
    hierarchical clustering so visually adjacent features are similar."""
    num = df.select_dtypes(include="number")
    corr = num.corr()

    if cluster_order and corr.shape[0] > 2:
        dist = 1.0 - corr.abs()
        np.fill_diagonal(dist.values, 0.0)
        order = leaves_list(linkage(squareform(dist.values, checks=False), method=method))
        corr = corr.iloc[order, :].iloc[:, order]

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Correlation heatmap ({'clustered' if cluster_order else 'original'} order)",
        height=700,
        width=750,
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def rolling_feature_stats(
    df: pd.DataFrame,
    feature: str,
    window: int = 60,
    date_col: str = DATE_COL,
    company_col: str = COMPANY_COL,
) -> go.Figure:
    """Rolling mean ± std of a feature over time. If a Company column is
    present, averages across companies per date before rolling so the line
    represents a market-wide trend."""
    cols = [date_col, feature]
    if company_col in df.columns:
        cols.append(company_col)
    data = df[cols].dropna().sort_values(date_col)

    if company_col in data.columns:
        data = data.groupby(date_col, as_index=False)[feature].mean()

    data["mean"] = data[feature].rolling(window, min_periods=window // 2).mean()
    data["std"] = data[feature].rolling(window, min_periods=window // 2).std()
    upper = data["mean"] + data["std"]
    lower = data["mean"] - data["std"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data[date_col],
            y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data[date_col],
            y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(31,119,180,0.2)",
            name=f"±1σ ({window}d)",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data[date_col],
            y=data["mean"],
            mode="lines",
            line=dict(color="#1f77b4"),
            name=f"rolling mean ({window}d)",
        )
    )
    fig.update_layout(
        title=f"{feature}: rolling stats over time (window={window})",
        xaxis_title="Date",
        yaxis_title=feature,
        height=380,
    )
    return fig
