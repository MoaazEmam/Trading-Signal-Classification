"""Unit tests for src/eda/plots.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.eda.plots import (
    _ordered_classes,
    class_balance_over_time,
    class_conditional_box,
    class_proportions_by_company,
    correlation_heatmap,
    mi_ranking_bar,
    rolling_feature_stats,
    train_test_timeline,
)


@pytest.fixture()
def labelled_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 300
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "Company": rng.choice(["AAPL", "MSFT", "NVDA"], size=n),
            "label": rng.choice(["Buy", "Hold", "Sell"], size=n),
            "rsi": rng.normal(50, 10, n),
            "momentum": rng.normal(0, 1, n),
            "volume_z": rng.normal(0, 1, n),
        }
    )


# ---------------------------------------------------------------------------
# _ordered_classes
# ---------------------------------------------------------------------------


def test_ordered_classes_puts_known_classes_first():
    assert _ordered_classes(["Sell", "Other", "Buy"]) == ["Buy", "Sell", "Other"]


def test_ordered_classes_keeps_full_order():
    assert _ordered_classes(["Hold", "Sell", "Buy"]) == ["Buy", "Hold", "Sell"]


# ---------------------------------------------------------------------------
# train_test_timeline
# ---------------------------------------------------------------------------


def test_train_test_timeline_has_both_splits(labelled_df):
    train, test = labelled_df.iloc[:200], labelled_df.iloc[200:]
    fig = train_test_timeline(train, test)
    assert isinstance(fig, go.Figure)
    splits = {y for trace in fig.data for y in trace.y}
    assert splits == {"train_val", "test"}


# ---------------------------------------------------------------------------
# class_balance_over_time / class_proportions_by_company
# ---------------------------------------------------------------------------


def test_class_balance_over_time_one_trace_per_class(labelled_df):
    fig = class_balance_over_time(labelled_df)
    assert [t.name for t in fig.data] == ["Buy", "Hold", "Sell"]
    stacked = sum(np.asarray(t.y) for t in fig.data)
    assert np.allclose(stacked, 1.0)


def test_class_proportions_by_company_sums_to_one(labelled_df):
    fig = class_proportions_by_company(labelled_df)
    assert fig.layout.barmode == "stack"
    stacked = sum(np.asarray(t.y) for t in fig.data)
    assert np.allclose(stacked, 1.0)
    assert set(fig.data[0].x) == {"AAPL", "MSFT", "NVDA"}


# ---------------------------------------------------------------------------
# class_conditional_box
# ---------------------------------------------------------------------------


def test_class_conditional_box_uses_box_by_default(labelled_df):
    fig = class_conditional_box(labelled_df, "rsi")
    assert all(isinstance(t, go.Box) for t in fig.data)
    assert [t.name for t in fig.data] == ["Buy", "Hold", "Sell"]


def test_class_conditional_box_violin_and_sampling(labelled_df):
    fig = class_conditional_box(labelled_df, "rsi", sample=50, use_violin=True)
    assert all(isinstance(t, go.Violin) for t in fig.data)
    assert sum(len(t.y) for t in fig.data) == 50


# ---------------------------------------------------------------------------
# mi_ranking_bar
# ---------------------------------------------------------------------------


def test_mi_ranking_bar_top_n_highest_first():
    mi = pd.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "mutual_info": [0.3, 0.2, 0.1],
            "f_stat": [10.0, 5.0, 1.0],
            "f_pvalue": [0.001, 0.01, 0.5],
        }
    )
    fig = mi_ranking_bar(mi, top_n=2)
    bar = fig.data[0]
    # reversed so the highest MI renders at the top of a horizontal chart
    assert list(bar.y) == ["b", "a"]
    assert fig.layout.height == 400


# ---------------------------------------------------------------------------
# correlation_heatmap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cluster_order", [True, False])
def test_correlation_heatmap_is_square_numeric(labelled_df, cluster_order):
    fig = correlation_heatmap(labelled_df, cluster_order=cluster_order)
    heat = fig.data[0]
    assert set(heat.x) == {"rsi", "momentum", "volume_z"}
    assert list(heat.x) == list(heat.y)
    assert np.allclose(np.diag(np.asarray(heat.z)), 1.0)


# ---------------------------------------------------------------------------
# rolling_feature_stats
# ---------------------------------------------------------------------------


def test_rolling_feature_stats_band_and_mean(labelled_df):
    fig = rolling_feature_stats(labelled_df, "momentum", window=20)
    assert len(fig.data) == 3
    assert fig.data[2].name == "rolling mean (20d)"


def test_rolling_feature_stats_without_company(labelled_df):
    df = labelled_df.drop(columns="Company")
    fig = rolling_feature_stats(df, "rsi", window=10)
    assert len(fig.data[2].x) == len(df)
