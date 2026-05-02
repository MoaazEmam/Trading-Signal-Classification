import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.loaders import load_feature_summary

st.set_page_config(page_title="Feature Explorer", layout="wide")
st.title("Feature Explorer")
st.caption(
    "Summary statistics for all selected features. "
    "Values are winsorized and standardized, so mean ≈ 0 and std ≈ 1 for most features."
)

summary = load_feature_summary()

st.subheader("All features")
st.dataframe(summary.style.format("{:.4f}"), use_container_width=True)

st.subheader("Low-cardinality features (effectively categorical or macro)")
st.caption(
    "Low `n_unique` flags features that look continuous but behave categorically "
    "(e.g. `sma_cross_*`, `month`, `vol_regime`) or macro features that repeat "
    "across companies (`vix`, `fed_funds_rate`, `sp500_level`)."
)
low_card = summary.sort_values("n_unique").head(15)[["n_unique", "min", "max", "mean", "std"]]
st.dataframe(low_card.style.format("{:.4f}"), use_container_width=True)

st.subheader("Extreme skew / kurtosis (tail survivors after winsorization)")
st.caption(
    "Features whose tails survived winsorization. "
    "Consider tighter clipping or a log-transform upstream if skew is extreme."
)
extreme = (
    summary.assign(abs_skew=summary["skew"].abs())[
        ["mean", "std", "min", "max", "skew", "kurtosis", "abs_skew"]
    ]
    .sort_values("abs_skew", ascending=False)
    .head(10)
)
st.dataframe(extreme.style.format("{:.4f}"), use_container_width=True)
