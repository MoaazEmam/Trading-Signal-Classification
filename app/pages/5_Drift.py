import sys
from pathlib import Path

import streamlit as st

from app.loaders import load_mi, load_train
from src.eda.plots import class_balance_over_time, rolling_feature_stats

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


st.set_page_config(page_title="Drift", layout="wide")
st.title("Feature & Class Drift")

train = load_train()
mi = load_mi()

st.subheader("Rolling feature drift")
st.caption(
    "Rolling mean ± 1σ averaged market-wide across companies. "
    "A trending mean or expanding variance band indicates non-stationarity — "
    "a risk for test-set generalization."
)

feature = st.selectbox("Feature", options=mi["feature"].tolist())
window = st.slider(
    "Rolling window (days)", min_value=20, max_value=120, value=60, step=10
)
st.plotly_chart(
    rolling_feature_stats(train, feature, window=window), use_container_width=True
)

st.divider()

st.subheader("Class proportion drift (market-wide)")
st.caption(
    "Monthly Buy / Hold / Sell proportions across all companies. "
    "A flat chart means the model faces a stable target distribution over time. "
    "Sustained regime shifts suggest considering time-weighted losses or recalibration."
)
st.plotly_chart(class_balance_over_time(train), use_container_width=True)
