import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.loaders import load_test, load_train
from src.eda.plots import (
    class_balance_over_time,
    class_proportions_by_company,
    train_test_timeline,
)

st.set_page_config(page_title="Trading Signal EDA", layout="wide")

st.title("Trading Signal Classification — EDA")
st.caption(
    "Phases 1–5: orientation, target structure, feature snapshot, "
    "feature ↔ target, redundancy & drift."
)

train = load_train()
test = load_test()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Train rows", f"{len(train):,}")
col2.metric("Test rows", f"{len(test):,}")
col3.metric("Features", len(train.columns) - 3)
col4.metric("Companies", train["Company"].nunique())

st.divider()

st.subheader("Train / Test timeline")
st.plotly_chart(train_test_timeline(train, test), use_container_width=True)

st.subheader("Class proportions per company")
st.plotly_chart(class_proportions_by_company(train), use_container_width=True)

st.subheader("Class proportions over time (training set)")
st.plotly_chart(class_balance_over_time(train), use_container_width=True)
