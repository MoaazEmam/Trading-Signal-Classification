import sys
from pathlib import Path

import streamlit as st

from app.loaders import load_mi, load_train
from src.eda.plots import class_conditional_box
from src.eda.stats import class_conditional_stats

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


st.set_page_config(page_title="Class-Conditional View", layout="wide")
st.title("Class-Conditional View")
st.caption(
    "Distribution of a feature split by Buy / Hold / Sell. "
    "Features are shown on the standardized scale; select from the MI-ranked list."
)

train = load_train()
mi = load_mi()

feature = st.selectbox("Feature", options=mi["feature"].tolist())

col_plot, col_stats = st.columns([2, 1])

with col_plot:
    st.plotly_chart(class_conditional_box(train, feature), use_container_width=True)

with col_stats:
    X = train.drop(columns=["Date", "Company", "label"])
    y = train["label"]
    stats = class_conditional_stats(X[[feature]], y)

    st.subheader("Per-class mean")
    st.dataframe(
        stats.pivot(index="feature", columns="class", values="mean").style.format(
            "{:.4f}"
        ),
        use_container_width=True,
    )
    st.subheader("Per-class median")
    st.dataframe(
        stats.pivot(index="feature", columns="class", values="median").style.format(
            "{:.4f}"
        ),
        use_container_width=True,
    )
    st.subheader("Per-class std")
    st.dataframe(
        stats.pivot(index="feature", columns="class", values="std").style.format(
            "{:.4f}"
        ),
        use_container_width=True,
    )
