import sys
from pathlib import Path

import requests
import streamlit as st

from app.loaders import load_test, load_train
from src.config import settings
from src.eda.plots import (
    class_balance_over_time,
    class_proportions_by_company,
    train_test_timeline,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


st.set_page_config(
    page_title="Trading Signal Classifier",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    [data-testid="block-container"] { padding-top: 1.5rem; }
    .card {
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.12);
        background: var(--background-secondary, #f9f9f9);
        margin-bottom: 0.75rem;
    }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def _api_health() -> bool:
    try:
        r = requests.get(f"{settings.api_url}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


with st.sidebar:
    st.markdown("## 📈 Trading Signal Classifier")
    st.markdown("---")
    st.markdown("**Navigation**")
    st.page_link("streamlit_app.py", label="EDA Overview", icon="🏠")
    st.page_link("pages/1_Feature_Explorer.py", label="Feature Explorer", icon="🔍")
    st.page_link("pages/2_MI_Importance.py", label="MI & Importance", icon="📊")
    st.page_link("pages/3_Class_Conditional.py", label="Class Conditional", icon="📦")
    st.page_link("pages/4_Correlation.py", label="Correlation Lab", icon="🔗")
    st.page_link("pages/5_Drift.py", label="Drift", icon="📉")
    st.page_link("pages/6_Predictions.py", label="Daily Predictions", icon="🎯")
    st.page_link("pages/7_Model_Performance.py", label="Model Performance", icon="🏆")
    st.markdown("---")

    healthy = _api_health()
    status_color = "🟢" if healthy else "🔴"
    status_text = "API online" if healthy else "API offline"
    st.markdown(f"{status_color} {status_text}")

    predictions_dir = ROOT / "predictions" / "history"
    if predictions_dir.exists():
        archives = sorted(predictions_dir.glob("*.json"))
        if archives:
            st.caption(f"Last prediction run: **{archives[-1].stem}**")


st.header("Trading Signal Classification — EDA")
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
