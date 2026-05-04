import sys
from pathlib import Path

import streamlit as st

from app.loaders import load_train
from src.eda.plots import correlation_heatmap
from src.eda.stats import correlation_clusters

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Correlation Lab", layout="wide")
st.title("Correlation Lab")
st.caption(
    "The selector removed pairs at |corr| ≥ 0.95. "
    "Everything in the 0.70–0.95 band is still present and represents real redundancy the model must resolve."
)

train = load_train()
X = train.drop(columns=["Date", "Company", "label"])

st.subheader("Correlation heatmap")
cluster_order = st.toggle("Clustered order", value=True)
st.plotly_chart(
    correlation_heatmap(X, cluster_order=cluster_order), use_container_width=True
)

st.divider()

st.subheader("Correlation clusters")
threshold = st.slider(
    "Cluster threshold (|corr| ≥)",
    min_value=0.50,
    max_value=0.95,
    value=0.70,
    step=0.05,
)

clusters = correlation_clusters(X, corr_threshold=threshold)
multi = clusters[clusters["n_in_cluster"] > 1]
n_singletons = clusters["cluster_id"].nunique() - multi["cluster_id"].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("Total clusters", clusters["cluster_id"].nunique())
col2.metric("Redundancy groups (n > 1)", multi["cluster_id"].nunique())
col3.metric("Singleton features", n_singletons)

tab1, tab2 = st.tabs(["Redundancy groups", "All clusters"])
with tab1:
    st.caption(
        "Features in the same group are interchangeable from a linear-model perspective."
    )
    st.dataframe(
        multi.sort_values(["n_in_cluster", "cluster_id"], ascending=[False, True]),
        use_container_width=True,
    )
with tab2:
    st.dataframe(clusters, use_container_width=True)
