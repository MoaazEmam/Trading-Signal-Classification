import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.loaders import load_mi, load_selector
from src.eda.plots import mi_ranking_bar

st.set_page_config(page_title="MI & Importance", layout="wide")
st.title("Mutual Information & LightGBM Gain")
st.caption(
    "MI measures marginal dependence with the target. "
    "LightGBM gain measures contextual split utility. "
    "Disagreements reveal interaction-driven features and marginal-signal features the selector under-used."
)

mi = load_mi()

st.subheader("Feature ranking by mutual information")
top_n = st.slider("Top N features", min_value=5, max_value=len(mi), value=25, step=5)
st.plotly_chart(mi_ranking_bar(mi, top_n=top_n), use_container_width=True)

st.divider()

st.subheader("MI vs LightGBM gain")
selector = load_selector()

if selector is None:
    st.warning("feature_selector.pkl not found in models/artifacts/. Run the selection pipeline first.")
else:
    gain = selector.importance_scores_.rename("lgbm_gain").reset_index()
    gain.columns = ["feature", "lgbm_gain"]

    compare = mi.merge(gain, on="feature", how="left")
    compare["lgbm_gain"] = compare["lgbm_gain"].fillna(0)
    compare["mi_rank"] = compare["mutual_info"].rank(ascending=False).astype(int)
    compare["gain_rank"] = compare["lgbm_gain"].rank(ascending=False).astype(int)
    compare["rank_delta"] = compare["gain_rank"] - compare["mi_rank"]

    COLS = ["feature", "mutual_info", "mi_rank", "lgbm_gain", "gain_rank", "rank_delta"]
    FLOAT_COLS = {"mutual_info": "{:.4f}", "lgbm_gain": "{:.4f}"}

    tab1, tab2, tab3 = st.tabs(
        ["MI > Gain (marginal signal)", "Gain > MI (interaction-driven)", "Full table"]
    )
    with tab1:
        st.caption(
            "Strong marginal signal that LightGBM ranked lower — "
            "likely redundant with a correlated feature it preferred."
        )
        st.dataframe(
            compare.sort_values("rank_delta", ascending=False).head(10)[COLS].style.format(FLOAT_COLS),
            use_container_width=True,
        )
    with tab2:
        st.caption(
            "Little marginal signal but high split utility — "
            "earns gain through interactions with other features."
        )
        st.dataframe(
            compare.sort_values("rank_delta").head(10)[COLS].style.format(FLOAT_COLS),
            use_container_width=True,
        )
    with tab3:
        st.dataframe(
            compare.sort_values("mi_rank")[COLS].style.format(FLOAT_COLS),
            use_container_width=True,
        )
