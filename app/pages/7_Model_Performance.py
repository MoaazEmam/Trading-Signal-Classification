from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from src.config import settings

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


st.set_page_config(page_title="Model Performance", layout="wide")

API_BASE = settings.api_url

CLASS_COLORS = {"Buy": "#2ca02c", "Hold": "#7f7f7f", "Sell": "#d62728"}
CHART_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]

BACKTEST_PATH = ROOT / "models" / "artifacts" / "backtest_results.json"
TRAINING_PATH = ROOT / "models" / "artifacts" / "training_results.json"


# ── data loaders ─────────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def _load_mlflow_runs() -> pd.DataFrame:
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        runs = mlflow.search_runs(
            experiment_names=["Trading-Signal-Classification"],
            order_by=["start_time DESC"],
        )
        return runs
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _load_backtest_results() -> dict:
    if not BACKTEST_PATH.exists():
        return {}
    import json

    return json.loads(BACKTEST_PATH.read_text())


@st.cache_data(ttl=300)
def _load_training_results() -> dict:
    if not TRAINING_PATH.exists():
        return {}
    import json

    data = json.loads(TRAINING_PATH.read_text())
    return data.get("models", {})


# ── page header ──────────────────────────────────────────────────────────────

st.header("Model Performance")
st.caption("MLflow experiment history, backtest results, and on-demand backtest runs.")

tab1, tab2, tab3 = st.tabs(["MLflow Runs", "Backtest Dashboard", "Run Backtest"])


# ── Tab 1: MLflow Run Browser ─────────────────────────────────────────────────

with tab1:
    st.subheader("MLflow Run Browser")
    st.caption(
        "All logged training runs from the Trading-Signal-Classification experiment."
    )

    runs_df = _load_mlflow_runs()

    if runs_df.empty:
        st.info(
            "No MLflow runs found. Run the training pipeline to populate experiment history."
        )
    else:
        display_cols = {}
        for col in runs_df.columns:
            if col == "tags.mlflow.runName":
                display_cols["Run Name"] = col
            elif col == "params.model_name":
                display_cols["Model"] = col
            elif col == "start_time":
                display_cols["Start Time"] = col
            elif "f1" in col.lower() and "weighted" in col.lower():
                display_cols["F1 (weighted)"] = col
            elif "mcc" in col.lower():
                display_cols["MCC"] = col
            elif "auc" in col.lower() and "pr" in col.lower():
                display_cols["AUC-PR"] = col
            elif "total_return" in col.lower():
                display_cols["Backtest Return %"] = col
            elif "sharpe" in col.lower():
                display_cols["Sharpe Ratio"] = col
            elif "win_rate" in col.lower():
                display_cols["Win Rate"] = col
            elif "profit_factor" in col.lower():
                display_cols["Profit Factor"] = col

        available = {
            label: col for label, col in display_cols.items() if col in runs_df.columns
        }
        table_df = runs_df[list(available.values())].rename(
            columns={v: k for k, v in available.items()}
        )

        st.dataframe(table_df, use_container_width=True)

        run_names = runs_df.get(
            "tags.mlflow.runName", runs_df.get("run_id", runs_df.index.astype(str))
        ).tolist()
        selected_run_name = st.selectbox("Select a run to inspect", options=run_names)

        if selected_run_name:
            mask = (
                runs_df.get("tags.mlflow.runName", runs_df.index.astype(str))
                == selected_run_name
            )
            selected_row = runs_df[mask].iloc[0] if mask.any() else None

            if selected_row is not None:
                param_cols = [c for c in runs_df.columns if c.startswith("params.")]
                metric_cols = [c for c in runs_df.columns if c.startswith("metrics.")]

                ml_metrics = [
                    c
                    for c in metric_cols
                    if not any(
                        k in c for k in ["return", "sharpe", "win_rate", "profit"]
                    )
                ]
                fin_metrics = [
                    c
                    for c in metric_cols
                    if any(k in c for k in ["return", "sharpe", "win_rate", "profit"])
                ]

                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown("**Parameters & ML Metrics**")
                    param_data = {
                        c.replace("params.", ""): selected_row[c]
                        for c in param_cols
                        if pd.notna(selected_row[c])
                    }
                    ml_data = {
                        c.replace("metrics.", ""): round(float(selected_row[c]), 4)
                        for c in ml_metrics
                        if pd.notna(selected_row[c])
                    }
                    combined = {**param_data, **ml_data}
                    st.dataframe(
                        pd.DataFrame(combined.items(), columns=["Key", "Value"]),
                        use_container_width=True,
                    )

                with col_right:
                    st.markdown("**Financial Metrics**")
                    if fin_metrics:
                        fin_data = {
                            c.replace("metrics.", ""): round(float(selected_row[c]), 4)
                            for c in fin_metrics
                            if pd.notna(selected_row[c])
                        }
                        st.dataframe(
                            pd.DataFrame(fin_data.items(), columns=["Metric", "Value"]),
                            use_container_width=True,
                        )
                    else:
                        st.info("No financial metrics logged for this run.")


# ── Tab 2: Backtest Results Dashboard ────────────────────────────────────────

with tab2:
    st.subheader("Backtest Results")
    st.caption(
        "Simulated trading performance across all trained models vs. the naive buy baseline."
    )

    backtest = _load_backtest_results()

    if not backtest:
        st.info("No backtest results found. Run the backtest pipeline first.")
    else:
        rows = []
        for model_name, data in backtest.items():
            m = data.get("metrics", {})
            rows.append(
                {
                    "Model": model_name,
                    "Total Return %": m.get("total_return_pct"),
                    "Sharpe Ratio": m.get("sharpe_ratio"),
                    "Win Rate": m.get("win_rate"),
                    "Profit Factor": m.get("profit_factor"),
                    "Total Trades": data.get("total_trades"),
                    "Final Capital": data.get("final_capital"),
                }
            )

        comp_df = pd.DataFrame(rows)
        best_idx = comp_df["Total Return %"].idxmax()
        best_model = comp_df.loc[best_idx, "Model"]
        best_return = comp_df.loc[best_idx, "Total Return %"]

        st.markdown(f"**Best model:** `{best_model}` — {best_return:.2f}% return")

        def _highlight_best(row):
            return [
                (
                    "background-color: #1a472a; color: white"
                    if row["Model"] == best_model
                    else ""
                )
                for _ in row
            ]

        styled = comp_df.style.apply(_highlight_best, axis=1).format(
            {
                "Total Return %": "{:.2f}%",
                "Sharpe Ratio": "{:.3f}",
                "Win Rate": "{:.2%}",
                "Profit Factor": "{:.3f}",
                "Final Capital": "${:,.0f}",
            }
        )
        st.dataframe(styled, use_container_width=True)

        st.divider()

        model_selector = st.selectbox(
            "Select model to inspect",
            options=list(backtest.keys()),
            index=(
                list(backtest.keys()).index(best_model) if best_model in backtest else 0
            ),
        )
        selected_data = backtest[model_selector]
        selected_metrics = selected_data.get("metrics", {})
        best_metrics = backtest.get(best_model, {}).get("metrics", {})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Total Return",
            f"{selected_metrics.get('total_return_pct', 0):.2f}%",
            delta=(
                f"{selected_metrics.get('total_return_pct', 0) - best_metrics.get('total_return_pct', 0):.2f}%"
                if model_selector != best_model
                else None
            ),
        )
        c2.metric(
            "Sharpe Ratio",
            f"{selected_metrics.get('sharpe_ratio', 0):.3f}",
            delta=(
                f"{selected_metrics.get('sharpe_ratio', 0) - best_metrics.get('sharpe_ratio', 0):.3f}"
                if model_selector != best_model
                else None
            ),
        )
        c3.metric(
            "Win Rate",
            f"{selected_metrics.get('win_rate', 0):.2%}",
            delta=(
                f"{(selected_metrics.get('win_rate', 0) - best_metrics.get('win_rate', 0)):.2%}"
                if model_selector != best_model
                else None
            ),
        )
        c4.metric(
            "Profit Factor",
            f"{selected_metrics.get('profit_factor', 0):.3f}",
            delta=(
                f"{selected_metrics.get('profit_factor', 0) - best_metrics.get('profit_factor', 0):.3f}"
                if model_selector != best_model
                else None
            ),
        )

        daily_values = selected_data.get("daily_portfolio_values", [])
        trades_list = selected_data.get("trades", [])

        col_chart, col_hist = st.columns(2)

        with col_chart:
            st.subheader("Equity Curve")
            if daily_values:
                equity_df = pd.DataFrame(
                    {"Day": range(len(daily_values)), "Portfolio Value": daily_values}
                )
                fig_equity = px.line(
                    equity_df,
                    x="Day",
                    y="Portfolio Value",
                    color_discrete_sequence=["#1f77b4"],
                )
                fig_equity.update_layout(height=350, yaxis_tickformat="$,.0f")
                st.plotly_chart(
                    fig_equity,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.info("Equity curve data not available in backtest_results.json.")

        with col_hist:
            st.subheader("Trade Return Distribution")
            if trades_list:
                trades_df = pd.DataFrame(trades_list)
                if (
                    "return_pct" in trades_df.columns
                    and "direction" in trades_df.columns
                ):
                    fig_hist = px.histogram(
                        trades_df,
                        x="return_pct",
                        color="direction",
                        nbins=60,
                        color_discrete_map={
                            "buy": CLASS_COLORS["Buy"],
                            "sell": CLASS_COLORS["Sell"],
                        },
                        barmode="overlay",
                        opacity=0.75,
                    )
                    fig_hist.update_layout(
                        height=350, xaxis_title="Return %", yaxis_title="Trades"
                    )
                    st.plotly_chart(
                        fig_hist,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                else:
                    st.info("Trade-level data not available.")
            else:
                st.info("No trade records in backtest_results.json.")


# ── Tab 3: Manual Backtest Trigger ────────────────────────────────────────────

with tab3:
    st.subheader("Run Backtest")
    st.caption(
        "Trigger a fresh backtest against the test dataset. Results update backtest_results.json."
    )

    training = _load_training_results()
    model_options = list(training.keys()) if training else ["best_model"]

    with st.form("backtest_form"):
        model_choice = st.selectbox("Model", options=model_options)
        capital = st.number_input(
            "Initial Capital ($)",
            min_value=10_000.0,
            max_value=10_000_000.0,
            value=100_000.0,
            step=10_000.0,
        )
        hold_days = st.slider("Hold Days", min_value=5, max_value=20, value=10)
        submitted = st.form_submit_button("Run Backtest")

    if submitted:
        with st.spinner(f"Running backtest for {model_choice}…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/backtest/run",
                    json={
                        "model_name": model_choice,
                        "initial_capital": capital,
                        "hold_days": hold_days,
                    },
                    timeout=300,
                )
                resp.raise_for_status()
                result = resp.json()
                st.session_state["last_backtest_result"] = result
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
                result = None

        if result:
            st.success(f"Backtest complete in {result['run_duration_seconds']}s")
            m = result.get("metrics", {})
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Total Return", f"{m.get('total_return_pct', 0):.2f}%")
            r2.metric("Sharpe Ratio", f"{m.get('sharpe_ratio', 0):.3f}")
            r3.metric("Win Rate", f"{m.get('win_rate', 0):.2%}")
            r4.metric("Profit Factor", f"{m.get('profit_factor', 0):.3f}")
            st.caption(
                f"Trades executed: {result.get('trades_count', 0):,} | Run at {result.get('timestamp', '')}"
            )

    if "last_backtest_result" in st.session_state and not submitted:
        prev = st.session_state["last_backtest_result"]
        st.info(
            f"Last run: {prev.get('model_name')} — {prev['metrics'].get('total_return_pct', 0):.2f}% return"
        )
