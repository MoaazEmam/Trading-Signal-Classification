from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from src.backtesting.metrics import compute_metrics
from src.utils import load_best_model_info

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_LABEL_COL = "label"
_COMPANY_COL = "Company"
_DATE_COL = "Date"


def run_baseline_backtest(
    initial_capital: float = 500_000.0,
    test_features_path: str = "data/processed/test_selected.csv",
    prices_path: str = "data/raw/market_data_merged.csv",
    label_encoder_path: str = "models/artifacts/label_encoder.pkl",
    output_path: str = "models/artifacts/backtest_results.json",
) -> dict:
    """
    Naive baseline: predicts Buy for every row regardless of features.
    Used to check whether real models add value over 'always be long'.
    If real models don't beat this, the signal is not informative.
    """

    test_features_path = _PROJECT_ROOT / test_features_path
    prices_path = _PROJECT_ROOT / prices_path
    output_path = _PROJECT_ROOT / output_path

    logger.info("Running naive Buy-everything baseline backtest")

    test_features_df = pd.read_csv(test_features_path)

    if _DATE_COL not in test_features_df.columns:
        test_raw_path = test_features_path.parent / "test.csv"
        logger.info("Date column absent in baseline — loading from %s", test_raw_path)
        test_features_df[_DATE_COL] = pd.read_csv(test_raw_path, usecols=[_DATE_COL])[
            _DATE_COL
        ].values

    # Keep only the columns we need
    test_features_df = test_features_df[[_COMPANY_COL, _DATE_COL]]
    test_features_df[_DATE_COL] = pd.to_datetime(test_features_df[_DATE_COL])
    if test_features_df[_DATE_COL].dt.tz is not None:
        test_features_df[_DATE_COL] = test_features_df[_DATE_COL].dt.tz_localize(None)

    # Load and filter prices
    prices_raw = pd.read_csv(
        prices_path, usecols=[_COMPANY_COL, _DATE_COL, "Open", "Close"]
    )
    prices_raw[_DATE_COL] = pd.to_datetime(prices_raw[_DATE_COL])
    if prices_raw[_DATE_COL].dt.tz is not None:
        prices_raw[_DATE_COL] = prices_raw[_DATE_COL].dt.tz_localize(None)

    test_mi = pd.MultiIndex.from_arrays(
        [test_features_df[_COMPANY_COL], test_features_df[_DATE_COL]]
    )
    prices_raw = prices_raw.set_index([_COMPANY_COL, _DATE_COL])
    prices_df = prices_raw[prices_raw.index.isin(test_mi)].copy()
    if prices_df.index.duplicated().any():
        prices_df = prices_df[~prices_df.index.duplicated(keep="first")]

    prices_index_set = set(prices_df.index.tolist())

    # Predict Buy for everything
    predictions_df = test_features_df[[_COMPANY_COL, _DATE_COL]].copy()
    predictions_df["signal"] = "Buy"

    # Same simulation loop as run_backtest
    trading_calendar = sorted(predictions_df[_DATE_COL].unique())
    portfolio = initial_capital
    daily_portfolio_values: list[float] = []
    trades: list[dict] = []

    for i, day in enumerate(trading_calendar[:-1]):
        next_day = trading_calendar[i + 1]
        day_predictions = predictions_df[predictions_df[_DATE_COL] == day]
        active = day_predictions  # all rows are Buy

        if active.empty:
            daily_portfolio_values.append(portfolio)
            continue

        capital_per = portfolio / len(active)

        for _, row in active.iterrows():
            company = row[_COMPANY_COL]
            if (company, next_day) not in prices_index_set:
                continue
            entry = float(prices_df.loc[(company, next_day), "Open"])
            exit_ = float(prices_df.loc[(company, next_day), "Close"])
            ret = (exit_ - entry) / entry
            pnl = capital_per * ret
            portfolio += pnl
            trades.append(
                {
                    "company": company,
                    "entry_date": day.isoformat(),
                    "exit_date": next_day.isoformat(),
                    "entry_price": round(entry, 6),
                    "exit_price": round(exit_, 6),
                    "shares": round(capital_per / entry, 6),
                    "pnl": round(pnl, 6),
                    "return_pct": round(ret * 100, 6),
                    "direction": "buy",
                }
            )

        daily_portfolio_values.append(portfolio)

    daily_portfolio_values.append(portfolio)
    metrics = compute_metrics(daily_portfolio_values, trades)

    results = {
        "model_name": "naive_buy_baseline",
        "initial_capital": initial_capital,
        "final_capital": portfolio,
        "total_trades": len(trades),
        "metrics": metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
    existing["naive_buy_baseline"] = _serialize_results(results)
    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info(
        "Baseline complete — return=%.2f%% sharpe=%.3f win_rate=%s trades=%d",
        metrics.get("total_return_pct", float("nan")),
        metrics.get("sharpe_ratio", float("nan")) or 0.0,
        (
            f"{metrics['win_rate']:.2%}"
            if metrics["win_rate"] == metrics["win_rate"]
            else "nan"
        ),
        len(trades),
    )

    return results


def validate_alignment(test_df: pd.DataFrame, prices_df: pd.DataFrame) -> None:
    missing = test_df.index.difference(prices_df.index)
    total = len(test_df)
    matched = total - len(missing)
    logger.info("Alignment check: %d pairs checked, %d matched", total, matched)
    if len(missing) > 0:
        logger.warning(
            "%d (Company, Date) pairs in test features have no price data. First 10: %s",
            len(missing),
            missing[:10].tolist(),
        )


def _serialize_results(results: dict) -> dict:
    def _convert(obj):
        if isinstance(obj, float):
            if obj == float("inf"):
                return "inf"
            if obj == float("-inf"):
                return "-inf"
            if obj != obj:
                return "nan"
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return obj

    def _walk(node):
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(i) for i in node]
        return _convert(node)

    return _walk(results)


def run_backtest(
    model=None,
    model_name: str = None,
    initial_capital: float = 500_000.0,
    hold_days: int = 1,
    test_features_path: str = "data/processed/test_selected.csv",
    prices_path: str = "data/raw/market_data_merged.csv",
    label_encoder_path: str = "models/artifacts/label_encoder.pkl",
    output_path: str = "models/artifacts/backtest_results.json",
) -> dict:
    test_features_path = _PROJECT_ROOT / test_features_path
    prices_path = _PROJECT_ROOT / prices_path
    label_encoder_path = _PROJECT_ROOT / label_encoder_path
    output_path = _PROJECT_ROOT / output_path

    # --- Load test features ---
    logger.info("Loading test features from %s", test_features_path)
    test_features_df = pd.read_csv(test_features_path)

    if _DATE_COL not in test_features_df.columns:
        test_raw_path = test_features_path.parent / "test.csv"
        logger.info("Date column absent — loading from %s", test_raw_path)
        test_features_df[_DATE_COL] = pd.read_csv(test_raw_path, usecols=[_DATE_COL])[
            _DATE_COL
        ].values

    test_features_df[_DATE_COL] = pd.to_datetime(test_features_df[_DATE_COL])
    if test_features_df[_DATE_COL].dt.tz is not None:
        test_features_df[_DATE_COL] = test_features_df[_DATE_COL].dt.tz_localize(None)

    # --- Load and filter prices to the test period ---
    logger.info("Loading prices from %s", prices_path)
    prices_raw = pd.read_csv(
        prices_path, usecols=[_COMPANY_COL, _DATE_COL, "Open", "Close"]
    )
    prices_raw[_DATE_COL] = pd.to_datetime(prices_raw[_DATE_COL])
    if prices_raw[_DATE_COL].dt.tz is not None:
        prices_raw[_DATE_COL] = prices_raw[_DATE_COL].dt.tz_localize(None)

    test_mi = pd.MultiIndex.from_arrays(
        [test_features_df[_COMPANY_COL], test_features_df[_DATE_COL]]
    )
    prices_raw = prices_raw.set_index([_COMPANY_COL, _DATE_COL])
    prices_df = prices_raw[prices_raw.index.isin(test_mi)].copy()

    if prices_df.index.duplicated().any():
        prices_df = prices_df[~prices_df.index.duplicated(keep="first")]

    test_features_df = test_features_df.set_index([_COMPANY_COL, _DATE_COL])

    # --- Strip label and validate alignment ---
    x_df = test_features_df.drop(columns=[_LABEL_COL], errors="ignore")
    assert (
        _LABEL_COL not in x_df.columns
    ), "Label must not be present in features for inference"

    validate_alignment(x_df, prices_df)
    prices_index_set = set(prices_df.index.tolist())

    # --- Load model and label encoder ---
    if model is None:
        if model_name is not None:
            model_path = _PROJECT_ROOT / "models" / f"{model_name}.pkl"
        else:
            info = load_best_model_info()
            model_name = info["model_name"]
            model_path = _PROJECT_ROOT / info["model_path"]
        model = joblib.load(model_path)
        logger.info("Loaded model: %s", model_name)
    elif model_name is None:
        model_name = "best_model"

    label_encoder = joblib.load(label_encoder_path)

    # --- Run inference ---
    x = x_df.drop(
        columns=[c for c in [_COMPANY_COL, _DATE_COL, _LABEL_COL] if c in x_df.columns]
    )
    signals = label_encoder.inverse_transform(model.predict(x))

    predictions_df = x_df.index.to_frame(index=False)
    predictions_df["signal"] = signals
    print(
        f"PREDICTION DISTRIBUTION FOR MODEL {model_name}",
        predictions_df["signal"].value_counts(normalize=True),
    )
    # --- Simple simulation: each signal → 1-day trade on next day's open→close ---
    trading_calendar = sorted(predictions_df[_DATE_COL].unique())
    portfolio = initial_capital
    daily_portfolio_values: list[float] = []
    trades: list[dict] = []

    for i, day in enumerate(trading_calendar[:-1]):
        next_day = trading_calendar[i + 1]
        day_predictions = predictions_df[predictions_df[_DATE_COL] == day]
        active = day_predictions[day_predictions["signal"] != "Hold"]

        if active.empty:
            daily_portfolio_values.append(portfolio)
            continue

        capital_per = portfolio / len(active)

        for _, row in active.iterrows():
            company = row[_COMPANY_COL]
            signal = row["signal"]

            if (company, next_day) not in prices_index_set:
                continue

            entry = float(prices_df.loc[(company, next_day), "Open"])
            exit_ = float(prices_df.loc[(company, next_day), "Close"])

            if signal == "Buy":
                direction = 1
            elif signal == "Sell":
                direction = -1

            ret = direction * (exit_ - entry) / entry
            pnl = capital_per * ret
            portfolio += pnl

            trades.append(
                {
                    "company": company,
                    "entry_date": day.isoformat(),
                    "exit_date": next_day.isoformat(),
                    "entry_price": round(entry, 6),
                    "exit_price": round(exit_, 6),
                    "shares": round(capital_per / entry, 6),
                    "pnl": round(pnl, 6),
                    "return_pct": round(ret * 100, 6),
                    "direction": "buy" if signal == "Buy" else "sell",
                }
            )

        daily_portfolio_values.append(portfolio)

    daily_portfolio_values.append(portfolio)

    # --- Metrics and output ---
    metrics = compute_metrics(daily_portfolio_values, trades)

    results = {
        "model_name": model_name,
        "initial_capital": initial_capital,
        "final_capital": portfolio,
        "total_trades": len(trades),
        "metrics": metrics,
        "trades": trades,
        "daily_portfolio_values": daily_portfolio_values,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
    summary = {
        k: v
        for k, v in results.items()
        if k not in ("trades", "daily_portfolio_values")
    }
    existing[model_name] = _serialize_results(summary)
    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2)

    logger.info(
        "Backtest complete — model=%s return=%.2f%% sharpe=%.3f win_rate=%s trades=%d",
        model_name,
        metrics.get("total_return_pct", float("nan")),
        metrics.get("sharpe_ratio", float("nan")) or 0.0,
        (
            f"{metrics['win_rate']:.2%}"
            if metrics["win_rate"] == metrics["win_rate"]
            else "nan"
        ),
        len(trades),
    )

    return results
