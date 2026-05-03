import logging
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from fredapi import Fred

from src.config import settings
from src.utils import RAW_DATA_PATH

logger = logging.getLogger(__name__)

FRED_TICKER_MAP = {
    "VIXCLS": "vix",
    "DFF": "fed_funds_rate",
    "DGS10": "treasury_10y",
    "SP500": "sp500_level",
}

fred = Fred(api_key=settings.fred_api_key)

RAW_SCHEMA_COLS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Dividends",
    "Stock Splits",
    "Company",
    "vix",
    "fed_funds_rate",
    "treasury_10y",
    "sp500_level",
    "fear_greed_score",
    "fear_greed_label",
]


def _get_active_companies(path: Path = RAW_DATA_PATH) -> list[str]:
    project_root = Path(__file__).resolve().parent.parent.parent
    absolute_path = project_root / path
    df = pd.read_csv(absolute_path, usecols=["Company"])
    return df["Company"].unique().tolist()


def _fetch_daily_yfinance(ticker: str, date: str) -> pd.DataFrame:
    try:
        # yfinance end is exclusive so we need the next calendar day
        end = (pd.Timestamp(date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=date, end=end, auto_adjust=True, progress=False)
        if df is None or df.empty:
            logger.warning(f"{ticker}: no data for {date}, skipping")
            return pd.DataFrame()
        df.columns = [col[0] for col in df.columns]  # flatten MultiIndex
        for col in ["Dividends", "Stock Splits"]:
            if col not in df.columns:
                df[col] = 0.0
        df["Company"] = ticker
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
        return df[
            [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Dividends",
                "Stock Splits",
                "Company",
            ]
        ]
    except Exception as e:
        logger.error(f"Skipped {ticker} due to error: {e}", exc_info=True)
        return pd.DataFrame()


def _fetch_all_yfinance(companies: list[str], date: str) -> pd.DataFrame:
    dfs = [_fetch_daily_yfinance(ticker, date) for ticker in companies]
    valid = [df for df in dfs if not df.empty]
    if not valid:
        raise RuntimeError(
            f"No yfinance data returned for any company on {date}. "
            "This is likely a weekend or market holiday."
        )
    result = pd.concat(valid, ignore_index=True)
    logger.info(f"Done — got data for {result['Company'].nunique()} companies")
    return result


def _fetch_fred_series_single(
    series_id: str, date: str, name: str
) -> pd.DataFrame | None:
    try:
        s = fred.get_series(series_id, observation_start=date, observation_end=date)
        if s.empty:
            # fetch the last known value if yesterday isnt available
            s = fred.get_series(series_id, observation_end=date)
            s = s.dropna().tail(1)
        if s.empty:
            logger.warning(
                f"FRED {series_id}: no data available up to {date}, skipping"
            )
            return None
        df_fred = s.reset_index()
        df_fred.columns = ["Date", name]
        df_fred["Date"] = (
            pd.to_datetime(df_fred["Date"]).dt.tz_localize(None).dt.normalize()
        )
        df_fred["Date"] = pd.Timestamp(date)
        return df_fred[["Date", name]]
    except Exception as e:
        logger.error(f"Skipping FRED {series_id} for {date}: {e}", exc_info=True)
        return None


def _fetch_fred_macros_daily(date: str) -> pd.DataFrame:
    results = {
        name: _fetch_fred_series_single(series_id, date, name)
        for series_id, name in FRED_TICKER_MAP.items()
    }

    if all(df is None for df in results.values()):
        raise RuntimeError(f"All FRED series failed to fetch for {date}.")

    row: dict = {"Date": pd.Timestamp(date)}
    for name, df in results.items():
        if df is not None and not df.empty:
            row[name] = df[name].iloc[0]
        else:
            # if series failed find the nearest date that wont
            series_id = next(k for k, v in FRED_TICKER_MAP.items() if v == name)
            try:
                s = fred.get_series(series_id, observation_end=date).dropna()
                row[name] = float(s.iloc[-1]) if not s.empty else float("nan")
                logger.warning(
                    f"FRED {name}: used last known value ({row[name]:.4f}) for {date}"
                )
            except Exception as e:
                logger.error(
                    f"FRED {name}: fallback fetch failed ({e}), defaulting to 0.0",
                    exc_info=True,
                )
                row[name] = 0.0

    merged = pd.DataFrame([row])

    for name in FRED_TICKER_MAP.values():
        if merged[name].isna().any():
            logger.warning(
                f"FRED {name}: still NaN after all fallbacks, defaulting to 0.0"
            )
            merged[name] = 0.0

    return merged


def _fetch_fear_greed_daily(date: str) -> pd.DataFrame:
    """
    the api is queried with limit=10 to get recent values; we then filter to the
    closest available date (same-day or most recent prior day).
    """
    url = "https://api.alternative.me/fng/?limit=10&format=json"
    data = requests.get(url).json()["data"]
    df_fg = pd.DataFrame(data)
    df_fg["Date"] = (
        pd.to_datetime(df_fg["timestamp"].astype(int), unit="s")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    df_fg["fear_greed_score"] = df_fg["value"].astype(int)
    df_fg["fear_greed_label"] = df_fg["value_classification"]
    df_fg = df_fg[["Date", "fear_greed_score", "fear_greed_label"]]

    target = pd.Timestamp(date)
    # use exact match if available, otherwise most recent prior entry
    match = df_fg[df_fg["Date"] <= target].sort_values("Date").tail(1).copy()
    if match.empty:
        raise RuntimeError(f"No Fear & Greed data available for or before {date}.")
    match["Date"] = target  # align date to target
    return match.reset_index(drop=True)


def run_daily_fetch(date: str) -> pd.DataFrame:
    """Fetch all data needed for prediction on a single date.

    Returns a DataFrame with the exact schema of market_data_merged.csv
    one row per active company.
    """
    companies = _get_active_companies()
    ohlcv_df = _fetch_all_yfinance(companies, date)  # raises if no data at all
    fred_df = _fetch_fred_macros_daily(date)
    fg_df = _fetch_fear_greed_daily(date)

    merged = ohlcv_df.merge(fred_df, on="Date", how="left")
    merged = merged.merge(fg_df, on="Date", how="left")

    # ensure column order matches raw schema exactly (no label column)
    for col in RAW_SCHEMA_COLS:
        if col not in merged.columns:
            merged[col] = None
    merged = merged[RAW_SCHEMA_COLS]

    logger.info(f"Daily fetch complete — {merged.shape[0]} rows for {date}")
    return merged
