import logging
import os
from datetime import date
from functools import reduce
from pathlib import Path

import kagglehub
import pandas as pd
import requests
import yfinance as yf
from fredapi import Fred

from src.config import settings
from src.utils import _save_to_csv

logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw/market_data_merged.csv")
os.environ["KAGGLE_USERNAME"] = settings.kaggle_username
os.environ["KAGGLE_API_KEY"] = settings.kaggle_api_token
FRED_TICKER_MAP = {
    "VIXCLS": "vix",
    "DFF": "fed_funds_rate",
    "DGS10": "treasury_10y",
    "SP500": "sp500_level",
}
fred = Fred(api_key=settings.fred_api_key)


SAMPLE_PATH = Path("data/samples/market_data_sample.csv")
SAMPLE_N_ROWS = 30  # rows per company kept in the sample


def save_sample(
    raw_path: Path = RAW_DATA_PATH,
    sample_path: Path = SAMPLE_PATH,
    n_rows: int = SAMPLE_N_ROWS,
) -> Path:
    df = pd.read_csv(raw_path, parse_dates=["Date"])

    sample = (
        df.sort_values(["Company", "Date"])
        .groupby("Company", group_keys=False)
        .apply(lambda g: g.head(n_rows))
    )

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False)

    logger.info(
        f"Sample saved → {sample_path}  ({sample.shape[0]:,} rows × {sample.shape[1]} cols)"
    )
    return sample_path


def _fetch_yfinance_ticker(ticker: str, last_date: pd.Timestamp) -> pd.DataFrame:
    today = date.today().strftime("%Y-%m-%d")
    try:
        start = last_date + pd.Timedelta(days=1)
        ticker_df = yf.download(
            ticker, start=start, end=today, auto_adjust=True, progress=False
        )
        if ticker_df is None or ticker_df.empty:
            logger.warning(f"{ticker}: no new data since {start.date()}, skipping")
            return pd.DataFrame()
        else:
            ticker_df.columns = [col[0] for col in ticker_df.columns]  # flatten df
            for col in ["Dividends", "Stock Splits"]:
                if col not in ticker_df.columns:
                    ticker_df[col] = 0.0
            ticker_df["Company"] = ticker
            ticker_df = ticker_df.reset_index()
        return ticker_df
    except Exception as e:
        logger.error(f"Skipped {ticker} due to errors: {e}", exc_info=True)
        return pd.DataFrame()


def _fetch_yfinance_data(tickers: pd.Series, last_date: pd.Timestamp) -> pd.DataFrame:
    dfs = tickers.map(lambda ticker: _fetch_yfinance_ticker(ticker, last_date))
    valid_dfs = [df for df in dfs.tolist() if not df.empty]
    if not valid_dfs:
        logger.warning("No data fetched for any ticker")
        return pd.DataFrame()
    return pd.concat(valid_dfs, ignore_index=True)


def _fetch_kaggle_dataset() -> pd.DataFrame:
    path = kagglehub.dataset_download("iveeaten3223times/massive-yahoo-finance-dataset")
    csv_path = Path(path) / "stock_details_5_years.csv"
    df = pd.read_csv(csv_path, compression="infer")
    df["Date"] = (
        pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    )
    return df


def _fetch_fred_series(
    series_id: str, start: str, end: str, name: str
) -> pd.DataFrame | None:
    try:
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        df_fred = s.reset_index()
        df_fred.columns = ["Date", name]
        df_fred["Date"] = (
            pd.to_datetime(df_fred["Date"]).dt.tz_localize(None).dt.normalize()
        )
        return df_fred
    except Exception as e:
        logger.error(f"skipping fred for date {start}: {e}", exc_info=True)
        return None


def _fetch_fred_macros(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dfs = [
        _fetch_fred_series(
            series_id, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), name
        )
        for series_id, name in FRED_TICKER_MAP.items()
    ]
    valid_dfs = [df for df in dfs if df is not None]

    if not valid_dfs:
        raise RuntimeError("All FRED series failed to fetch.")
    return reduce(
        lambda left, right: pd.merge(left, right, on="Date", how="outer"), dfs
    )


def _fetch_fear_greed(limit: int = 3000) -> pd.DataFrame:
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    data = requests.get(url).json()["data"]
    df_fg = pd.DataFrame(data)
    df_fg["Date"] = (
        pd.to_datetime(df_fg["timestamp"].astype(int), unit="s")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    df_fg["fear_greed_score"] = df_fg["value"].astype(int)
    df_fg["fear_greed_label"] = df_fg["value_classification"]
    return df_fg[["Date", "fear_greed_score", "fear_greed_label"]]  # type: ignore


def _normalize_date(df: pd.DataFrame, date_col: str = "Date") -> pd.Series:
    return df[date_col].dt.tz_localize(None).dt.normalize()


def _merge_on_date(
    df1: pd.DataFrame, df2: pd.DataFrame, date_col: str = "Date"
) -> pd.DataFrame:
    return df1.merge(df2, on=date_col, how="left")


def _get_date_limits(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return df["Date"].min(), df["Date"].max()  # type: ignore


def run_ingestion():
    kaggle_df = _fetch_kaggle_dataset()
    # get date limits incase dataset dynamically incase dataset is updated
    start, end = _get_date_limits(kaggle_df)
    tickers = pd.Series(kaggle_df["Company"].unique())
    # enriching dataset with new data
    yfinance_df = _fetch_yfinance_data(tickers, end)
    kaggle_df = pd.concat([kaggle_df, yfinance_df], ignore_index=True).sort_values(
        "Date"
    )
    # get dates again
    start, end = _get_date_limits(kaggle_df)

    fred_df = _fetch_fred_macros(start, end)

    fg_df = _fetch_fear_greed()
    fg_df = fg_df[(fg_df["Date"] >= start) & (fg_df["Date"] <= end)]  # cap fg_df

    merged_df = _merge_on_date(kaggle_df, fred_df)
    merged_df = _merge_on_date(merged_df, fg_df)  # type: ignore
    _save_to_csv(merged_df, RAW_DATA_PATH)
    logger.info("Ingestion complete.")
    return merged_df


if __name__ == "__main__":
    run_ingestion()
    save_sample()
