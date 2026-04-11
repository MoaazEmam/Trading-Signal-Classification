import os
from functools import reduce
from pathlib import Path

import kagglehub
import pandas as pd
import requests
from fredapi import Fred

from src.config import settings

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
    print(f"Reading full dataset from {raw_path} …")
    df = pd.read_csv(raw_path, parse_dates=["Date"])

    sample = df.sort_values(["Company", "Date"]).groupby("Company", group_keys=False).apply(lambda g: g.head(n_rows))

    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False)

    print(f"Sample saved → {sample_path}  ({sample.shape[0]:,} rows × {sample.shape[1]} cols)")
    return sample_path


def _fetch_kaggle_dataset() -> pd.DataFrame:
    print("Fetching kaggle dataset.....")
    path = kagglehub.dataset_download("iveeaten3223times/massive-yahoo-finance-dataset")
    csv_path = Path(path) / "stock_details_5_years.csv"
    df = pd.read_csv(csv_path, compression="infer")
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    print("Done")
    return df


def _fetch_fred_series(series_id: str, start: str, end: str, name: str) -> pd.DataFrame:
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    df_fred = s.reset_index()
    df_fred.columns = ["Date", name]
    df_fred["Date"] = pd.to_datetime(df_fred["Date"]).dt.tz_localize(None).dt.normalize()
    return df_fred


def _fetch_fred_macros(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    print("Fetching fred macros.......")
    dfs = [
        _fetch_fred_series(series_id, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), name)
        for series_id, name in FRED_TICKER_MAP.items()
    ]
    print("Done")
    return reduce(lambda left, right: pd.merge(left, right, on="Date", how="outer"), dfs)


def _fetch_fear_greed(limit: int = 3000) -> pd.DataFrame:
    print("Fetching fear&greed macros........")
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    data = requests.get(url).json()["data"]
    df_fg = pd.DataFrame(data)
    df_fg["Date"] = pd.to_datetime(df_fg["timestamp"].astype(int), unit="s").dt.tz_localize(None).dt.normalize()
    df_fg["fear_greed_score"] = df_fg["value"].astype(int)
    df_fg["fear_greed_label"] = df_fg["value_classification"]
    print("Done")
    return df_fg[["Date", "fear_greed_score", "fear_greed_label"]]  # type: ignore


def _normalize_date(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    return df[date_col].dt.tz_localize(None).dt.normalize()


def _merge_on_date(df1: pd.DataFrame, df2: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    return df1.merge(df2, on=date_col, how="left")


def _get_date_limits(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return df["Date"].min(), df["Date"].max()  # type: ignore


def _save_to_csv(df: pd.DataFrame) -> None:
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Successfully saved merged dataset to: {RAW_DATA_PATH}")
    print(f"📊 Final Dataset Shape: {df.shape}")


def run_ingestion():
    kaggle_df = _fetch_kaggle_dataset()
    # get date limits incase dataset dynamically incase dataset is updated
    start, end = _get_date_limits(kaggle_df)

    fred_df = _fetch_fred_macros(start, end)

    fg_df = _fetch_fear_greed()
    fg_df = fg_df[(fg_df["Date"] >= start) & (fg_df["Date"] <= end)]  # cap fg_df

    # merge all 3
    print("Merging all three......")
    merged_df = _merge_on_date(kaggle_df, fred_df)
    merged_df = _merge_on_date(merged_df, fg_df)  # type: ignore
    _save_to_csv(merged_df)
    print("Ingestion Complete.")


if __name__ == "__main__":
    run_ingestion()
    save_sample()
