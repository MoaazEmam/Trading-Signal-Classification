from pathlib import Path

import numpy as np
import pandas as pd
import pytest

COMPANIES = ["AAPL", "MSFT"]
BASE_DATE = pd.Timestamp("2020-01-02")
N_DAYS = 60
SAMPLE_PATH = Path("data/samples/market_data_sample.csv")


def _base_dates(n: int = N_DAYS) -> pd.DatetimeIndex:
    return pd.bdate_range(start=BASE_DATE, periods=n)


@pytest.fixture(scope="session")
def sample_df() -> pd.DataFrame:
    """
    Loads data/samples/market_data_sample.csv — produced by save_sample()
    after ingestion AND labeling have both been run.
    Skips automatically if the file is missing.
    """
    if not SAMPLE_PATH.exists():
        pytest.skip(
            f"Sample file not found at {SAMPLE_PATH}. " "Run save_sample() after ingestion + labeling to generate it."
        )
    return pd.read_csv(SAMPLE_PATH, parse_dates=["Date"])


@pytest.fixture()
def raw_kaggle_df() -> pd.DataFrame:
    dates = _base_dates()
    rows = []
    for company in COMPANIES:
        for date in dates:
            rows.append(
                {
                    "Date": date,
                    "Open": 100.0,
                    "High": 105.0,
                    "Low": 95.0,
                    "Close": 102.0,
                    "Volume": 1_000_000,
                    "Dividends": 0.0,
                    "Stock Splits": 0.0,
                    "Company": company,
                }
            )
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    return df


@pytest.fixture()
def raw_fred_df() -> pd.DataFrame:
    dates = _base_dates()
    return pd.DataFrame(
        {
            "Date": dates,
            "vix": np.linspace(15, 25, len(dates)),
            "fed_funds_rate": np.linspace(0.5, 1.5, len(dates)),
            "treasury_10y": np.linspace(1.5, 2.5, len(dates)),
            "sp500_level": np.linspace(3000, 3500, len(dates)),
        }
    )


@pytest.fixture()
def raw_fear_greed_df() -> pd.DataFrame:
    dates = _base_dates()
    return pd.DataFrame(
        {
            "Date": dates,
            "fear_greed_score": np.tile([30, 50, 70], len(dates))[: len(dates)],
            "fear_greed_label": np.tile(["Fear", "Neutral", "Greed"], len(dates))[: len(dates)],
        }
    )


@pytest.fixture()
def merged_df(raw_kaggle_df, raw_fred_df, raw_fear_greed_df) -> pd.DataFrame:
    df = raw_kaggle_df.merge(raw_fred_df, on="Date", how="left")
    df = df.merge(raw_fear_greed_df, on="Date", how="left")
    return df


@pytest.fixture()
def labeling_buy_df() -> pd.DataFrame:
    """
    20 warmup rows at 100, then immediate spike to 300.
    Volatility from warmup ≈ 0 → barriers extremely tight → 300 > upper → Buy.
    """
    return _make_labeling_df([100.0] * 20 + [300.0] * 20)


def _make_labeling_df(price_sequence: list, company: str = "TEST") -> pd.DataFrame:
    n = len(price_sequence)
    dates = _base_dates(n)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": price_sequence,
            "Stock Splits": [0.0] * n,
            "Dividends": [0.0] * n,
            # High/Low set equal to Close so barriers are computed only from volatility,
            # not from intraday range — keeps fixture behavior predictable
            "High": price_sequence,
            "Low": price_sequence,
            "Close": price_sequence,
            "Volume": [1_000_000] * n,
            "Company": [company] * n,
        }
    )


@pytest.fixture()
def labeling_sell_df() -> pd.DataFrame:
    """
    20 warmup rows at 100, then immediate crash to 1.
    1 is below ANY lower barrier, and since it appears before 300 never does,
    Sell fires on the first labeled row.
    Uses [1.0] not [20.0] — 20 is only an 80% drop but the labeler checks
    upper barrier first, so we need a value that is unambiguously below lower
    and cannot accidentally trigger upper (1.0 << lower barrier).
    """
    return _make_labeling_df([100.0] * 20 + [1.0] * 20)


@pytest.fixture()
def labeling_hold_df() -> pd.DataFrame:
    """
    Perfectly flat prices — zero volatility after warmup.
    With M=2 and volatility≈0, barriers collapse to exactly Close[t],
    but future prices are also exactly Close[t], so neither barrier is
    ever breached → Hold.
    We add a small amount of noise during warmup only so volatility isn't
    literally zero (which would cause division issues), but keep post-warmup
    prices perfectly flat.
    """
    warmup = [100.0 + np.random.uniform(-0.01, 0.01) for _ in range(20)]
    flat = [100.0] * 30  # perfectly flat — no barrier ever hit
    return _make_labeling_df(warmup + flat)


@pytest.fixture()
def labeling_multi_company_df() -> pd.DataFrame:
    df_buy = _make_labeling_df([100.0] * 20 + [300.0] * 20, company="BUY_CO")
    df_sell = _make_labeling_df([100.0] * 20 + [1.0] * 20, company="SELL_CO")
    return pd.concat([df_buy, df_sell], ignore_index=True)


@pytest.fixture()
def clean_validation_df() -> pd.DataFrame:
    dates = _base_dates(40)
    rows = []
    for company in COMPANIES:
        for date in dates:
            rows.append(
                {
                    "Date": date,
                    "Open": 100.0,
                    "High": 105.0,
                    "Low": 95.0,
                    "Close": 102.0,
                    "Volume": 500_000,
                    "Dividends": 0.0,
                    "Stock Splits": 0.0,
                    "Company": company,
                    "vix": 18.0,
                    "fed_funds_rate": 1.0,
                    "treasury_10y": 2.0,
                    "sp500_level": 3200.0,
                    "fear_greed_score": 50,
                    "fear_greed_label": "Neutral",
                    "label": "Hold",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def missing_values_df(clean_validation_df) -> pd.DataFrame:
    df = clean_validation_df.copy()
    df.loc[df.index[: int(len(df) * 0.25)], "treasury_10y"] = np.nan
    return df


@pytest.fixture()
def duplicate_rows_df(clean_validation_df) -> pd.DataFrame:
    return pd.concat([clean_validation_df, clean_validation_df.iloc[:5]], ignore_index=True)


@pytest.fixture()
def sanity_fail_df(clean_validation_df) -> pd.DataFrame:
    df = clean_validation_df.copy()
    df.loc[df.index[:3], "Close"] = 999.0
    return df


@pytest.fixture()
def stale_data_df(clean_validation_df) -> pd.DataFrame:
    df = clean_validation_df.copy()
    df.loc[df.index[:5], ["Open", "High", "Low", "Close"]] = 100.0
    df.loc[df.index[:5], "Volume"] = 0
    return df


@pytest.fixture()
def price_spike_df(clean_validation_df) -> pd.DataFrame:
    """
    Injects a >50% price spike with no stock split.
    Uses the 10th AAPL row (not 5th) to ensure there is a prior row
    for pct_change() to compare against after sort_values().
    """
    df = clean_validation_df.copy().sort_values(["Company", "Date"]).reset_index(drop=True)
    idx = df[df["Company"] == "AAPL"].index[10]
    df.loc[idx, "Close"] = 999.0  # >50% jump from prior row's 102.0
    df.loc[idx, "High"] = 999.0
    return df


@pytest.fixture()
def imbalanced_labels_df(clean_validation_df) -> pd.DataFrame:
    df = clean_validation_df.copy()
    df["label"] = "Buy"
    df.loc[df.index[-2:], "label"] = "Sell"
    return df


@pytest.fixture()
def wrong_dtypes_df(clean_validation_df) -> pd.DataFrame:
    df = clean_validation_df.copy()
    df["Close"] = df["Close"].astype(str)
    return df


# ---------------------------------------------------------------------------
# Splitting fixtures
# ---------------------------------------------------------------------------

N_SPLIT_DAYS = 100  # enough for lookahead buffer (LOOKAHEAD_DAYS=10) to kick in


@pytest.fixture()
def splitting_labeled_df() -> pd.DataFrame:
    """
    100 business-day range, two companies, all rows labeled.
    Suitable for temporal_split / save_splits tests.
    """
    dates = pd.bdate_range(start=BASE_DATE, periods=N_SPLIT_DAYS)
    rows = []
    for company in COMPANIES:
        for date in dates:
            rows.append(
                {
                    "Date": date,
                    "Close": 100.0,
                    "Company": company,
                    "label": "Hold",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def splitting_unlabeled_df(splitting_labeled_df) -> pd.DataFrame:
    """Same shape as splitting_labeled_df but every label is NaN."""
    df = splitting_labeled_df.copy()
    df["label"] = np.nan
    return df


@pytest.fixture()
def splitting_few_dates_df() -> pd.DataFrame:
    """
    Only 12 business days — after an 80/20 split, train_val has ~9 unique dates,
    which is <= LOOKAHEAD_DAYS (10).  Used to test that the lookahead buffer is
    skipped gracefully when there are not enough training dates.
    """
    dates = pd.bdate_range(start=BASE_DATE, periods=12)
    rows = []
    for date in dates:
        rows.append({"Date": date, "Close": 100.0, "Company": "AAPL", "label": "Hold"})
    return pd.DataFrame(rows)
