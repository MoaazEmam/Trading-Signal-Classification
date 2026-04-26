import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_cleaned_labeled

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ENGINEERED_PATH = Path("data/processed/market_data_with_features.csv")


def _calculate_price_momentum(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    for n in [1, 2, 5, 10, 20]:
        features[f"return_{n}d"] = df["Close"].pct_change(n)
    features["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    features["gap_open"] = df["Open"] / df["Close"].shift(1) - 1
    features["intraday_return"] = df["Close"] / df["Open"] - 1
    return features


def _calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    for n in [5, 10, 20, 50, 200]:
        features[f"sma_{n}"] = df["Close"].rolling(n).mean()

    for n in [9, 21, 50]:
        features[f"ema_{n}"] = df["Close"].ewm(span=n, adjust=False).mean()
    features["price_to_sma20"] = df["Close"] / (features["sma_20"] + 1e-9)
    features["sma_cross_20_50"] = (
        (features["sma_20"] > features["sma_50"]).astype(int).replace(0, -1)
    )
    features["sma_cross_50_200"] = (
        (features["sma_50"] > features["sma_200"]).astype(int).replace(0, -1)
    )
    return features


def _calculate_rsi(close: pd.Series, period: int):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_rsi_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    for n in [7, 14]:
        features[f"rsi_{n}"] = _calculate_rsi(df["Close"], n)
    features["rsi_divergence"] = features["rsi_14"] - features["rsi_14"].shift(5)
    return features


def _calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    features["macd_line"] = macd_line
    features["macd_signal"] = macd_signal
    features["macd_histogram"] = macd_hist

    features["macd_cross"] = (macd_line > macd_signal).astype(int).replace(0, -1)
    return features


def _calculate_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    sma_20 = df["Close"].rolling(20).mean()
    std_20 = df["Close"].rolling(20).std()

    bb_mid = sma_20
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20

    features["bb_mid"] = bb_mid
    features["bb_upper"] = bb_upper
    features["bb_lower"] = bb_lower

    features["bb_width"] = (bb_upper - bb_lower) / bb_mid

    features["bb_position"] = (df["Close"] - bb_lower) / (
        bb_upper - bb_lower + 1e-9
    )  # avoid divide by zero

    return features


def _calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    volume_sma_20 = df["Volume"].rolling(20).mean()
    features["volume_sma_20"] = volume_sma_20

    features["volume_ratio"] = df["Volume"] / (volume_sma_20 + 1e-9)

    direction = np.sign(df["Close"].diff()).fillna(0)

    features["obv"] = (direction * df["Volume"]).cumsum()

    vwap = (df["Close"] * df["Volume"]).rolling(20).sum() / (
        df["Volume"].rolling(20).sum() + 1e-9
    )

    features["vwap"] = vwap
    features["price_to_vwap"] = df["Close"] / (vwap + 1e-9)

    return features


def _calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    prev_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_14 = tr.rolling(14).mean()

    features["atr_14"] = atr_14
    features["atr_ratio"] = atr_14 / (df["Close"] + 1e-9)

    returns = df["Close"].pct_change()

    realized_vol_10 = returns.rolling(10).std()
    realized_vol_20 = returns.rolling(20).std()

    features["realized_vol_10"] = realized_vol_10
    features["realized_vol_20"] = realized_vol_20

    features["high_low_range"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)

    features["vol_regime"] = realized_vol_10 / (realized_vol_20 + 1e-9)

    return features


def _calculate_vix_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    vix = df["vix"]

    features["vix_change"] = vix.pct_change()

    vix_sma_20 = vix.rolling(20).mean()
    features["vix_sma_20"] = vix_sma_20

    features["vix_ratio"] = vix / (vix_sma_20 + 1e-9)

    features["vix_percentile"] = vix.rolling(252).rank(pct=True)

    features["vix_regime"] = vix.apply(
        lambda x: 0 if x < 15 else (1 if x < 25 else (2 if x < 35 else 3))
    )

    if "realized_vol_20" in df.columns:
        features["vix_vs_realized"] = vix / (df["realized_vol_20"] + 1e-9)

    return features


def _calculate_sp500_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    spx = df["sp500_level"]
    stock = df["Close"]

    spx_ret_1d = spx.pct_change()

    features["sp500_return_1d"] = spx_ret_1d
    features["sp500_return_5d"] = spx.pct_change(5)
    features["sp500_return_20d"] = spx.pct_change(20)

    stock_ret_5d = stock.pct_change(5)

    features["relative_return_5d"] = stock_ret_5d - features["sp500_return_5d"]

    stock_ret_1d = stock.pct_change()
    market_ret_1d = spx_ret_1d

    cov = stock_ret_1d.rolling(20).cov(market_ret_1d)
    var = market_ret_1d.rolling(20).var()

    features["beta_rolling_20"] = cov / (var + 1e-9)

    features["stock_to_sp500"] = stock / (spx + 1e-9)

    return features


def _calculate_interest_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    fed = df["fed_funds_rate"]
    ten_y = df["treasury_10y"]

    features["yield_spread"] = ten_y - fed

    features["rate_change_fed"] = fed.diff(5)
    features["rate_change_10y"] = ten_y.diff(5)

    features["yield_curve_regime"] = (
        (features["yield_spread"] > 0).astype(int).replace(0, -1)
    )

    if "realized_vol_20" in df.columns:
        features["real_rate_proxy"] = ten_y - df["realized_vol_20"]

    return features


def _calculate_fear_greed_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    fg = df["fear_greed_score"]

    features["fg_change_5d"] = fg.diff(5)

    features["fg_sma_10"] = fg.rolling(10).mean()

    features["fg_extreme_fear"] = (fg < 25).astype(int)
    features["fg_extreme_greed"] = (fg > 75).astype(int)

    features["fg_momentum"] = fg - fg.shift(10)

    if "vix_percentile" in df.columns:
        vix_proxy = 100 - (df["vix_percentile"] * 100)
        features["fg_vix_divergence"] = fg - vix_proxy

    return features


def _calculate_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    # open is builtin so it had to be something else
    o = df["Open"]
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    features["body_size"] = (close - o).abs() / (close + 1e-9)

    features["upper_wick"] = (high - np.maximum(o, close)) / (close + 1e-9)
    features["lower_wick"] = (np.minimum(o, close) - low) / (close + 1e-9)

    features["is_doji"] = (features["body_size"] < 0.001).astype(int)

    features["is_bullish_candle"] = (close > o).astype(int)

    direction = np.sign(close.diff()).fillna(0)

    streak = np.zeros(len(df))
    current = 0

    for i in range(len(df)):
        if direction.iloc[i] > 0:
            current = current + 1 if current > 0 else 1
        elif direction.iloc[i] < 0:
            current = current - 1 if current < 0 else -1
        else:
            current = 0
        streak[i] = current

    features["candle_direction_streak"] = streak

    return features


def _calculate_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    date = df["Date"]

    features["day_of_week"] = date.dt.day_of_week

    features["month"] = date.dt.month

    features["is_month_end"] = date.dt.is_month_end.astype(int)
    features["is_quarter_end"] = date.dt.is_quarter_end.astype(int)

    features["week_of_year"] = date.dt.isocalendar().week.astype(int)

    return features


def _calculate_cross_feature_interactions(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)

    if "rsi_14" in df.columns and "bb_position" in df.columns:
        features["rsi_bb_position"] = df["rsi_14"] * df["bb_position"]

    if "volume_ratio" in df.columns:
        daily_return = df["Close"].pct_change()
        features["volume_price_trend"] = df["volume_ratio"] * daily_return

    if "vix_percentile" in df.columns and "fear_greed_score" in df.columns:
        fg_scaled = df["fear_greed_score"] / 100.0
        features["macro_risk_score"] = df["vix_percentile"] * (1 - fg_scaled)

    if "price_to_sma20" in df.columns and "rsi_14" in df.columns:
        features["trend_strength"] = (df["price_to_sma20"] - 1) * df["rsi_14"]

    if "fed_funds_rate" in df.columns:
        daily_return = df["Close"].pct_change()
        features["carry_spread"] = daily_return - (df["fed_funds_rate"] / 252.0)

    return features


def _calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    t = time.time()
    price_momentum = _calculate_price_momentum(df)
    logger.info(f"  price momentum         : {time.time() - t:.2f}s")

    t = time.time()
    moving_averages = _calculate_moving_averages(df)
    logger.info(f"  moving averages        : {time.time() - t:.2f}s")

    t = time.time()
    rsi_features = _calculate_rsi_features(df)
    logger.info(f"  RSI                    : {time.time() - t:.2f}s")

    t = time.time()
    macd_features = _calculate_macd(df)
    logger.info(f"  MACD                   : {time.time() - t:.2f}s")

    t = time.time()
    bb_features = _calculate_bollinger_bands(df)
    logger.info(f"  Bollinger bands        : {time.time() - t:.2f}s")

    t = time.time()
    volume_features = _calculate_volume_features(df)
    logger.info(f"  volume                 : {time.time() - t:.2f}s")

    t = time.time()
    volatility_features = _calculate_volatility_features(df)
    logger.info(f"  volatility             : {time.time() - t:.2f}s")

    df_aug = df.copy()
    df_aug["realized_vol_20"] = volatility_features["realized_vol_20"]
    df_aug["vix_percentile"] = None

    t = time.time()
    vix_features = _calculate_vix_features(df_aug)
    logger.info(f"  VIX                    : {time.time() - t:.2f}s")

    df_aug["vix_percentile"] = vix_features["vix_percentile"]

    t = time.time()
    sp500_features = _calculate_sp500_relative_features(df)
    logger.info(f"  S&P500 relative        : {time.time() - t:.2f}s")

    t = time.time()
    rate_features = _calculate_interest_rate_features(df_aug)
    logger.info(f"  interest rates         : {time.time() - t:.2f}s")

    t = time.time()
    fg_features = _calculate_fear_greed_features(df_aug)
    logger.info(f"  fear & greed           : {time.time() - t:.2f}s")

    t = time.time()
    candle_features = _calculate_candlestick_features(df)
    logger.info(f"  candlestick            : {time.time() - t:.2f}s")

    t = time.time()
    calendar_features = _calculate_calendar_features(df)
    logger.info(f"  calendar               : {time.time() - t:.2f}s")

    cross_input = df.copy()
    cross_input["rsi_14"] = rsi_features["rsi_14"]
    cross_input["bb_position"] = bb_features["bb_position"]
    cross_input["volume_ratio"] = volume_features["volume_ratio"]
    cross_input["vix_percentile"] = vix_features["vix_percentile"]
    cross_input["price_to_sma20"] = moving_averages["price_to_sma20"]

    t = time.time()
    cross_features = _calculate_cross_feature_interactions(cross_input)
    logger.info(f"  cross-feature interactions: {time.time() - t:.2f}s")

    return pd.concat(
        [
            price_momentum,
            moving_averages,
            rsi_features,
            macd_features,
            bb_features,
            volume_features,
            volatility_features,
            vix_features,
            sp500_features,
            rate_features,
            fg_features,
            candle_features,
            calendar_features,
            cross_features,
        ],
        axis=1,
    )


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the full multi-company DataFrame (sorted by Company, Date).
    Returns the original df columns + all engineered features
    """
    start = time.time()
    df = df.sort_values(["Company", "Date"]).reset_index(drop=True)

    feature_blocks = df.groupby("Company", group_keys=False).apply(
        _calculate_features, include_groups=False
    )

    feature_blocks = feature_blocks.reset_index(drop=True)

    assert len(feature_blocks) == len(df), (
        f"Row count mismatch after feature engineering: "
        f"{len(feature_blocks)} features vs {len(df)} original rows"
    )

    result = pd.concat([df, feature_blocks], axis=1)
    logger.info(
        f"Feature engineering complete: {len(result.columns)} total columns — {time.time() - start:.2f}s"
    )
    return result


def run_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("run_engineering: building feature matrix...")
    result = build_feature_matrix(df)
    before = len(result)
    result = result.dropna()
    logger.info(
        "run_engineering: dropped %d NaN rows (rolling-window warmup)",
        before - len(result),
    )

    project_root = Path(__file__).resolve().parent.parent.parent
    out_path = project_root / ENGINEERED_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    logger.info("run_engineering: saved to %s", out_path)
    return result


if __name__ == "__main__":
    logger.info("Loading dataset...")
    t = time.time()
    df = load_cleaned_labeled()
    logger.info(
        f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns — {time.time() - t:.2f}s"
    )
    run_engineering(df)
