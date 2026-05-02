"""
Feature pipeline composition.

build_pipeline() returns a fresh, unfitted sklearn Pipeline wiring up every
stateful transformation applied between the cleaned splits and model input:

    1. GroupedWinsorizer   -- per-Company cap on Volume outliers
    1. GlobalWinsorizer    -- global cap
    2. ColumnDropper       -- drops redundant / non-feature columns
    3. ColumnTransformer   -- scale numeric, passthrough flags and group key

The same builder is called by the feature runner to fit on train_val, and
later at inference time to reconstruct the object before loading fitted state
from the joblib artifact. Keeping construction in one place guarantees
training and inference see identical pipeline topology.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, StandardScaler

from src.features.transformers import ColumnDropper, GlobalWinsorizer, GroupedWinsorizer

GROUP_COL = "Company"

GROUPED_WINSORIZE_COLS: list[str] = ["Volume"]

GLOBAL_WINSORIZE_COLS: list[str] = [
    "obv",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "body_size",
    "upper_wick",
    "lower_wick",
    "volume_price_trend",
    "return_1d",
    "return_2d",
    "return_5d",
    "return_10d",
    "return_20d",
    "log_return",
    "gap_open",
    "intraday_return",
    "sp500_return_1d",
    "sp500_return_5d",
    "sp500_return_20d",
    "relative_return_5d",
    "vix_change",
    "fg_change_5d",
    "fg_momentum",
    "beta_rolling_20",
]

ROBUST_COLS: list[str] = [
    "Volume",
    "volume_sma_20",
    "volume_ratio",
    "obv",
    "atr_14",
    "atr_ratio",
    "high_low_range",
    "realized_vol_10",
    "realized_vol_20",
    "bb_width",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "body_size",
    "upper_wick",
    "lower_wick",
    "volume_price_trend",
    "beta_rolling_20",
    "vix_change",
    "fg_change_5d",
    "fg_momentum",
]

STANDARD_COLS: list[str] = [
    "return_1d",
    "return_2d",
    "return_5d",
    "return_10d",
    "return_20d",
    "log_return",
    "gap_open",
    "intraday_return",
    "price_to_sma20",
    "rsi_7",
    "rsi_14",
    "rsi_divergence",
    "bb_position",
    "price_to_vwap",
    "vix",
    "vix_sma_20",
    "vix_ratio",
    "vix_percentile",
    "vix_vs_realized",
    "sp500_return_1d",
    "sp500_return_5d",
    "sp500_return_20d",
    "relative_return_5d",
    "stock_to_sp500",
    "fed_funds_rate",
    "treasury_10y",
    "yield_spread",
    "rate_change_fed",
    "rate_change_10y",
    "real_rate_proxy",
    "carry_spread",
    "fear_greed_score",
    "fg_sma_10",
    "fg_vix_divergence",
    "rsi_bb_position",
    "macro_risk_score",
    "trend_strength",
]

PASSTHROUGH_COLS: list[str] = [
    "is_doji",
    "is_bullish_candle",
    "fg_extreme_fear",
    "fg_extreme_greed",
    "sma_cross_20_50",
    "sma_cross_50_200",
    "macd_cross",
    "vol_regime",
    "vix_regime",
    "yield_curve_regime",
    "candle_direction_streak",
    "day_of_week",
    "month",
    "is_month_end",
    "is_quarter_end",
    "week_of_year",
]

DROP_COLS: list[str] = [
    "Date",
    "fear_greed_label",
    "Open",
    "High",
    "Low",
    "Close",
    "vwap",
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "sma_200",
    "ema_9",
    "ema_21",
    "ema_50",
    "bb_mid",
    "bb_upper",
    "bb_lower",
    "sp500_level",
]


def build_pipeline(
    scale: bool = True,
    winsorize_q: float = 0.99,
    encode_company: bool = False,
) -> Pipeline:
    grouped_winsorizer = GroupedWinsorizer(
        group_col=GROUP_COL,
        cols=GROUPED_WINSORIZE_COLS,
        q=winsorize_q,
        unseen_group_policy="global",
    )

    global_winsorizer = GlobalWinsorizer(
        cols=GLOBAL_WINSORIZE_COLS,
        upper_q=winsorize_q,
        lower_q=1.0 - winsorize_q,
    )

    dropper = ColumnDropper(cols=DROP_COLS)

    if encode_company:
        company_transformer = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    else:
        company_transformer = "passthrough"

    if scale:
        num_transformers = [
            ("robust", RobustScaler(), ROBUST_COLS),
            ("standard", StandardScaler(), STANDARD_COLS),
        ]
    else:
        num_transformers = [
            ("passthrough_num", "passthrough", ROBUST_COLS + STANDARD_COLS),
        ]

    column_transformer = ColumnTransformer(
        transformers=num_transformers
        + [
            ("passthrough_flags", "passthrough", PASSTHROUGH_COLS),
            ("group", company_transformer, [GROUP_COL]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    column_transformer.set_output(transform="pandas")

    pipeline = Pipeline(
        steps=[
            ("winsorize_grouped", grouped_winsorizer),
            ("winsorize_global", global_winsorizer),
            ("drop", dropper),
            ("columns", column_transformer),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def feature_input_columns() -> list[str]:
    return (
        [GROUP_COL]
        + GROUPED_WINSORIZE_COLS
        + GLOBAL_WINSORIZE_COLS
        + ROBUST_COLS
        + STANDARD_COLS
        + PASSTHROUGH_COLS
        + DROP_COLS
    )
