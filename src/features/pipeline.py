"""
Feature pipeline composition.

build_pipeline() returns a fresh, unfitted sklearn Pipeline wiring up every
stateful transformation applied between the cleaned splits and model input:

    1. GroupedWinsorizer   -- per-Company cap on Volume outliers
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

from src.features.transformers import ColumnDropper, GroupedWinsorizer

# -- column groups ---------------------------------------------------------

GROUP_COL = "Company"

WINSORIZE_COLS: list[str] = ["Volume"]

# RobustScaler — legitimate heavy tails, not errors
ROBUST_COLS: list[str] = [
    "Volume",
    "volume_sma_20",
    "volume_ratio",
    "obv",
    "atr_14",
    "atr_ratio",
    "high_low_range",
]

# StandardScaler — price-derived, returns, technical indicators, macro
STANDARD_COLS: list[str] = [
    "Open",
    "High",
    "Low",
    "Close",
    "vix",
    "fed_funds_rate",
    "treasury_10y",
    "sp500_level",
    "fear_greed_score",
    "return_1d",
    "return_2d",
    "return_5d",
    "return_10d",
    "return_20d",
    "log_return",
    "gap_open",
    "intraday_return",
    "sma_5",
    "sma_10",
    "sma_20",
    "sma_50",
    "sma_200",
    "ema_9",
    "ema_21",
    "ema_50",
    "price_to_sma20",
    "sma_cross_20_50",
    "sma_cross_50_200",
    "rsi_7",
    "rsi_14",
    "rsi_divergence",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "macd_cross",
    "bb_mid",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "bb_position",
    "vwap",
    "price_to_vwap",
    "realized_vol_10",
    "realized_vol_20",
    "vix_change",
    "vix_sma_20",
    "vix_ratio",
    "vix_percentile",
    "vix_vs_realized",
    "sp500_return_1d",
    "sp500_return_5d",
    "sp500_return_20d",
    "relative_return_5d",
    "beta_rolling_20",
    "stock_to_sp500",
    "yield_spread",
    "rate_change_fed",
    "rate_change_10y",
    "real_rate_proxy",
    "fg_change_5d",
    "fg_sma_10",
    "fg_momentum",
    "fg_vix_divergence",
    "body_size",
    "upper_wick",
    "lower_wick",
    "candle_direction_streak",
    "rsi_bb_position",
    "volume_price_trend",
    "macro_risk_score",
    "trend_strength",
    "carry_spread",
]

# Passthrough — binary flags, regime labels, calendar features.
# Scaling these would distort their semantics.
PASSTHROUGH_COLS: list[str] = [
    "is_doji",
    "is_bullish_candle",
    "fg_extreme_fear",
    "fg_extreme_greed",
    "vol_regime",
    "vix_regime",
    "yield_curve_regime",
    "day_of_week",
    "month",
    "is_month_end",
    "is_quarter_end",
    "week_of_year",
]

# Dropped at transform time — redundant or non-feature columns.
# fear_greed_label is a deterministic bucket of fear_greed_score.
# Date is a raw timestamp; date features are already in PASSTHROUGH_COLS.
DROP_COLS: list[str] = ["Date", "fear_greed_label"]


# -- builder ---------------------------------------------------------------


def build_pipeline(
    scale: bool = True,
    winsorize_q: float = 0.99,
    encode_company: bool = False,
) -> Pipeline:
    """
    Build the feature pipeline.

    Parameters
    ----------
    scale : bool
        If True, apply RobustScaler to ROBUST_COLS and StandardScaler to
        STANDARD_COLS. Set False for tree-based models (XGBoost / LightGBM /
        RandomForest) which are scale-invariant.
    winsorize_q : float
        Upper quantile used by GroupedWinsorizer. 0.99 caps the top 1%
        of Volume per company.
    encode_company : bool
        If True, ordinal-encode the Company column (tickers seen at fit are
        mapped to integers 0..N-1; tickers not seen at fit are encoded as
        -1). If False, Company is passed through as a string.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted. Call .fit(X_train) then .transform(X_test), and persist
        with joblib.dump so inference uses the exact fitted state.
    """
    winsorizer = GroupedWinsorizer(
        group_col=GROUP_COL,
        cols=WINSORIZE_COLS,
        q=winsorize_q,
        unseen_group_policy="global",
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
            ("winsorize", winsorizer),
            ("drop", dropper),
            ("columns", column_transformer),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def feature_input_columns() -> list[str]:
    """Columns the pipeline expects in X (everything except the label)."""
    return [GROUP_COL] + ROBUST_COLS + STANDARD_COLS + PASSTHROUGH_COLS + DROP_COLS
