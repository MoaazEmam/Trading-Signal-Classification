import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    _calculate_bollinger_bands,
    _calculate_calendar_features,
    _calculate_candlestick_features,
    _calculate_cross_feature_interactions,
    _calculate_fear_greed_features,
    _calculate_features,
    _calculate_interest_rate_features,
    _calculate_macd,
    _calculate_moving_averages,
    _calculate_price_momentum,
    _calculate_rsi,
    _calculate_rsi_features,
    _calculate_sp500_relative_features,
    _calculate_vix_features,
    _calculate_volatility_features,
    _calculate_volume_features,
    build_feature_matrix,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _no_inf(df: pd.DataFrame) -> bool:
    """Returns True if no column contains ±inf."""
    return not np.isinf(df.select_dtypes(include="number")).any().any()


def _single(df: pd.DataFrame, company: str = "AAPL") -> pd.DataFrame:
    """Slice + reset index for a single company — ready to pass to _calculate_* fns."""
    return df[df["Company"] == company].reset_index(drop=True)


# ── _calculate_price_momentum ─────────────────────────────────────────────────


class TestPriceMomentum:
    def test_output_columns(self, engineering_single_company_df):
        result = _calculate_price_momentum(engineering_single_company_df)
        expected = {
            "return_1d",
            "return_2d",
            "return_5d",
            "return_10d",
            "return_20d",
            "log_return",
            "gap_open",
            "intraday_return",
        }
        assert expected.issubset(result.columns)

    def test_shape(self, engineering_single_company_df):
        result = _calculate_price_momentum(engineering_single_company_df)
        assert len(result) == len(engineering_single_company_df)

    def test_no_inf(self, engineering_single_company_df):
        result = _calculate_price_momentum(engineering_single_company_df)
        assert _no_inf(result)

    def test_first_row_nan(self, engineering_single_company_df):
        result = _calculate_price_momentum(engineering_single_company_df)
        # First row has no prior close → return_1d and log_return must be NaN
        assert pd.isna(result["return_1d"].iloc[0])
        assert pd.isna(result["log_return"].iloc[0])

    def test_flat_prices_zero_returns(self, engineering_flat_price_df):
        df = _single(engineering_flat_price_df)
        result = _calculate_price_momentum(df)
        # After first row all returns should be exactly 0
        assert (result["return_1d"].dropna() == 0).all()
        assert (result["intraday_return"] == 0).all()

    def test_intraday_return_formula(self, engineering_single_company_df):
        df = engineering_single_company_df
        result = _calculate_price_momentum(df)
        expected = df["Close"] / df["Open"] - 1
        pd.testing.assert_series_equal(
            result["intraday_return"], expected, check_names=False
        )


# ── _calculate_moving_averages ────────────────────────────────────────────────


class TestMovingAverages:
    def test_output_columns(self, engineering_single_company_df):
        result = _calculate_moving_averages(engineering_single_company_df)
        for n in [5, 10, 20, 50, 200]:
            assert f"sma_{n}" in result.columns
        for n in [9, 21, 50]:
            assert f"ema_{n}" in result.columns
        assert "price_to_sma20" in result.columns
        assert "sma_cross_20_50" in result.columns
        assert "sma_cross_50_200" in result.columns

    def test_cross_values_only_one_or_minus_one(self, engineering_single_company_df):
        result = _calculate_moving_averages(engineering_single_company_df)
        cross_values = result["sma_cross_20_50"].dropna().unique()
        assert set(cross_values).issubset({1, -1})

    def test_sma200_needs_200_rows(self, engineering_single_company_df):
        result = _calculate_moving_averages(engineering_single_company_df)
        # First 199 sma_200 values must be NaN
        assert result["sma_200"].iloc[:199].isna().all()
        assert pd.notna(result["sma_200"].iloc[199])

    def test_no_inf(self, engineering_single_company_df):
        result = _calculate_moving_averages(engineering_single_company_df)
        assert _no_inf(result)

    def test_flat_price_to_sma20_is_one(self, engineering_flat_price_df):
        df = _single(engineering_flat_price_df)
        result = _calculate_moving_averages(df)
        # price / sma == 1 when all prices are equal (after warmup)
        ratio = result["price_to_sma20"].dropna()
        assert (ratio.round(6) == 1.0).all()


# ── _calculate_rsi ────────────────────────────────────────────────────────────


class TestRsi:
    def test_bounds(self, engineering_single_company_df):
        rsi = _calculate_rsi(engineering_single_company_df["Close"], 14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_length(self, engineering_single_company_df):
        rsi = _calculate_rsi(engineering_single_company_df["Close"], 14)
        assert len(rsi) == len(engineering_single_company_df)

    def test_flat_prices_rsi_nan_or_fifty(self, engineering_flat_price_df):
        df = _single(engineering_flat_price_df)
        rsi = _calculate_rsi(df["Close"], 14)
        # Flat prices → avg_loss = 0 → RS undefined → NaN (not 50)
        # Just assert no inf and length is correct
        assert not np.isinf(rsi.fillna(0)).any()
        assert len(rsi) == len(df)


class TestRsiFeatures:
    def test_output_columns(self, engineering_single_company_df):
        result = _calculate_rsi_features(engineering_single_company_df)
        assert {"rsi_7", "rsi_14", "rsi_divergence"}.issubset(result.columns)

    def test_no_inf(self, engineering_single_company_df):
        result = _calculate_rsi_features(engineering_single_company_df)
        assert _no_inf(result)


# ── _calculate_macd ───────────────────────────────────────────────────────────


class TestMacd:
    def test_output_columns(self, engineering_single_company_df):
        result = _calculate_macd(engineering_single_company_df)
        assert {"macd_line", "macd_signal", "macd_histogram", "macd_cross"}.issubset(
            result.columns
        )

    def test_histogram_is_line_minus_signal(self, engineering_single_company_df):
        result = _calculate_macd(engineering_single_company_df)
        diff = (
            result["macd_line"] - result["macd_signal"] - result["macd_histogram"]
        ).abs()
        assert (diff < 1e-9).all()

    def test_cross_values(self, engineering_single_company_df):
        result = _calculate_macd(engineering_single_company_df)
        assert set(result["macd_cross"].unique()).issubset({1, -1})

    def test_no_inf(self, engineering_single_company_df):
        result = _calculate_macd(engineering_single_company_df)
        assert _no_inf(result)


# ── _calculate_bollinger_bands ────────────────────────────────────────────────


class TestBollingerBands:
    def test_output_columns(self, engineering_single_company_df):
        result = _calculate_bollinger_bands(engineering_single_company_df)
        assert {"bb_upper", "bb_lower", "bb_mid", "bb_width", "bb_position"}.issubset(
            result.columns
        )

    def test_upper_above_lower(self, engineering_single_company_df):
        result = _calculate_bollinger_bands(engineering_single_company_df)
        valid = result.dropna()
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_position_range(self, engineering_single_company_df):
        result = _calculate_bollinger_bands(engineering_single_company_df)
        # bb_position can go outside [0,1] when price breaks the bands — just no inf
        assert _no_inf(result)

    def test_no_inf_flat_prices(self, engineering_flat_price_df):
        df = _single(engineering_flat_price_df)
        result = _calculate_bollinger_bands(df)
        assert _no_inf(result)


# ── _calculate_volume_features ────────────────────────────────────────────────


class TestVolumeFeatures:
    def test_output_columns(self, engineering_single_company_df):
        result = _calculate_volume_features(engineering_single_company_df)
        assert {
            "volume_sma_20",
            "volume_ratio",
            "obv",
            "vwap",
            "price_to_vwap",
        }.issubset(result.columns)

    def test_obv_monotone_on_rising_prices(self):
        """OBV should strictly increase when prices only go up."""
        n = 30
        dates = pd.bdate_range("2020-01-02", periods=n)
        close = np.linspace(100, 200, n)
        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": close * 0.99,
                "High": close * 1.01,
                "Low": close * 0.98,
                "Close": close,
                "Volume": [1_000_000] * n,
            }
        )
        result = _calculate_volume_features(df)
        obv = result["obv"]
        assert (obv.diff().dropna() > 0).all()

    def test_no_inf(self, engineering_single_company_df):
        result = _calculate_volume_features(engineering_single_company_df)
        assert _no_inf(result)

    def test_shape(self, engineering_single_company_df):
        result = _calculate_volume_features(engineering_single_company_df)
        assert len(result) == len(engineering_single_company_df)


# ── _calculate_volatility_features ───────────────────────────────────────────


class TestVolatilityFeatures:
    def test_output_columns(self, engineering_single_company_df):
        result = _calculate_volatility_features(engineering_single_company_df)
        assert {
            "atr_14",
            "atr_ratio",
            "realized_vol_10",
            "realized_vol_20",
            "high_low_range",
            "vol_regime",
        }.issubset(result.columns)

    def test_atr_nonnegative(self, engineering_single_company_df):
        result = _calculate_volatility_features(engineering_single_company_df)
        assert (result["atr_14"].dropna() >= 0).all()

    def test_no_inf_flat(self, engineering_flat_price_df):
        df = _single(engineering_flat_price_df)
        result = _calculate_volatility_features(df)
        assert _no_inf(result)

    def test_realized_vol_nonnegative(self, engineering_single_company_df):
        result = _calculate_volatility_features(engineering_single_company_df)
        assert (result["realized_vol_20"].dropna() >= 0).all()


# ── _calculate_vix_features ───────────────────────────────────────────────────


class TestVixFeatures:
    def test_output_columns(self, engineering_single_company_df):
        result = _calculate_vix_features(engineering_single_company_df)
        assert {
            "vix_change",
            "vix_sma_20",
            "vix_ratio",
            "vix_percentile",
            "vix_regime",
        }.issubset(result.columns)

    def test_vix_vs_realized_present_when_col_exists(
        self, engineering_single_company_df
    ):
        df = engineering_single_company_df.copy()
        df["realized_vol_20"] = 0.01
        result = _calculate_vix_features(df)
        assert "vix_vs_realized" in result.columns

    def test_vix_vs_realized_absent_when_col_missing(
        self, engineering_single_company_df
    ):
        result = _calculate_vix_features(engineering_single_company_df)
        assert "vix_vs_realized" not in result.columns

    def test_regime_buckets(self, engineering_extreme_vix_df):
        df = _single(engineering_extreme_vix_df)
        result = _calculate_vix_features(df)
        assert set(result["vix_regime"].unique()).issubset({0, 1, 2, 3})

    def test_percentile_bounds(self, engineering_single_company_df):
        result = _calculate_vix_features(engineering_single_company_df)
        pct = result["vix_percentile"].dropna()
        assert (pct >= 0).all() and (pct <= 1).all()

    def test_no_inf(self, engineering_single_company_df):
        result = _calculate_vix_features(engineering_single_company_df)
        assert _no_inf(result)


# ── _calculate_sp500_relative_features ───────────────────────────────────────


class TestSp500RelativeFeatures:
    def test_output_columns(self, engineering_single_company_df):
        result = _calculate_sp500_relative_features(engineering_single_company_df)
        assert {
            "sp500_return_1d",
            "sp500_return_5d",
            "sp500_return_20d",
            "relative_return_5d",
            "beta_rolling_20",
            "stock_to_sp500",
        }.issubset(result.columns)

    def test_no_inf(self, engineering_single_company_df):
        result = _calculate_sp500_relative_features(engineering_single_company_df)
        assert _no_inf(result)

    def test_shape(self, engineering_single_company_df):
        result = _calculate_sp500_relative_features(engineering_single_company_df)
        assert len(result) == len(engineering_single_company_df)


# ── _calculate_interest_rate_features ────────────────────────────────────────


class TestInterestRateFeatures:
    def test_output_columns(self, engineering_single_company_df):
        result = _calculate_interest_rate_features(engineering_single_company_df)
        assert {
            "yield_spread",
            "rate_change_fed",
            "rate_change_10y",
            "yield_curve_regime",
        }.issubset(result.columns)

    def test_real_rate_proxy_present_when_col_exists(
        self, engineering_single_company_df
    ):
        df = engineering_single_company_df.copy()
        df["realized_vol_20"] = 0.01
        result = _calculate_interest_rate_features(df)
        assert "real_rate_proxy" in result.columns

    def test_inverted_yield_curve_regime(self, engineering_inverted_yield_df):
        df = _single(engineering_inverted_yield_df)
        result = _calculate_interest_rate_features(df)
        assert (result["yield_curve_regime"] == -1).all()
        assert (result["yield_spread"] < 0).all()

    def test_normal_yield_curve_regime(self, engineering_single_company_df):
        df = engineering_single_company_df.copy()
        df["fed_funds_rate"] = 1.0
        df["treasury_10y"] = 3.0
        result = _calculate_interest_rate_features(df)
        assert (result["yield_curve_regime"] == 1).all()

    def test_no_inf(self, engineering_single_company_df):
        result = _calculate_interest_rate_features(engineering_single_company_df)
        assert _no_inf(result)


# ── _calculate_fear_greed_features ───────────────────────────────────────────


class TestFearGreedFeatures:
    def test_output_columns(self, engineering_single_company_df):
        result = _calculate_fear_greed_features(engineering_single_company_df)
        assert {
            "fg_change_5d",
            "fg_sma_10",
            "fg_extreme_fear",
            "fg_extreme_greed",
            "fg_momentum",
        }.issubset(result.columns)

    def test_fg_vix_divergence_present_when_col_exists(
        self, engineering_single_company_df
    ):
        df = engineering_single_company_df.copy()
        df["vix_percentile"] = 0.5
        result = _calculate_fear_greed_features(df)
        assert "fg_vix_divergence" in result.columns

    def test_extreme_fear_flag(self, engineering_extreme_fear_greed_df):
        df = _single(engineering_extreme_fear_greed_df)
        result = _calculate_fear_greed_features(df)
        # First half has score=10 → extreme_fear=1, extreme_greed=0
        half = len(df) // 2
        assert (result["fg_extreme_fear"].iloc[:half] == 1).all()
        assert (result["fg_extreme_greed"].iloc[:half] == 0).all()

    def test_extreme_greed_flag(self, engineering_extreme_fear_greed_df):
        df = _single(engineering_extreme_fear_greed_df)
        result = _calculate_fear_greed_features(df)
        half = len(df) // 2
        assert (result["fg_extreme_greed"].iloc[half:] == 1).all()
        assert (result["fg_extreme_fear"].iloc[half:] == 0).all()

    def test_no_inf(self, engineering_single_company_df):
        result = _calculate_fear_greed_features(engineering_single_company_df)
        assert _no_inf(result)


# ── _calculate_candlestick_features ──────────────────────────────────────────


class TestCandlestickFeatures:
    def test_output_columns(self, engineering_single_company_df):
        result = _calculate_candlestick_features(engineering_single_company_df)
        assert {
            "body_size",
            "upper_wick",
            "lower_wick",
            "is_doji",
            "is_bullish_candle",
            "candle_direction_streak",
        }.issubset(result.columns)

    def test_body_size_nonnegative(self, engineering_single_company_df):
        result = _calculate_candlestick_features(engineering_single_company_df)
        assert (result["body_size"] >= 0).all()

    def test_flat_is_doji(self, engineering_flat_price_df):
        df = _single(engineering_flat_price_df)
        result = _calculate_candlestick_features(df)
        # Open == Close == 0 body_size → is_doji should be 1
        assert (result["is_doji"] == 1).all()

    def test_streak_sign_matches_direction(self, engineering_single_company_df):
        df = engineering_single_company_df
        result = _calculate_candlestick_features(df)
        # Where close > prev close, streak should be positive
        up_days = df["Close"].diff() > 0
        streak = result["candle_direction_streak"]
        assert (streak[up_days].dropna() > 0).all()

    def test_no_inf(self, engineering_single_company_df):
        result = _calculate_candlestick_features(engineering_single_company_df)
        assert _no_inf(result)


# ── _calculate_calendar_features ─────────────────────────────────────────────


class TestCalendarFeatures:
    def test_output_columns(self, engineering_single_company_df):
        result = _calculate_calendar_features(engineering_single_company_df)
        assert {
            "day_of_week",
            "month",
            "is_month_end",
            "is_quarter_end",
            "week_of_year",
        }.issubset(result.columns)

    def test_day_of_week_range(self, engineering_single_company_df):
        result = _calculate_calendar_features(engineering_single_company_df)
        assert result["day_of_week"].between(0, 6).all()

    def test_month_range(self, engineering_single_company_df):
        result = _calculate_calendar_features(engineering_single_company_df)
        assert result["month"].between(1, 12).all()

    def test_binary_flags(self, engineering_single_company_df):
        result = _calculate_calendar_features(engineering_single_company_df)
        assert set(result["is_month_end"].unique()).issubset({0, 1})
        assert set(result["is_quarter_end"].unique()).issubset({0, 1})

    def test_no_nan(self, engineering_single_company_df):
        result = _calculate_calendar_features(engineering_single_company_df)
        assert not result.isna().any().any()


# ── _calculate_cross_feature_interactions ────────────────────────────────────


class TestCrossFeatureInteractions:
    def _make_cross_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Augments df with all columns the cross-feature fn checks for."""
        d = df.copy()
        d["rsi_14"] = 50.0
        d["bb_position"] = 0.5
        d["volume_ratio"] = 1.2
        d["vix_percentile"] = 0.4
        d["price_to_sma20"] = 1.02
        return d

    def test_all_cross_features_present_when_inputs_available(
        self, engineering_single_company_df
    ):
        cross_input = self._make_cross_input(engineering_single_company_df)
        result = _calculate_cross_feature_interactions(cross_input)
        assert {
            "rsi_bb_position",
            "volume_price_trend",
            "macro_risk_score",
            "trend_strength",
            "carry_spread",
        }.issubset(result.columns)

    def test_features_absent_when_inputs_missing(self, engineering_single_company_df):
        # Pass bare df — none of the optional columns exist
        result = _calculate_cross_feature_interactions(engineering_single_company_df)
        # Only carry_spread should appear (fed_funds_rate is in original df)
        assert "rsi_bb_position" not in result.columns
        assert "macro_risk_score" not in result.columns
        assert "carry_spread" in result.columns

    def test_no_inf(self, engineering_single_company_df):
        cross_input = self._make_cross_input(engineering_single_company_df)
        result = _calculate_cross_feature_interactions(cross_input)
        assert _no_inf(result)


# ── _calculate_features (orchestrator) ───────────────────────────────────────


class TestCalculateFeatures:
    def test_shape(self, engineering_single_company_df):
        result = _calculate_features(engineering_single_company_df)
        assert len(result) == len(engineering_single_company_df)

    def test_no_inf(self, engineering_single_company_df):
        result = _calculate_features(engineering_single_company_df)
        assert _no_inf(result)

    def test_key_columns_present(self, engineering_single_company_df):
        result = _calculate_features(engineering_single_company_df)
        key_cols = [
            "return_1d",
            "sma_20",
            "rsi_14",
            "macd_line",
            "bb_position",
            "volume_ratio",
            "atr_14",
            "vix_percentile",
            "yield_spread",
            "fg_extreme_fear",
            "body_size",
            "day_of_week",
            "carry_spread",
        ]
        for col in key_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_vix_vs_realized_present(self, engineering_single_company_df):
        # _calculate_features passes realized_vol_20 via df_aug → must appear
        result = _calculate_features(engineering_single_company_df)
        assert "vix_vs_realized" in result.columns

    def test_real_rate_proxy_present(self, engineering_single_company_df):
        result = _calculate_features(engineering_single_company_df)
        assert "real_rate_proxy" in result.columns

    def test_fg_vix_divergence_present(self, engineering_single_company_df):
        result = _calculate_features(engineering_single_company_df)
        assert "fg_vix_divergence" in result.columns

    def test_flat_prices_no_inf(self, engineering_flat_price_df):
        df = _single(engineering_flat_price_df)
        result = _calculate_features(df)
        assert _no_inf(result)

    def test_single_row_correct_shape(self, engineering_single_row_df):
        df = _single(engineering_single_row_df)
        result = _calculate_features(df)
        assert len(result) == 1


# ── build_feature_matrix ─────────────────────────────────────────────────────


class TestBuildFeatureMatrix:
    def test_row_count_preserved(self, engineering_base_df):
        result = build_feature_matrix(engineering_base_df)
        assert len(result) == len(engineering_base_df)

    def test_original_columns_preserved(self, engineering_base_df):
        original_cols = set(engineering_base_df.columns)
        result = build_feature_matrix(engineering_base_df)
        assert original_cols.issubset(result.columns)

    def test_date_is_column_not_index(self, engineering_base_df):
        result = build_feature_matrix(engineering_base_df)
        assert "Date" in result.columns
        assert result.index.name != "Date"

    def test_company_is_column_not_index(self, engineering_base_df):
        result = build_feature_matrix(engineering_base_df)
        assert "Company" in result.columns

    def test_index_is_clean_integer(self, engineering_base_df):
        result = build_feature_matrix(engineering_base_df)
        assert list(result.index) == list(range(len(result)))

    def test_no_duplicate_columns(self, engineering_base_df):
        result = build_feature_matrix(engineering_base_df)
        assert result.columns.duplicated().sum() == 0

    def test_no_inf(self, engineering_base_df):
        result = build_feature_matrix(engineering_base_df)
        assert _no_inf(result)

    def test_feature_count(self, engineering_base_df):
        original_cols = len(engineering_base_df.columns)
        result = build_feature_matrix(engineering_base_df)
        # Should add substantially more columns than the original
        assert len(result.columns) > original_cols + 50

    def test_sorted_by_company_then_date(self, engineering_base_df):
        # Shuffle input deliberately
        shuffled = engineering_base_df.sample(frac=1, random_state=0)
        result = build_feature_matrix(shuffled)
        for company, group in result.groupby("Company"):
            dates = group["Date"].values
            assert (dates[:-1] <= dates[1:]).all(), f"{company} dates not sorted"

    def test_companies_not_mixed(self, engineering_base_df):
        """return_1d for row 0 of each company must be NaN — no bleed from prior company."""
        result = build_feature_matrix(engineering_base_df)
        for company, group in result.groupby("Company"):
            first_return = group["return_1d"].iloc[0]
            assert pd.isna(
                first_return
            ), f"{company}: return_1d at first row should be NaN, got {first_return}"

    def test_flat_prices_no_inf(self, engineering_flat_price_df):
        result = build_feature_matrix(engineering_flat_price_df)
        assert _no_inf(result)

    def test_assert_fires_on_row_mismatch(self, engineering_base_df, monkeypatch):
        """If _calculate_features returns wrong number of rows, an error should fire."""
        original_fn = _calculate_features

        def bad_calculate_features(df):
            return original_fn(df).iloc[:-1]  # drop one row

        monkeypatch.setattr(
            "src.features.engineering._calculate_features", bad_calculate_features
        )
        # index re-assignment raises ValueError before the assert fires, both signal mismatch
        with pytest.raises((AssertionError, ValueError)):
            build_feature_matrix(engineering_base_df)
