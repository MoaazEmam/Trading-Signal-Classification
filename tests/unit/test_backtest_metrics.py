"""Unit tests for src/backtesting/metrics.py"""

from __future__ import annotations

import math

import pytest

from src.backtesting.metrics import compute_metrics


def _make_trades(returns: list[float], capital_per: float = 1000.0) -> list[dict]:
    return [
        {
            "return_pct": r * 100,
            "pnl": capital_per * r,
        }
        for r in returns
    ]


class TestComputeMetricsPortfolio:
    def test_empty_inputs_returns_dict(self):
        result = compute_metrics([], [])
        assert isinstance(result, dict)

    def test_single_value_portfolio_zero_return(self):
        result = compute_metrics([100_000.0], [])
        assert result["total_return_pct"] == pytest.approx(0.0)

    def test_two_equal_values_zero_return(self):
        result = compute_metrics([100_000.0, 100_000.0], [])
        assert result["total_return_pct"] == pytest.approx(0.0)

    def test_positive_total_return(self):
        result = compute_metrics([100_000.0, 110_000.0], [])
        assert result["total_return_pct"] == pytest.approx(10.0)

    def test_negative_total_return(self):
        result = compute_metrics([100_000.0, 90_000.0], [])
        assert result["total_return_pct"] == pytest.approx(-10.0)

    def test_total_return_pct_multiday(self):
        # 100k → 105k → 115k: total return = 15%
        result = compute_metrics([100_000.0, 105_000.0, 115_000.0], [])
        assert result["total_return_pct"] == pytest.approx(15.0)

    def test_sharpe_nan_when_single_value(self):
        result = compute_metrics([100_000.0], [])
        assert math.isnan(result["sharpe_ratio"])

    def test_sharpe_nan_when_zero_std(self):
        # All returns identical → std=0 → sharpe=nan
        result = compute_metrics([100_000.0] * 10, [])
        assert math.isnan(result["sharpe_ratio"])

    def test_sharpe_positive_when_consistently_up(self):
        # Steady upward trend → positive sharpe
        values = [100_000.0 * (1.001**i) for i in range(100)]
        result = compute_metrics(values, [])
        assert result["sharpe_ratio"] > 0

    def test_all_keys_present(self):
        result = compute_metrics([100_000.0, 102_000.0], [])
        assert "total_return_pct" in result
        assert "sharpe_ratio" in result
        assert "win_rate" in result
        assert "profit_factor" in result


class TestComputeMetricsTrades:
    def test_no_trades_win_rate_nan(self):
        result = compute_metrics([100_000.0, 101_000.0], [])
        assert math.isnan(result["win_rate"])

    def test_no_trades_profit_factor_nan(self):
        result = compute_metrics([100_000.0, 101_000.0], [])
        assert math.isnan(result["profit_factor"])

    def test_all_winning_trades_win_rate_one(self):
        trades = _make_trades([0.01, 0.02, 0.03])
        result = compute_metrics([100_000.0, 101_000.0], trades)
        assert result["win_rate"] == pytest.approx(1.0)

    def test_all_losing_trades_win_rate_zero(self):
        trades = _make_trades([-0.01, -0.02])
        result = compute_metrics([100_000.0, 98_000.0], trades)
        assert result["win_rate"] == pytest.approx(0.0)

    def test_mixed_trades_win_rate(self):
        trades = _make_trades([0.01, -0.01, 0.01, -0.01])
        result = compute_metrics([100_000.0, 100_000.0], trades)
        assert result["win_rate"] == pytest.approx(0.5)

    def test_all_winning_no_losses_profit_factor_inf(self):
        trades = _make_trades([0.01, 0.02])
        result = compute_metrics([100_000.0, 102_000.0], trades)
        assert result["profit_factor"] == float("inf")

    def test_all_losing_no_gains_profit_factor_zero(self):
        trades = _make_trades([-0.01, -0.02])
        result = compute_metrics([100_000.0, 98_000.0], trades)
        assert result["profit_factor"] == pytest.approx(0.0)

    def test_profit_factor_ratio(self):
        # 200 profit vs 100 loss → PF = 2.0
        trades = [
            {"return_pct": 10.0, "pnl": 200.0},
            {"return_pct": -5.0, "pnl": -100.0},
        ]
        result = compute_metrics([100_000.0, 101_000.0], trades)
        assert result["profit_factor"] == pytest.approx(2.0)

    def test_win_rate_with_single_winning_trade(self):
        trades = _make_trades([0.05])
        result = compute_metrics([100_000.0, 105_000.0], trades)
        assert result["win_rate"] == pytest.approx(1.0)

    def test_large_number_of_trades(self):
        rng_returns = [0.001 * i for i in range(1, 201)]
        trades = _make_trades(rng_returns)
        result = compute_metrics([100_000.0 + i * 10 for i in range(201)], trades)
        assert result["win_rate"] == pytest.approx(1.0)
        assert result["profit_factor"] == float("inf")
