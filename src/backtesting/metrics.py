import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def compute_metrics(daily_portfolio_values: list[float], trades: list[dict]) -> dict:
    if len(daily_portfolio_values) >= 2:
        arr = np.array(daily_portfolio_values, dtype=float)
        daily_returns = np.diff(arr) / arr[:-1]
        mean_r = float(np.mean(daily_returns))
        std_r = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0
    else:
        mean_r = 0.0
        std_r = 0.0

    initial = daily_portfolio_values[0] if daily_portfolio_values else 0.0
    final = daily_portfolio_values[-1] if daily_portfolio_values else 0.0
    total_return_pct = (final / initial - 1) * 100 if initial else float("nan")

    sharpe_ratio = (mean_r / std_r) * math.sqrt(252) if std_r != 0 else float("nan")

    if not trades:
        win_rate = float("nan")
        profit_factor = float("nan")
    else:
        wins = sum(1 for t in trades if t["return_pct"] > 0)
        win_rate = wins / len(trades)
        pos_pnl = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        neg_pnl = sum(t["pnl"] for t in trades if t["pnl"] < 0)
        profit_factor = pos_pnl / abs(neg_pnl) if neg_pnl != 0 else float("inf")

    return {
        "total_return_pct": total_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }
