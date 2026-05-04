"""
scripts/run_backtest.py
=======================
Standalone CLI for running the backtesting pipeline.

Usage
-----
    python scripts/run_backtest.py
    python scripts/run_backtest.py --model-name random_forest
    python scripts/run_backtest.py --model-name lightgbm --initial-capital 50000
    python scripts/run_backtest.py --all-models
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from src.backtesting.engine import run_backtest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _model_names_from_training_results() -> list[str]:
    path = _PROJECT_ROOT / "models" / "artifacts" / "training_results.json"
    if not path.exists():
        raise FileNotFoundError(f"training_results.json not found at {path}")
    with open(path) as f:
        data = json.load(f)
    return list(data.get("models", {}).keys())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtesting pipeline")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--hold-days", type=int, default=10)
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run backtest for every model in training_results.json",
    )
    args = parser.parse_args()

    if args.all_models:
        model_names = _model_names_from_training_results()
        logger.info("Running backtest for %d models: %s", len(model_names), model_names)
    elif args.model_name:
        model_names = [args.model_name]
    else:
        model_names = [None]

    for name in model_names:
        start = time.time()
        label = name or "best_model"
        logger.info("=== Starting backtest: %s ===", label)

        results = run_backtest(
            model_name=name,
            initial_capital=args.initial_capital,
            hold_days=args.hold_days,
        )

        elapsed = time.time() - start
        m = results["metrics"]
        logger.info(
            "=== Finished: %s (%.1fs) | return=%.2f%% sharpe=%.3f win_rate=%s profit_factor=%s trades=%d ===",
            label,
            elapsed,
            m.get("total_return_pct", float("nan")),
            m.get("sharpe_ratio", float("nan")) or 0.0,
            f"{m['win_rate']:.2%}" if m.get("win_rate") == m.get("win_rate") else "nan",
            (
                f"{m['profit_factor']:.4f}"
                if isinstance(m.get("profit_factor"), float)
                and m["profit_factor"] == m["profit_factor"]
                and m["profit_factor"] != float("inf")
                else str(m.get("profit_factor"))
            ),
            results["total_trades"],
        )


if __name__ == "__main__":
    main()
