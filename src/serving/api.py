from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
BACKTEST_RESULTS_PATH = PROJECT_ROOT / "models" / "artifacts" / "backtest_results.json"


class BacktestRequest(BaseModel):
    model_name: str | None = None
    initial_capital: float = 100_000.0
    hold_days: int = 10


app = FastAPI(
    title="Trading Signal Prediction API",
    description="Returns daily Buy / Hold / Sell signals for ~500 stocks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predictions/latest")
def get_latest_predictions():
    """Return today's predictions. Falls back to most recent archive if latest.json missing"""
    latest = PREDICTIONS_DIR / "latest.json"

    if not latest.exists():
        # try to serve most recent archive instead
        history_dir = PREDICTIONS_DIR / "history"
        if history_dir.exists():
            archives = sorted(history_dir.glob("*.json"))
            if archives:
                data = json.loads(archives[-1].read_text())
                return {
                    "source": "archive",
                    "date": archives[-1].stem,
                    "predictions": data,
                    "count": len(data),
                }
        raise HTTPException(
            status_code=404,
            detail="No predictions available yet. Run the prediction pipeline first.",
        )

    data = json.loads(latest.read_text())
    date = data[0]["Date"] if data else "unknown"
    return {
        "source": "latest",
        "date": date,
        "predictions": data,
        "count": len(data),
    }


@app.get("/predictions/history/{date}")
def get_predictions_by_date(date: str):
    """
    Return archived predictions for a specific date.
    Date format: YYYY-MM-DD
    """
    archive = PREDICTIONS_DIR / "history" / f"{date}.json"
    if not archive.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for {date}. Available dates are in predictions/history/.",
        )
    data = json.loads(archive.read_text())
    return {
        "source": "archive",
        "date": date,
        "predictions": data,
        "count": len(data),
    }


@app.get("/predictions/dates")
def list_available_dates():
    """List all dates for which archived predictions exist."""
    history_dir = PREDICTIONS_DIR / "history"
    if not history_dir.exists():
        return {"dates": []}
    dates = sorted([p.stem for p in history_dir.glob("*.json")], reverse=True)
    return {"dates": dates, "count": len(dates)}


@app.post("/predictions/trigger")
def trigger_prediction(date: str | None = None):
    """
    Manually trigger the prediction pipeline.
    Optionally pass a date (YYYY-MM-DD) to predict for a specific day.
    In prod this api isnt available, its only for testing
    """
    logger.info(f"Manual prediction trigger requested for date: {date or 'yesterday'}")
    cmd = ["python", "-m", "src.pipelines.predict"]
    if date:
        cmd += ["--date", date]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.stdout:
            logger.info(f"Pipeline stdout:\n{result.stdout}")
        if result.returncode != 0:
            logger.error(f"Prediction failed:\n{result.stderr}")
            raise HTTPException(
                status_code=500,
                detail=result.stderr[-2000:] or result.stdout[-2000:],
            )
        return {
            "status": "success",
            "date": date or "yesterday",
            "output": result.stdout[-2000:],
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Prediction pipeline timed out.")


@app.post("/backtest/run")
async def run_backtest_endpoint(req: BacktestRequest):
    from src.backtesting.engine import run_backtest

    logger.info(
        "Backtest requested — model=%s capital=%.0f hold_days=%d",
        req.model_name or "best_model",
        req.initial_capital,
        req.hold_days,
    )
    start = time.monotonic()

    def _run():
        kwargs = dict(
            initial_capital=req.initial_capital,
            hold_days=req.hold_days,
        )
        if req.model_name:
            kwargs["model_name"] = req.model_name
        return run_backtest(**kwargs)

    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, _run)
    except Exception as exc:
        logger.error("Backtest failed: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc)})

    duration = time.monotonic() - start
    metrics = results.get("metrics", {})
    logger.info(
        "Backtest complete — duration=%.1fs return=%.2f%%",
        duration,
        metrics.get("total_return_pct", 0),
    )

    return {
        "status": "success",
        "model_name": results.get("model_name", req.model_name or "best_model"),
        "metrics": {
            "total_return_pct": metrics.get("total_return_pct"),
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "win_rate": metrics.get("win_rate"),
            "profit_factor": metrics.get("profit_factor"),
        },
        "trades_count": results.get("total_trades", 0),
        "run_duration_seconds": round(duration, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/backtest/results")
def get_backtest_results():
    if not BACKTEST_RESULTS_PATH.exists():
        raise HTTPException(status_code=404, detail="backtest_results.json not found.")
    return json.loads(BACKTEST_RESULTS_PATH.read_text())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8000, reload=False)
