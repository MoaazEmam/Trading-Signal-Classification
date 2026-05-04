from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=8000, reload=False)
