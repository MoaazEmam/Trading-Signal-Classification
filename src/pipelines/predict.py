import json
import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd

from src.data.fetcher import run_daily_fetch
from src.data.preprocessing import _ExtendedCleaner
from src.features.engineering import run_engineering_predict
from src.features.features_selection.selector import FeatureSelector
from src.utils import (
    PREDICTIONS_DIR,
    RAW_DATA_PATH,
    load_best_model_info,
    load_last_n_rows_per_company,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

FEATURE_PIPELINE_PATH = Path("models/artifacts/feature_pipeline.pkl")
LABEL_ENCODER_PATH = Path("models/artifacts/label_encoder.pkl")
FEATURE_SELECTOR_PATH = Path("models/artifacts/feature_selector.pkl")
CONTEXT_ROWS = 300


def _resolve_target_date(target_date: str | None) -> str:
    if target_date is not None:
        return target_date
    return (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")


def _clean_daily(df: pd.DataFrame) -> pd.DataFrame:
    cleaner = _ExtendedCleaner(df)
    cleaned = cleaner.run_all()
    return cleaned


def _transform_features(df: pd.DataFrame) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parent.parent.parent
    pipeline = joblib.load(project_root / FEATURE_PIPELINE_PATH)
    drop_cols = [
        c for c in ["label", "Dividends", "Stock Splits", "index"] if c in df.columns
    ]
    x = df.drop(columns=drop_cols)
    x_transformed = pipeline.transform(x)
    return x_transformed


def _select_features(x: pd.DataFrame) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parent.parent.parent
    selector: FeatureSelector = joblib.load(project_root / FEATURE_SELECTOR_PATH)
    result = selector.transform(x)
    return result.drop(columns=["Company"], errors="ignore")


def _build_predictions_df(
    companies: pd.Series,
    date: str,
    labels: list,
    probabilities,
    label_encoder,
) -> pd.DataFrame:
    class_names = label_encoder.classes_.tolist()
    proba_cols = {
        f"proba_{c.lower()}": probabilities[:, i] for i, c in enumerate(class_names)
    }
    result = pd.DataFrame(
        {"Company": companies.values, "Date": date, "predicted_label": labels}
    )
    for col, vals in proba_cols.items():
        result[col] = vals
    return result


def _write_predictions(predictions_df: pd.DataFrame, date: str) -> None:
    """Write latest.json, latest.csv, and the dated archive entry"""
    project_root = Path(__file__).resolve().parent.parent.parent
    pred_dir = project_root / PREDICTIONS_DIR
    history_dir = pred_dir / "history"
    pred_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    records = predictions_df.to_dict(orient="records")

    latest_json = pred_dir / "latest.json"
    latest_csv = pred_dir / "latest.csv"
    archive_json = history_dir / f"{date}.json"

    latest_json.write_text(json.dumps(records, indent=2))
    predictions_df.to_csv(latest_csv, index=False)
    archive_json.write_text(json.dumps(records, indent=2))

    logger.info(f"Predictions written — latest.json, latest.csv, history/{date}.json")


def _copy_latest_from_archive(predictions_dir: Path) -> None:
    """On etch failure, copy the most recent archived prediction to latest"""
    history_dir = predictions_dir / "history"
    archives = sorted(history_dir.glob("*.json"))
    if not archives:
        logger.warning("No archived predictions found — latest.* not updated")
        return
    most_recent = archives[-1]
    (predictions_dir / "latest.json").write_text(most_recent.read_text())
    logger.info(f"Served latest predictions from archive: {most_recent.name}")


def _append_to_merged(daily_df: pd.DataFrame) -> None:
    """Append the new day's raw rows to market_data_merged.csv without reloading the full file"""
    project_root = Path(__file__).resolve().parent.parent.parent
    path = project_root / RAW_DATA_PATH
    daily_df.to_csv(path, mode="a", header=False, index=False)
    logger.info(f"Appended {len(daily_df)} rows to {path}")


def run_predict_pipeline(target_date: str | None = None) -> pd.DataFrame | None:
    """
    runs the full daily prediction pipeline
    steps:
        1. get target date (default: yesterday)
        2. fetch data's raw data
        3. clean daily fetch
        4. load last 300 rows per company for feature engineering
        5. concat and engineer features
        6. drop the extra rows so that only the new row continues down the pipeline
        7. transform features using saved transformers
        8. select features using saved selector
        9. load best model and predict using it
        10. write prediction and append new row to merged data for future retraining
    Returns the predictions DataFrame on success, None on failure.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    pred_dir = project_root / PREDICTIONS_DIR

    date = _resolve_target_date(target_date)
    logger.info(f"Running prediction pipeline for date: {date}")

    try:
        daily_df = run_daily_fetch(date)
    except RuntimeError as e:
        logger.error(f"Daily fetch failed for {date}: {e}")
        _copy_latest_from_archive(pred_dir)
        return None

    try:
        cleaned_daily = _clean_daily(daily_df)
        if cleaned_daily.empty:
            raise RuntimeError("Cleaning removed all rows from daily fetch.")

        history_df = load_last_n_rows_per_company(n_rows=CONTEXT_ROWS)
        history_df = history_df[history_df["Date"] < pd.Timestamp(date)]

        combined_df = pd.concat([history_df, cleaned_daily], ignore_index=True)
        combined_df = combined_df.sort_values(["Company", "Date"]).reset_index(
            drop=True
        )
        featured_df = run_engineering_predict(combined_df, date)

        todays_features = featured_df[featured_df["Date"] == pd.Timestamp(date)].copy()
        if todays_features.empty:
            raise RuntimeError(
                f"Feature engineering produced no rows for {date}. "
                "Context rows may be insufficient or the date was dropped during NaN removal."
            )
        logger.info(f"Features ready for {len(todays_features)} companies")

        companies = todays_features["Company"].reset_index(drop=True)

        x_transformed = _transform_features(todays_features)
        x_selected = _select_features(x_transformed)

        best_model_info = load_best_model_info()
        model_path = project_root / best_model_info["model_path"]
        logger.info(f"Loading model: {best_model_info['model_name']} from {model_path}")
        model = joblib.load(model_path)
        label_encoder = joblib.load(project_root / LABEL_ENCODER_PATH)

        labels_encoded = model.predict(x_selected)
        probabilities = model.predict_proba(x_selected)
        labels = label_encoder.inverse_transform(labels_encoded)

        predictions_df = _build_predictions_df(
            companies, date, labels, probabilities, label_encoder
        )
        _write_predictions(predictions_df, date)

    except Exception:
        logger.error(
            f"Prediction pipeline failed for {date}:\n{traceback.format_exc()}"
        )
        return None

    _append_to_merged(cleaned_daily)
    logger.info(f"Prediction pipeline complete for {date}")
    return predictions_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the daily prediction pipeline.")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date in YYYY-MM-DD format. Defaults to yesterday.",
    )
    args = parser.parse_args()
    run_predict_pipeline(target_date=args.date)
