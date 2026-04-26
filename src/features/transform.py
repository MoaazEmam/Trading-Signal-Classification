"""
Feature transformation runner.

Fits the stateful feature pipeline on train_val, applies it to both splits,
and persists the transformed files and the fitted pipeline artifact. The
fitted pipeline is re-loaded at inference time so serving sees the same
caps and scaler parameters learned during training.

This stage does not create new features -- it only applies winsorization,
scaling, and column selection. New-feature construction (if any) belongs
in a separate stage upstream of this one.

Inputs  : data/processed/train_val.csv, data/processed/test.csv
Outputs : data/processed/train_val_transformed.csv
          data/processed/test_transformed.csv
          models/artifacts/feature_pipeline.pkl
          models/artifacts/label_encoder.pkl
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.features.pipeline import build_pipeline

logger = logging.getLogger(__name__)

# -- paths -----------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
TRAIN_VAL_PATH = PROCESSED_DIR / "train_val.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
TRAIN_VAL_OUT = PROCESSED_DIR / "train_val_transformed.csv"
TEST_OUT = PROCESSED_DIR / "test_transformed.csv"

ARTIFACT_DIR = Path("models/artifacts")
PIPELINE_PATH = ARTIFACT_DIR / "feature_pipeline.pkl"
LABEL_ENCODER_PATH = ARTIFACT_DIR / "label_encoder.pkl"

LABEL_COL = "label"


def _load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    logger.info("loading %s", path)
    df = pd.read_csv(path, parse_dates=["Date"])
    y = df[LABEL_COL]
    X = df.drop(columns=[LABEL_COL])
    logger.info("  %d rows x %d feature columns", len(X), X.shape[1])
    return X, y


def run_transform(
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    scale: bool = True,
    winsorize_q: float = 0.99,
    encode_company: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sep = "=" * 60
    logger.info(sep)
    logger.info(
        "FEATURE TRANSFORM START (scale=%s, q=%s, encode_company=%s)",
        scale,
        winsorize_q,
        encode_company,
    )
    logger.info(sep)

    if train_df is not None and test_df is not None:
        y_train = train_df[LABEL_COL]
        X_train = train_df.drop(columns=[LABEL_COL])
        y_test = test_df[LABEL_COL]
        X_test = test_df.drop(columns=[LABEL_COL])
        logger.info(
            "  train_val: %d rows x %d feature columns", len(X_train), X_train.shape[1]
        )
        logger.info(
            "  test     : %d rows x %d feature columns", len(X_test), X_test.shape[1]
        )
    else:
        X_train, y_train = _load_split(TRAIN_VAL_PATH)
        X_test, y_test = _load_split(TEST_PATH)

    pipeline = build_pipeline(
        scale=scale,
        winsorize_q=winsorize_q,
        encode_company=encode_company,
    )

    logger.info("fitting pipeline on train_val...")
    pipeline.fit(X_train, y_train)

    logger.info("transforming train_val and test...")
    X_train_out = pipeline.transform(X_train)
    X_test_out = pipeline.transform(X_test)
    logger.info("  train_val: %d rows x %d cols", *X_train_out.shape)
    logger.info("  test     : %d rows x %d cols", *X_test_out.shape)

    # Label encoding: fit on train_val only so test cannot influence the
    # class ordering.
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    logger.info(
        "label classes (encoded 0..%d): %s",
        len(label_encoder.classes_) - 1,
        list(label_encoder.classes_),
    )

    train_out = X_train_out.assign(**{LABEL_COL: y_train_enc})
    test_out = X_test_out.assign(**{LABEL_COL: y_test_enc})

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(TRAIN_VAL_OUT, index=False)
    test_out.to_csv(TEST_OUT, index=False)
    logger.info("saved %s", TRAIN_VAL_OUT)
    logger.info("saved %s", TEST_OUT)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    logger.info("saved fitted pipeline to %s", PIPELINE_PATH)
    logger.info("saved label encoder to %s", LABEL_ENCODER_PATH)

    logger.info(sep)
    logger.info("FEATURE TRANSFORM COMPLETE")
    logger.info(sep)
    return train_out, test_out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_transform()
