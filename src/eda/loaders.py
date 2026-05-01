

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "artifacts"

LABEL_COL = "label"
DATE_COL = "Date"
COMPANY_COL = "Company"

Split = Literal["train", "test"] #2 valid values: train / test


#return path for train_val_selected vs test_selected
def _selected_path(split: Split) -> Path:
    name = "train_val_selected.csv" if split == "train" else "test_selected.csv"
    return PROCESSED_DIR / name

#return path for train_val vs test
def _split_path(split: Split) -> Path:
    name = "train_val.csv" if split == "train" else "test.csv"
    return PROCESSED_DIR / name


def load_label_encoder() -> LabelEncoder:
    return joblib.load(ARTIFACT_DIR / "label_encoder.pkl")


def decode_labels(y: pd.Series) -> pd.Series:
    """Map encoded integer labels (0/1/2) back to {Buy, Hold, Sell}."""
    encoder = load_label_encoder()
    decoded = encoder.inverse_transform(y.to_numpy())
    return pd.Series(decoded, index=y.index, name=y.name)

#return df with decoded labels
def load_selected(split: Split = "train", decode: bool = True) -> pd.DataFrame:
    """Load the post-selection split with standardized features."""
    df = pd.read_csv(_selected_path(split))
    if decode and LABEL_COL in df.columns:
        df[LABEL_COL] = decode_labels(df[LABEL_COL])
    return df


def load_with_dates(split: Split = "train", decode: bool = True) -> pd.DataFrame:
    """Load selected split and attach Date by row-aligned join with the
    pre-transform split file. Verifies alignment via the Company column."""
    selected = load_selected(split, decode=decode)
    anchor = pd.read_csv(
        _split_path(split),
        parse_dates=[DATE_COL],
        usecols=[DATE_COL, COMPANY_COL],
    )

    #check that row count in selected and anchor are the same
    if len(anchor) != len(selected):
        raise ValueError(
            f"row count mismatch: selected={len(selected)}, "
            f"split={len(anchor)} — pipeline may have reordered rows."
        )
    #verify company col identical in both
    if not (anchor[COMPANY_COL].to_numpy() == selected[COMPANY_COL].to_numpy()).all():
        raise ValueError(
            "Company column does not align row-by-row between selected "
            "and pre-transform split files — index-based join is unsafe."
        )
    #attach date
    selected.insert(0, DATE_COL, anchor[DATE_COL].to_numpy())
    return selected


def load_raw_for_features(
    features: list[str],
    split: Split,
) -> pd.DataFrame:
    """Load (Date, Company, *features, label) from the engineered file,
    filtered to the date range of the given split.

    Use this to plot on the original feature scale, since values in the
    selected CSVs are winsorized and standardized.
    """
    keep = {DATE_COL, COMPANY_COL, LABEL_COL, *features}
    df = pd.read_csv(
        PROCESSED_DIR / "market_data_with_features.csv",
        parse_dates=[DATE_COL],
        usecols=lambda c: c in keep,
    )

    anchor = pd.read_csv(
        _split_path(split),
        parse_dates=[DATE_COL],
        usecols=[DATE_COL],
    )
    lo, hi = anchor[DATE_COL].min(), anchor[DATE_COL].max()
    df = df[(df[DATE_COL] >= lo) & (df[DATE_COL] <= hi)].reset_index(drop=True)

    return df
