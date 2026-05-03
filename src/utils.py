import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("data/raw/market_data_merged.csv")
PREDICTIONS_DIR = Path("predictions")


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    project_root = Path(__file__).resolve().parent.parent
    train_path = project_root / "data" / "processed" / "train_val.csv"
    test_path = project_root / "data" / "processed" / "test.csv"
    train_df = pd.read_csv(train_path, parse_dates=["Date"])
    test_df = pd.read_csv(test_path, parse_dates=["Date"])
    return train_df, test_df


def load_train_test_transformed() -> tuple[pd.DataFrame, pd.DataFrame]:
    project_root = Path(__file__).resolve().parent.parent
    train_path = project_root / "data" / "processed" / "train_val_transformed.csv"
    test_path = project_root / "data" / "processed" / "test_transformed.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def load_cleaned_labeled() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "data" / "processed" / "market_data_labeled.csv"
    df = pd.read_csv(file_path, parse_dates=["Date"])
    return df


def load_data_with_features() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "data" / "processed" / "market_data_with_features.csv"
    df = pd.read_csv(file_path, parse_dates=["Date"])
    return df


def get_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include="number")


def get_nonnumeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(exclude="number")


def _save_to_csv(df: pd.DataFrame, path: Path) -> None:
    absolute_path = Path(__file__).resolve().parent.parent / path
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(absolute_path, index=False)
    logger.info(f"Successfully saved merged dataset to: {absolute_path}")
    logger.info(f"Final Dataset Shape: {df.shape}")


def load_last_n_rows_per_company(
    path: Path = RAW_DATA_PATH,
    n_rows: int = 300,
) -> pd.DataFrame:

    project_root = Path(__file__).resolve().parent.parent
    absolute_path = project_root / path

    df = pd.read_csv(absolute_path, parse_dates=["Date"])
    df = (
        df.sort_values(["Company", "Date"])
        .groupby("Company", group_keys=False)
        .apply(lambda g: g.tail(n_rows), include_groups=True)
        .reset_index(drop=True)
    )
    return df


def load_best_model_info() -> dict:
    """Load the best model metadata from models/artifacts/best_model.json"""
    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "models" / "artifacts" / "best_model.json"
    if not path.exists():
        raise FileNotFoundError(
            f"best_model.json not found at {path}. "
            "Run the training and evaluation pipeline first to generate this file."
        )
    with open(path) as f:
        return json.load(f)
