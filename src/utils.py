from pathlib import Path

import pandas as pd


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    project_root = Path(__file__).resolve().parent.parent
    train_path = project_root / "data" / "processed" / "train_val.csv"
    test_path = project_root / "data" / "processed" / "test.csv"
    train_df = pd.read_csv(train_path, parse_dates=["Date"])
    test_df = pd.read_csv(test_path, parse_dates=["Date"])
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


def _save_to_csv(df: pd.DataFrame, path: Path) -> None:
    absolute_path = Path(__file__).resolve().parent.parent / path
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(absolute_path, index=False)
    print(f"Successfully saved merged dataset to: {absolute_path}")
    print(f"📊 Final Dataset Shape: {df.shape}")
