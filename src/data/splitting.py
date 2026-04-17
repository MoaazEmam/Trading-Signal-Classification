from pathlib import Path

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

PROCESSED_PATH = Path("data/processed")
TRAIN_VAL_PATH = PROCESSED_PATH / "train_val.csv"
TEST_PATH = PROCESSED_PATH / "test.csv"

TEST_SIZE = 0.2
N_CV_SPLITS = 5
LOOKAHEAD_DAYS = 10


def compute_test_cutoff(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    date_col: str = "Date",
) -> pd.Timestamp:
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    total_span = max_date - min_date
    cutoff = min_date + total_span * (1 - test_size)
    print(
        f"Computed test cutoff: {cutoff.date()} "
        f"(data range {min_date.date()} → {max_date.date()}, "
        f"test_size={test_size})"
    )
    return cutoff


def temporal_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    date_col: str = "Date",
    label_col: str = "label",
    apply_lookahead_buffer: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a labeled dataset into (train_val, test) by a date cutoff.

    The cutoff is computed automatically from the data as
    min_date + (max_date - min_date) * (1 - test_size), so the split
    stays correct as the dataset grows over time without any hardcoded dates.

    - Drops unlabeled rows before splitting.
    - Sorts by date to preserve temporal order.
    - Optionally removes the last LOOKAHEAD_DAYS trading days from train_val
      to prevent label leakage (labels are computed using N future days, so
      rows near the cutoff have labels that peek into the test window).

    Args:
        df: The full merged and labeled DataFrame.
        test_size: Fraction of the total time range to reserve for test.
        date_col: Name of the datetime column.
        label_col: Name of the target column; rows with NaN are dropped.
        apply_lookahead_buffer: If True, drops the last LOOKAHEAD_DAYS
            business days from train_val to eliminate label leakage.

    Returns:
        (train_val, test) as a tuple of DataFrames.
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame.")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

    df = df.dropna(subset=[label_col]).copy()
    df = df.sort_values(date_col).reset_index(drop=True)

    if df.empty:
        raise ValueError("DataFrame is empty after dropping unlabeled rows.")

    cutoff = compute_test_cutoff(df, test_size=test_size, date_col=date_col)
    train_val = pd.DataFrame(df[df[date_col] < cutoff])
    test = pd.DataFrame(df[df[date_col] >= cutoff])

    if train_val.empty:
        raise ValueError(
            f"train_val is empty after splitting at cutoff '{cutoff.date()}'. "
            "Check that test_size is not too large for this dataset range."
        )
    if test.empty:
        raise ValueError(
            f"test is empty after splitting at cutoff '{cutoff.date()}'. "
            "Check that test_size is not too small for this dataset range."
        )

    if apply_lookahead_buffer:
        unique_train_dates = sorted(train_val[date_col].dt.normalize().unique().tolist())
        if len(unique_train_dates) > LOOKAHEAD_DAYS:
            buffer_cutoff = unique_train_dates[-LOOKAHEAD_DAYS - 1]
            rows_before = len(train_val)
            train_val = pd.DataFrame(train_val[train_val[date_col] <= buffer_cutoff])
            rows_dropped = rows_before - len(train_val)
            print(f"Lookahead buffer: dropped {rows_dropped:,} rows (last {LOOKAHEAD_DAYS} trading days of train_val).")

    return train_val, test


def get_time_series_cv(n_splits: int = N_CV_SPLITS) -> TimeSeriesSplit:
    """
    Returns a TimeSeriesSplit object for use during hyperparameter tuning.
    """
    return TimeSeriesSplit(n_splits=n_splits)


def save_splits(
    train_val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path = PROCESSED_PATH,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_val_path = output_dir / "train_val.csv"
    test_path = output_dir / "test.csv"
    train_val.to_csv(train_val_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"train_val → {train_val_path}  ({train_val.shape[0]:,} rows)")
    print(f"test      → {test_path}  ({test.shape[0]:,} rows)")


def run_splitting(
    input_path: str = "data/processed/market_data_cleaned.csv",
    output_dir: str = str(PROCESSED_PATH),
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Loading dataset from {input_path} ...")
    df = pd.read_csv(input_path, parse_dates=["Date"])
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    train_val, test = temporal_split(df, test_size=test_size)

    save_splits(train_val, test, output_dir=Path(output_dir))

    print("\nSplitting complete.")
    return train_val, test


if __name__ == "__main__":
    run_splitting()
