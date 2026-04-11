import os

import pandas as pd

from .validation_helper import Validator

DATA_PATH = os.path.join("data", "raw", "market_data_merged.csv")
REPORT_PATH = os.path.join("reports", "validation_report.txt")


# helper functions
def section(title: str, width: int = 70) -> str:
    line = "=" * width
    return f"\n{line}\n  {title}\n{line}\n"


def write(f, text: str) -> None:
    print(text)
    f.write(text + "\n")


def run_validation():
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        # Load dataset
        write(f, section("0. LOADING DATASET"))
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        write(f, f"Loaded from: {DATA_PATH}")
        write(f, f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        # Column types
        write(f, section("1. Column names and data types"))
        dtype_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Dtype": df.dtypes.values,
                "Non-Null Count": df.notnull().sum().values,
                "Null Count": df.isnull().sum().values,
            }
        )
        write(f, dtype_df.to_string(index=False))
        # data quality checks
        write(f, section("2. Data quality checks"))
        write(f, "Status: Initializing Validator...")
        validator = Validator(df)

        write(f, "Status: Running all automated checks (Nulls, Outliers, Sanity)...")
        issues = validator.run_all()

        write(f, f"Status: Complete. {len(issues)} potential issues flagged.")

        # quality issues summary
        write(f, section("3. Quality issues summary"))
        if not issues:
            write(f, "No quality issues found. Dataset looks clean.")
        else:
            write(f, f"Found {len(issues)} issue(s):\n")
            for i, issue in enumerate(issues, 1):
                write(f, f"  {i:>2}. {issue}")

        write(f, "\n" + "-" * 70)
        write(f, "Report saved to: " + REPORT_PATH)


if __name__ == "__main__":
    run_validation()
