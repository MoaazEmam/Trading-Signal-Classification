import logging
import os

import pandas as pd

from .validation_helper import Validator

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join("data", "raw", "market_data_merged.csv")
REPORT_PATH = os.path.join("reports", "validation_report.txt")


def _section(title: str, width: int = 70) -> str:
    line = "=" * width
    return f"\n{line}\n  {title}\n{line}\n"


def _write(f, text: str) -> None:
    print(text)
    f.write(text + "\n")


def run_validation(df: pd.DataFrame, stage: str) -> list[str]:
    logger.info("--- Validation: %s ---", stage)
    validator = Validator(df)

    checks = [
        validator.check_dtypes,
        validator.check_missing,
        validator.check_duplicates,
        validator.check_company_rows,
        validator.check_date_gaps,
        validator.check_outliers,
        validator.check_sanity,
        validator.check_stale_data,
        validator.check_correlations,
    ]

    if "Stock Splits" in df.columns:
        checks.append(validator.check_price_spikes)

    if "label" in df.columns:
        checks.extend(
            [
                validator.check_class_distribution,
                validator.check_label_consistency,
                validator.check_label_leakage,
            ]
        )

    for check in checks:
        check()

    if validator.issues:
        for issue in validator.issues:
            logger.warning("[%s] %s", stage, issue)
    else:
        logger.info("[%s] all checks passed", stage)

    return validator.issues


def run_validation_report():
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        _write(f, _section("0. LOADING DATASET"))
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        _write(f, f"Loaded from: {DATA_PATH}")
        _write(f, f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

        _write(f, _section("1. Column names and data types"))
        dtype_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Dtype": df.dtypes.values,
                "Non-Null Count": df.notnull().sum().values,
                "Null Count": df.isnull().sum().values,
            }
        )
        _write(f, dtype_df.to_string(index=False))

        _write(f, _section("2. Data quality checks"))
        validator = Validator(df)
        issues = validator.run_all()
        _write(f, f"Status: Complete. {len(issues)} potential issues flagged.")

        _write(f, _section("3. Quality issues summary"))
        if not issues:
            _write(f, "No quality issues found. Dataset looks clean.")
        else:
            _write(f, f"Found {len(issues)} issue(s):\n")
            for i, issue in enumerate(issues, 1):
                _write(f, f"  {i:>2}. {issue}")

        _write(f, "\n" + "-" * 70)
        _write(f, "Report saved to: " + REPORT_PATH)


if __name__ == "__main__":
    run_validation_report()
