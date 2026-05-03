import logging
import os

import numpy as np
import pandas as pd

from src.data.validation import Validator

logger = logging.getLogger(__name__)


class Cleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()  # never touch raw data
        self.quarantine = []  # rejected rows go here
        self.log = []  # decision log

    # 1. Accuracy — Fix/Reject bad records
    def drop_invalid_prices(self):
        """Drops rows with invalid price relationships and logs the actions."""
        # Check High < Low
        condition = self.df["High"] < self.df["Low"]
        if condition.any():
            invalid_rows = self.df[condition]
            self.quarantine.append(invalid_rows)
            self.df = self.df[~condition]
            self.log.append(
                f"drop_invalid_prices: Dropped {len(invalid_rows)} rows where High < Low"
            )

        # Check Close outside High/Low
        condition = (self.df["Close"] > self.df["High"]) | (
            self.df["Close"] < self.df["Low"]
        )
        if condition.any():
            invalid_rows = self.df[condition]
            self.quarantine.append(invalid_rows)
            self.df = self.df[~condition]
            self.log.append(
                f"drop_invalid_prices: Dropped {len(invalid_rows)} rows where Close outside High/Low"
            )
            # Drop rows where prices are negative or zero
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            condition = self.df[col] <= 0
            if condition.any():
                invalid_rows = self.df[condition]
                self.quarantine.append(invalid_rows)
                self.df = self.df[~condition]
                self.log.append(
                    f"drop_invalid_prices: Dropped {len(invalid_rows)} rows where {col} <= 0"
                )

    # 2. Consistency — Fix dtypes & standardize formats
    def fix_dtypes(self):
        """Coerces numeric columns to numeric types,string columns to string types and parses dates."""

        # Replace all implicit missing values with NaN first
        implicit_missing = [
            "",
            " ",
            "N/A",
            "n/a",
            "NA",
            "null",
            "NULL",
            "None",
            "none",
            "NaN",
            "nan",
            "?",
            "missing",
            "Missing",
            "-",
        ]
        self.df = self.df.replace(implicit_missing, pd.NA)
        self.log.append("fix_dtypes: replaced implicit missing values with NaN")

        # Numeric columns
        numeric_cols = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Dividends",
            "Stock Splits",
            "vix",
            "fed_funds_rate",
            "treasury_10y",
            "sp500_level",
            "fear_greed_score",
        ]
        for col in numeric_cols:
            if col in self.df.columns:
                before = self.df[col].isnull().sum()
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                after = self.df[col].isnull().sum()
                new_nulls = after - before
                if new_nulls > 0:
                    self.log.append(
                        f"fix_dtypes: '{col}' — {new_nulls} unparseable values set to NaN"
                    )
        # Date column
        before = self.df["Date"].isnull().sum()
        self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")
        after_coerce = self.df["Date"].isnull().sum()
        new_nulls = after_coerce - before
        if new_nulls > 0:
            self.log.append(
                f"fix_dtypes: 'Date' — {new_nulls} unparseable dates set to NaT"
            )
        # String columns
        # convert actual NaN before astype to avoid converting them to "nan"
        string_cols = ["Company", "fear_greed_label", "label"]
        for col in string_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].where(self.df[col].notna(), other=np.nan)
                self.df[col] = self.df[col].astype(str).str.strip().str.title()
                self.df[col] = self.df[col].replace({"None": np.nan, "Nan": np.nan})

    # 3. Completeness — Handle missing values
    # For non-critical columns with <5% missing; drop rows (MCAR)
    def drop_missing(self):
        """Drops rows missing critical fields and rows where non-critical columns have <5% missing."""

        # Critical columns — drop regardless of percentage
        critical_cols = ["Date", "Company", "Close", "label"]
        condition = self.df[critical_cols].isnull().any(axis=1)
        if condition.any():
            missing_rows = self.df[condition]
            self.quarantine.append(missing_rows)
            self.df = self.df[~condition]
            self.log.append(
                f"drop_missing: {len(missing_rows)} rows missing critical fields removed"
            )

        # Non-critical columns — only drop if <5% missing (MCAR)
        non_critical_cols = [
            "vix",
            "fed_funds_rate",
            "treasury_10y",
            "sp500_level",
            "fear_greed_score",
            "fear_greed_label",
        ]
        for col in non_critical_cols:
            if col in self.df.columns:
                missing_pct = self.df[col].isnull().sum() / len(self.df) * 100
                if 0 < missing_pct < 5:
                    condition = self.df[col].isnull()
                    missing_rows = self.df[condition]
                    self.quarantine.append(missing_rows)
                    self.df = self.df[~condition]
                    self.log.append(
                        f"drop_missing: '{col}' had {missing_pct:.2f}% missing (MCAR) "
                        f"— {len(missing_rows)} rows removed"
                    )
                elif missing_pct >= 5:
                    self.log.append(
                        f"drop_missing: '{col}' has {missing_pct:.2f}% missing "
                        f"— too high to drop, left for preprocessing step"
                    )

    # 4. Uniqueness — Remove duplicates
    def drop_duplicates(self):
        """Removes fully duplicate rows and duplicate (Date, Company) pairs."""
        full_duplicates = self.df[self.df.duplicated(keep="first")]
        self.quarantine.append(full_duplicates)
        self.df = self.df.drop_duplicates()
        self.log.append(
            f"drop_duplicates: Dropped {len(full_duplicates)} duplicate rows"
        )

        # Drop fuzzy duplicate: (Date, Company) pairs, keep first
        fuzzy_duplicates = self.df[
            self.df.duplicated(subset=["Date", "Company"], keep="first")
        ]
        self.quarantine.append(fuzzy_duplicates)
        self.df = self.df.drop_duplicates(subset=["Date", "Company"], keep="first")
        self.log.append(
            f"drop_duplicates: Dropped {len(fuzzy_duplicates)} duplicate (Date, Company) rows"
        )

    # 5. Timeliness — Handle stale data

    # Stale row detection uses two conditions:
    # 1. Flat per day (Open=High=Low=Close) + Volume=0: catches large cap stocks
    #    where any price freeze is impossible given their trading volume.
    # 2. Identical to previous day + Volume=0: catches cases where only Open!=Close
    #    but all other values are copied from yesterday.
    # Both are safe for the 491 large/mega cap companies (AAPL, MSFT, etc.)
    # verified via df["Company"].unique(): no small/micro cap stocks present.

    def drop_stale_rows(self):
        """Removes rows where data didn't update from previous trading day and
        price is frozen with 0 volume within the same day."""

        temp = self.df.sort_values(["Company", "Date"]).copy()
        # condition 1: flat within same day + 0 volume
        flat_within_day = (
            (temp["Open"] == temp["Close"])
            & (temp["High"] == temp["Low"])
            & (temp["Open"] == temp["High"])
            & (temp["Volume"] == 0)
        )

        # condition 2: exact copy of previous day + 0 volume
        previous_day = temp.groupby("Company")
        identical_to_yesterday = (
            (temp["Open"] == previous_day["Open"].shift(1))
            & (temp["High"] == previous_day["High"].shift(1))
            & (temp["Low"] == previous_day["Low"].shift(1))
            & (temp["Close"] == previous_day["Close"].shift(1))
            & (temp["Volume"] == 0)
        )

        stale_mask = flat_within_day | identical_to_yesterday
        stale_rows = temp[stale_mask].copy()
        self.quarantine.append(stale_rows)
        number_of_stale_rows = len(stale_rows)
        self.df = temp[~stale_mask]
        self.log.append(
            f"drop_stale_rows: {number_of_stale_rows} stale rows removed — price frozen + volume 0"
        )
        logger.info(
            f"drop_stale_rows: complete — {number_of_stale_rows} rows quarantined"
        )

    def _save_quarantine(self):
        """Saves all rejected rows to a quarantine CSV file for review."""

        if not self.quarantine:
            return

        quarantine_df = pd.concat(self.quarantine, ignore_index=True).drop_duplicates()
        os.makedirs(os.path.join("data", "processed"), exist_ok=True)
        quarantine_path = os.path.join("data", "processed", "quarantine.csv")
        quarantine_df.to_csv(quarantine_path, index=False)

        self.log.append(
            f"_save_quarantine: {len(quarantine_df)} total rows saved to {quarantine_path}"
        )
        logger.info(f"_save_quarantine: complete — {len(quarantine_df)} rows saved")

    def run_all(self) -> pd.DataFrame:
        self.fix_dtypes()  # 1. fix types first
        self.drop_missing()  # 2. drop missing before any comparisons
        self.drop_duplicates()  # 3. then duplicates
        self.drop_invalid_prices()  # 4. then invalid prices
        self.drop_stale_rows()  # 5. stale last (needs sorted numeric data)
        self._save_quarantine()
        return self.df


DATA_PATH = os.path.join("data", "raw", "market_data_merged.csv")
CLEANED_PATH = os.path.join("data", "processed", "market_data_cleaned.csv")
LOG_PATH = os.path.join("reports", "cleaning_log.txt")


def run_cleaning():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    cleaner = Cleaner(df)
    clean_df = cleaner.run_all()
    logger.info(f"run_cleaning: cleaning complete — {len(clean_df):,} rows remaining")

    assert (
        clean_df["Close"].isnull().sum() == 0
    ), "Close still has nulls after cleaning!"
    assert (clean_df["High"] < clean_df["Low"]).sum() == 0, "High < Low still present!"
    assert clean_df.duplicated().sum() == 0, "Duplicates still present!"

    validator = Validator(clean_df)
    remaining_issues = validator.run_all()
    if remaining_issues:
        logger.warning(
            f"run_cleaning: {len(remaining_issues)} issues still flagged after cleaning:"
        )
        for issue in remaining_issues:
            logger.warning(f"  - {issue}")

    os.makedirs(os.path.dirname(CLEANED_PATH), exist_ok=True)
    clean_df.to_csv(CLEANED_PATH, index=False)

    # save decision log
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("CLEANING DECISION LOG\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Rows before cleaning: {len(df):,}\n")
        f.write(f"Rows after cleaning:  {len(clean_df):,}\n")
        f.write(f"Rows removed:         {len(df) - len(clean_df):,}\n\n")
        f.write("=" * 70 + "\n\n")
        for entry in cleaner.log:
            f.write(f"  - {entry}\n")

        # validation check results after cleaning
        f.write("\n" + "=" * 70 + "\n")
        f.write("POST-CLEANING VALIDATION\n")
        f.write("=" * 70 + "\n\n")
        if remaining_issues:
            f.write(f"  {len(remaining_issues)} issues still flagged:\n")
            for issue in remaining_issues:
                f.write(f"  - {issue}\n")
        else:
            f.write("  All validation checks passed \n")

    return clean_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_cleaning()
