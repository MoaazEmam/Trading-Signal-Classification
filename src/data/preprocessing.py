"""
Preprocessing pipeline.

Orchestrates the full pipeline.

Stages:
    0. Raw validation  -- catch problems in raw data before cleaning
    1. Cleaning        -- raw CSV -> data/processed/market_data_cleaned.csv
    2. Post-clean validation -- confirm fixes (embedded in cleaning log)
    3. Labeling        -- cleaned -> data/processed/market_data_labeled.csv
    4. Splitting       -- labeled -> data/processed/train_val.csv + test.csv
"""

import datetime
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cleaning import Cleaner
from src.data.labeling import label as _label
from src.data.splitting import save_splits, temporal_split
from src.data.validation import Validator

logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────
RAW_PATH = Path("data/raw/market_data_merged.csv")
CLEANED_PATH = Path("data/processed/market_data_cleaned.csv")
LABELED_PATH = Path("data/processed/market_data_labeled.csv")
PROCESSED_PATH = Path("data/processed")
LOG_PATH = Path("reports/cleaning_log.txt")

# Single source of truth shared by labeling and splitting.
# Must equal the N passed to _label() so the lookahead buffer in run_splitting
# covers exactly the rows whose labels peek into the test window.
LABEL_LOOKAHEAD_N: int = 10


# -- fear & greed score -> label mapping ------------------------------------

_FEAR_GREED_BOUNDS: list[tuple[int, int, str]] = [
    (0, 24, "Extreme Fear"),
    (25, 44, "Fear"),
    (45, 55, "Neutral"),
    (56, 75, "Greed"),
    (76, 100, "Extreme Greed"),
]


def _score_to_fg_label(score: float) -> str:
    for lo, hi, lbl in _FEAR_GREED_BOUNDS:
        if lo <= score <= hi:
            return lbl
    return "Extreme Greed" if score > 100 else "Extreme Fear"


# -- extended cleaner -------------------------------------------------------

class _ExtendedCleaner(Cleaner):
    """
    Extends Cleaner with additional steps and bug fixes.
    All overrides are documented with the reason for the change.
    """

    # -- overrides --

    def fix_dtypes(self) -> None:
        """
        Same as Cleaner.fix_dtypes but Company is NOT title-cased.
        Ticker symbols (e.g. 'AAPL') must remain uppercase; the original
        applies str.title() to all string columns including Company.
        """
        implicit_missing = [
            "", " ", "N/A", "n/a", "NA", "null", "NULL",
            "None", "none", "NaN", "nan", "?", "missing", "Missing", "-",
        ]
        self.df = self.df.replace(implicit_missing, pd.NA)
        self.log.append("fix_dtypes: replaced implicit missing values with NaN")

        numeric_cols = [
            "Open", "High", "Low", "Close", "Volume", "Dividends",
            "Stock Splits", "vix", "fed_funds_rate", "treasury_10y",
            "sp500_level", "fear_greed_score",
        ]
        for col in numeric_cols:
            if col in self.df.columns:
                before = self.df[col].isnull().sum()
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                new_nulls = self.df[col].isnull().sum() - before
                if new_nulls > 0:
                    self.log.append(
                        f"fix_dtypes: '{col}' - {new_nulls} unparseable values set to NaN"
                    )

        before = self.df["Date"].isnull().sum()
        self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")
        new_nulls = self.df["Date"].isnull().sum() - before
        if new_nulls > 0:
            self.log.append(f"fix_dtypes: 'Date' - {new_nulls} unparseable dates set to NaT")

        # Company: strip whitespace only -- do NOT title-case ticker symbols
        if "Company" in self.df.columns:
            self.df["Company"] = self.df["Company"].where(
                self.df["Company"].notna(), other=np.nan
            )
            self.df["Company"] = self.df["Company"].astype(str).str.strip()
            self.df["Company"] = self.df["Company"].replace(
                {"None": np.nan, "Nan": np.nan, "nan": np.nan}
            )

        # Non-ticker string columns: keep title-case normalisation
        for col in ["fear_greed_label", "label"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].where(self.df[col].notna(), other=np.nan)
                self.df[col] = self.df[col].astype(str).str.strip().str.title()
                self.df[col] = self.df[col].replace({"None": np.nan, "Nan": np.nan})

        logger.info("fix_dtypes: complete")

    def drop_missing(self) -> None:
        """
        Same as Cleaner.drop_missing but 'label' is NOT a critical column.
        The original treats 'label' as critical, which drops every row because
        the label column does not exist at cleaning stage (labeling runs after).
        """
        critical_cols = ["Date", "Company", "Close"]
        condition = self.df[critical_cols].isnull().any(axis=1)
        if condition.any():
            missing_rows = self.df[condition]
            self.quarantine.append(missing_rows)
            self.df = self.df[~condition]
            self.log.append(
                f"drop_missing: {len(missing_rows)} rows missing critical fields removed"
            )

        non_critical_cols = [
            "vix", "fed_funds_rate", "treasury_10y",
            "sp500_level", "fear_greed_score", "fear_greed_label",
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
                        f"drop_missing: '{col}' had {missing_pct:.2f}% missing "
                        f"(MCAR) - {len(missing_rows)} rows removed"
                    )
                elif missing_pct >= 5:
                    self.log.append(
                        f"drop_missing: '{col}' has {missing_pct:.2f}% missing "
                        f"- too high to drop, left for preprocessing step"
                    )

    def drop_invalid_prices(self) -> None:
        """
        Extends Cleaner.drop_invalid_prices to also check Open outside High/Low.
        Calls super() for the original checks, then adds the Open check.
        """
        super().drop_invalid_prices()

        # Open outside High/Low (not in original)
        condition = (self.df["Open"] > self.df["High"]) | (self.df["Open"] < self.df["Low"])
        if condition.any():
            invalid_rows = self.df[condition]
            self.quarantine.append(invalid_rows)
            self.df = self.df[~condition]
            self.log.append(
                f"drop_invalid_prices: Dropped {len(invalid_rows)} rows where Open outside High/Low"
            )

    def run_all(self) -> pd.DataFrame:
        """Runs all cleaning steps in order, recording per-step row counts in self.step_rows."""
        self.step_rows: list[tuple[str, int, int]] = []
        steps = [
            ("FIX DATA TYPES",            self.fix_dtypes),
            ("DROP MISSING VALUES",        self.drop_missing),
            ("DROP DUPLICATES",            self.drop_duplicates),
            ("DROP INVALID PRICES",        self.drop_invalid_prices),
            ("DROP STALE ROWS",            self.drop_stale_rows),
            ("DROP ZERO-VOLUME MOVEMENT",  self.drop_zero_volume_movement),
            ("HANDLE PRICE SPIKES",        self.handle_price_spikes),
            ("WINSORIZE VOLUME OUTLIERS",  self.winsorize_outliers),
            ("FIX FEAR & GREED LABELS",    self.fix_fear_greed_labels),
            ("SAVE QUARANTINE",            self._save_quarantine),
        ]
        for name, fn in steps:
            before = len(self.df)
            fn()
            self.step_rows.append((name, before, len(self.df)))
        return self.df

    # -- new methods --

    def drop_zero_volume_movement(self) -> None:
        """Removes rows where Volume=0 but Close changed from the previous trading day."""
        temp = self.df.sort_values(["Company", "Date"]).copy()
        prev_close = temp.groupby("Company")["Close"].shift(1)
        has_movement = (temp["Close"] != prev_close) & prev_close.notna()
        zero_vol = temp["Volume"] == 0
        condition = zero_vol & has_movement
        bad_rows = temp[condition].copy()
        if len(bad_rows) > 0:
            self.quarantine.append(bad_rows)
            self.df = temp[~condition]
            self.log.append(
                f"drop_zero_volume_movement: {len(bad_rows)} rows removed "
                f"(Volume=0 with Close change)"
            )
        else:
            self.log.append(
                "drop_zero_volume_movement: no anomalous zero-volume + price-movement rows found"
            )
        logger.info(f"drop_zero_volume_movement: complete -- {len(bad_rows)} rows quarantined")

    def handle_price_spikes(self, threshold: float = 0.50) -> None:
        """Removes spike rows where the next day reverses >30% (data errors); keeps persistent spikes."""
        temp = self.df.sort_values(["Company", "Date"]).copy()
        pct_change = temp.groupby("Company")["Close"].pct_change()
        pct_change_next = temp.groupby("Company")["Close"].pct_change(-1)
        spike_mask = (pct_change.abs() > threshold) & (temp["Stock Splits"] == 0)
        reversal_mask = pct_change_next.abs() > 0.30
        data_error_mask = spike_mask & reversal_mask
        error_rows = temp[data_error_mask].copy()
        kept_spikes = int(spike_mask.sum()) - len(error_rows)
        if len(error_rows) > 0:
            self.quarantine.append(error_rows)
            self.df = temp[~data_error_mask]
            self.log.append(
                f"handle_price_spikes: {len(error_rows)} spike-rows removed "
                f"(reversed next day - data error); "
                f"{kept_spikes} persistent spike(s) kept as legitimate events"
            )
        else:
            self.log.append(
                f"handle_price_spikes: 0 data-error spikes; "
                f"{int(spike_mask.sum())} persistent spike(s) kept as legitimate events"
            )
        logger.info(f"handle_price_spikes: complete -- {len(error_rows)} removed, {kept_spikes} kept")

    def winsorize_outliers(self) -> None:
        """Caps Volume per company at its 99th percentile."""
        temp = self.df.copy()
        cap_99 = temp.groupby("Company")["Volume"].transform(lambda s: s.quantile(0.99))
        over_cap = temp["Volume"] > cap_99
        n_capped = int(over_cap.sum())
        if n_capped > 0:
            temp.loc[over_cap, "Volume"] = cap_99[over_cap].astype(temp["Volume"].dtype)
            self.df = temp
            self.log.append(
                f"winsorize_outliers: {n_capped} Volume values capped at per-company 99th percentile"
            )
        else:
            self.log.append(
                "winsorize_outliers: no Volume values exceeded per-company 99th percentile cap"
            )
        logger.info(f"winsorize_outliers: complete -- {n_capped} values capped")

    def fix_fear_greed_labels(self) -> None:
        """Regenerates fear_greed_label from fear_greed_score using canonical non-overlapping boundaries."""
        if "fear_greed_score" not in self.df.columns or "fear_greed_label" not in self.df.columns:
            return
        before_mismatches = self._count_fg_mismatches()
        self.df["fear_greed_label"] = self.df["fear_greed_score"].apply(_score_to_fg_label)
        after_mismatches = self._count_fg_mismatches()
        self.log.append(
            f"fix_fear_greed_labels: regenerated fear_greed_label from score "
            f"(mismatches: {before_mismatches} -> {after_mismatches})"
        )
        logger.info(f"fix_fear_greed_labels: complete -- {before_mismatches} mismatches resolved")

    def _count_fg_mismatches(self) -> int:
        mismatches = 0
        for lo, hi, label_name in _FEAR_GREED_BOUNDS:
            mask = self.df["fear_greed_label"] == label_name
            mismatches += int((~self.df["fear_greed_score"].between(lo, hi) & mask).sum())
        return mismatches


# -- cleaning log writer ----------------------------------------------------

_W = 78  # log line width

_STEP_PREFIX_MAP = {
    "FIX DATA TYPES":            "fix_dtypes",
    "DROP MISSING VALUES":       "drop_missing",
    "DROP DUPLICATES":           "drop_duplicates",
    "DROP INVALID PRICES":       "drop_invalid_prices",
    "DROP STALE ROWS":           "drop_stale_rows",
    "DROP ZERO-VOLUME MOVEMENT": "drop_zero_volume_movement",
    "HANDLE PRICE SPIKES":       "handle_price_spikes",
    "WINSORIZE VOLUME OUTLIERS": "winsorize_outliers",
    "FIX FEAR & GREED LABELS":   "fix_fear_greed_labels",
    "SAVE QUARANTINE":           "_save_quarantine",
}

def _percentage_removed(n:int, raw_df: pd.DataFrame)->string:
    return f"{n / len(raw_df) * 100:.2f}%" if len(raw_df) else "n/a"
def _write_cleaning_log(
    path: str,
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    cleaner: _ExtendedCleaner,
    remaining_issues: list[str],
) -> None:
    total_removed = len(raw_df) - len(clean_df)
    pct = _percentage_removed(n,raw_df)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    step_log: dict[str, list[str]] = {k: [] for k in _STEP_PREFIX_MAP}
    for entry in cleaner.log:
        for step, prefix in _STEP_PREFIX_MAP.items():
            if entry.startswith(prefix + ":"):
                step_log[step].append(entry[len(prefix) + 1:].strip())
                break

    quarantine_count = 0
    for entry in cleaner.log:
        if entry.startswith("_save_quarantine:"):
            m = re.search(r"(\d[\d,]*) total rows", entry)
            if m:
                quarantine_count = int(m.group(1).replace(",", ""))

    removal_steps = [
        "DROP MISSING VALUES", "DROP DUPLICATES", "DROP INVALID PRICES",
        "DROP STALE ROWS", "DROP ZERO-VOLUME MOVEMENT", "HANDLE PRICE SPIKES",
    ]
    step_rows_map = {name: (b, a) for name, b, a in cleaner.step_rows}

    categories: dict[str, list[str]] = {
        "DATA COVERAGE": [],
        "RESIDUAL OUTLIERS (post-winsorize; expected for equity data)": [],
        "SANITY": [],
        "CORRELATIONS (expected structural patterns)": [],
    }
    for issue in remaining_issues:
        if issue.startswith("Date"):
            categories["DATA COVERAGE"].append(issue)
        elif issue.startswith("Outliers"):
            categories["RESIDUAL OUTLIERS (post-winsorize; expected for equity data)"].append(issue)
        elif issue.lower().startswith("sanity"):
            categories["SANITY"].append(issue)
        else:
            categories["CORRELATIONS (expected structural patterns)"].append(issue)

    sep = "=" * _W
    thin = "-" * _W

    with open(path, "w", encoding="utf-8") as f:
        w = f.write
        w("CLEANING DECISION LOG\n")
        w(f"Generated: {now}\n")
        w(sep + "\n\n")
        w("SUMMARY\n")
        w(thin + "\n")
        w(f"  Input rows:             {len(raw_df):>10,}\n")
        w(f"  Output rows:            {len(clean_df):>10,}\n")
        w(f"  Total rows removed:     {total_removed:>10,}  ({pct(total_removed)})\n")
        w(f"  Rows quarantined:       {quarantine_count:>10,}\n")
        w("\n")
        w("  Removal by step:\n")
        col_w = max(len(s) for s in removal_steps) + 4
        for step in removal_steps:
            if step in step_rows_map:
                before, after = step_rows_map[step]
                n = before - after
                label = step.title().replace("  ", " ")
                dots = "." * (col_w - len(label))
                w(f"    {label} {dots}  {n:>7,}  ({pct(n)})\n")
        w("\n")
        w(sep + "\n\n")
        w("STEP-BY-STEP DETAIL\n")
        w(sep + "\n")
        for i, (step_name, before, after) in enumerate(cleaner.step_rows, 1):
            if step_name == "SAVE QUARANTINE":
                continue
            delta = before - after
            change = "(no rows removed)" if delta == 0 else f"(-{delta:,} rows)"
            header = f"[{i}] {step_name}"
            counts = f"{before:,} -> {after:,}  {change}"
            padding = _W - len(header) - len(counts)
            w(f"\n{header}{' ' * max(padding, 2)}{counts}\n")
            for note in step_log.get(step_name, []):
                w(f"    - {note}\n")
        w("\n")
        w(sep + "\n\n")
        w("POST-CLEANING VALIDATION\n")
        w(sep + "\n")
        if not remaining_issues:
            w("\n  All validation checks passed.\n")
        else:
            w(f"\n  {len(remaining_issues)} residual issues (flagged for awareness):\n")
            for cat, issues in categories.items():
                if not issues:
                    continue
                w(f"\n  {cat}:\n")
                for issue in issues:
                    w(f"    - {issue}\n")
        w("\n")


# -- validation helper -----------------------------------------------------

def _run_validator_checks(df: pd.DataFrame) -> list[str]:
    """
    Run the 10 label-independent Validator checks and return the issues list.

    Skips check_class_distribution, check_label_consistency, and
    check_label_leakage — all three require a label column that does not
    exist at either the raw-data or post-cleaning stage.
    """
    validator = Validator(df)
    for check in [
        validator.check_dtypes,
        validator.check_missing,
        validator.check_duplicates,
        validator.check_company_rows,
        validator.check_date_gaps,
        validator.check_outliers,
        validator.check_price_spikes,
        validator.check_sanity,
        validator.check_stale_data,
        validator.check_correlations,
    ]:
        check()
    return validator.issues


# -- raw validation stage --------------------------------------------------

def run_raw_validation() -> list[str]:
    """
    Validates raw data BEFORE cleaning to document pre-existing problems.

    Runs _run_validator_checks on the raw CSV so there is a clear before
    picture of what the cleaner is expected to fix. Results are logged as
    WARNINGs and the same checks re-run after cleaning (inside run_cleaning)
    to confirm which issues were resolved.
    """
    logger.info("run_raw_validation: loading raw data from %s", RAW_PATH)
    df = pd.read_csv(str(RAW_PATH), low_memory=False, on_bad_lines="skip")
    logger.info("run_raw_validation: loaded %d rows x %d columns", len(df), df.shape[1])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    issues = _run_validator_checks(df)
    if issues:
        logger.warning(
            "run_raw_validation: %d issues found in raw data (cleaning will address these):",
            len(issues),
        )
        for issue in issues:
            logger.warning("  - %s", issue)
    else:
        logger.info("run_raw_validation: no issues found in raw data")
    return issues


# -- cleaning stage ---------------------------------------------------------

def run_cleaning() -> pd.DataFrame:
    """
    Extended cleaning stage using _ExtendedCleaner (subclass of teammate's Cleaner).

    Differences from the original cleaning.py run_cleaning():
    - Reads raw CSV with on_bad_lines="skip" (handles 73 fused rows).
    - Uses _ExtendedCleaner which adds new steps and fixes existing ones.
    - Calls _run_validator_checks (post-clean) to confirm fixes; skips the 3
      label-dependent checks that raise KeyError without a label column.
    - Writes a structured cleaning log with step table and categorised issues.
    """
    logger.info("run_cleaning: loading raw data...")
    df = pd.read_csv(str(RAW_PATH), low_memory=False, on_bad_lines="skip")
    logger.info(f"run_cleaning: loaded {len(df):,} rows x {df.shape[1]} columns")

    cleaner = _ExtendedCleaner(df)
    clean_df = cleaner.run_all()
    logger.info(f"run_cleaning: cleaning complete -- {len(clean_df):,} rows remaining")

    assert clean_df["Close"].isnull().sum() == 0, "Close still has nulls after cleaning!"
    assert (clean_df["High"] < clean_df["Low"]).sum() == 0, "High < Low still present!"
    assert clean_df.duplicated().sum() == 0, "Duplicates still present!"
    logger.info("run_cleaning: post-cleaning checks passed")

    remaining_issues = _run_validator_checks(clean_df)
    if remaining_issues:
        logger.warning(f"run_cleaning: {len(remaining_issues)} issues still flagged after cleaning:")
        for issue in remaining_issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("run_cleaning: full validation passed -- no issues remaining")

    CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(str(CLEANED_PATH), index=False)
    logger.info(f"run_cleaning: saved cleaned data to {CLEANED_PATH}")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_cleaning_log(str(LOG_PATH), df, clean_df, cleaner, remaining_issues)
    logger.info(f"run_cleaning: log saved to {LOG_PATH}")

    return clean_df


# ── labeling ───────────────────────────────────────────────────────────────

def run_labeling(cleaned_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Label the cleaned dataset using the teammate's fixed label() and write to labeled CSV."""
    if cleaned_df is None:
        logger.info("run_labeling: loading cleaned data from %s", CLEANED_PATH)
        cleaned_df = pd.read_csv(CLEANED_PATH)

    logger.info("run_labeling: applying triple-barrier labels (N=%d, rolling_window=20)...", LABEL_LOOKAHEAD_N)
    labeled_df = _label(cleaned_df, N=LABEL_LOOKAHEAD_N, M=2)

    LABELED_PATH.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(str(LABELED_PATH), index=False)

    dist = labeled_df["label"].value_counts().to_string()
    logger.info("run_labeling: done -- label distribution:\n%s", dist)
    return labeled_df


# ── splitting ──────────────────────────────────────────────────────────────

def run_splitting(labeled_df: pd.DataFrame | None = None, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split labeled data into train_val / test using the teammate's temporal_split and write both CSVs."""
    if labeled_df is None:
        logger.info("run_splitting: loading labeled data from %s", LABELED_PATH)
        labeled_df = pd.read_csv(str(LABELED_PATH), parse_dates=["Date"])
    else:
        labeled_df = labeled_df.copy()
        labeled_df["Date"] = pd.to_datetime(labeled_df["Date"])

    train_val, test = temporal_split(labeled_df, test_size=test_size)
    save_splits(train_val, test)
    logger.info(
        "run_splitting: complete -- train_val %d rows, test %d rows",
        len(train_val), len(test),
    )
    return train_val, test


# ── full pipeline ──────────────────────────────────────────────────────────

def run_preprocessing() -> None:
    """Run the complete preprocessing pipeline.

    Order:
        Stage 0 -- raw validation    (catch problems before cleaning)
        Stage 1 -- cleaning          (fix them)
        Stage 2 -- post-clean validation  (confirm fixes, embedded in cleaning log)
        Stage 3 -- labeling          (on clean data — no alignment risk)
        Stage 4 -- splitting
    """
    sep = "=" * 60
    logger.info(sep)
    logger.info("PREPROCESSING PIPELINE START")
    logger.info(sep)

    # Stage 0 -- raw validation (catch problems before cleaning)
    logger.info("--- Stage 0: Raw Validation ---")
    run_raw_validation()

    # Stage 1+2 -- cleaning + post-clean validation
    logger.info("--- Stage 1: Cleaning ---")
    logger.info("--- Stage 2: Post-clean Validation ---")
    clean_df = run_cleaning()

    # Stage 3 -- labeling on clean data (indices alignment bug fixed)
    logger.info("--- Stage 3: Labeling ---")
    labeled_df = run_labeling(clean_df)

    # Stage 4 -- splitting (trading-day-aware cutoff + lookahead buffer)
    logger.info("--- Stage 4: Splitting ---")
    labeled_df["Date"] = pd.to_datetime(labeled_df["Date"])
    run_splitting(labeled_df)

    logger.info(sep)
    logger.info("PREPROCESSING PIPELINE COMPLETE")
    logger.info("Outputs:")
    logger.info("  %s", CLEANED_PATH)
    logger.info("  %s", LABELED_PATH)
    logger.info("  %s", PROCESSED_PATH / "train_val.csv")
    logger.info("  %s", PROCESSED_PATH / "test.csv")
    logger.info(sep)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_preprocessing()
