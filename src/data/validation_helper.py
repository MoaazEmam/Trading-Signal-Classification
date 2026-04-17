import numpy as np
import pandas as pd


class Validator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.issues = []

    # 1. check for missing values (%)
    def check_missing(self):
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100).round(2)
        for col, pct in missing_pct.items():
            if pct > 0:
                sev = "HIGH ⚠️" if pct > 20 else ("MEDIUM ⚠️" if pct > 5 else "LOW ⚠️")
                self.issues.append(f"missing: '{col}' missing {pct}% — {sev}")

    def check_dtypes(self):
        expected_numeric = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "vix",
            "fed_funds_rate",
            "treasury_10y",
            "sp500_level",
            "fear_greed_score",
        ]
        expected_datetime = ["Date"]
        expected_string = ["Company", "fear_greed_label", "label"]

        for col in expected_numeric:
            if col in self.df.columns and not pd.api.types.is_numeric_dtype(self.df[col]):
                self.issues.append(
                    f"Dtype: '{col}' expected numeric but found {self.df[col].dtype} — "
                    f"possible parsing issue (commas, symbols, or mixed values)."
                )

        for col in expected_datetime:
            if col in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df[col]):
                self.issues.append(
                    f"Dtype: '{col}' expected datetime but found {self.df[col].dtype} — "
                    f"Date parsing may have failed."
                )

        for col in expected_string:
            if col in self.df.columns and not pd.api.types.is_object_dtype(self.df[col]):
                self.issues.append(f"Dtype: '{col}' expected string/object but found {self.df[col].dtype}.")

    # 2. duplicates
    def check_duplicates(self):
        n_dup = self.df.duplicated().sum()
        if n_dup > 0:
            self.issues.append(f"Duplicates: {n_dup} fully duplicate rows found.")

        key_dup = self.df.duplicated(
            subset=["Date", "Company"]
        ).sum()  # because date,company should be a unique combination
        if key_dup > 0:
            self.issues.append(f"Duplicates: {key_dup} duplicate (Date, Company) pairs found.")

    # 3. class distribution
    def check_class_distribution(self):
        label_pct = self.df["label"].value_counts(normalize=True)
        if len(label_pct) < 2:
            return
        imbalance_ratio = round(label_pct.max() / label_pct.min(), 2)
        if imbalance_ratio > 1.5:
            self.issues.append(f"Balance: Class imbalance detected — ratio {imbalance_ratio}x.")

    # 4. checks done per company

    def check_company_rows(self):
        n_companies = self.df["Company"].nunique()
        rows_per_date = self.df.groupby("Date").size()
        bad_dates = rows_per_date[rows_per_date != n_companies]
        if len(bad_dates) > 0:
            self.issues.append(f"Date coverage; {len(bad_dates)} dates have inconsistent company coverage.")

    # 5. date time

    def check_date_gaps(self):
        date_min, date_max = self.df["Date"].min(), self.df["Date"].max()
        all_dates = pd.bdate_range(start=date_min, end=date_max)
        present_dates = self.df["Date"].dt.normalize().unique()
        missing_dates = sorted(set(all_dates.normalize()) - set(present_dates))  # type: ignore
        if len(missing_dates) > 20:
            self.issues.append(f"Date gaps: {len(missing_dates)} missing business days.")

    # 6. feature distributions and outliers

    def check_outliers(self):
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        for col in numeric_cols:
            q1, q3 = self.df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            pct_out = round(n_out / len(self.df) * 100, 2)
            if pct_out > 5:
                self.issues.append(f"Outliers: '{col}' has {pct_out}% outliers.")

    # spikes->not logicLy possible for a stock to jump 50%+ in one day without a stock split or major corporate action.
    def check_price_spikes(self):
        # Ensure data is ordered so we compare consecutive days for the same company
        temp_df = self.df.sort_values(["Company", "Date"])

        # pct_change() calculates: (Current - Previous) / Previous
        pct_change = temp_df.groupby("Company")["Close"].pct_change()

        # We only flag if there was NO stock split recorded that day
        mask = (pct_change.abs() > 0.50) & (self.df["Stock Splits"] == 0)
        spike_count = mask.sum()

        if spike_count > 0:
            self.issues.append(f"Sanity: {spike_count} suspicious price jumps >50% without a stock split.")

    # 7. domain specific sanity checks

    def check_sanity(self):
        # Prices positive
        for col in ["Open", "High", "Low", "Close"]:
            n_bad = (self.df[col] <= 0).sum()
            if n_bad > 0:
                self.issues.append(f"Sanity: '{col}' has {n_bad} non-positive values.")
        # High >= Low
        n_hl = (self.df["High"] < self.df["Low"]).sum()
        if n_hl > 0:
            self.issues.append(f"Sanity: {n_hl} rows where High < Low.")
        # Close within High/Low
        n_close = ((self.df["Close"] > self.df["High"]) | (self.df["Close"] < self.df["Low"])).sum()
        if n_close > 0:
            self.issues.append(f"Sanity:{n_close} rows where Close is outside High/Low.")
        # Volume, VIX, Fear/Greed range
        ranges = {"Volume": (1, np.inf), "vix": (5, 100), "fear_greed_score": (0, 100)}
        for col, (lo, hi) in ranges.items():
            n_bad = ((self.df[col] < lo) | (self.df[col] > hi)).sum()
            if n_bad > 0:
                self.issues.append(f"Sanity:{n_bad} rows where '{col}' outside [{lo},{hi}]")
        # Fear/Greed label consistency
        label_map = {
            "Extreme Fear": (0, 25),
            "Fear": (25, 45),
            "Neutral": (45, 55),
            "Greed": (55, 75),
            "Extreme Greed": (75, 100),
        }
        mismatches = 0
        for label_name, (lo, hi) in label_map.items():
            mask = self.df["fear_greed_label"] == label_name
            mismatches += (~self.df["fear_greed_score"].between(lo, hi) & mask).sum()
        if mismatches > 0:
            self.issues.append(f"Sanity:{mismatches} Fear/Greed score-label mismatches.")

    # data likely repeats itself
    def check_stale_data(self):
        stale_mask = (
            (self.df["Open"] == self.df["Close"])
            & (self.df["High"] == self.df["Low"])
            & (self.df["Open"] == self.df["High"])
            & (self.df["Volume"] == 0)
        )
        stale_count = stale_mask.sum()

        if stale_count > 0:
            self.issues.append(f"Sanity: {stale_count} 'stale' rows found (Price frozen + 0 Volume).")

    # 8. Correlations
    def check_correlations(self):
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()

        pearson_corr = self.df[numeric_cols].corr(method="pearson")  # type: ignore
        spearman_corr = self.df[numeric_cols].corr(method="spearman")  # type: ignore

        price_cols = ["Open", "High", "Low", "Close"]

        for i, col1 in enumerate(pearson_corr.columns):
            for col2 in pearson_corr.columns[i + 1 :]:
                p = pearson_corr.loc[col1, col2]
                s = spearman_corr.loc[col1, col2]
                assert isinstance(p, float)
                assert isinstance(s, float)

                # Handle price columns separately-expected
                if col1 in price_cols and col2 in price_cols:
                    self.issues.append(
                        f"Correlation (Expected - Price Features): {col1} ↔ {col2} — "
                        f"Pearson={round(p, 3)}, Spearman={round(s, 3)}"
                    )
                    continue

                # high
                if abs(p) > 0.85 or abs(s) > 0.85:
                    self.issues.append(
                        f"Correlation: {col1} ↔ {col2} — " f"Pearson={round(p, 3)}, Spearman={round(s, 3)}"
                    )

                # non-linear or there are outliers
                if abs(p - s) > 0.1:
                    self.issues.append(
                        f"Correlation divergence: {col1} ↔ {col2} — " f"Pearson={round(p, 3)} vs Spearman={round(s, 3)}"
                    )

    # 9. label consistency
    def check_label_consistency(self):
        multi_label = self.df.groupby(["Date", "Company"])["label"].nunique()
        inconsistent = multi_label[multi_label > 1]
        if len(inconsistent) > 0:
            self.issues.append(f"Consistency: {len(inconsistent)} (Date, Company) pairs have conflicting labels.")

    def check_label_leakage(self):
        # is the target label is too highly correlated with current price?
        # We only check this if the label has been converted to numbers (0 or 1)
        if pd.api.types.is_numeric_dtype(self.df["label"]):
            correlation = self.df["Close"].corr(self.df["label"])  # type: ignore
            if abs(correlation) > 0.90:
                self.issues.append(
                    f"Leakage: 'label' and 'Close' correlation is {round(correlation, 3)}. Target might be leaking!"
                )

    def run_all(self):
        self.check_dtypes()
        self.check_missing()

        self.check_duplicates()
        self.check_company_rows()
        self.check_date_gaps()
        self.check_class_distribution()

        self.check_outliers()
        self.check_price_spikes()

        self.check_sanity()
        self.check_stale_data()

        self.check_correlations()
        self.check_label_consistency()
        self.check_label_leakage()
        return self.issues
