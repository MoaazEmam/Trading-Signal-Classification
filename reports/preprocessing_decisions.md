# Preprocessing Pipeline — Design Decisions & Fixes
**Author:** Ruaa  
**File:** `src/data/preprocessing.py`  
**Run:** `python -m src.data.preprocessing`

---

## Overview

Two teammates' files were patched directly to fix bugs that could not be
worked around from outside:
- **`labeling.py`** — indices alignment bug fixed at source (see Stage 3).
- **`splitting.py`** — Unicode `→` characters replaced with `->` (Windows
  `cp1252` encoding crash).

All other teammates' files (`cleaning.py`, `ingestion.py`,
`validation_helper.py`, `config.py`) are **left completely untouched**.

For `cleaning.py`, the approach is to **subclass and wrap**: every fix lives
in `_ExtendedCleaner(Cleaner)` inside `preprocessing.py`.

The pipeline runs in five stages:

```
Raw CSV
   ↓
[Stage 0: Raw Validation]        ← catch problems before touching data
   ↓
[Stage 1: Cleaning]              ← fix them (_ExtendedCleaner)
   ↓
[Stage 2: Post-clean Validation] ← confirm fixes (embedded in cleaning log)
   ↓
[Stage 3: Labeling]              ← on clean data — no alignment risk
   ↓
[Stage 4: Splitting]             ← train_val + test
```

**Orchestrator entry point:** [`preprocessing.py` L555 — `run_preprocessing()`](../src/data/preprocessing.py#L555)

---

## Stage 0 — Raw Validation

> **Code:** [`preprocessing.py` L442–L481 — `run_raw_validation()`](../src/data/preprocessing.py#L442-L481)

Before any cleaning is done, the raw CSV is loaded and the same 10
label-independent `Validator` checks are run. This creates a clear before
picture of what problems exist in raw data, so cleaning decisions can be
justified against it.

Results are logged as WARNINGs. The same checks then run again after cleaning
(Stage 2) so you can see exactly which issues were resolved.

The 3 label-dependent checks (`check_class_distribution`,
`check_label_consistency`, `check_label_leakage`) are still skipped — the raw
data has no `label` column either.

---

## Stage 1 — Cleaning

### The Approach: `_ExtendedCleaner` subclass

> **Code:** [`preprocessing.py` L69 — `class _ExtendedCleaner(Cleaner)`](../src/data/preprocessing.py#L69)

Instead of editing `cleaning.py`, we create `_ExtendedCleaner(Cleaner)` — a
subclass that overrides broken methods and adds new ones. The teammate's
`Cleaner` is imported and extended, so all their logic still runs;
we only replace or add what's needed.

---

### Fix 1 — `fix_dtypes`: Don't title-case Company (ticker symbols)

> **Code:** [`preprocessing.py` L77–L128 — `_ExtendedCleaner.fix_dtypes()`](../src/data/preprocessing.py#L77-L128)

**Problem:** The original `fix_dtypes` applies `.str.title()` to all string
columns, including `Company`. That converts `"AAPL"` → `"Aapl"`, `"GOOGL"` → `"Googl"`, etc.
Ticker symbols are identifiers, not natural-language text — title-casing them
corrupts groupby operations and joins downstream.

**Fix:** `Company` column gets `.str.strip()` only (whitespace removal).
Title-casing is kept for `fear_greed_label` and `label` since those are
natural-language category values.

---

### Fix 2 — `drop_missing`: Don't treat `label` as a critical column

> **Code:** [`preprocessing.py` L130–L165 — `_ExtendedCleaner.drop_missing()`](../src/data/preprocessing.py#L130-L165)

**Problem:** The original `drop_missing` marks `label` as a critical column,
meaning any row with a null `label` is quarantined. But at the cleaning stage,
the `label` column does not exist yet — labeling runs in Stage 3. The result
was that **every single row was dropped**, producing an empty dataset.

**Fix:** Critical columns are reduced to only `["Date", "Company", "Close"]`.
`label` is removed from the critical list entirely. Non-critical columns
(`vix`, `fed_funds_rate`, `fear_greed_score`, etc.) with low missingness (<5%)
are still dropped per the original logic.

---

### Fix 3 — `drop_invalid_prices`: Add Open outside High/Low check

> **Code:** [`preprocessing.py` L167–L183 — `_ExtendedCleaner.drop_invalid_prices()`](../src/data/preprocessing.py#L167-L183)

**Problem:** The original only checks:
- `High < Low`
- `Close` outside `[Low, High]`

But it never checks whether `Open` is within the day's `[Low, High]` range.
An `Open` outside the bar boundaries is physically impossible in market data
and indicates a data error.

**Fix:** Calls `super().drop_invalid_prices()` to run the parent's three
checks unchanged, then adds one new check: any row where `Open > High` or
`Open < Low` is quarantined. This avoids re-implementing the parent logic.

---

### New Method — `drop_zero_volume_movement`

> **Code:** [`preprocessing.py` L207–L227 — `_ExtendedCleaner.drop_zero_volume_movement()`](../src/data/preprocessing.py#L207-L227)

**Problem:** Some rows show `Volume = 0` but `Close` changed from the
previous trading day. A stock cannot move in price with zero volume —
this is a data recording error, not a legitimate trading halt (which would
show `Volume = 0` and `Close == prev_Close`).

**Fix:** For each company, the previous day's Close is compared. Rows with
`Volume = 0` AND `Close ≠ prev_Close` are quarantined. Zero-volume rows
where price didn't change are kept (those represent genuine no-activity days).

---

### New Method — `handle_price_spikes`

> **Code:** [`preprocessing.py` L228–L252 — `_ExtendedCleaner.handle_price_spikes(threshold=0.50)`](../src/data/preprocessing.py#L228-L252)

**Problem:** Some rows have price jumps >50% in a single day. This could be:
- **Legitimate**: Earnings surprise, FDA approval, acquisition — the move
  persists or reverses slowly.
- **Data error**: The price was entered incorrectly — it reverses sharply
  (>30%) the very next day.

Dropping all >50% moves would remove valid events. Keeping all of them
pollutes the dataset with recording errors.

**Fix:** A spike row is only removed if the *next day reverses >30%*
(indicating a data entry mistake that was corrected). Persistent spikes —
where the next day does not reverse sharply — are kept. This is gated on
`Stock Splits == 0` to avoid misclassifying legitimate split-adjusted prices.

Result: 1 data-error spike removed, 36 legitimate spikes kept.

---

### New Method — `winsorize_outliers`

> **Code:** [`preprocessing.py` L253–L269 — `_ExtendedCleaner.winsorize_outliers()`](../src/data/preprocessing.py#L253-L269)

**Problem:** Volume has extreme right-tail outliers. Some companies have
single-day volume spikes 10x–100x above their normal range (e.g., earnings,
options expiry). The original cleaning drops nothing for Volume outliers.
These don't break the model but inflate feature ranges and skew
normalization.

**Fix:** Volume is capped at the **per-company 99th percentile** (not a
global percentile). This is important because a small-cap company's 99th
percentile volume is very different from a mega-cap's.  
9,100 values were capped across the dataset.

Price columns (`Open`, `High`, `Low`, `Close`) are **not winsorized** — those
extremes are often genuine market events and should remain for labeling.

---

### New Method — `fix_fear_greed_labels`

> **Code:** [`preprocessing.py` L271–L291 — `_ExtendedCleaner.fix_fear_greed_labels()`](../src/data/preprocessing.py#L271-L291)  
> **Canonical bounds:** [`preprocessing.py` L50–L56 — `_FEAR_GREED_BOUNDS`](../src/data/preprocessing.py#L50-L56)  
> **Score mapper:** [`preprocessing.py` L59–L63 — `_score_to_fg_label()`](../src/data/preprocessing.py#L59-L63)

**Problem:** The `fear_greed_label` column (text category) was inconsistent
with `fear_greed_score` (numeric 0–100). For example, a score of 45 was
labeled `"Fear"` instead of `"Neutral"`. There were 42,752 such mismatches.
This happened because the original ingestion used overlapping or shifting
label boundaries.

**Fix:** We define a single canonical non-overlapping mapping:

| Score Range | Label         |
|-------------|---------------|
| 0 – 24      | Extreme Fear  |
| 25 – 44     | Fear          |
| 45 – 55     | Neutral       |
| 56 – 75     | Greed         |
| 76 – 100    | Extreme Greed |

`fear_greed_label` is **fully regenerated** from `fear_greed_score` using
this map. After the fix, mismatches = 0.

---

### Fix 4 — `run_all`: Track per-step row counts for logging

> **Code:** [`preprocessing.py` L184–L205 — `_ExtendedCleaner.run_all()`](../src/data/preprocessing.py#L184-L205)

**Problem:** The original `run_all` runs all steps in sequence but records
no information about how many rows each step removed.

**Fix:** The override adds `self.step_rows: list[tuple[str, int, int]]` —
a list of `(step_name, rows_before, rows_after)` tuples — populated by
recording `len(self.df)` before and after each step. This data feeds the
structured cleaning log.

---

### Fix 5 — Raw CSV read with `on_bad_lines="skip"`

> **Code:** [`preprocessing.py` L455 — `pd.read_csv(..., on_bad_lines="skip")`](../src/data/preprocessing.py#L455) inside [`run_cleaning()` L442](../src/data/preprocessing.py#L442)

**Problem:** The raw CSV produced by `ingestion.py` contains 73 fused rows
where a newline character was missing in the `fear_greed_label` field during
the data merge. These rows have too many columns and cause pandas to raise a
`ParserError` with the default `on_bad_lines="error"`.

**Fix:** `run_cleaning()` reads the CSV with `on_bad_lines="skip"`, which
silently discards the 73 malformed rows rather than crashing.  
This was chosen over patching `ingestion.py` (teammate's file).

---

### Fix 6 — Validator called per-check, not via `run_all()`

> **Code:** [`preprocessing.py` L463–L476 — individual `validator.check_*()` calls in `run_cleaning()`](../src/data/preprocessing.py#L463-L476)

**Problem:** After cleaning, we validate using the teammate's `Validator`.
But `validator.run_all()` calls `check_class_distribution()`,
`check_label_consistency()`, and `check_label_leakage()`, all of which
require a `label` column. At cleaning stage, that column doesn't exist yet,
so all three raise `KeyError` and crash.

**Fix:** Instead of `validator.run_all()`, we call each check individually,
skipping the three label-dependent ones. The 10 safe checks still run:
`check_dtypes`, `check_missing`, `check_duplicates`, `check_company_rows`,
`check_date_gaps`, `check_outliers`, `check_price_spikes`, `check_sanity`,
`check_stale_data`, `check_correlations`.

---

### Cleaning Log — `_write_cleaning_log`

> **Code:** [`preprocessing.py` L310–L413 — `_write_cleaning_log()`](../src/data/preprocessing.py#L310-L413)  
> **Step prefix map:** [`preprocessing.py` L296–L308 — `_STEP_PREFIX_MAP`](../src/data/preprocessing.py#L296-L308)

A structured log is written to `reports/cleaning_log.txt` after every run.
It contains:

1. **Summary table** — input rows, output rows, total removed, rows
   quarantined, and a breakdown of rows removed per step.
2. **Step-by-step detail** — for each cleaning step, the before/after row
   count and the notes logged during that step.
3. **Post-cleaning validation** — remaining issues from the Validator,
   categorised as:
   - Data Coverage (inconsistent company coverage, missing business days)
   - Residual Outliers (expected for equity data; price outliers are kept)
   - Sanity (remaining suspicious price jumps)
   - Correlations (structural patterns like price-column perfect correlations)

**Why do residual issues remain?** The 20 flagged issues after cleaning are
expected:
- Price outliers (7–8%) are intentionally kept — extreme moves are valid
  market events.
- Missing business days (86) are market holidays, not gaps.
- Inconsistent company coverage (1,787 dates) is expected — companies have
  different IPO/delisting dates.
- Price correlations near 1.0 are correct (Open/High/Low/Close are naturally
  correlated).

---

## Stage 2 — Post-clean Validation

> **Code:** [`preprocessing.py` L497–L513 — Validator calls inside `run_cleaning()`](../src/data/preprocessing.py#L497-L513)

Immediately after cleaning, the same 10 checks run again on the cleaned
DataFrame. Results are embedded in `reports/cleaning_log.txt` under
"POST-CLEANING VALIDATION". Any issues still flagged after cleaning are
documented as residual (see "What Was NOT Changed and Why" section below).

---

## Stage 3 — Labeling

### Fix 7 — Indices captured after `dropna`/`iloc` (alignment bug)

> **Code:** [`labeling.py` — `label()`, indices line](../src/data/labeling.py) (fixed directly in source)

**Problem:** In the original `labeling.py`, for each company group the code
did:
```python
indices = group.index           # captured BEFORE filtering
group = group.dropna(...)
group = group.iloc[:-N]
labels[indices[i]] = row_label  # writes to WRONG rows (warmup period rows)
```
`dropna` and `iloc[:-N]` shrink the group, but `indices` still pointed to the
original (larger) index. Label `i` was written to the `i`-th row of the
*original* group — a warmup row — not the row the label was computed from.
This silently assigned labels to the wrong time periods.

**Fix applied directly in `labeling.py`:** `indices = group.index` is now
captured **after** both `dropna` and `iloc[:-N]`, so each label maps to
exactly the row it was computed from.

`run_labeling()` in `preprocessing.py` calls `labeling.label()` directly —
no local copy of the labeling logic.

---

### Fix 8 — Write to labeled CSV path (not raw CSV path)

> **Code:** [`preprocessing.py` L516–L531 — `run_labeling()`](../src/data/preprocessing.py#L516-L531)  
> **Output path constant:** [`preprocessing.py` L38 — `LABELED_PATH`](../src/data/preprocessing.py#L38)

**Problem:** The original `labeling.py` `__main__` block writes the labeled
DataFrame to `data/raw/market_data_merged.csv` — overwriting the raw input
file. Any subsequent run would then process the already-labeled (and
partially corrupted) file.

**Fix:** `run_labeling()` writes to `data/processed/market_data_labeled.csv`,
leaving the raw file untouched.

**Label distribution** (880,129 total):
| Label | Rows    |
|-------|---------|
| Buy   | 358,178 |
| Sell  | 291,327 |
| Hold  | 215,403 |

---

## Stage 4 — Splitting

> **Code:** [`preprocessing.py` L535–L552 — `run_splitting()`](../src/data/preprocessing.py#L535-L552)

`run_splitting()` calls `splitting.temporal_split()` and `splitting.save_splits()`
directly — no local split logic.

Two bugs that originally prevented this were fixed at source in `splitting.py`:

### Fix 9 — Unicode `→` crash on Windows fixed in `splitting.py`

> **Code:** [`splitting.py`](../src/data/splitting.py) — `compute_test_cutoff()` and `save_splits()` print lines

**Problem:** Both `compute_test_cutoff()` and `save_splits()` in the teammate's
file printed an arrow character (`→`) to stdout. On Windows with `cp1252`
encoding this raises `UnicodeEncodeError` and crashes immediately.

**Fix applied directly in `splitting.py`:** Both `→` characters replaced
with `->`. `save_splits()` and `temporal_split()` are now called normally
from `run_splitting()`.

### Fix 10 — Lookahead buffer

> **Shared constant:** [`preprocessing.py` L45 — `LABEL_LOOKAHEAD_N = 10`](../src/data/preprocessing.py#L45)

`temporal_split()` already accepts `apply_lookahead_buffer=True` (the
default) and uses `LOOKAHEAD_DAYS = 10`, which matches `LABEL_LOOKAHEAD_N`.
No extra logic needed in `preprocessing.py`.

**Final split:**
| Set       | Rows    |
|-----------|---------|
| train_val | 688,527 |
| test      | 171,601 |

---

## What Was NOT Changed (and Why)

| Issue | Decision |
|---|---|
| 86 missing business days | These are market holidays — no forward fill needed |
| 1,787 dates with inconsistent coverage | Expected: different companies have different active date ranges |
| Price column outliers (~7%) | Kept intentionally — extreme price moves are real market events |
| 36 residual suspicious price spikes | Persistent moves (no reversal next day) — likely legitimate events |
| Ingestion fused rows (73) | Fixed at read time with `on_bad_lines="skip"`, not in `ingestion.py` |
| `splitting.py` calendar-day cutoff | Accepted — `temporal_split()` cutoff is close enough for this dataset |

---

## File Outputs

| File | Description |
|------|-------------|
| `data/processed/market_data_cleaned.csv` | 880,129 rows after all cleaning steps |
| `data/processed/market_data_labeled.csv` | Same rows with `label` column (Buy/Sell/Hold) |
| `data/processed/train_val.csv` | 688,527 rows for training and validation |
| `data/processed/test.csv` | 171,601 rows held out for final evaluation |
| `reports/cleaning_log.txt` | Structured log of every cleaning decision |
