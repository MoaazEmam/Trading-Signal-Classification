from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit

from src.data.splitting import (
    N_CV_SPLITS,
    compute_test_cutoff,
    get_time_series_cv,
    run_splitting,
    save_splits,
    temporal_split,
)


class TestComputeTestCutoff:
    def test_cutoff_is_before_max_date(self, splitting_labeled_df):
        cutoff = compute_test_cutoff(splitting_labeled_df)
        assert cutoff < splitting_labeled_df["Date"].max()

    def test_cutoff_is_after_min_date(self, splitting_labeled_df):
        cutoff = compute_test_cutoff(splitting_labeled_df)
        assert cutoff > splitting_labeled_df["Date"].min()

    def test_cutoff_fraction_matches_test_size(self, splitting_labeled_df):
        df = splitting_labeled_df
        test_size = 0.3
        cutoff = compute_test_cutoff(df, test_size=test_size)
        min_date = df["Date"].min()
        max_date = df["Date"].max()
        expected = min_date + (max_date - min_date) * (1 - test_size)
        assert cutoff == expected

    def test_default_test_size_gives_20_pct(self, splitting_labeled_df):
        """Default test_size=0.2 → cutoff at 80% of the time span."""
        df = splitting_labeled_df
        cutoff = compute_test_cutoff(df)
        min_date = df["Date"].min()
        max_date = df["Date"].max()
        expected = min_date + (max_date - min_date) * 0.8
        assert cutoff == expected

    def test_custom_date_column(self, splitting_labeled_df):
        df = splitting_labeled_df.rename(columns={"Date": "timestamp"})
        cutoff = compute_test_cutoff(df, date_col="timestamp")
        assert isinstance(cutoff, pd.Timestamp)


class TestTemporalSplitValidation:
    def test_raises_if_date_col_missing(self, splitting_labeled_df):
        df = splitting_labeled_df.drop(columns=["Date"])
        with pytest.raises(ValueError, match="Date column"):
            temporal_split(df)

    def test_raises_if_label_col_missing(self, splitting_labeled_df):
        df = splitting_labeled_df.drop(columns=["label"])
        with pytest.raises(ValueError, match="Label column"):
            temporal_split(df)

    def test_raises_if_all_labels_nan(self, splitting_unlabeled_df):
        with pytest.raises(ValueError, match="empty after dropping"):
            temporal_split(splitting_unlabeled_df)

    def test_raises_if_train_val_empty(self, splitting_labeled_df):
        """test_size > 1 pushes the cutoff before min_date → train_val is empty."""
        with pytest.raises(ValueError, match="train_val is empty"):
            temporal_split(splitting_labeled_df, test_size=1.1)

    def test_raises_if_test_empty(self, splitting_labeled_df):
        """Negative test_size pushes cutoff beyond max_date → test is empty."""
        with pytest.raises(ValueError, match="test is empty"):
            temporal_split(splitting_labeled_df, test_size=-0.1)


class TestTemporalSplitOutput:
    def test_returns_two_dataframes(self, splitting_labeled_df):
        result = temporal_split(splitting_labeled_df)
        assert isinstance(result, tuple) and len(result) == 2
        train_val, test = result
        assert isinstance(train_val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_combined_row_count_leq_original(self, splitting_labeled_df):
        """
        train_val + test ≤ original because the lookahead buffer removes some
        rows from train_val.
        """
        train_val, test = temporal_split(splitting_labeled_df)
        assert len(train_val) + len(test) <= len(splitting_labeled_df)

    def test_no_date_overlap_between_splits(self, splitting_labeled_df):
        train_val, test = temporal_split(splitting_labeled_df)
        train_dates = set(train_val["Date"])
        test_dates = set(test["Date"])
        assert train_dates.isdisjoint(test_dates)

    def test_all_train_dates_before_all_test_dates(self, splitting_labeled_df):
        train_val, test = temporal_split(splitting_labeled_df)
        assert train_val["Date"].max() < test["Date"].min()

    def test_train_val_is_sorted_by_date(self, splitting_labeled_df):
        train_val, _ = temporal_split(splitting_labeled_df)
        assert train_val["Date"].is_monotonic_increasing

    def test_test_is_sorted_by_date(self, splitting_labeled_df):
        _, test = temporal_split(splitting_labeled_df)
        assert test["Date"].is_monotonic_increasing

    def test_no_nan_labels_in_output(self, splitting_labeled_df):
        """NaN-labeled rows are dropped before splitting."""
        import numpy as np

        df = splitting_labeled_df.copy()
        df.loc[df.index[:5], "label"] = np.nan
        train_val, test = temporal_split(df)
        assert bool(train_val["label"].notna().all())
        assert bool(test["label"].notna().all())

    def test_approximate_test_size_fraction(self, splitting_labeled_df):
        """
        Without lookahead buffer, test rows ≈ test_size of all rows.
        With buffer the fraction is slightly smaller; we just check it's
        in the right ballpark (>5% and <40% for default test_size=0.2).
        """
        train_val, test = temporal_split(splitting_labeled_df, apply_lookahead_buffer=False)
        total = len(train_val) + len(test)
        test_fraction = len(test) / total
        assert 0.05 < test_fraction < 0.40


class TestLookaheadBuffer:
    def test_buffer_removes_rows_from_train_val(self, splitting_labeled_df):
        train_no_buf, _ = temporal_split(splitting_labeled_df, apply_lookahead_buffer=False)
        train_with_buf, _ = temporal_split(splitting_labeled_df, apply_lookahead_buffer=True)
        assert len(train_with_buf) < len(train_no_buf)

    def test_buffer_does_not_affect_test(self, splitting_labeled_df):
        _, test_no_buf = temporal_split(splitting_labeled_df, apply_lookahead_buffer=False)
        _, test_with_buf = temporal_split(splitting_labeled_df, apply_lookahead_buffer=True)
        pd.testing.assert_frame_equal(
            test_no_buf.reset_index(drop=True),
            test_with_buf.reset_index(drop=True),
        )

    def test_buffer_skipped_when_too_few_train_dates(self, splitting_few_dates_df):
        """
        When train_val has <= LOOKAHEAD_DAYS unique dates the buffer branch is
        not entered and no rows are dropped from train_val.
        """
        train_no_buf, _ = temporal_split(splitting_few_dates_df, apply_lookahead_buffer=False)
        train_with_buf, _ = temporal_split(splitting_few_dates_df, apply_lookahead_buffer=True)
        assert len(train_with_buf) == len(train_no_buf)

    def test_buffer_false_preserves_all_train_rows(self, splitting_labeled_df):
        """apply_lookahead_buffer=False → combined rows equal original labeled rows."""

        # no NaN labels in this fixture so combined should equal original
        train_val, test = temporal_split(splitting_labeled_df, apply_lookahead_buffer=False)
        assert len(train_val) + len(test) == len(splitting_labeled_df)


class TestGetTimeSeriesCv:
    def test_returns_time_series_split(self):
        cv = get_time_series_cv()
        assert isinstance(cv, TimeSeriesSplit)

    def test_default_n_splits(self):
        cv = get_time_series_cv()
        assert cv.n_splits == N_CV_SPLITS

    def test_custom_n_splits(self):
        cv = get_time_series_cv(n_splits=3)
        assert cv.n_splits == 3


class TestSaveSplits:
    def test_creates_train_val_file(self, tmp_path, splitting_labeled_df):
        train_val, test = temporal_split(splitting_labeled_df)
        save_splits(train_val, test, output_dir=tmp_path)
        assert (tmp_path / "train_val.csv").exists()

    def test_creates_test_file(self, tmp_path, splitting_labeled_df):
        train_val, test = temporal_split(splitting_labeled_df)
        save_splits(train_val, test, output_dir=tmp_path)
        assert (tmp_path / "test.csv").exists()

    def test_saved_train_val_row_count(self, tmp_path, splitting_labeled_df):
        train_val, test = temporal_split(splitting_labeled_df)
        save_splits(train_val, test, output_dir=tmp_path)
        loaded = pd.read_csv(tmp_path / "train_val.csv")
        assert len(loaded) == len(train_val)

    def test_saved_test_row_count(self, tmp_path, splitting_labeled_df):
        train_val, test = temporal_split(splitting_labeled_df)
        save_splits(train_val, test, output_dir=tmp_path)
        loaded = pd.read_csv(tmp_path / "test.csv")
        assert len(loaded) == len(test)

    def test_creates_output_directory_if_missing(self, tmp_path, splitting_labeled_df):
        nested = tmp_path / "nested" / "output"
        train_val, test = temporal_split(splitting_labeled_df)
        save_splits(train_val, test, output_dir=nested)
        assert nested.exists()

    def test_saved_columns_match_input(self, tmp_path, splitting_labeled_df):
        train_val, test = temporal_split(splitting_labeled_df)
        save_splits(train_val, test, output_dir=tmp_path)
        loaded_train = pd.read_csv(tmp_path / "train_val.csv")
        assert set(loaded_train.columns) == set(train_val.columns)


class TestRunSplitting:
    def _write_temp_csv(self, path: Path, df: pd.DataFrame) -> None:
        df.to_csv(path, index=False)

    def test_returns_two_dataframes(self, tmp_path, splitting_labeled_df):
        csv_path = tmp_path / "input.csv"
        self._write_temp_csv(csv_path, splitting_labeled_df)
        train_val, test = run_splitting(
            input_path=str(csv_path),
            output_dir=str(tmp_path / "output"),
        )
        assert isinstance(train_val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_output_files_are_written(self, tmp_path, splitting_labeled_df):
        csv_path = tmp_path / "input.csv"
        out_dir = tmp_path / "output"
        self._write_temp_csv(csv_path, splitting_labeled_df)
        run_splitting(input_path=str(csv_path), output_dir=str(out_dir))
        assert (out_dir / "train_val.csv").exists()
        assert (out_dir / "test.csv").exists()

    def test_custom_test_size_respected(self, tmp_path, splitting_labeled_df):
        """A larger test_size should yield more test rows than the default."""
        csv_path = tmp_path / "input.csv"
        self._write_temp_csv(csv_path, splitting_labeled_df)
        _, test_default = run_splitting(
            input_path=str(csv_path),
            output_dir=str(tmp_path / "out_default"),
            test_size=0.2,
        )
        _, test_larger = run_splitting(
            input_path=str(csv_path),
            output_dir=str(tmp_path / "out_larger"),
            test_size=0.4,
        )
        assert len(test_larger) > len(test_default)
