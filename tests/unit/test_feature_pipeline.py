"""Unit tests for src/features/pipeline.py and src/features/transform.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features.pipeline import (
    DROP_COLS,
    GLOBAL_WINSORIZE_COLS,
    GROUP_COL,
    GROUPED_WINSORIZE_COLS,
    PASSTHROUGH_COLS,
    ROBUST_COLS,
    STANDARD_COLS,
    build_pipeline,
    feature_input_columns,
)
from src.features.transformers import ColumnDropper, GlobalWinsorizer, GroupedWinsorizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline_df(n: int = 100) -> pd.DataFrame:
    """Minimal DataFrame with all columns the pipeline expects."""
    rng = np.random.default_rng(42)
    companies = ["AAPL"] * (n // 2) + ["MSFT"] * (n // 2)
    data: dict = {"Company": companies}

    all_numeric_cols = list(
        dict.fromkeys(
            GROUPED_WINSORIZE_COLS
            + GLOBAL_WINSORIZE_COLS
            + ROBUST_COLS
            + STANDARD_COLS
            + PASSTHROUGH_COLS
        )
    )
    for col in all_numeric_cols:
        if col == GROUP_COL:
            continue
        data[col] = rng.normal(0, 1, n)

    for col in DROP_COLS:
        if col not in data and col != GROUP_COL:
            data[col] = rng.normal(100, 10, n)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------


class TestBuildPipeline:
    def test_returns_sklearn_pipeline(self):
        pipeline = build_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_step_names(self):
        pipeline = build_pipeline()
        step_names = [s[0] for s in pipeline.steps]
        assert "winsorize_grouped" in step_names
        assert "winsorize_global" in step_names
        assert "drop" in step_names
        assert "columns" in step_names

    def test_winsorize_grouped_is_correct_type(self):
        pipeline = build_pipeline()
        assert isinstance(pipeline.named_steps["winsorize_grouped"], GroupedWinsorizer)

    def test_winsorize_global_is_correct_type(self):
        pipeline = build_pipeline()
        assert isinstance(pipeline.named_steps["winsorize_global"], GlobalWinsorizer)

    def test_drop_step_is_column_dropper(self):
        pipeline = build_pipeline()
        assert isinstance(pipeline.named_steps["drop"], ColumnDropper)

    def test_scale_false_produces_passthrough(self):
        pipeline = build_pipeline(scale=False)
        ct = pipeline.named_steps["columns"]
        transformer_names = [t[0] for t in ct.transformers]
        assert "passthrough_num" in transformer_names

    def test_scale_true_includes_robust_and_standard(self):
        pipeline = build_pipeline(scale=True)
        ct = pipeline.named_steps["columns"]
        transformer_names = [t[0] for t in ct.transformers]
        assert "robust" in transformer_names
        assert "standard" in transformer_names

    def test_encode_company_true_adds_ordinal_encoder(self):
        from sklearn.preprocessing import OrdinalEncoder

        pipeline = build_pipeline(encode_company=True)
        ct = pipeline.named_steps["columns"]
        group_transformer = next(t for t in ct.transformers if t[0] == "group")
        assert isinstance(group_transformer[1], OrdinalEncoder)

    def test_encode_company_false_uses_passthrough(self):
        pipeline = build_pipeline(encode_company=False)
        ct = pipeline.named_steps["columns"]
        group_transformer = next(t for t in ct.transformers if t[0] == "group")
        assert group_transformer[1] == "passthrough"

    def test_pipeline_fits_and_transforms(self):
        df = _make_pipeline_df(n=100)
        pipeline = build_pipeline(scale=True)
        pipeline.fit(df)
        out = pipeline.transform(df)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 100

    def test_pipeline_drops_expected_columns(self):
        df = _make_pipeline_df(n=100)
        pipeline = build_pipeline()
        pipeline.fit(df)
        out = pipeline.transform(df)
        for col in DROP_COLS:
            assert col not in out.columns

    def test_pipeline_output_has_no_date_column(self):
        df = _make_pipeline_df(n=100)
        pipeline = build_pipeline()
        pipeline.fit(df)
        out = pipeline.transform(df)
        assert "Date" not in out.columns

    def test_winsorize_q_param_is_passed(self):
        pipeline = build_pipeline(winsorize_q=0.95)
        gw = pipeline.named_steps["winsorize_grouped"]
        assert gw.q == 0.95


# ---------------------------------------------------------------------------
# feature_input_columns
# ---------------------------------------------------------------------------


class TestFeatureInputColumns:
    def test_returns_list(self):
        cols = feature_input_columns()
        assert isinstance(cols, list)

    def test_returns_strings(self):
        for col in feature_input_columns():
            assert isinstance(col, str)

    def test_includes_group_col(self):
        assert GROUP_COL in feature_input_columns()

    def test_includes_all_constant_lists(self):
        cols = set(feature_input_columns())
        for col in (
            GROUPED_WINSORIZE_COLS
            + GLOBAL_WINSORIZE_COLS
            + ROBUST_COLS
            + STANDARD_COLS
            + PASSTHROUGH_COLS
            + DROP_COLS
        ):
            assert col in cols

    def test_non_empty(self):
        assert len(feature_input_columns()) > 0


# ---------------------------------------------------------------------------
# run_transform
# ---------------------------------------------------------------------------


class TestRunTransform:
    @pytest.fixture()
    def split_dfs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = _make_pipeline_df(n=100)
        df["label"] = ["Hold"] * 50 + ["Buy"] * 25 + ["Sell"] * 25
        train = df.iloc[:80].reset_index(drop=True)
        test = df.iloc[80:].reset_index(drop=True)
        return train, test

    def test_returns_two_dataframes(self, split_dfs):
        from src.features.transform import run_transform

        train_df, test_df = split_dfs
        train_out, test_out = run_transform(train_df=train_df, test_df=test_df)
        assert isinstance(train_out, pd.DataFrame)
        assert isinstance(test_out, pd.DataFrame)

    def test_output_has_label_column(self, split_dfs):
        from src.features.transform import run_transform

        train_df, test_df = split_dfs
        train_out, test_out = run_transform(train_df=train_df, test_df=test_df)
        assert "label" in train_out.columns
        assert "label" in test_out.columns

    def test_label_is_encoded_as_integer(self, split_dfs):
        from src.features.transform import run_transform

        train_df, test_df = split_dfs
        train_out, _ = run_transform(train_df=train_df, test_df=test_df)
        assert pd.api.types.is_integer_dtype(train_out["label"]) or train_out[
            "label"
        ].dtype in [int, np.int64]

    def test_train_row_count_preserved(self, split_dfs):
        from src.features.transform import run_transform

        train_df, test_df = split_dfs
        train_out, _ = run_transform(train_df=train_df, test_df=test_df)
        assert len(train_out) == len(train_df)

    def test_test_row_count_preserved(self, split_dfs):
        from src.features.transform import run_transform

        train_df, test_df = split_dfs
        _, test_out = run_transform(train_df=train_df, test_df=test_df)
        assert len(test_out) == len(test_df)

    def test_scale_false_produces_output(self, split_dfs):
        from src.features.transform import run_transform

        train_df, test_df = split_dfs
        train_out, test_out = run_transform(
            train_df=train_df, test_df=test_df, scale=False
        )
        assert isinstance(train_out, pd.DataFrame)

    def test_saves_files_to_processed_dir(self, split_dfs, tmp_path, monkeypatch):
        from src.features import transform as transform_mod

        monkeypatch.setattr(transform_mod, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(transform_mod, "ARTIFACT_DIR", tmp_path / "artifacts")
        monkeypatch.setattr(
            transform_mod,
            "TRAIN_VAL_OUT",
            tmp_path / "processed" / "train_val_transformed.csv",
        )
        monkeypatch.setattr(
            transform_mod, "TEST_OUT", tmp_path / "processed" / "test_transformed.csv"
        )
        monkeypatch.setattr(
            transform_mod,
            "PIPELINE_PATH",
            tmp_path / "artifacts" / "feature_pipeline.pkl",
        )
        monkeypatch.setattr(
            transform_mod,
            "LABEL_ENCODER_PATH",
            tmp_path / "artifacts" / "label_encoder.pkl",
        )

        train_df, test_df = split_dfs
        transform_mod.run_transform(train_df=train_df, test_df=test_df)

        assert (tmp_path / "processed" / "train_val_transformed.csv").exists()
        assert (tmp_path / "processed" / "test_transformed.csv").exists()
        assert (tmp_path / "artifacts" / "feature_pipeline.pkl").exists()
        assert (tmp_path / "artifacts" / "label_encoder.pkl").exists()
