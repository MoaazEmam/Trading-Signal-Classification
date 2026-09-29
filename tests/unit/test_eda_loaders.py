"""Unit tests for src/eda/loaders.py"""

from __future__ import annotations

import joblib
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from src.eda import loaders


@pytest.fixture()
def data_dirs(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    artifacts = tmp_path / "artifacts"
    processed.mkdir()
    artifacts.mkdir()
    monkeypatch.setattr(loaders, "PROCESSED_DIR", processed)
    monkeypatch.setattr(loaders, "ARTIFACT_DIR", artifacts)

    encoder = LabelEncoder().fit(["Buy", "Hold", "Sell"])
    joblib.dump(encoder, artifacts / "label_encoder.pkl")

    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    companies = ["AAPL", "MSFT", "AAPL", "MSFT"]
    pd.DataFrame(
        {"Company": companies, "rsi": [0.1, -0.2, 0.3, 0.0], "label": [0, 1, 2, 0]}
    ).to_csv(processed / "train_val_selected.csv", index=False)
    pd.DataFrame({"Date": dates, "Company": companies}).to_csv(
        processed / "train_val.csv", index=False
    )
    pd.DataFrame(
        {
            "Date": pd.date_range("2023-12-30", periods=8, freq="D"),
            "Company": ["AAPL"] * 8,
            "rsi": range(8),
            "unused": range(8),
            "label": ["Buy"] * 8,
        }
    ).to_csv(processed / "market_data_with_features.csv", index=False)
    return processed


def test_selected_and_split_paths():
    assert loaders._selected_path("train").name == "train_val_selected.csv"
    assert loaders._selected_path("test").name == "test_selected.csv"
    assert loaders._split_path("train").name == "train_val.csv"
    assert loaders._split_path("test").name == "test.csv"


def test_decode_labels_maps_back_to_names(data_dirs):
    y = pd.Series([0, 1, 2], index=[5, 6, 7], name="label")
    decoded = loaders.decode_labels(y)
    assert list(decoded) == ["Buy", "Hold", "Sell"]
    assert list(decoded.index) == [5, 6, 7]
    assert decoded.name == "label"


def test_load_selected_decodes_by_default(data_dirs):
    df = loaders.load_selected("train")
    assert list(df["label"]) == ["Buy", "Hold", "Sell", "Buy"]


def test_load_selected_can_skip_decoding(data_dirs):
    df = loaders.load_selected("train", decode=False)
    assert list(df["label"]) == [0, 1, 2, 0]


def test_load_with_dates_attaches_date_first(data_dirs):
    df = loaders.load_with_dates("train")
    assert df.columns[0] == "Date"
    assert df["Date"].iloc[0] == pd.Timestamp("2024-01-01")


def test_load_with_dates_rejects_row_count_mismatch(data_dirs):
    anchor = pd.read_csv(data_dirs / "train_val.csv")
    anchor.iloc[:3].to_csv(data_dirs / "train_val.csv", index=False)
    with pytest.raises(ValueError, match="row count mismatch"):
        loaders.load_with_dates("train")


def test_load_with_dates_rejects_misaligned_companies(data_dirs):
    anchor = pd.read_csv(data_dirs / "train_val.csv")
    anchor["Company"] = anchor["Company"][::-1].to_numpy()
    anchor.to_csv(data_dirs / "train_val.csv", index=False)
    with pytest.raises(ValueError, match="does not align"):
        loaders.load_with_dates("train")


def test_load_raw_for_features_filters_to_split_range(data_dirs):
    df = loaders.load_raw_for_features(["rsi"], "train")
    assert set(df.columns) == {"Date", "Company", "rsi", "label"}
    assert df["Date"].min() == pd.Timestamp("2024-01-01")
    assert df["Date"].max() == pd.Timestamp("2024-01-04")
    assert len(df) == 4
