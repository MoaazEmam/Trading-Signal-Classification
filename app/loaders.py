from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eda.loaders import load_with_dates
from src.eda.stats import correlation_clusters, feature_summary, mutual_info_table

PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "models" / "artifacts"


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def _get_train(mtime: float) -> pd.DataFrame:
    return load_with_dates("train")


@st.cache_data
def _get_test(mtime: float) -> pd.DataFrame:
    return load_with_dates("test")


@st.cache_data
def _get_mi(mtime: float) -> pd.DataFrame:
    train = _get_train(mtime)
    X = train.drop(columns=["Date", "Company", "label"])
    y = train["label"]
    return mutual_info_table(X, y, n_samples=100_000)


@st.cache_data
def _get_feature_summary(mtime: float) -> pd.DataFrame:
    train = _get_train(mtime)
    X = train.drop(columns=["Date", "Company", "label"])
    return feature_summary(X)


def load_train() -> pd.DataFrame:
    return _get_train(_mtime(PROCESSED_DIR / "train_val_selected.csv"))


def load_test() -> pd.DataFrame:
    return _get_test(_mtime(PROCESSED_DIR / "test_selected.csv"))


def load_mi() -> pd.DataFrame:
    return _get_mi(_mtime(PROCESSED_DIR / "train_val_selected.csv"))


def load_feature_summary() -> pd.DataFrame:
    return _get_feature_summary(_mtime(PROCESSED_DIR / "train_val_selected.csv"))


def load_selector():
    path = ARTIFACT_DIR / "feature_selector.pkl"
    return joblib.load(path) if path.exists() else None
