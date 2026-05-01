from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.features.features_selection.selector import FeatureSelector
from src.utils import load_train_test_transformed

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
ARTIFACT_DIR = Path("models/artifacts")

SELECTOR_PATH = ARTIFACT_DIR / "feature_selector.pkl"
TRAIN_VAL_SELECTED_PATH = PROCESSED_DIR / "train_val_selected.csv"
TEST_SELECTED_PATH = PROCESSED_DIR / "test_selected.csv"

LABEL_COL = "label"


def load_selector(path: Path | None = None) -> FeatureSelector:
    if path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        path = project_root / "models" / "artifacts" / "feature_selector.pkl"
    return FeatureSelector.load(path)


def _split_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    y = df[LABEL_COL]
    x = df.drop(columns=[LABEL_COL])
    return x, y


def run_selection(
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    filter_thresholds: tuple[float, float, float] = (1e-4, 0.80, 0.01),
    importance_threshold: float = 0.02,
    min_features: int = 10,
    max_features: int = 15,
    mi_top_n: int | None = None,
    mi_top_pct: float | None = None,
    importance_top_n: int | None = None,
    importance_top_pct: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sep = "=" * 60
    logger.info(sep)
    logger.info("FEATURE SELECTION START")
    logger.info(sep)

    if train_df is not None and test_df is not None:
        train_data, test_data = train_df, test_df
    else:
        train_data, test_data = load_train_test_transformed()
    logger.info("train_val: %d rows x %d cols", *train_data.shape)
    logger.info("test     : %d rows x %d cols", *test_data.shape)

    x_train, y_train = _split_x_y(train_data)
    x_test, y_test = _split_x_y(test_data)

    selector = FeatureSelector(
        filter_thresholds=filter_thresholds,
        importance_threshold=importance_threshold,
        min_features=min_features,
        max_features=max_features,
        mi_top_n=mi_top_n,
        mi_top_pct=mi_top_pct,
        importance_top_n=importance_top_n,
        importance_top_pct=importance_top_pct,
    )

    x_train_out = selector.fit_transform(x_train, y_train)
    x_test_out = selector.transform(x_test)
    logger.info("train_val: %d rows x %d cols", *x_train_out.shape)
    logger.info("test     : %d rows x %d cols", *x_test_out.shape)

    train_out = x_train_out.assign(**{LABEL_COL: y_train.values})
    test_out = x_test_out.assign(**{LABEL_COL: y_test.values})

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(TRAIN_VAL_SELECTED_PATH, index=False)
    test_out.to_csv(TEST_SELECTED_PATH, index=False)
    logger.info("saved %s", TRAIN_VAL_SELECTED_PATH)
    logger.info("saved %s", TEST_SELECTED_PATH)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    selector.save(SELECTOR_PATH)

    logger.info(sep)
    logger.info("FEATURE SELECTION COMPLETE")
    logger.info(sep)

    return train_out, test_out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_selection(mi_top_pct=0.50, importance_top_n=40)
