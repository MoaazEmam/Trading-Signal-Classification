"""
Feature pipeline composition.

build_pipeline() returns a fresh, unfitted sklearn Pipeline wiring up every
stateful transformation applied between the cleaned splits and model input:

    1. GroupedWinsorizer   -- per-Company cap on Volume outliers
    2. ColumnTransformer   -- scale numeric, drop redundant or non-feature
                              columns (Company is kept as a passthrough key
                              so it can be used for group-aware CV; drop it
                              downstream if the model should not see it)

The same builder is called by the feature runner (build_features.py) to fit
on train_val, and later at inference time to reconstruct the object before
loading fitted state from the joblib artifact. Keeping construction in one
place guarantees training and inference see identical pipeline topology.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.features.transformers import GroupedWinsorizer

# -- column groups ---------------------------------------------------------
# Columns present in data/processed/train_val.csv after the stateless stage.

GROUP_COL = "Company"

WINSORIZE_COLS: list[str] = ["Volume"]

NUMERIC_COLS: list[str] = [
    "Open", "High", "Low", "Close", "Volume",
    "vix", "fed_funds_rate", "treasury_10y", "sp500_level",
    "fear_greed_score",
]

# fear_greed_label is a deterministic function of fear_greed_score (see
# _FEAR_GREED_BOUNDS in preprocessing.py); keeping both would duplicate info.
# Date is a raw timestamp; any date-derived feature should be built upstream.
DROP_COLS: list[str] = ["Date", "fear_greed_label"]


# -- builder ---------------------------------------------------------------

def build_pipeline(
    scale: bool = True,
    winsorize_q: float = 0.99,
    encode_company: bool = False,
) -> Pipeline:
    """
    Build the feature pipeline.

    Parameters
    ----------
    scale : bool
        If True, apply StandardScaler to NUMERIC_COLS. Set False for
        tree-based models (XGBoost / LightGBM / RandomForest) which are
        scale-invariant.
    winsorize_q : float
        Upper quantile used by GroupedWinsorizer. 0.99 caps the top 1%
        of Volume per company.
    encode_company : bool
        If True, ordinal-encode the Company column (tickers seen at fit are
        mapped to integers 0..N-1; tickers not seen at fit are encoded as
        -1). If False, Company is passed through as a string so downstream
        code can decide how to handle it (group key, target encoding inside
        the CV loop, or native categorical support in the model).

        Ordinal encoding imposes an arbitrary numeric order on tickers, so
        it is only appropriate for tree-based models where that order is
        treated as a split threshold, not as a magnitude.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted. Call .fit(X_train) then .transform(X_test), and persist
        with joblib.dump so inference uses the exact fitted state.
    """
    winsorizer = GroupedWinsorizer(
        group_col=GROUP_COL,
        cols=WINSORIZE_COLS,
        q=winsorize_q,
        unseen_group_policy="global",
    )

    if encode_company:
        company_transformer = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    else:
        company_transformer = "passthrough"

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler() if scale else "passthrough", NUMERIC_COLS),
            ("group", company_transformer, [GROUP_COL]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    column_transformer.set_output(transform="pandas")

    pipeline = Pipeline(
        steps=[
            ("winsorize", winsorizer),
            ("columns", column_transformer),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def feature_input_columns() -> list[str]:
    """Columns the pipeline expects in X (everything except the label)."""
    return [GROUP_COL] + NUMERIC_COLS + DROP_COLS