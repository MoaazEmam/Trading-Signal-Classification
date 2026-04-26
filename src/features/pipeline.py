from __future__ import annotations

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.features.transformers import AdaptiveScaler, GroupedWinsorizer

GROUP_COL = "Company"
WINSORIZE_COLS: list[str] = ["Volume"]


def build_pipeline(
    scale: bool = True,
    winsorize_q: float = 0.99,
    encode_company: bool = False,
) -> Pipeline:
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
            (
                "num",
                AdaptiveScaler() if scale else "passthrough",
                make_column_selector(dtype_include="number"),
            ),
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
    return [GROUP_COL]
