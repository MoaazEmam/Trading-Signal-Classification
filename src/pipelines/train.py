import logging

from src.data.ingestion import run_ingestion
from src.data.preprocessing import run_cleaning, run_labeling, run_splitting
from src.data.validation import run_validation
from src.features.engineering import run_engineering
from src.features.selection_runner import run_selection
from src.features.transform import run_transform
from src.models.trainer import train_all

logger = logging.getLogger(__name__)


def run_train_pipeline() -> tuple[object, object, dict]:
    sep = "=" * 60
    logger.info(sep)
    logger.info("TRAINING PIPELINE START")
    logger.info(sep)

    raw_df = run_ingestion()
    run_validation(raw_df, "post-ingestion")

    clean_df = run_cleaning(raw_df)
    run_validation(clean_df, "post-cleaning")

    labeled_df = run_labeling(clean_df)
    run_validation(labeled_df, "post-labeling")

    feat_df = run_engineering(labeled_df)

    train_df, test_df = run_splitting(labeled_df=feat_df)

    train_t, test_t = run_transform(train_df=train_df, test_df=test_df)

    train_s, test_s = run_selection(train_df=train_t, test_df=test_t)

    logger.info(sep)
    logger.info("DATA PREPROCESSING COMPLETE")
    logger.info("  train_selected: %d rows x %d cols", *train_s.shape)
    logger.info("  test_selected : %d rows x %d cols", *test_s.shape)

    training_results = train_all(train_df=train_s, test_df=test_s)
    logger.info(sep)

    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info(sep)
    return train_s, test_s, training_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    run_train_pipeline()
