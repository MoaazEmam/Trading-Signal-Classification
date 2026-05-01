.PHONY: install lint format test test-unit test-integration type-check \
        ingest clean-data label engineer split transform select train\
        train-pipeline validate clean app

# dev

install:
	poetry install

lint:
	poetry run ruff check src/ tests/

format:
	poetry run black src/ tests/

test:
	poetry run pytest tests/ -v

test-unit:
	poetry run pytest tests/unit/ -v

test-integration:
	poetry run pytest tests/integration/ -v

type-check:
	poetry run pyright

# pipeline stages

ingest:    data/raw/market_data_merged.csv
clean-data: data/processed/market_data_cleaned.csv
label:     data/processed/market_data_labeled.csv
engineer:  data/processed/market_data_with_features.csv
split:     data/processed/train_val.csv
transform: data/processed/train_val_transformed.csv
select:    data/processed/train_val_selected.csv
train:     models/artifacts/training_results.json

data/raw/market_data_merged.csv: src/data/ingestion.py
	poetry run python -m src.data.ingestion

data/processed/market_data_cleaned.csv: data/raw/market_data_merged.csv src/data/cleaning.py
	poetry run python -c "from src.data.preprocessing import run_cleaning; run_cleaning()"

data/processed/market_data_labeled.csv: data/processed/market_data_cleaned.csv src/data/labeling.py
	poetry run python -c "from src.data.preprocessing import run_labeling; run_labeling()"

data/processed/market_data_with_features.csv: data/processed/market_data_labeled.csv src/features/engineering.py
	poetry run python -m src.features.engineering

data/processed/train_val.csv data/processed/test.csv &: data/processed/market_data_with_features.csv src/data/splitting.py
	poetry run python -c "from src.data.preprocessing import run_splitting; run_splitting()"

data/processed/train_val_transformed.csv data/processed/test_transformed.csv &: data/processed/train_val.csv data/processed/test.csv src/features/pipeline.py src/features/transform.py src/features/transformers.py
	poetry run python -m src.features.transform

data/processed/train_val_selected.csv data/processed/test_selected.csv &: data/processed/train_val_transformed.csv data/processed/test_transformed.csv src/features/selection_runner.py src/features/features_selection/filter.py src/features/features_selection/importance.py src/features/features_selection/selector.py
	poetry run python -m src.features.selection_runner

models/artifacts/training_results.json &: data/processed/train_val_selected.csv data/processed/test_selected.csv src/models/trainer.py src/pipelines/train.py
	poetry run python -m src.models.trainer

# full training pipeline
# Depends on the final output files — only rebuilds stages whose inputs changed.

train-pipeline: models/artifacts/training_results.json
	@echo "Training pipeline complete."

# standalone validation report

validate:
	poetry run python -m src.data.validation

app:
	poetry run streamlit run app/streamlit_app.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f .coverage coverage.xml
	rm -rf htmlcov/
