.PHONY: install lint format test test-unit test-integration type-check \
        ingest clean-data label engineer split transform select train\
        train-pipeline validate clean app mlflow-log mlflow-server evaluate \
        predict full-pipeline

# dev

install:
	poetry install

lint:
	poetry run ruff check --fix src/ tests/

format:
	poetry run black src/ tests/

test:
	poetry run pytest tests/ -v --cov=src --cov-fail-under=60

test-unit:
	poetry run pytest tests/unit/ -v --cov=src/models --cov-fail-under=60

test-integration:
	poetry run pytest tests/integration/ -v --cov=src --cov-fail-under=60

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
predict: 	predictions/latest.json

evaluate:
	poetry run python -m src.models.run_evaluation

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

models/artifacts/model_comparison.csv &: models/artifacts/training_results.json data/processed/train_val_transformed.csv data/processed/test_transformed.csv src/models/run_evaluation.py
	poetry run python -m src.models.run_evaluation

# pipelines

train-pipeline: models/artifacts/training_results.json
	@echo "Training pipeline complete."

full-pipeline: models/artifacts/model_comparison.csv
	@echo "Full pipeline complete — results in models/artifacts/"

predictions/latest.json &:
	poetry run python -m src.pipelines.predict

# standalone targets

validate:
	poetry run python -m src.data.validation

mlflow-log: models/artifacts/training_results.json
	poetry run python -m src.models.ml_flow

mlflow-server:
	poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

app:
	poetry run streamlit run app/streamlit_app.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f .coverage coverage.xml
	rm -rf htmlcov/
