install:
	poetry install

lint:
	poetry run ruff check src/ tests/

format:
	poetry run ruff format src/ tests/

test:
	poetry run pytest tests/ -v

test-unit:
	poetry run pytest tests/unit/ -v

type-check:
	poetry run pyright

test-integration:
	poetry run pytest tests/integration/ -v

ingest:
	poetry run python -m src.data.ingestion

label:
	poetry run python -m src.data.labeling

validate:
	poetry run python -m src.data.validation

phase2: ingest label validate
	@echo "Phase 2 pipeline complete — check reports/"
split:
	poetry run python -m src.data.splitting
data-clean:
	poetry run python -m src.data.cleaning

preprocess:
	poetry run python -m src.data.preprocessing

# Full pipeline: ingest raw data, then run all preprocessing stages.
# Stages inside preprocess: raw validation → cleaning → post-clean validation → labeling → splitting
pipeline: ingest preprocess

transform:
	poetry run python -m src.features.transform

train:
	poetry run python -m src.models.train

evaluate:
	poetry run python -m src.models.evaluate

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -f .coverage coverage.xml
	rm -rf htmlcov/