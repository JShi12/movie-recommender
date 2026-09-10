# MovieLens two-stage recommender — common tasks.
# `make help` lists targets. Python is whatever `python` resolves to in your env.

PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help setup setup-pipeline data retrieval ranking eval test test-pipeline lint format typecheck serve ui demo clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Install core + dev + serve + ui deps (modern Python)
	$(PYTHON) -m pip install -e ".[dev,serve,ui]"
	pre-commit install

setup-pipeline: ## Install the TFX 1.14 pipeline stack (Python 3.9-3.10 only)
	$(PYTHON) -m pip install -e ".[pipeline,dev]"

data: ## Build MovieLens CSV/parquet splits from ml-100k/ (needs [pipeline])
	$(PYTHON) -m retrieval.training.prepare_data
	$(PYTHON) -m ranking.training.prepare_ranking_data

retrieval: ## Run the TFX retrieval pipeline + export embeddings/ANN index (needs [pipeline])
	$(PYTHON) -m retrieval.training.run_local_pipeline
	$(PYTHON) -m retrieval.export_artifacts

ranking: ## Train the LightGBM ranker (needs [pipeline] for candidate generation)
	$(PYTHON) -m ranking.training.train_ranker

eval: ## Run retrieval, ranking, and end-to-end offline evaluation
	$(PYTHON) -m retrieval.evaluate
	$(PYTHON) -m ranking.evaluate_ranker
	$(PYTHON) -m ranking.evaluate_end_to_end

bundle: ## Rebuild serving/model_bundle/ from the latest artifacts (needs [pipeline] outputs)
	$(PYTHON) -m serving.build_bundle

test: ## Run the fast unit tests (no TFX/TensorFlow required)
	$(PYTHON) -m pytest -m "not pipeline"

test-pipeline: ## Run the slow pipeline smoke tests (needs [pipeline])
	$(PYTHON) -m pytest -m pipeline

lint: ## Lint with ruff
	ruff check .

format: ## Auto-format with ruff
	ruff format .
	ruff check --fix .

typecheck: ## Static type-check with mypy
	mypy

serve: ## Run the FastAPI inference service on :8000
	uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload

ui: ## Run the Streamlit demo UI on :8501 (expects `make serve` running)
	streamlit run ui/app.py

demo: ## Bring up API + UI together via docker compose
	docker compose up --build

clean: ## Remove caches and generated pipeline output
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__ .coverage
	rm -rf tfx_pipeline_output
