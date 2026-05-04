# TFX MovieLens Recommender

Two-stage MovieLens 100k recommender:

- Retrieval: TFX two-tower TensorFlow model that generates candidate movies.
- Ranking: LightGBM LambdaRank model that reranks retrieval candidates.

## Setup

Use Python 3.9 or 3.10. TFX 1.14 does not support Python 3.11+.

```bash
pip install -r requirements.txt
```

Windows users can also run:

```bat
setup_env.bat
```

## Run

Train the retrieval pipeline locally:

```bash
python retrieval/training/run_local_pipeline.py
```

Evaluate retrieval:

```bash
python retrieval/evaluate.py
```

Prepare ranking data from retrieval candidates:

```bash
python ranking/training/prepare_ranking_data.py
```

Train and evaluate the ranker:

```bash
python ranking/training/train_ranker.py
python ranking/evaluate_ranker.py
python ranking/evaluate_end_to_end.py
```

Compile Kubeflow pipeline packages:

```bash
python retrieval/training/compile_kubeflow_pipeline.py
python ranking/training/compile_kubeflow_pipeline.py
```

## Layout

```text
retrieval/   TFX retrieval model, candidate generation, retrieval evaluation
ranking/     LightGBM ranking data prep, training, and evaluation
shared/      MovieLens loading and shared feature utilities
data/        Prepared datasets
artifacts/   Models, metrics, and generated outputs
ml-100k/     MovieLens 100k source data
```

## Outputs

- Retrieval metrics: `artifacts/retrieval/retrieval_metrics.json`
- Ranking metrics: `artifacts/ranking/ranking_metrics.json`
- End-to-end metrics: `artifacts/ranking/end_to_end_metrics.json`
- Kubeflow YAMLs: `movielens-recommender-pipeline.yaml`, `movielens-ranking-pipeline.yaml`

The project uses the MovieLens 100k dataset. Cite the dataset if publishing results.
