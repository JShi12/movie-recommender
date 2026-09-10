# Roadmap

Concrete follow-ups, roughly in priority order.

## 1. Lift retrieval off the TFX component layer onto plain KFP

**Why.** The `tfx==1.14` meta-package pins the training environment to Python 3.9–3.10
and drags in a large, brittle dependency set. Keeping `tensorflow_transform`, TFDV and
TFMA as *standalone libraries* preserves data validation and train/serve-skew
prevention while dropping the `tfx` orchestration layer.

**Estimate.** ~5–7 focused days (~1.5–2 weeks calendar).

**Steps.**

1. Standalone transform — a `tft_beam.AnalyzeAndTransformDataset` script reusing the
   `preprocessing_fn` from `retrieval/training/transform_module.py` almost verbatim;
   write the transform graph + transformed TFRecords to a run directory.
2. `run_fn(fn_args: FnArgs)` → a plain `retrieval/training/train.py`: load CSV → apply
   the transform graph → `tf.data` → `build_two_tower_model(...)` (reused unchanged) →
   fit → export the SavedModel with the existing three-signature block (unchanged).
3. Data validation as a TFDV library step, or swap to Great Expectations / pandera.
4. `retrieval/training/validate_and_push.py` — compute AUC, compare to the previous
   pushed model, gate the copy; mirror `ranking/training/push_ranker.py`.
5. `retrieval/training/pipeline_definition.py` → a KFP `ContainerOp` DAG copying the
   pattern already in `ranking/training/pipeline_definition.py`; keep a local runner.
6. Update the directory-layout helpers in `retrieval/config.py` /
   `retrieval/candidates.py` (`Trainer/model/<n>/Format-Serving`,
   `Transform/transform_graph/<n>` disappear).
7. Drop the `tfx` / `kfp==1.8` pins; move `tensorflow` + `tensorflow-transform` +
   `apache-beam` forward so the core installs on Python 3.11+.

**Unaffected:** `build_two_tower_model`, the SavedModel signature export, the entire
ranking stage, `retrieval/evaluate.py` metric logic, `shared/`.

## 2. Online user tower for cold start

Serving currently rejects users absent from `user_embeddings.parquet`. Export the user
tower as its own SavedModel (or a small ONNX graph) and compute the user vector at
request time from raw profile features, falling back to a popularity prior when a user
has no history.

## 3. Experiment tracking

Wire an MLflow file backend into `retrieval/evaluate.py`,
`ranking/training/train_ranker.py` and `ranking/evaluate_end_to_end.py` (params +
metrics + the feature list as an artifact). Keep `artifacts/ranker/pushed/` as the
promotion record; MLflow becomes the run history behind it.

## 4. Retrieval index

Swap scikit-learn `NearestNeighbors` (brute-force cosine) for FAISS or ScaNN so
`n_candidates` and catalogue size stop being linear-scan bound; benchmark recall vs.
latency. A vector database (Vespa, Qdrant) would also carry the movie metadata.

## 5. Evaluation depth

Bootstrap confidence intervals on the headline metrics; slice by user activity level
and by movie popularity; add a diversity / catalogue-coverage objective to the
ranking sweep.
