# Architecture

Two independent stages joined by exported artifacts.

```text
ml-100k/  ──prepare_data──▶  data/retrieval/{train,val,test}.csv
                                     │
                          TFX pipeline (retrieval/training/pipeline_definition.py)
                                     │
                 tfx_pipeline_output/  (SavedModel, transform graph, MLMD)
                                     │
                          export_artifacts
                                     ▼
        artifacts/retrieval/{user,movie}_embeddings.parquet + movie_ann_index.joblib
                                     │
                          prepare_ranking_data  ──▶  data/ranking/{train,val,test}.parquet
                                     │
                          train_ranker (LightGBM)  ──▶  artifacts/ranker/lgbm_ranker.txt
                                     │
                          evaluate_end_to_end → push_ranker (gate)
                                     ▼
                    artifacts/ranker/pushed/<version>/   (the model registry)
                                     │
                          serving/build_bundle.py
                                     ▼
                    serving/model_bundle/   (committed, ~2.4 MB)
```

## Stage 1 — retrieval (TFX)

`retrieval/training/pipeline_definition.py` wires the standard TFX components:

| Component | Role |
|---|---|
| `CsvExampleGen` | ingest the three chronological CSV splits as TFRecord |
| `StatisticsGen` → `SchemaGen` | profile features, infer a schema |
| `ExampleValidator` | flag anomalies / skew against the schema |
| `Transform` | `tft.compute_and_apply_vocabulary` for the id/categorical/genre features, `tft.bucketize(age, 6)`; emits a **transform graph** reused at serving |
| `Trainer` | the two-tower model (`trainer_module.py`), exports `serving_default` + `user_embedding` + `movie_embedding` signatures |
| `Resolver` + `Evaluator` | TFMA sliced metrics; **blesses** the model only if AUC ≥ 0.6 and it is no worse than the last blessed model |
| `Pusher` | copies the SavedModel to `serving_model/` only when blessed |

Runners: `run_local_pipeline.py` (`LocalDagRunner`, MLMD in `metadata.sqlite`) and
`compile_kubeflow_pipeline.py` (`KubeflowDagRunner` → an Argo/KFP YAML). The cluster
path is compile-only in this repo (placeholder `gs://` roots).

### Two-tower model (`trainer_module.py`)

```text
user tower                              movie tower
  user_id  → Embedding(·, 32)             movie_id → Embedding(·, 32)
  age      → Embedding(6,  8)             genres   → Embedding(·, 12) → mean-pool
  gender   → Embedding(·,  4)
  occ.     → Embedding(·, 12)
  concat → Dropout(0.2) → Dense(32, relu) → L2-normalize      (both towers)

score = sigmoid( 5.0 · dot(user_vec, movie_vec) )
loss  = binary cross-entropy      metrics = AUC, Recall
optimizer = Adam(1e-3)   EarlyStopping(val_auc, patience 1)   ReduceLROnPlateau
```

`export_artifacts.py` runs every user and every movie through the tower signatures,
writes the 32-d vectors to parquet, and fits a cosine `NearestNeighbors` index over
the movie vectors.

## Stage 2 — ranking (LightGBM)

`prepare_ranking_data.py` takes the retrieval top-K per user, joins the observed
labels, adds time-aware features (see [MODELING.md](MODELING.md)), and writes grouped
parquet splits. `train_ranker.py` fits an `LGBMRanker` (`objective=lambdarank`,
`metric=ndcg`) with one group per user and per-row sample weights.

`push_ranker.py` is the release gate: it copies the booster + metrics + a
`manifest.json` into `artifacts/ranker/pushed/<UTC timestamp>/` only when the
end-to-end metrics clear their thresholds **and** beat the previous pushed model's
nDCG@10. `ranking/training/pipeline_definition.py` expresses the same steps as a plain
Kubeflow `ContainerOp` DAG.

## Serving (`serving/`, `ranking/inference.py`)

No TensorFlow at request time — MovieLens 100k is fully in-vocabulary and the towers
are precomputed, so retrieval becomes a lookup:

```text
GET /recommend/{user_id}
  user vector  ← user_embeddings.parquet
  ANN.kneighbors(user_vector, n_candidates)   → candidate movie_ids + cosine scores
  exclude already-rated (from history.parquet)
  build features                              → ranking.features (identical to training)
     · time-aware *_before stats from history.parquet
     · genre affinity, context (hour/day)
     · 4×32 retrieval-vector block from the embedding parquets
  booster.predict(frame[RANKING_FEATURES])    → sort desc → top-k
  join titles                                 → JSON
```

`serving/build_bundle.py` assembles `serving/model_bundle/`:

| File | From |
|---|---|
| `user_embeddings.parquet`, `movie_embeddings.parquet` | `artifacts/retrieval/` (verbatim) |
| `movie_ann_index.joblib` | refitted with the installed scikit-learn |
| `history.parquet`, `movies.parquet` | `shared.feature_tables` (pre-joined, so no raw `ml-100k/` needed) |
| `lgbm_ranker.txt`, `features.json`, `*metrics.json`, `manifest.json` | latest `artifacts/ranker/pushed/<version>/` |

`serving/app.py` loads the bundle once at startup (`lru_cache`d dependency) and fails
fast if it is missing. `docker-compose.yml` runs the API and the Streamlit UI from one
image.

### Deploying the two processes separately

`ui/app.py` is a thin HTTP client (`streamlit`, `pandas`, `requests` only) — it never
imports `fastapi`/`numpy`/`scikit-learn`/`lightgbm`, so it does not need the API's
Docker image or its dependency footprint. On a host that only exposes one process's
memory per instance (e.g. two free Render web services), run:

- **API** — `serving/Dockerfile`, start command `uvicorn serving.app:app --host 0.0.0.0 --port $PORT`.
- **UI** — either the same Docker image with start command
  `streamlit run ui/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`,
  or a slim native-runtime deploy using `ui/requirements.txt` (build command
  `pip install -r ui/requirements.txt`) with the same start command — smaller image,
  faster build, identical runtime memory either way since only imported packages
  consume RAM. Both need `API_URL` set to the API service's URL.

`serving/start.sh` bundles both processes into a single container (API backgrounded
and private on `127.0.0.1:8000`, Streamlit foregrounded and public) for hosts that only
allow one process/instance — note this halves the memory available to each process
compared to deploying them separately.
