"""Configuration for the LightGBM ranking stage."""

from pathlib import Path

from retrieval import config as retrieval_config
from shared import config as shared_config


PROJECT_ROOT = shared_config.PROJECT_ROOT
RAW_DATA_DIR = shared_config.RAW_DATA_DIR
TRAIN_FRACTION = shared_config.TRAIN_FRACTION
VALIDATION_FRACTION = shared_config.VALIDATION_FRACTION
POSITIVE_RATING_THRESHOLD = shared_config.POSITIVE_RATING_THRESHOLD
FINAL_EMBEDDING_DIM = retrieval_config.FINAL_EMBEDDING_DIM

RANKING_DATA_DIR = shared_config.DATA_ROOT / "ranking"
RANKER_ARTIFACT_DIR = shared_config.PROJECT_ROOT / "artifacts" / "ranker"
PUSHED_RANKER_DIR = RANKER_ARTIFACT_DIR / "pushed"
RETRIEVAL_ARTIFACT_DIR = retrieval_config.RETRIEVAL_ARTIFACT_DIR

RANKING_CANDIDATES_PER_USER = 1000
RANDOM_SEED = 42

OBSERVED_CANDIDATE_WEIGHT = 1.0
UNOBSERVED_CANDIDATE_WEIGHT = 0.3

TRAIN_FILE = RANKING_DATA_DIR / "ranking_train.parquet"
VALIDATION_FILE = RANKING_DATA_DIR / "ranking_val.parquet"
TEST_FILE = RANKING_DATA_DIR / "ranking_test.parquet"
MODEL_FILE = RANKER_ARTIFACT_DIR / "lgbm_ranker.txt"
FEATURES_FILE = RANKER_ARTIFACT_DIR / "features.json"
METRICS_FILE = RANKER_ARTIFACT_DIR / "metrics.json"
END_TO_END_METRICS_FILE = RANKER_ARTIFACT_DIR / "end_to_end_metrics.json"
END_TO_END_CANDIDATES_FILE = RANKER_ARTIFACT_DIR / "end_to_end_scored_candidates.parquet"
PUSHED_MODEL_FILE_NAME = "lgbm_ranker.txt"
PUSHED_JOBLIB_FILE_NAME = "lgbm_ranker.joblib"
PUSHED_FEATURES_FILE_NAME = "features.json"
PUSHED_METRICS_FILE_NAME = "metrics.json"
PUSHED_END_TO_END_METRICS_FILE_NAME = "end_to_end_metrics.json"
MOVIE_EMBEDDINGS_FILE = retrieval_config.MOVIE_EMBEDDINGS_FILE
USER_EMBEDDINGS_FILE = retrieval_config.USER_EMBEDDINGS_FILE
ANN_INDEX_FILE = retrieval_config.ANN_INDEX_FILE
RETRIEVAL_METRICS_FILE = retrieval_config.RETRIEVAL_METRICS_FILE
RETRIEVAL_EVAL_CANDIDATES_FILE = retrieval_config.RETRIEVAL_EVAL_CANDIDATES_FILE

MIN_END_TO_END_NDCG_AT_10 = 0.0
MIN_END_TO_END_RECALL_AT_100 = 0.0

KUBEFLOW_RANKING_PIPELINE_NAME = "movielens-ranking-pipeline"
KUBEFLOW_RANKING_PIPELINE_FILE = (
    shared_config.PROJECT_ROOT / f"{KUBEFLOW_RANKING_PIPELINE_NAME}.yaml"
)
KUBEFLOW_RANKING_IMAGE = "movielens-ranking:latest"
KUBEFLOW_RANKING_PIPELINE_ROOT = "/tmp/movielens-ranking-pipeline"
KUBEFLOW_RAW_DATA_DIR = "/data/ml-100k"

LIGHTGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 30,
    "max_depth": -1,
    "min_child_samples": 40,
    "subsample": 0.7,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


def latest_numeric_subdir(parent: Path) -> Path:
    subdirs = [path for path in parent.iterdir() if path.is_dir() and path.name.isdigit()]
    if not subdirs:
        raise FileNotFoundError(f"No numeric run directories found under {parent}")
    return sorted(subdirs, key=lambda path: int(path.name))[-1]


def latest_retrieval_model_dir() -> Path:
    return latest_numeric_subdir(retrieval_config.PIPELINE_ROOT / "Trainer" / "model") / "Format-Serving"


def latest_transform_graph_dir() -> Path:
    return latest_numeric_subdir(retrieval_config.PIPELINE_ROOT / "Transform" / "transform_graph")
