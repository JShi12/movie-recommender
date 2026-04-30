"""Configuration for the LightGBM ranking stage."""

from pathlib import Path
import importlib.util


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_CONFIG_PATH = PROJECT_ROOT / "config.py"
_spec = importlib.util.spec_from_file_location("project_config", _PROJECT_CONFIG_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load project config from {_PROJECT_CONFIG_PATH}")
project_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(project_config)


RANKING_DATA_DIR = project_config.DATA_ROOT / "ranking"
RANKER_ARTIFACT_DIR = project_config.PROJECT_ROOT / "artifacts" / "ranker"
RETRIEVAL_ARTIFACT_DIR = project_config.RETRIEVAL_ARTIFACT_DIR

CANDIDATES_PER_USER = project_config.CANDIDATES_PER_USER
RANDOM_SEED = 42

POSITIVE_WEIGHT = 2.0
STRONG_NEGATIVE_WEIGHT = 1.5
WEAK_NEGATIVE_WEIGHT = 1.0

TRAIN_FILE = RANKING_DATA_DIR / "ranking_train.parquet"
VALIDATION_FILE = RANKING_DATA_DIR / "ranking_val.parquet"
TEST_FILE = RANKING_DATA_DIR / "ranking_test.parquet"
MODEL_FILE = RANKER_ARTIFACT_DIR / "lgbm_ranker.txt"
FEATURES_FILE = RANKER_ARTIFACT_DIR / "features.json"
METRICS_FILE = RANKER_ARTIFACT_DIR / "metrics.json"
MOVIE_EMBEDDINGS_FILE = project_config.MOVIE_EMBEDDINGS_FILE
USER_EMBEDDINGS_FILE = project_config.USER_EMBEDDINGS_FILE
ANN_INDEX_FILE = project_config.ANN_INDEX_FILE
RETRIEVAL_METRICS_FILE = project_config.RETRIEVAL_METRICS_FILE
RETRIEVAL_EVAL_CANDIDATES_FILE = project_config.RETRIEVAL_EVAL_CANDIDATES_FILE

LIGHTGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
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
    return latest_numeric_subdir(project_config.PIPELINE_ROOT / "Trainer" / "model") / "Format-Serving"


def latest_transform_graph_dir() -> Path:
    return latest_numeric_subdir(project_config.PIPELINE_ROOT / "Transform" / "transform_graph")
