"""Configuration for the retrieval stage and TFX pipeline."""

from shared.config import *  # noqa: F403
from shared import config as shared_config


RETRIEVAL_DATA_DIR = shared_config.DATA_ROOT / "retrieval"
PIPELINE_ROOT = shared_config.PROJECT_ROOT / "tfx_pipeline_output"
METADATA_PATH = PIPELINE_ROOT / "metadata.sqlite"
SERVING_MODEL_DIR = PIPELINE_ROOT / "serving_model"
RETRIEVAL_ARTIFACT_DIR = shared_config.PROJECT_ROOT / "artifacts" / "retrieval"

TRANSFORM_MODULE_FILE = (
    shared_config.PROJECT_ROOT / "retrieval" / "training" / "transform_module.py"
)
TRAINER_MODULE_FILE = (
    shared_config.PROJECT_ROOT / "retrieval" / "training" / "trainer_module.py"
)

PIPELINE_NAME = "movielens_recommender_pipeline"

TRAIN_STEPS = 1500
EVAL_STEPS = 150

BATCH_SIZE = 64
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 1

LEARNING_RATE = 0.001
DROPOUT_RATE = 0.20
L2_REGULARIZATION = 1e-6

USER_EMBEDDING_DIM = 32
MOVIE_EMBEDDING_DIM = 32
AGE_EMBEDDING_DIM = 8
GENDER_EMBEDDING_DIM = 4
OCCUPATION_EMBEDDING_DIM = 12
GENRE_EMBEDDING_DIM = 12
FINAL_EMBEDDING_DIM = 32

CANDIDATES_PER_USER = 1000
MOVIE_EMBEDDINGS_FILE = RETRIEVAL_ARTIFACT_DIR / "movie_embeddings.parquet"
USER_EMBEDDINGS_FILE = RETRIEVAL_ARTIFACT_DIR / "user_embeddings.parquet"
ANN_INDEX_FILE = RETRIEVAL_ARTIFACT_DIR / "movie_ann_index.joblib"
RETRIEVAL_METRICS_FILE = RETRIEVAL_ARTIFACT_DIR / "retrieval_metrics.json"
RETRIEVAL_EVAL_CANDIDATES_FILE = (
    RETRIEVAL_ARTIFACT_DIR / "retrieval_eval_candidates.parquet"
)

TRAINER_HYPERPARAMETERS = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "learning_rate": LEARNING_RATE,
    "dropout_rate": DROPOUT_RATE,
    "l2_regularization": L2_REGULARIZATION,
    "user_embedding_dim": USER_EMBEDDING_DIM,
    "movie_embedding_dim": MOVIE_EMBEDDING_DIM,
    "age_embedding_dim": AGE_EMBEDDING_DIM,
    "gender_embedding_dim": GENDER_EMBEDDING_DIM,
    "occupation_embedding_dim": OCCUPATION_EMBEDDING_DIM,
    "genre_embedding_dim": GENRE_EMBEDDING_DIM,
    "final_embedding_dim": FINAL_EMBEDDING_DIM,
}

KUBEFLOW_PIPELINE_NAME = "movielens-recommender-pipeline"
KUBEFLOW_PIPELINE_FILE = (
    shared_config.PROJECT_ROOT / f"{KUBEFLOW_PIPELINE_NAME}.yaml"
)
KUBEFLOW_NAMESPACE = "kubeflow"
KUBEFLOW_TFX_IMAGE = "tensorflow/tfx:1.14.0"
