"""Shared configuration defaults for the MovieLens TFX recommender pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA_DIR = PROJECT_ROOT / "ml-100k"
DATA_ROOT = PROJECT_ROOT / "data"
PIPELINE_ROOT = PROJECT_ROOT / "tfx_pipeline_output"
METADATA_PATH = PIPELINE_ROOT / "metadata.sqlite"
SERVING_MODEL_DIR = PIPELINE_ROOT / "serving_model"

TRANSFORM_MODULE_FILE = PROJECT_ROOT / "transform_module.py"
TRAINER_MODULE_FILE = PROJECT_ROOT / "trainer_module.py"

PIPELINE_NAME = "movielens_recommender_pipeline"

TRAIN_STEPS = 1500
EVAL_STEPS = 150

USE_USER_AWARE_ATTENTION = False

TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10
POSITIVE_RATING_THRESHOLD = 4
NEUTRAL_RATING = 3

BATCH_SIZE = 64
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 2

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

TRAINER_HYPERPARAMETERS = {
    "use_user_aware_attention": USE_USER_AWARE_ATTENTION,
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
KUBEFLOW_PIPELINE_FILE = PROJECT_ROOT / f"{KUBEFLOW_PIPELINE_NAME}.yaml"
KUBEFLOW_NAMESPACE = "kubeflow"
KUBEFLOW_TFX_IMAGE = "tensorflow/tfx:1.14.0"
