"""Shared project configuration used across retrieval and ranking."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "ml-100k"
DATA_ROOT = PROJECT_ROOT / "data"

TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10
POSITIVE_RATING_THRESHOLD = 4
NEUTRAL_RATING = 3
