"""Shared project configuration used across retrieval and ranking."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def relpath(path: "str | Path") -> str:
    """Stringify ``path`` relative to the repo root when possible.

    Keeps metrics/manifest JSON portable across machines instead of baking in
    absolute paths like ``C:\\Users\\...`` or ``/Users/...``.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()

RAW_DATA_DIR = PROJECT_ROOT / "ml-100k"
DATA_ROOT = PROJECT_ROOT / "data"

TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10
POSITIVE_RATING_THRESHOLD = 4
NEUTRAL_RATING = 3
