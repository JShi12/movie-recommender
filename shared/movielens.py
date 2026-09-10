"""Shared MovieLens 100k loading and split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from shared import config

GENRE_COLUMNS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


@dataclass(frozen=True)
class SplitFractions:
    train: float = config.TRAIN_FRACTION
    validation: float = config.VALIDATION_FRACTION

    @property
    def test(self) -> float:
        return 1.0 - self.train - self.validation


def load_movielens_100k(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw MovieLens 100k ratings, users, and movies files."""
    ratings = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    users = pd.read_csv(
        data_dir / "u.user",
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        data_dir / "u.item",
        sep="|",
        names=["movie_id", "title", "release_date", "video_release_date", "imdb_url"]
        + GENRE_COLUMNS,
        encoding="latin-1",
    )
    return ratings, users, movies


def active_genres(row: pd.Series) -> str:
    active = [genre for genre in GENRE_COLUMNS if row[genre] == 1]
    return "|".join(active) if active else "unknown"


def time_based_split(
    data: pd.DataFrame,
    fractions: SplitFractions = SplitFractions(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split records chronologically into train/validation/test dataframes.

    A single timestamp never spans two partitions: rows sharing the timestamp
    that lands on a nominal row-count boundary are all assigned to the earlier
    partition. This keeps the split genuinely leak-free for time-based eval.
    """
    if fractions.train <= 0 or fractions.validation <= 0 or fractions.test <= 0:
        raise ValueError(f"Invalid split fractions: {fractions}")

    sorted_data = data.sort_values("timestamp").reset_index(drop=True)
    timestamps = sorted_data["timestamp"].to_numpy()
    n_total = len(sorted_data)

    def _cut(fraction_end: float) -> int:
        nominal = int(n_total * fraction_end)
        if nominal <= 0:
            return 0
        if nominal >= n_total:
            return n_total
        # Advance past every row that shares the boundary timestamp.
        return int(np.searchsorted(timestamps, timestamps[nominal - 1], side="right"))

    train_end = _cut(fractions.train)
    val_end = max(_cut(fractions.train + fractions.validation), train_end)

    train = sorted_data.iloc[:train_end]
    validation = sorted_data.iloc[train_end:val_end]
    test = sorted_data.iloc[val_end:]

    for name, part in (("train", train), ("validation", validation), ("test", test)):
        if part.empty:
            raise ValueError(
                f"Chronological split produced an empty {name} partition; "
                "timestamps are too concentrated for the requested fractions."
            )
    # Whole timestamp groups stay together, so these must hold; keep them as an
    # invariant check on the cut logic above.
    if train["timestamp"].max() >= validation["timestamp"].min():
        raise ValueError("Temporal leakage detected between train and validation")
    if validation["timestamp"].max() >= test["timestamp"].min():
        raise ValueError("Temporal leakage detected between validation and test")

    return train, validation, test
