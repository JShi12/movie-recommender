"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shared.movielens import GENRE_COLUMNS

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_raw_dir() -> Path:
    """Directory holding a ~25-row MovieLens 100k lookalike (u.data/u.user/u.item)."""
    return FIXTURES / "ml-100k-mini"


@pytest.fixture
def observed_history() -> pd.DataFrame:
    """A small, chronologically ordered interaction log with genre one-hot columns.

    Two users, deterministic ratings, one movie per genre pattern, so that
    time-aware feature math can be checked by hand.
    """
    rows = [
        # user, movie, rating, timestamp
        (1, 10, 5, 100),
        (1, 11, 3, 200),
        (1, 12, 4, 300),
        (1, 13, 2, 400),
        (2, 10, 4, 150),
        (2, 12, 5, 250),
        (2, 14, 1, 350),
    ]
    frame = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    # Genre one-hots: movie 10 -> Action, 11 -> Comedy, 12 -> Action+Comedy,
    # 13 -> Drama, 14 -> Drama.
    genre_map = {
        10: {"Action"},
        11: {"Comedy"},
        12: {"Action", "Comedy"},
        13: {"Drama"},
        14: {"Drama"},
    }
    for genre in GENRE_COLUMNS:
        frame[genre] = frame["movie_id"].map(lambda mid, g=genre: 1 if g in genre_map[mid] else 0)
    return frame


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
