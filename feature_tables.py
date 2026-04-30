"""Shared raw feature tables used by retrieval and ranking utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from prepare_data import GENRE_COLUMNS, load_movielens_100k


def load_joined_movielens(raw_data_dir: Path = config.RAW_DATA_DIR) -> pd.DataFrame:
    """Load and join raw ratings, users, and movies with movie metadata."""
    ratings, users, movies = load_movielens_100k(raw_data_dir)
    movies = movies.copy()
    movies["release_year"] = pd.to_datetime(
        movies["release_date"],
        errors="coerce",
        dayfirst=True,
    ).dt.year
    movies["release_year"] = movies["release_year"].fillna(movies["release_year"].median())
    movies["genres"] = movies.apply(active_genres, axis=1)

    data = ratings.merge(users, on="user_id", how="left")
    data = data.merge(movies, on="movie_id", how="left")
    data["label"] = pd.NA
    data.loc[data["rating"] >= config.POSITIVE_RATING_THRESHOLD, "label"] = 1
    data.loc[data["rating"] <= 2, "label"] = 0
    return data


def active_genres(row: pd.Series) -> str:
    active = [genre for genre in GENRE_COLUMNS if row[genre] == 1]
    return "|".join(active) if active else "unknown"


def movie_feature_table(raw_data_dir: Path = config.RAW_DATA_DIR) -> pd.DataFrame:
    """Return one row per movie with static movie metadata."""
    _, _, movies = load_movielens_100k(raw_data_dir)
    movies = movies.copy()
    movies["release_year"] = pd.to_datetime(
        movies["release_date"],
        errors="coerce",
        dayfirst=True,
    ).dt.year
    movies["release_year"] = movies["release_year"].fillna(movies["release_year"].median())
    movies["genres"] = movies.apply(active_genres, axis=1)
    keep = ["movie_id", "title", "release_year", "genres"] + GENRE_COLUMNS
    return movies[keep].copy()


def user_feature_table(raw_data_dir: Path = config.RAW_DATA_DIR) -> pd.DataFrame:
    """Return one row per user with profile metadata."""
    _, users, _ = load_movielens_100k(raw_data_dir)
    return users.copy()
