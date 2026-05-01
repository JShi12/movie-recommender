"""Feature engineering for the LightGBM ranking model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import GENRE_COLUMNS, load_movielens_100k
from feature_tables import movie_feature_table, user_feature_table
from ranking.config import project_config


USER_RETRIEVAL_VECTOR_FEATURES = [
    f"user_retrieval_vector_{idx:02d}"
    for idx in range(project_config.FINAL_EMBEDDING_DIM)
]
MOVIE_RETRIEVAL_VECTOR_FEATURES = [
    f"movie_retrieval_vector_{idx:02d}"
    for idx in range(project_config.FINAL_EMBEDDING_DIM)
]

RANKING_FEATURES = [
    "candidate_score",
    "retrieval_rank",
    "user_avg_rating_before",
    "user_rating_count_before",
    "user_like_rate_before",
    "user_activity_gap_log",
    "movie_avg_rating_before",
    "movie_rating_count_before",
    "movie_like_rate_before",
    "movie_popularity_before",
    "release_year",
    "hour_of_day",
    "day_of_week",
    "user_avg_minus_movie_avg",
    "user_genre_affinity",
] + [f"genre_{genre}" for genre in GENRE_COLUMNS] + USER_RETRIEVAL_VECTOR_FEATURES + MOVIE_RETRIEVAL_VECTOR_FEATURES


def load_joined_movielens(raw_data_dir: Path = project_config.RAW_DATA_DIR) -> pd.DataFrame:
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
    data["label"] = np.where(
        data["rating"] >= project_config.POSITIVE_RATING_THRESHOLD,
        1,
        np.where(data["rating"] <= 2, 0, np.nan),
    )
    return data


def active_genres(row: pd.Series) -> str:
    active = [genre for genre in GENRE_COLUMNS if row[genre] == 1]
    return "|".join(active) if active else "unknown"


def add_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    dt = pd.to_datetime(frame["timestamp"], unit="s")
    frame["hour_of_day"] = dt.dt.hour
    frame["day_of_week"] = dt.dt.dayofweek
    return frame


def add_historical_observed_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add time-aware stats for observed interactions."""
    frame = frame.sort_values(["timestamp", "user_id", "movie_id"]).copy()
    frame["_is_like"] = (
        frame["rating"] >= project_config.POSITIVE_RATING_THRESHOLD
    ).astype(float)

    user_group = frame.groupby("user_id", sort=False)
    frame["user_rating_count_before"] = user_group.cumcount()
    frame["user_rating_sum_before"] = user_group["rating"].cumsum() - frame["rating"]
    frame["user_like_count_before"] = user_group["_is_like"].cumsum() - frame["_is_like"]
    frame["user_avg_rating_before"] = frame["user_rating_sum_before"] / frame[
        "user_rating_count_before"
    ].replace(0, np.nan)
    frame["user_like_rate_before"] = frame["user_like_count_before"] / frame[
        "user_rating_count_before"
    ].replace(0, np.nan)
    frame["previous_user_timestamp"] = user_group["timestamp"].shift(1)
    frame["user_activity_gap_log"] = np.log1p(
        (frame["timestamp"] - frame["previous_user_timestamp"]).clip(lower=0)
    ) # log(1+x)

    movie_group = frame.groupby("movie_id", sort=False)
    frame["movie_rating_count_before"] = movie_group.cumcount()
    frame["movie_rating_sum_before"] = movie_group["rating"].cumsum() - frame["rating"]
    frame["movie_like_count_before"] = movie_group["_is_like"].cumsum() - frame["_is_like"]
    frame["movie_avg_rating_before"] = frame["movie_rating_sum_before"] / frame[
        "movie_rating_count_before"
    ].replace(0, np.nan)
    frame["movie_like_rate_before"] = frame["movie_like_count_before"] / frame[
        "movie_rating_count_before"
    ].replace(0, np.nan)
    frame["movie_popularity_before"] = np.log1p(frame["movie_rating_count_before"])

    global_avg = frame["rating"].mean()
    global_like_rate = frame["_is_like"].mean()
    fill_values = {
        "user_avg_rating_before": global_avg,
        "user_like_rate_before": global_like_rate,
        "user_activity_gap_log": 0.0,
        "movie_avg_rating_before": global_avg,
        "movie_like_rate_before": global_like_rate,
    }
    frame = frame.fillna(fill_values)
    return frame.drop(columns=["_is_like"])


def user_genre_preferences(history: pd.DataFrame) -> pd.DataFrame:
    """Compute user genre preferences from positive historical interactions."""
    liked = history[history["rating"] >= project_config.POSITIVE_RATING_THRESHOLD]
    prefs = liked.groupby("user_id")[GENRE_COLUMNS].sum()
    totals = prefs.sum(axis=1).replace(0, np.nan)
    prefs = prefs.div(totals, axis=0).fillna(0.0)
    prefs.columns = [f"user_pref_{genre}" for genre in GENRE_COLUMNS]
    return prefs.reset_index()


def add_genre_affinity(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    prefs = user_genre_preferences(history)
    frame = frame.merge(prefs, on="user_id", how="left")
    pref_cols = [f"user_pref_{genre}" for genre in GENRE_COLUMNS]
    frame[pref_cols] = frame[pref_cols].fillna(0.0)
    frame["user_genre_affinity"] = 0.0
    for genre in GENRE_COLUMNS:
        frame["user_genre_affinity"] += frame[f"user_pref_{genre}"] * frame[genre]
    frame = frame.drop(columns=pref_cols)
    return frame


def finalize_features(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    frame = add_context_features(frame)
    frame = add_genre_affinity(frame, history)
    for genre in GENRE_COLUMNS:
        frame[f"genre_{genre}"] = frame[genre].fillna(0).astype(float)
    frame["user_avg_minus_movie_avg"] = (
        frame["user_avg_rating_before"] - frame["movie_avg_rating_before"]
    )
    for feature in RANKING_FEATURES:
        if feature not in frame.columns:
            frame[feature] = 0.0
    frame[RANKING_FEATURES] = frame[RANKING_FEATURES].replace([np.inf, -np.inf], np.nan)
    frame[RANKING_FEATURES] = frame[RANKING_FEATURES].fillna(0.0)
    return frame


def add_retrieval_embedding_features(
    frame: pd.DataFrame,
    batch_size: int = 4096,
) -> pd.DataFrame:
    """Add retrieval embedding features from exported tower signatures."""
    if frame.empty:
        return frame

    from retrieval_candidates import RetrievalCandidateScorer

    frame = frame.copy()
    scorer = RetrievalCandidateScorer()
    user_array, movie_array, _ = scorer.embedding_pairs(frame, frame, batch_size=batch_size)

    expected_width = project_config.FINAL_EMBEDDING_DIM
    if user_array.shape[1] != expected_width or movie_array.shape[1] != expected_width:
        raise ValueError(
            "Retrieval tower vector width does not match FINAL_EMBEDDING_DIM: "
            f"user={user_array.shape[1]}, movie={movie_array.shape[1]}, "
            f"expected={expected_width}"
        )

    frame[USER_RETRIEVAL_VECTOR_FEATURES] = user_array
    frame[MOVIE_RETRIEVAL_VECTOR_FEATURES] = movie_array
    return frame


def ndcg_at_k(labels: list[float], scores: list[float], k: int) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = np.asarray(labels, dtype=float)[order]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(np.asarray(labels, dtype=float))[::-1][:k]
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(labels: list[float], scores: list[float], k: int) -> float:
    positives = float(np.sum(labels))
    if positives == 0:
        return 0.0
    order = np.argsort(scores)[::-1][:k]
    return float(np.sum(np.asarray(labels)[order])) / positives
