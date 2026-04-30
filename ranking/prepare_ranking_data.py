"""Prepare balanced top-K candidate datasets for LightGBM ranking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from prepare_data import SplitFractions, time_based_split
from ranking import config as ranking_config
from ranking.config import project_config
from ranking.feature_builder import (
    add_retrieval_embedding_features,
    add_historical_observed_features,
    finalize_features,
    load_joined_movielens,
    movie_feature_table,
    user_feature_table,
)
from retrieval_candidates import generate_top_k_candidates


def split_observed_interactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labelled = raw[raw["label"].notna()].copy()
    return time_based_split(
        labelled,
        SplitFractions(project_config.TRAIN_FRACTION, project_config.VALIDATION_FRACTION),
    )


def add_known_candidate_scores(observed: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    frame = observed.copy()
    if "candidate_score" not in frame.columns:
        frame["candidate_score"] = np.nan
    if "retrieval_rank" not in frame.columns:
        frame["retrieval_rank"] = np.nan
    candidate_scores = candidates.rename(
        columns={
            "candidate_score": "retrieval_candidate_score",
            "retrieval_rank": "candidate_retrieval_rank",
        }
    )
    frame = frame.merge(candidate_scores, on=["user_id", "movie_id"], how="left")
    frame["candidate_score"] = frame["candidate_score"].fillna(
        frame["retrieval_candidate_score"]
    )
    frame["candidate_score"] = frame["candidate_score"].fillna(0.0)
    if "candidate_retrieval_rank" in frame.columns:
        frame["retrieval_rank"] = frame["retrieval_rank"].fillna(
            frame["candidate_retrieval_rank"]
        )
    frame["retrieval_rank"] = frame["retrieval_rank"].fillna(0.0)
    frame = frame.drop(
        columns=[
            col
            for col in ["retrieval_candidate_score", "candidate_retrieval_rank"]
            if col in frame.columns
        ]
    )
    return frame


def weak_negative_rows(
    observed_split: pd.DataFrame,
    candidates: pd.DataFrame,
    all_observed_pairs: set[tuple[int, int]],
    movies: pd.DataFrame,
    users: pd.DataFrame,
    n_rows: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if n_rows <= 0:
        return pd.DataFrame()

    user_times = (
        observed_split.groupby("user_id")["timestamp"]
        .max()
        .reset_index()
        .rename(columns={"timestamp": "candidate_timestamp"})
    )
    pool = candidates.merge(user_times, on="user_id", how="inner")
    pool = pool[
        ~pool[["user_id", "movie_id"]].apply(tuple, axis=1).isin(all_observed_pairs)
    ].copy()
    if pool.empty:
        return pd.DataFrame()

    sampled = pool.sample(n=min(n_rows, len(pool)), random_state=int(rng.integers(0, 1_000_000)))
    sampled = sampled.merge(movies, on="movie_id", how="left")
    sampled = sampled.merge(users, on="user_id", how="left")
    sampled["timestamp"] = sampled["candidate_timestamp"]
    sampled["rating"] = np.nan
    sampled["label"] = 0
    sampled["sample_weight"] = ranking_config.WEAK_NEGATIVE_WEIGHT
    sampled["label_source"] = "weak_negative"
    return sampled.drop(columns=["candidate_timestamp"])


def balance_split(
    observed_split: pd.DataFrame,
    candidates: pd.DataFrame,
    all_observed_pairs: set[tuple[int, int]],
    movies: pd.DataFrame,
    users: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    positives = observed_split[observed_split["label"] == 1].copy()
    strong = observed_split[observed_split["label"] == 0].copy()

    target_negative_count = len(positives)
    strong_count = min(len(strong), target_negative_count // 2)
    weak_count = target_negative_count - strong_count

    if strong_count:
        strong = strong.sample(
            n=strong_count,
            random_state=int(rng.integers(0, 1_000_000)),
        )
    else:
        strong = strong.iloc[0:0]

    positives["sample_weight"] = ranking_config.POSITIVE_WEIGHT
    positives["label_source"] = "positive"
    strong["sample_weight"] = ranking_config.STRONG_NEGATIVE_WEIGHT
    strong["label_source"] = "strong_negative"

    weak = weak_negative_rows(
        observed_split=observed_split,
        candidates=candidates,
        all_observed_pairs=all_observed_pairs,
        movies=movies,
        users=users,
        n_rows=weak_count,
        rng=rng,
    )
    frame = pd.concat([positives, strong, weak], ignore_index=True)
    return frame.sort_values(["user_id", "timestamp", "movie_id"]).reset_index(drop=True)


def build_ranking_split(
    balanced: pd.DataFrame,
    history: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    observed = balanced[balanced["label_source"] != "weak_negative"].copy()
    weak = balanced[balanced["label_source"] == "weak_negative"].copy()

    observed = add_historical_observed_features(observed)
    weak = fill_weak_historical_features(weak, history)

    frame = pd.concat([observed, weak], ignore_index=True)
    frame = add_known_candidate_scores(frame, candidates)
    frame = finalize_features(frame, history)
    frame = add_retrieval_embedding_features(frame)
    return frame.sort_values(["user_id", "label", "candidate_score"], ascending=[True, False, False])


def fill_weak_historical_features(weak: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    if weak.empty:
        return weak
    history_with_stats = add_historical_observed_features(history)
    latest_user = (
        history_with_stats.sort_values("timestamp")
        .groupby("user_id")
        .tail(1)[
            [
                "user_id",
                "user_avg_rating_before",
                "user_rating_count_before",
                "user_like_rate_before",
                "user_activity_gap_log",
            ]
        ]
    )
    latest_movie = (
        history_with_stats.sort_values("timestamp")
        .groupby("movie_id")
        .tail(1)[
            [
                "movie_id",
                "movie_avg_rating_before",
                "movie_rating_count_before",
                "movie_like_rate_before",
                "movie_popularity_before",
            ]
        ]
    )
    weak = weak.merge(latest_user, on="user_id", how="left")
    weak = weak.merge(latest_movie, on="movie_id", how="left")
    defaults = {
        "user_avg_rating_before": history["rating"].mean(),
        "user_rating_count_before": 0,
        "user_like_rate_before": history["label"].mean(),
        "user_activity_gap_log": 0,
        "movie_avg_rating_before": history["rating"].mean(),
        "movie_rating_count_before": 0,
        "movie_like_rate_before": history["label"].mean(),
        "movie_popularity_before": 0,
    }
    return weak.fillna(defaults)


def prepare_ranking_data(
    output_dir: Path = ranking_config.RANKING_DATA_DIR,
    candidates_per_user: int = ranking_config.CANDIDATES_PER_USER,
    refresh_candidates: bool = False,
) -> None:
    rng = np.random.default_rng(ranking_config.RANDOM_SEED)
    raw = load_joined_movielens()
    train, val, test = split_observed_interactions(raw)
    users = user_feature_table()
    movies = movie_feature_table()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = output_dir / "retrieval_candidates.parquet"
    if candidates_path.exists() and not refresh_candidates:
        print(f"Loading cached retrieval candidates: {candidates_path}")
        candidates = pd.read_parquet(candidates_path)
        if "retrieval_rank" not in candidates.columns:
            print("Cached candidates do not include retrieval_rank; regenerating.")
            candidates = generate_top_k_candidates(users, movies, k=candidates_per_user)
            candidates.to_parquet(candidates_path, index=False)
    else:
        print("Generating retrieval top-K candidates from latest two-tower model...")
        candidates = generate_top_k_candidates(users, movies, k=candidates_per_user)
        candidates.to_parquet(candidates_path, index=False)

    all_observed_pairs = set(raw[["user_id", "movie_id"]].apply(tuple, axis=1))
    splits = {
        "train": (train, train),
        "val": (val, pd.concat([train, val], ignore_index=True)),
        "test": (test, pd.concat([train, val, test], ignore_index=True)),
    }
    paths = {
        "train": ranking_config.TRAIN_FILE,
        "val": ranking_config.VALIDATION_FILE,
        "test": ranking_config.TEST_FILE,
    }

    for name, (observed_split, history) in splits.items():
        balanced = balance_split(observed_split, candidates, all_observed_pairs, movies, users, rng)
        ranking_frame = build_ranking_split(balanced, history, candidates)
        ranking_frame.to_parquet(paths[name], index=False)
        print(
            f"{name:<5}: {len(ranking_frame):>7,} rows, "
            f"positives={int(ranking_frame['label'].sum()):>6,}, "
            f"users={ranking_frame['user_id'].nunique():>4,}, path={paths[name]}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ranking_config.RANKING_DATA_DIR)
    parser.add_argument("--candidates-per-user", type=int, default=ranking_config.CANDIDATES_PER_USER)
    parser.add_argument("--refresh-candidates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_ranking_data(args.output_dir, args.candidates_per_user, args.refresh_candidates)


if __name__ == "__main__":
    main()
