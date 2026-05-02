"""Prepare retrieval-candidate datasets for LightGBM ranking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from retrieval.candidates import generate_top_k_candidates
from shared.movielens import SplitFractions
from shared.movielens import time_based_split
from ranking import config as ranking_config
from ranking.features import (
    add_retrieval_embedding_features,
    add_historical_observed_features,
    finalize_features,
    load_joined_movielens,
    movie_feature_table,
    user_feature_table,
)


def split_observed_interactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labelled = raw[raw["rating"].notna()].copy()
    return time_based_split(
        labelled,
        SplitFractions(ranking_config.TRAIN_FRACTION, ranking_config.VALIDATION_FRACTION),
    )


def rating_to_relevance(rating: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [rating >= 5, rating == 4, rating == 3],
            [3, 2, 1],
            default=0,
        ),
        index=rating.index,
        name=rating.name,
    )


def rating_to_label_source(rating: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                rating >= 5,
                rating == 4,
                rating == 3,
                rating <= 2,
            ],
            ["rating_5", "rating_4", "rating_3", "observed_negative"],
            default="unobserved",
        ),
        index=rating.index,
        name=rating.name,
    )


def candidate_request_times(target: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Use the latest available history timestamp as the candidate request time."""
    target_start = (
        target.groupby("user_id")["timestamp"]
        .min()
        .reset_index()
        .rename(columns={"timestamp": "target_start_timestamp"})
    )
    history_latest = (
        history.groupby("user_id")["timestamp"]
        .max()
        .reset_index()
        .rename(columns={"timestamp": "history_latest_timestamp"})
    )
    times = target_start.merge(history_latest, on="user_id", how="left")
    times["candidate_timestamp"] = times["history_latest_timestamp"].fillna(
        times["target_start_timestamp"]
    )
    return times[["user_id", "candidate_timestamp"]]


def build_candidate_ranking_split(
    observed_split: pd.DataFrame,
    history: pd.DataFrame,
    candidates: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame,
    retrieval_model_dir: Path | None = None,
    transform_graph_dir: Path | None = None,
    batch_size: int = 4096,
) -> pd.DataFrame:
    """Build ranking rows from the actual retrieval top-K candidate distribution."""
    target_users = observed_split["user_id"].unique()
    frame = candidates[candidates["user_id"].isin(target_users)].copy()

    observed_labels = observed_split[
        ["user_id", "movie_id", "rating", "timestamp"]
    ].rename(columns={"timestamp": "observed_timestamp"})
    frame = frame.merge(observed_labels, on=["user_id", "movie_id"], how="left")
    frame = frame.merge(movies, on="movie_id", how="left")
    frame = frame.merge(users, on="user_id", how="left")

    candidate_times = candidate_request_times(observed_split, history)
    frame = frame.merge(candidate_times, on="user_id", how="left")
    frame["timestamp"] = frame["observed_timestamp"].fillna(frame["candidate_timestamp"])
    frame["label"] = rating_to_relevance(frame["rating"]).astype(int)
    frame["label_source"] = rating_to_label_source(frame["rating"])
    frame["sample_weight"] = np.where(
        frame["rating"].notna(),
        ranking_config.OBSERVED_CANDIDATE_WEIGHT,
        ranking_config.UNOBSERVED_CANDIDATE_WEIGHT,
    )
    frame = frame.drop(columns=["observed_timestamp", "candidate_timestamp"])

    frame = fill_candidate_historical_features(frame, history)
    frame = finalize_features(frame, history)
    frame = add_retrieval_embedding_features(
        frame,
        batch_size=batch_size,
        retrieval_model_dir=retrieval_model_dir,
        transform_graph_dir=transform_graph_dir,
    )
    return frame.sort_values(
        ["user_id", "candidate_score"],
        ascending=[True, False],
    ).reset_index(drop=True)


def fill_candidate_historical_features(frame: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
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
    frame = frame.merge(latest_user, on="user_id", how="left")
    frame = frame.merge(latest_movie, on="movie_id", how="left")
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
    return frame.fillna(defaults)


def prepare_ranking_data(
    output_dir: Path = ranking_config.RANKING_DATA_DIR,
    candidates_per_user: int = ranking_config.RANKING_CANDIDATES_PER_USER,
    refresh_candidates: bool = False,
    raw_data_dir: Path = ranking_config.RAW_DATA_DIR,
    retrieval_model_dir: Path | None = None,
    transform_graph_dir: Path | None = None,
    ann_index_file: Path = ranking_config.ANN_INDEX_FILE,
    batch_size: int = 4096,
) -> None:
    raw = load_joined_movielens(raw_data_dir)
    train, val, test = split_observed_interactions(raw)
    users = user_feature_table(raw_data_dir)
    movies = movie_feature_table(raw_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = output_dir / "retrieval_candidates.parquet"
    if candidates_path.exists() and not refresh_candidates:
        print(f"Loading cached retrieval candidates: {candidates_path}")
        candidates = pd.read_parquet(candidates_path)
        if (
            "retrieval_rank" not in candidates.columns
            or int(candidates["retrieval_rank"].max()) < candidates_per_user
        ):
            print("Cached candidates are missing required ranks; regenerating.")
            candidates = generate_top_k_candidates(
                users,
                movies,
                k=candidates_per_user,
                batch_size=batch_size,
                model_dir=retrieval_model_dir,
                transform_graph_dir=transform_graph_dir,
                ann_index_file=ann_index_file,
            )
            candidates.to_parquet(candidates_path, index=False)
    else:
        print("Generating retrieval top-K candidates from latest two-tower model...")
        candidates = generate_top_k_candidates(
            users,
            movies,
            k=candidates_per_user,
            batch_size=batch_size,
            model_dir=retrieval_model_dir,
            transform_graph_dir=transform_graph_dir,
            ann_index_file=ann_index_file,
        )
        candidates.to_parquet(candidates_path, index=False)

    splits = {
        "train": (train, train),
        "val": (val, train),
        "test": (test, pd.concat([train, val], ignore_index=True)),
    }
    paths = {
        "train": output_dir / ranking_config.TRAIN_FILE.name,
        "val": output_dir / ranking_config.VALIDATION_FILE.name,
        "test": output_dir / ranking_config.TEST_FILE.name,
    }

    for name, (observed_split, history) in splits.items():
        ranking_frame = build_candidate_ranking_split(
            observed_split,
            history,
            candidates,
            movies,
            users,
            retrieval_model_dir=retrieval_model_dir,
            transform_graph_dir=transform_graph_dir,
            batch_size=batch_size,
        )
        ranking_frame.to_parquet(paths[name], index=False)
        print(
            f"{name:<5}: {len(ranking_frame):>7,} rows, "
            f"graded_label_sum={int(ranking_frame['label'].sum()):>6,}, "
            f"observed={int(ranking_frame['rating'].notna().sum()):>6,}, "
            f"users={ranking_frame['user_id'].nunique():>4,}, path={paths[name]}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ranking_config.RANKING_DATA_DIR)
    parser.add_argument(
        "--candidates-per-user",
        type=int,
        default=ranking_config.RANKING_CANDIDATES_PER_USER,
    )
    parser.add_argument("--refresh-candidates", action="store_true")
    parser.add_argument("--raw-data-dir", type=Path, default=ranking_config.RAW_DATA_DIR)
    parser.add_argument("--retrieval-model-dir", type=Path, default=None)
    parser.add_argument("--transform-graph-dir", type=Path, default=None)
    parser.add_argument("--ann-index-file", type=Path, default=ranking_config.ANN_INDEX_FILE)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_ranking_data(
        output_dir=args.output_dir,
        candidates_per_user=args.candidates_per_user,
        refresh_candidates=args.refresh_candidates,
        raw_data_dir=args.raw_data_dir,
        retrieval_model_dir=args.retrieval_model_dir,
        transform_graph_dir=args.transform_graph_dir,
        ann_index_file=args.ann_index_file,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
