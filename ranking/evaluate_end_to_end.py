"""Evaluate retrieval plus LightGBM ranking on held-out interactions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from prepare_data import SplitFractions, time_based_split
from ranking import config as ranking_config
from ranking.config import project_config
from ranking.feature_builder import (
    RANKING_FEATURES,
    add_retrieval_embedding_features,
    finalize_features,
    load_joined_movielens,
    movie_feature_table,
    user_feature_table,
)
from ranking.prepare_ranking_data import fill_candidate_historical_features
from ranking.prepare_ranking_data import candidate_request_times
from retrieval_candidates import generate_top_k_candidates


def split_observed_interactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labelled = raw[raw["label"].notna()].copy()
    return time_based_split(
        labelled,
        SplitFractions(project_config.TRAIN_FRACTION, project_config.VALIDATION_FRACTION),
    )


def split_and_history(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_joined_movielens()
    train, val, test = split_observed_interactions(raw)
    if split == "train":
        return train, train
    if split == "val":
        return val, train
    if split == "test":
        return test, pd.concat([train, val], ignore_index=True)
    raise ValueError(f"Unsupported split: {split}")


def positive_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["label"] == 1][["user_id", "movie_id"]].drop_duplicates().copy()


def build_end_to_end_candidate_features(
    target: pd.DataFrame,
    history: pd.DataFrame,
    candidates_per_user: int,
    batch_size: int,
    include_seen_history: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positives = positive_interactions(target)
    if positives.empty:
        raise ValueError("No positive interactions found in the selected split.")

    users = user_feature_table()
    movies = movie_feature_table()
    eval_users = users[users["user_id"].isin(positives["user_id"].unique())].copy()

    candidates = generate_top_k_candidates(
        eval_users,
        movies,
        k=candidates_per_user,
        batch_size=batch_size,
    )

    if not include_seen_history:
        seen_pairs = set(history[["user_id", "movie_id"]].apply(tuple, axis=1))
        candidates = candidates[
            ~candidates[["user_id", "movie_id"]].apply(tuple, axis=1).isin(seen_pairs)
        ].copy()

    eval_times = candidate_request_times(target, history)
    frame = candidates.merge(eval_times, on="user_id", how="inner")
    frame = frame.merge(movies, on="movie_id", how="left")
    frame = frame.merge(users, on="user_id", how="left")
    frame["timestamp"] = frame["candidate_timestamp"]
    frame["rating"] = np.nan
    frame["sample_weight"] = 1.0
    frame["label_source"] = "end_to_end_candidate"

    positive_pairs = set(positives.apply(tuple, axis=1))
    frame["label"] = (
        frame[["user_id", "movie_id"]].apply(tuple, axis=1).isin(positive_pairs).astype(float)
    )
    frame = frame.drop(columns=["candidate_timestamp"])

    frame = fill_candidate_historical_features(frame, history)
    frame = finalize_features(frame, history)
    frame = add_retrieval_embedding_features(frame, batch_size=batch_size)
    return frame, positives


def evaluate_ranked_candidates(
    frame: pd.DataFrame,
    positives: pd.DataFrame,
    ks: list[int],
) -> dict[str, float | int | str | None]:
    positives_by_user = {
        int(user_id): set(group["movie_id"].astype(int))
        for user_id, group in positives.groupby("user_id")
    }

    metrics: dict[str, float | int | str | None] = {
        "evaluated_users": len(positives_by_user),
        "positive_interactions": int(len(positives)),
        "candidate_rows": int(len(frame)),
        "candidate_users": int(frame["user_id"].nunique()),
        "candidate_movies": int(frame["movie_id"].nunique()),
        "retrieved_positive_interactions": int(frame["label"].sum()),
        "retrieval_micro_recall": float(frame["label"].sum() / len(positives)),
    }

    if frame["label"].nunique() > 1:
        metrics["candidate_auc"] = float(roc_auc_score(frame["label"], frame["ranker_score"]))
    else:
        metrics["candidate_auc"] = None

    for k in ks:
        user_recalls = []
        user_hits = []
        user_ndcgs = []
        micro_hits = 0
        positive_count = 0
        unique_recommendations = set()

        for user_id, positive_movies in positives_by_user.items():
            group = frame[frame["user_id"] == user_id].sort_values(
                "ranker_score",
                ascending=False,
            )
            top_movies = group.head(k)["movie_id"].astype(int).tolist()
            unique_recommendations.update(top_movies)
            hits = set(top_movies) & positive_movies
            user_recalls.append(len(hits) / len(positive_movies))
            user_hits.append(1.0 if hits else 0.0)
            micro_hits += len(hits)
            positive_count += len(positive_movies)

            gains = np.asarray([1.0 if movie_id in positive_movies else 0.0 for movie_id in top_movies])
            discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
            dcg = float(np.sum(gains * discounts))
            ideal_hits = min(k, len(positive_movies))
            ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_hits + 2))
            idcg = float(np.sum(ideal_discounts))
            user_ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        metrics[f"ndcg@{k}"] = float(np.mean(user_ndcgs))
        metrics[f"recall@{k}"] = float(np.mean(user_recalls))
        metrics[f"micro_recall@{k}"] = float(micro_hits / positive_count)
        metrics[f"hit_rate@{k}"] = float(np.mean(user_hits))
        metrics[f"unique_recommendations@{k}"] = len(unique_recommendations)

    return metrics


def evaluate_end_to_end(
    split: str = "test",
    candidates_per_user: int = ranking_config.RANKING_CANDIDATES_PER_USER,
    ks: list[int] | None = None,
    model_file: Path = ranking_config.MODEL_FILE,
    output_file: Path = ranking_config.END_TO_END_METRICS_FILE,
    scored_candidates_file: Path = ranking_config.END_TO_END_CANDIDATES_FILE,
    batch_size: int = 4096,
    include_seen_history: bool = False,
) -> dict[str, float | int | str | None]:
    ks = sorted(ks or [10, 20, 100])
    target, history = split_and_history(split)
    frame, positives = build_end_to_end_candidate_features(
        target=target,
        history=history,
        candidates_per_user=candidates_per_user,
        batch_size=batch_size,
        include_seen_history=include_seen_history,
    )

    booster = lgb.Booster(model_file=str(model_file))
    frame["ranker_score"] = booster.predict(frame[RANKING_FEATURES])
    frame = frame.sort_values(["user_id", "ranker_score"], ascending=[True, False])

    metrics = evaluate_ranked_candidates(frame, positives, ks)
    metrics["split"] = split
    metrics["candidates_per_user"] = candidates_per_user
    metrics["model_file"] = str(model_file)
    metrics["include_seen_history"] = include_seen_history

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    scored_candidates_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(scored_candidates_file, index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {output_file}")
    print(f"Scored candidates saved to {scored_candidates_file}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--candidates-per-user",
        type=int,
        default=ranking_config.RANKING_CANDIDATES_PER_USER,
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20, 100])
    parser.add_argument("--model-file", type=Path, default=ranking_config.MODEL_FILE)
    parser.add_argument("--output-file", type=Path, default=ranking_config.END_TO_END_METRICS_FILE)
    parser.add_argument(
        "--scored-candidates-file",
        type=Path,
        default=ranking_config.END_TO_END_CANDIDATES_FILE,
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--include-seen-history", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_end_to_end(
        split=args.split,
        candidates_per_user=args.candidates_per_user,
        ks=args.ks,
        model_file=args.model_file,
        output_file=args.output_file,
        scored_candidates_file=args.scored_candidates_file,
        batch_size=args.batch_size,
        include_seen_history=args.include_seen_history,
    )


if __name__ == "__main__":
    main()
