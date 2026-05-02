"""Evaluate offline top-K metrics for the retrieval model.

This script evaluates the latest two-tower retrieval SavedModel as a candidate
generator. It compares top-K retrieved movies against held-out positive
interactions and writes metrics to artifacts/retrieval/retrieval_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from retrieval import config
from retrieval.candidates import generate_top_k_candidates, latest_pushed_model_dir
from shared.feature_tables import load_joined_movielens, movie_feature_table, user_feature_table
from shared.movielens import SplitFractions
from shared.movielens import time_based_split


def split_observed_interactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return chronological train/validation/test splits for labelled interactions."""
    labelled = raw[raw["label"].notna()].copy()
    return time_based_split(
        labelled,
        SplitFractions(config.TRAIN_FRACTION, config.VALIDATION_FRACTION),
    )


def positive_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep positive held-out user/movie interactions used as retrieval targets."""
    return frame[frame["label"] == 1][["user_id", "movie_id"]].drop_duplicates().copy()


def evaluate_top_k(
    candidates: pd.DataFrame,
    positives: pd.DataFrame,
    catalog_size: int,
    ks: list[int],
) -> dict[str, float | int | str | None]:
    """Compute user-averaged and micro top-K retrieval metrics."""
    if positives.empty:
        raise ValueError("No positive interactions found in the selected evaluation split.")

    max_k = max(ks)
    candidates = candidates[candidates["retrieval_rank"] <= max_k].copy()
    positives_by_user = {
        int(user_id): set(group["movie_id"].astype(int))
        for user_id, group in positives.groupby("user_id")
    }

    metrics: dict[str, float | int | str | None] = {
        "evaluated_users": len(positives_by_user),
        "positive_interactions": int(len(positives)),
        "candidate_rows": int(len(candidates)),
        "catalog_size": int(catalog_size),
    }

    for k in ks:
        top_k = candidates[candidates["retrieval_rank"] <= k]
        retrieved_by_user = {
            int(user_id): list(group.sort_values("retrieval_rank")["movie_id"].astype(int))
            for user_id, group in top_k.groupby("user_id")
        }

        user_recalls = []
        user_hits = []
        micro_hits = 0
        positive_count = 0
        found_ranks = []

        rank_lookup = {
            (int(row.user_id), int(row.movie_id)): int(row.retrieval_rank)
            for row in top_k.itertuples(index=False)
        }

        for user_id, positive_movies in positives_by_user.items():
            retrieved_movies = set(retrieved_by_user.get(user_id, []))
            hits = positive_movies & retrieved_movies
            user_recalls.append(len(hits) / len(positive_movies))
            user_hits.append(1.0 if hits else 0.0)
            micro_hits += len(hits)
            positive_count += len(positive_movies)
            found_ranks.extend(
                rank_lookup[(user_id, movie_id)]
                for movie_id in hits
                if (user_id, movie_id) in rank_lookup
            )

        unique_candidates = int(top_k["movie_id"].nunique())
        metrics[f"recall@{k}"] = float(np.mean(user_recalls))
        metrics[f"micro_recall@{k}"] = float(micro_hits / positive_count)
        metrics[f"hit_rate@{k}"] = float(np.mean(user_hits))
        metrics[f"candidate_coverage@{k}"] = float(unique_candidates / catalog_size)
        metrics[f"unique_candidates@{k}"] = unique_candidates
        metrics[f"mean_positive_rank@{k}"] = (
            float(np.mean(found_ranks)) if found_ranks else None
        )

    return metrics


def load_or_generate_candidates(
    users: pd.DataFrame,
    movies: pd.DataFrame,
    k: int,
    candidates_file: Path,
    use_cached_candidates: bool,
    batch_size: int,
) -> pd.DataFrame:
    """Load cached candidates when requested, otherwise generate fresh top-K candidates."""
    if use_cached_candidates and candidates_file.exists():
        candidates = pd.read_parquet(candidates_file)
        if int(candidates["retrieval_rank"].max()) >= k:
            return candidates[candidates["retrieval_rank"] <= k].copy()
        print(f"Cached candidates have fewer than {k} ranks; regenerating.")

    candidates = generate_top_k_candidates(users, movies, k=k, batch_size=batch_size)
    candidates_file.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(candidates_file, index=False)
    return candidates


def evaluate_retrieval(
    split: str = "test",
    ks: list[int] | None = None,
    output_file: Path = config.RETRIEVAL_METRICS_FILE,
    candidates_file: Path = config.RETRIEVAL_EVAL_CANDIDATES_FILE,
    use_cached_candidates: bool = False,
    batch_size: int = 4096,
) -> dict[str, float | int | str | None]:
    """Evaluate retrieval metrics for validation or test held-out positives."""
    ks = sorted(ks or [10, 50, 100])
    raw = load_joined_movielens()
    train, validation, test = split_observed_interactions(raw)
    split_frame = {"train": train, "val": validation, "test": test}[split]
    positives = positive_interactions(split_frame)

    users = user_feature_table()
    users = users[users["user_id"].isin(positives["user_id"].unique())].copy()
    movies = movie_feature_table()

    candidates = load_or_generate_candidates(
        users=users,
        movies=movies,
        k=max(ks),
        candidates_file=candidates_file,
        use_cached_candidates=use_cached_candidates,
        batch_size=batch_size,
    )
    metrics = evaluate_top_k(candidates, positives, catalog_size=len(movies), ks=ks)
    metrics["split"] = split
    metrics["max_k"] = max(ks)
    metrics["model_dir"] = str(latest_pushed_model_dir())
    metrics["candidates_file"] = str(candidates_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Retrieval metrics saved to {output_file}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--output-file", type=Path, default=config.RETRIEVAL_METRICS_FILE)
    parser.add_argument(
        "--candidates-file",
        type=Path,
        default=config.RETRIEVAL_EVAL_CANDIDATES_FILE,
    )
    parser.add_argument("--use-cached-candidates", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_retrieval(
        split=args.split,
        ks=args.ks,
        output_file=args.output_file,
        candidates_file=args.candidates_file,
        use_cached_candidates=args.use_cached_candidates,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
