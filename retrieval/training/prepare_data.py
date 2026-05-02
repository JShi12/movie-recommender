"""Prepare MovieLens 100k CSV splits for TFX ingestion.

This is the productionized version of the data-loading and time-split notebook
cells. It reads the raw MovieLens 100k files from ``ml-100k`` and writes:

- data/retrieval/train.csv
- data/retrieval/val.csv
- data/retrieval/test.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval import config
from shared.movielens import SplitFractions
from shared.movielens import active_genres
from shared.movielens import load_movielens_100k
from shared.movielens import time_based_split


def build_model_dataframe(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
) -> pd.DataFrame:
    """Join raw data and return the model feature dataframe."""
    merged = ratings.merge(users, on="user_id", how="left")
    merged = merged.merge(movies, on="movie_id", how="left")
    merged = merged[merged["rating"] != config.NEUTRAL_RATING].copy()
    merged["label"] = (
        merged["rating"] >= config.POSITIVE_RATING_THRESHOLD
    ).astype(int)
    merged["genres"] = merged.apply(active_genres, axis=1)

    feature_cols = [
        "user_id",
        "movie_id",
        "age",
        "gender",
        "occupation",
        "genres",
        "timestamp",
        "label",
    ]
    return merged[feature_cols].copy()


def write_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write TFX input CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_dir / "train.csv", index=False)
    validation.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)


def _print_split_summary(name: str, frame: pd.DataFrame, n_total: int) -> None:
    start = pd.to_datetime(frame["timestamp"].min(), unit="s")
    end = pd.to_datetime(frame["timestamp"].max(), unit="s")
    print(f"{name:<10}: {len(frame):>7,} rows ({len(frame) / n_total:.1%})")
    print(f"  timestamps: {start} to {end}")
    print(f"  like rate : {frame['label'].mean():.1%}")


def prepare_data(
    raw_data_dir: Path = config.RAW_DATA_DIR,
    output_dir: Path = config.RETRIEVAL_DATA_DIR,
    fractions: SplitFractions = SplitFractions(),
) -> None:
    """Run the full data-preparation workflow."""
    ratings, users, movies = load_movielens_100k(raw_data_dir)
    data = build_model_dataframe(ratings, users, movies)
    train, validation, test = time_based_split(data, fractions)
    write_splits(train, validation, test, output_dir)

    print("=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    print(f"Raw data dir : {raw_data_dir}")
    print(f"Output dir   : {output_dir}")
    print(f"Total records: {len(data):,}")
    _print_split_summary("train", train, len(data))
    _print_split_summary("val", validation, len(data))
    _print_split_summary("test", test, len(data))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", type=Path, default=config.RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=config.RETRIEVAL_DATA_DIR)
    parser.add_argument("--train-fraction", type=float, default=config.TRAIN_FRACTION)
    parser.add_argument("--validation-fraction", type=float, default=config.VALIDATION_FRACTION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_data(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        fractions=SplitFractions(
            train=args.train_fraction,
            validation=args.validation_fraction,
        ),
    )


if __name__ == "__main__":
    main()
