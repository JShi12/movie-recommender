"""Export retrieval tower embeddings and a local ANN-style movie index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from retrieval import config
from retrieval.candidates import RetrievalCandidateScorer
from shared.feature_tables import movie_feature_table, user_feature_table


def _embedding_columns(prefix: str, width: int) -> list[str]:
    return [f"{prefix}_{idx:02d}" for idx in range(width)]


def export_retrieval_artifacts(
    movie_embeddings_file: Path = config.MOVIE_EMBEDDINGS_FILE,
    user_embeddings_file: Path = config.USER_EMBEDDINGS_FILE,
    ann_index_file: Path = config.ANN_INDEX_FILE,
) -> None:
    scorer = RetrievalCandidateScorer()
    if not scorer.has_embedding_signatures:
        raise RuntimeError(
            "Latest retrieval SavedModel does not expose user_embedding/movie_embedding "
            "signatures. Rerun the TFX retrieval pipeline after the trainer_module.py "
            "signature update, then rerun this exporter."
        )

    movies = movie_feature_table().reset_index(drop=True)
    users = user_feature_table().reset_index(drop=True)

    movie_vectors = scorer.movie_embeddings(movies)
    movie_cols = _embedding_columns("movie_vector", movie_vectors.shape[1])
    movie_embedding_frame = pd.concat(
        [
            movies[["movie_id", "title"]],
            pd.DataFrame(movie_vectors, columns=movie_cols),
        ],
        axis=1,
    )

    user_vectors = scorer.user_embeddings(users)
    user_cols = _embedding_columns("user_vector", user_vectors.shape[1])
    user_embedding_frame = pd.concat(
        [
            users[["user_id"]],
            pd.DataFrame(user_vectors, columns=user_cols),
        ],
        axis=1,
    )

    movie_embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    movie_embedding_frame.to_parquet(movie_embeddings_file, index=False)
    user_embedding_frame.to_parquet(user_embeddings_file, index=False)

    index = NearestNeighbors(
        n_neighbors=min(config.CANDIDATES_PER_USER, len(movie_embedding_frame)),
        metric="cosine",
        algorithm="brute",
    )
    index.fit(movie_vectors)
    joblib.dump(
        {
            "index": index,
            "movie_ids": movie_embedding_frame["movie_id"].to_numpy(),
            "embedding_columns": movie_cols,
            "metric": "cosine",
        },
        ann_index_file,
    )

    print(f"Movie embeddings saved to {movie_embeddings_file}")
    print(f"User embeddings saved to {user_embeddings_file}")
    print(f"Movie ANN index saved to {ann_index_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie-embeddings-file", type=Path, default=config.MOVIE_EMBEDDINGS_FILE)
    parser.add_argument("--user-embeddings-file", type=Path, default=config.USER_EMBEDDINGS_FILE)
    parser.add_argument("--ann-index-file", type=Path, default=config.ANN_INDEX_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_retrieval_artifacts(
        movie_embeddings_file=args.movie_embeddings_file,
        user_embeddings_file=args.user_embeddings_file,
        ann_index_file=args.ann_index_file,
    )


if __name__ == "__main__":
    main()
