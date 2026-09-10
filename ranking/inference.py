"""TensorFlow-free recommendation path: embedding lookup -> ANN -> LightGBM rerank.

Every MovieLens 100k user and movie is in-vocabulary and the two-tower model
already exports per-user / per-movie embeddings, so serving does not need
TensorFlow. This module reuses the offline feature engineering
(:mod:`ranking.features`, :mod:`ranking.training.prepare_ranking_data`) and swaps
the model call for a precomputed-embedding lookup, keeping training and serving
features identical.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ranking.features import (
    MOVIE_RETRIEVAL_VECTOR_FEATURES,
    RANKING_FEATURES,
    RETRIEVAL_VECTOR_ABS_DIFF_FEATURES,
    RETRIEVAL_VECTOR_PRODUCT_FEATURES,
    USER_RETRIEVAL_VECTOR_FEATURES,
    fill_candidate_historical_features,
    finalize_features,
)
from shared.config import PROJECT_ROOT

RETRIEVAL_LOGIT_SCALE = 5.0
_EMBED_DIM = 32
USER_VECTOR_COLUMNS = [f"user_vector_{i:02d}" for i in range(_EMBED_DIM)]
MOVIE_VECTOR_COLUMNS = [f"movie_vector_{i:02d}" for i in range(_EMBED_DIM)]

DEFAULT_BUNDLE_DIR = PROJECT_ROOT / "serving" / "model_bundle"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Recommendation:
    movie_id: int
    title: str
    genres: str
    ranker_score: float
    retrieval_rank: int


@dataclass
class Recommender:
    """Loads a serving bundle and answers ``recommend`` / ``user_history`` queries."""

    booster: Any  # lightgbm.Booster; kept loose so the module imports without lightgbm
    user_embeddings: pd.DataFrame
    movie_embeddings: pd.DataFrame
    ann_index: Any  # sklearn NearestNeighbors
    ann_movie_ids: np.ndarray
    movies: pd.DataFrame
    history: pd.DataFrame
    model_version: str
    metrics: dict

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_bundle(cls, bundle_dir: Path = DEFAULT_BUNDLE_DIR) -> Recommender:
        import lightgbm as lgb

        bundle_dir = Path(bundle_dir)
        booster = lgb.Booster(model_file=str(bundle_dir / "lgbm_ranker.txt"))

        user_embeddings = pd.read_parquet(bundle_dir / "user_embeddings.parquet")
        movie_embeddings = pd.read_parquet(bundle_dir / "movie_embeddings.parquet")

        ann = joblib.load(bundle_dir / "movie_ann_index.joblib")
        movies = pd.read_parquet(bundle_dir / "movies.parquet")

        history = pd.read_parquet(bundle_dir / "history.parquet")
        history["label"] = pd.to_numeric(history["label"], errors="coerce")

        manifest_path = bundle_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        metrics_path = bundle_dir / "end_to_end_metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

        return cls(
            booster=booster,
            user_embeddings=user_embeddings,
            movie_embeddings=movie_embeddings,
            ann_index=ann["index"],
            ann_movie_ids=np.asarray(ann["movie_ids"]),
            movies=movies,
            history=history,
            model_version=str(manifest.get("version", "unknown")),
            metrics=metrics,
        )

    # -- queries ------------------------------------------------------------

    @property
    def known_user_ids(self) -> list[int]:
        return sorted(int(u) for u in self.user_embeddings["user_id"].unique())

    def has_user(self, user_id: int) -> bool:
        return bool((self.user_embeddings["user_id"] == user_id).any())

    def user_history(self, user_id: int, limit: int = 20) -> pd.DataFrame:
        """Movies the user rated, most recent first (with a 'liked' flag)."""
        seen = self.history[self.history["user_id"] == user_id].copy()
        seen = seen.sort_values("timestamp", ascending=False)
        seen["liked"] = seen["rating"] >= 4
        cols = ["movie_id", "title", "genres", "rating", "liked", "timestamp"]
        return seen[cols].head(limit).reset_index(drop=True)

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        n_candidates: int = 200,
        exclude_seen: bool = True,
    ) -> list[Recommendation]:
        if not self.has_user(user_id):
            raise KeyError(f"Unknown user_id {user_id}")

        candidates = self._retrieve(user_id, n_candidates)
        if exclude_seen:
            seen = set(self.history.loc[self.history["user_id"] == user_id, "movie_id"])
            candidates = candidates[~candidates["movie_id"].isin(seen)]
        if candidates.empty:
            return []

        frame = self._build_features(user_id, candidates)
        frame["ranker_score"] = self.booster.predict(frame[RANKING_FEATURES])
        top = frame.sort_values("ranker_score", ascending=False).head(k)

        return [
            Recommendation(
                movie_id=int(row["movie_id"]),
                title=str(row["title"]),
                genres=str(row["genres"]),
                ranker_score=float(row["ranker_score"]),
                retrieval_rank=int(row["retrieval_rank"]),
            )
            for row in top[
                ["movie_id", "title", "genres", "ranker_score", "retrieval_rank"]
            ].to_dict("records")
        ]

    # -- internals --------------------------------------------------------

    def _retrieve(self, user_id: int, n_candidates: int) -> pd.DataFrame:
        user_vec = (
            self.user_embeddings.loc[
                self.user_embeddings["user_id"] == user_id, USER_VECTOR_COLUMNS
            ]
            .to_numpy()
            .reshape(1, -1)
        )
        n = min(n_candidates, len(self.ann_movie_ids))
        distances, indices = self.ann_index.kneighbors(user_vec, n_neighbors=n)
        distances, indices = distances[0], indices[0]
        # ANN metric is cosine distance; mirror retrieval scoring: sigmoid(5 * cos_sim).
        scores = _sigmoid(RETRIEVAL_LOGIT_SCALE * (1.0 - distances))
        return pd.DataFrame(
            {
                "user_id": user_id,
                "movie_id": self.ann_movie_ids[indices].astype(int),
                "candidate_score": scores,
                "retrieval_rank": np.arange(1, len(indices) + 1),
            }
        )

    def _build_features(self, user_id: int, candidates: pd.DataFrame) -> pd.DataFrame:
        frame = candidates.merge(self.movies, on="movie_id", how="left")
        frame["timestamp"] = int(time.time())
        frame["rating"] = np.nan

        frame = fill_candidate_historical_features(frame, self.history)
        frame = finalize_features(frame, self.history)
        frame = self._add_retrieval_vector_features(user_id, frame)
        return frame

    def _add_retrieval_vector_features(self, user_id: int, frame: pd.DataFrame) -> pd.DataFrame:
        user_vec = (
            self.user_embeddings.loc[
                self.user_embeddings["user_id"] == user_id, USER_VECTOR_COLUMNS
            ]
            .to_numpy()
            .reshape(1, -1)
        )
        movie_vecs = (
            frame[["movie_id"]]
            .merge(
                self.movie_embeddings[["movie_id", *MOVIE_VECTOR_COLUMNS]],
                on="movie_id",
                how="left",
            )[MOVIE_VECTOR_COLUMNS]
            .to_numpy()
        )
        user_rep = np.repeat(user_vec, len(frame), axis=0)
        block = np.hstack(
            [
                user_rep,
                movie_vecs,
                user_rep * movie_vecs,
                np.abs(user_rep - movie_vecs),
            ]
        )
        columns = (
            USER_RETRIEVAL_VECTOR_FEATURES
            + MOVIE_RETRIEVAL_VECTOR_FEATURES
            + RETRIEVAL_VECTOR_PRODUCT_FEATURES
            + RETRIEVAL_VECTOR_ABS_DIFF_FEATURES
        )
        frame = frame.drop(columns=columns, errors="ignore")
        vectors = pd.DataFrame(block, columns=columns, index=frame.index)
        return pd.concat([frame, vectors], axis=1)
