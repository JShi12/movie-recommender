"""Generate top-K retrieval candidates using the two-tower SavedModel."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_transform as tft

import config


RETRIEVAL_LOGIT_SCALE = 5.0


def _sigmoid_scaled_dot(dot_product: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(RETRIEVAL_LOGIT_SCALE * dot_product)))


class RetrievalCandidateScorer:
    """Score raw user-movie pairs with the latest pushed TFX retrieval model."""

    def __init__(
        self,
        model_dir: Path | None = None,
        transform_graph_dir: Path | None = None,
        use_trainer_model: bool = False,
    ) -> None:
        if model_dir is not None:
            self.model_dir = model_dir
        elif use_trainer_model:
            self.model_dir = latest_retrieval_model_dir()
        else:
            self.model_dir = latest_pushed_model_dir()
        self.transform_graph_dir = transform_graph_dir or latest_transform_graph_dir()
        self.transform_output = tft.TFTransformOutput(str(self.transform_graph_dir))
        self.model = tf.saved_model.load(str(self.model_dir))
        self.signature = self.model.signatures["serving_default"]
        self.user_embedding_signature = self.model.signatures.get("user_embedding")
        self.movie_embedding_signature = self.model.signatures.get("movie_embedding")
        self._embedding_variables = None

    @property
    def has_embedding_signatures(self) -> bool:
        return (
            self.user_embedding_signature is not None
            and self.movie_embedding_signature is not None
        )

    def score_pairs(
        self,
        users: pd.DataFrame,
        movies: pd.DataFrame,
        batch_size: int = 4096,
    ) -> np.ndarray:
        scores = []
        for start in range(0, len(users), batch_size):
            end = min(start + batch_size, len(users))
            transformed = self._transform_batch(users.iloc[start:end], movies.iloc[start:end])
            serialized = self._serialize_transformed(transformed, end - start)
            outputs = self.signature(examples=tf.constant(serialized))
            scores.append(outputs["outputs"].numpy().reshape(-1))
        return np.concatenate(scores) if scores else np.asarray([], dtype=float)

    def embedding_pairs(
        self,
        users: pd.DataFrame,
        movies: pd.DataFrame,
        batch_size: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return user vectors, movie vectors, and sigmoid dot-product scores."""
        if not self.has_embedding_signatures:
            return self._embedding_pairs_from_variables(users, movies, batch_size=batch_size)

        user_vectors = []
        movie_vectors = []
        scores = []
        for start in range(0, len(users), batch_size):
            end = min(start + batch_size, len(users))
            transformed = self._transform_batch(users.iloc[start:end], movies.iloc[start:end])
            user_serialized = self._serialize_user_transformed(transformed, end - start)
            movie_serialized = self._serialize_movie_transformed(transformed, end - start)
            user_batch = self.user_embedding_signature(
                examples=tf.constant(user_serialized)
            )["user_embedding"].numpy()
            movie_batch = self.movie_embedding_signature(
                examples=tf.constant(movie_serialized)
            )["movie_embedding"].numpy()
            user_vectors.append(user_batch)
            movie_vectors.append(movie_batch)
            scores.append(_sigmoid_scaled_dot(np.sum(user_batch * movie_batch, axis=1)))

        return np.vstack(user_vectors), np.vstack(movie_vectors), np.concatenate(scores)

    def movie_embeddings(
        self,
        movies: pd.DataFrame,
        batch_size: int = 4096,
        users: pd.DataFrame | None = None,
    ) -> np.ndarray:
        if not self.has_embedding_signatures:
            if users is None:
                raise ValueError("users are required when using an older model without embedding signatures")
            _, movie_vectors, _ = self._embedding_pairs_from_variables(users, movies, batch_size)
            return movie_vectors

        vectors = []
        for start in range(0, len(movies), batch_size):
            end = min(start + batch_size, len(movies))
            transformed = self._transform_movie_batch(movies.iloc[start:end])
            serialized = self._serialize_movie_transformed(transformed, end - start)
            vectors.append(
                self.movie_embedding_signature(examples=tf.constant(serialized))[
                    "movie_embedding"
                ].numpy()
            )
        return np.vstack(vectors)

    def user_embeddings(
        self,
        users: pd.DataFrame,
        batch_size: int = 4096,
        movies: pd.DataFrame | None = None,
    ) -> np.ndarray:
        if not self.has_embedding_signatures:
            if movies is None:
                raise ValueError("movies are required when using an older model without embedding signatures")
            user_vectors, _, _ = self._embedding_pairs_from_variables(users, movies, batch_size)
            return user_vectors

        vectors = []
        for start in range(0, len(users), batch_size):
            end = min(start + batch_size, len(users))
            transformed = self._transform_user_batch(users.iloc[start:end])
            serialized = self._serialize_user_transformed(transformed, end - start)
            vectors.append(
                self.user_embedding_signature(examples=tf.constant(serialized))[
                    "user_embedding"
                ].numpy()
            )
        return np.vstack(vectors)

    def _embedding_pairs_from_variables(
        self,
        users: pd.DataFrame,
        movies: pd.DataFrame,
        batch_size: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fallback for older SavedModels that lack tower embedding signatures."""
        variables = self._load_embedding_variables()
        user_values = []
        movie_values = []
        genre_values = []
        for start in range(0, len(users), batch_size):
            end = min(start + batch_size, len(users))
            transformed = self._transform_batch(users.iloc[start:end], movies.iloc[start:end])
            user_ids = transformed["user_id"].numpy().reshape(-1)
            movie_ids = transformed["movie_id"].numpy().reshape(-1)
            user_values.append(variables["user_embedding/embeddings"][user_ids])
            movie_values.append(variables["movie_embedding/embeddings"][movie_ids])

            sparse_genres = transformed["genres"]
            genre_batch = np.zeros((end - start, variables["genre_embedding/embeddings"].shape[1]))
            genre_counts = np.zeros(end - start)
            for index, value in zip(sparse_genres.indices.numpy(), sparse_genres.values.numpy()):
                row = int(index[0])
                genre_batch[row] += variables["genre_embedding/embeddings"][int(value)]
                genre_counts[row] += 1
            genre_values.append(genre_batch / np.maximum(genre_counts[:, None], 1.0))

        user_array = np.vstack(user_values)
        movie_array = np.vstack(movie_values)
        movie_with_genres = np.concatenate([movie_array, np.vstack(genre_values)], axis=1)
        scores = self.score_pairs(users, movies, batch_size=batch_size)
        return user_array, movie_with_genres, scores

    def _load_embedding_variables(self) -> dict[str, np.ndarray]:
        if self._embedding_variables is None:
            self._embedding_variables = {
                variable.name.split(":")[0]: variable.numpy()
                for variable in self.model.variables
            }
        return self._embedding_variables

    def _transform_batch(self, users: pd.DataFrame, movies: pd.DataFrame) -> dict:
        raw = {
            "user_id": tf.constant(users["user_id"].to_numpy().reshape(-1, 1), dtype=tf.int64),
            "movie_id": tf.constant(movies["movie_id"].to_numpy().reshape(-1, 1), dtype=tf.int64),
            "age": tf.constant(users["age"].to_numpy().reshape(-1, 1), dtype=tf.int64),
            "gender": tf.constant(users["gender"].astype(str).to_numpy().reshape(-1, 1)),
            "occupation": tf.constant(users["occupation"].astype(str).to_numpy().reshape(-1, 1)),
            "genres": tf.constant(movies["genres"].astype(str).to_numpy().reshape(-1, 1)),
            "label": tf.zeros((len(users), 1), dtype=tf.int64),
            "timestamp": tf.zeros((len(users), 1), dtype=tf.int64),
        }
        return self.transform_output.transform_raw_features(raw)

    def _transform_user_batch(self, users: pd.DataFrame) -> dict:
        n_rows = len(users)
        raw = {
            "user_id": tf.constant(users["user_id"].to_numpy().reshape(-1, 1), dtype=tf.int64),
            "movie_id": tf.ones((n_rows, 1), dtype=tf.int64),
            "age": tf.constant(users["age"].to_numpy().reshape(-1, 1), dtype=tf.int64),
            "gender": tf.constant(users["gender"].astype(str).to_numpy().reshape(-1, 1)),
            "occupation": tf.constant(users["occupation"].astype(str).to_numpy().reshape(-1, 1)),
            "genres": tf.fill((n_rows, 1), "unknown"),
            "label": tf.zeros((n_rows, 1), dtype=tf.int64),
            "timestamp": tf.zeros((n_rows, 1), dtype=tf.int64),
        }
        return self.transform_output.transform_raw_features(raw)

    def _transform_movie_batch(self, movies: pd.DataFrame) -> dict:
        n_rows = len(movies)
        raw = {
            "user_id": tf.ones((n_rows, 1), dtype=tf.int64),
            "movie_id": tf.constant(movies["movie_id"].to_numpy().reshape(-1, 1), dtype=tf.int64),
            "age": tf.zeros((n_rows, 1), dtype=tf.int64),
            "gender": tf.fill((n_rows, 1), ""),
            "occupation": tf.fill((n_rows, 1), ""),
            "genres": tf.constant(movies["genres"].astype(str).to_numpy().reshape(-1, 1)),
            "label": tf.zeros((n_rows, 1), dtype=tf.int64),
            "timestamp": tf.zeros((n_rows, 1), dtype=tf.int64),
        }
        return self.transform_output.transform_raw_features(raw)

    def _serialize_transformed(
        self,
        transformed: dict,
        n_rows: int,
    ) -> list[bytes]:
        return self._serialize_feature_subset(
            transformed,
            n_rows,
            dense_keys=["user_id", "movie_id", "age", "gender", "occupation"],
            include_genres=True,
        )

    def _serialize_user_transformed(self, transformed: dict, n_rows: int) -> list[bytes]:
        return self._serialize_feature_subset(
            transformed,
            n_rows,
            dense_keys=["user_id", "age", "gender", "occupation"],
            include_genres=False,
        )

    def _serialize_movie_transformed(self, transformed: dict, n_rows: int) -> list[bytes]:
        return self._serialize_feature_subset(
            transformed,
            n_rows,
            dense_keys=["movie_id"],
            include_genres=True,
        )

    @staticmethod
    def _serialize_feature_subset(
        transformed: dict,
        n_rows: int,
        dense_keys: list[str],
        include_genres: bool,
    ) -> list[bytes]:
        genre_rows = [[] for _ in range(n_rows)]
        if include_genres:
            genre_sparse = transformed["genres"]
            for index, value in zip(genre_sparse.indices.numpy(), genre_sparse.values.numpy()):
                genre_rows[int(index[0])].append(int(value))

        serialized = []
        dense_values = {key: transformed[key].numpy() for key in dense_keys}
        for row_idx in range(n_rows):
            features = {
                key: tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(dense_values[key][row_idx, 0])])
                )
                for key in dense_keys
            }
            if include_genres:
                features["genres"] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=genre_rows[row_idx])
                )
            example = tf.train.Example(features=tf.train.Features(feature=features))
            serialized.append(example.SerializeToString())
        return serialized


def generate_top_k_candidates(
    users: pd.DataFrame,
    movies: pd.DataFrame,
    k: int = config.CANDIDATES_PER_USER,
    batch_size: int = 4096,
) -> pd.DataFrame:
    """Return top K movies per user, using an ANN index when available."""
    scorer = RetrievalCandidateScorer()
    if scorer.has_embedding_signatures and config.ANN_INDEX_FILE.exists():
        return generate_top_k_candidates_from_ann(scorer, users, k=k, batch_size=batch_size)

    return generate_top_k_candidates_bruteforce(scorer, users, movies, k=k, batch_size=batch_size)


def generate_top_k_candidates_from_ann(
    scorer: RetrievalCandidateScorer,
    users: pd.DataFrame,
    k: int = config.CANDIDATES_PER_USER,
    batch_size: int = 4096,
) -> pd.DataFrame:
    """Query the exported movie nearest-neighbor index with user embeddings."""
    artifact = joblib.load(config.ANN_INDEX_FILE)
    index = artifact["index"]
    movie_ids = artifact["movie_ids"]
    user_rows = users.reset_index(drop=True)

    user_vectors = scorer.user_embeddings(user_rows, batch_size=batch_size)
    distances, indices = index.kneighbors(user_vectors, n_neighbors=k)

    frames = []
    for row_idx, user_id in enumerate(user_rows["user_id"].astype(int)):
        frames.append(
            pd.DataFrame(
                {
                    "user_id": user_id,
                    "movie_id": movie_ids[indices[row_idx]].astype(int),
                    "candidate_score": _sigmoid_scaled_dot(1.0 - distances[row_idx]),
                    "retrieval_rank": np.arange(1, len(indices[row_idx]) + 1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def generate_top_k_candidates_bruteforce(
    scorer: RetrievalCandidateScorer,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    k: int = config.CANDIDATES_PER_USER,
    batch_size: int = 4096,
) -> pd.DataFrame:
    """Score all user/movie pairs from tower signatures and keep top K movies per user."""
    if not scorer.has_embedding_signatures:
        raise RuntimeError(
            "Latest retrieval SavedModel does not expose user_embedding/movie_embedding "
            "signatures. Rerun the TFX retrieval pipeline after the trainer_module.py "
            "signature update."
        )

    candidate_frames = []
    users = users.reset_index(drop=True)
    movies = movies.reset_index(drop=True)

    if users.empty or movies.empty:
        return pd.DataFrame(
            columns=["user_id", "movie_id", "candidate_score", "retrieval_rank"]
        )

    top_k = min(k, len(movies))
    movie_vectors = scorer.movie_embeddings(movies, batch_size=batch_size)
    user_vectors = scorer.user_embeddings(users, batch_size=batch_size)

    for row_idx, user in users.iterrows():
        scores = _sigmoid_scaled_dot(movie_vectors @ user_vectors[row_idx])
        top_idx = np.argsort(scores)[::-1][:top_k]
        candidate_frames.append(
            pd.DataFrame(
                {
                    "user_id": int(user["user_id"]),
                    "movie_id": movies.iloc[top_idx]["movie_id"].astype(int).to_numpy(),
                    "candidate_score": scores[top_idx],
                    "retrieval_rank": np.arange(1, len(top_idx) + 1),
                }
            )
        )

    return pd.concat(candidate_frames, ignore_index=True)


def latest_numeric_subdir(parent: Path) -> Path:
    subdirs = [path for path in parent.iterdir() if path.is_dir() and path.name.isdigit()]
    if not subdirs:
        raise FileNotFoundError(f"No numeric run directories found under {parent}")
    return sorted(subdirs, key=lambda path: int(path.name))[-1]


def latest_retrieval_model_dir() -> Path:
    return latest_numeric_subdir(config.PIPELINE_ROOT / "Trainer" / "model") / "Format-Serving"


def latest_pushed_model_dir() -> Path:
    return latest_numeric_subdir(config.SERVING_MODEL_DIR)


def latest_transform_graph_dir() -> Path:
    return latest_numeric_subdir(config.PIPELINE_ROOT / "Transform" / "transform_graph")
