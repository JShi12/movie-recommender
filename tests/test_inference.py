"""Tests for the TensorFlow-free serving path (ranking.inference)."""

from __future__ import annotations

import numpy as np
import pytest

from ranking.features import RANKING_FEATURES


class TestRecommenderQueries:
    def test_knows_all_movielens_users(self, recommender):
        ids = recommender.known_user_ids
        assert len(ids) == 943
        assert recommender.has_user(1)
        assert not recommender.has_user(10_000)

    def test_user_history_is_recent_first_with_liked_flag(self, recommender):
        hist = recommender.user_history(1, limit=10)
        assert list(hist.columns) == ["movie_id", "title", "genres", "rating", "liked", "timestamp"]
        assert len(hist) <= 10
        assert hist["timestamp"].is_monotonic_decreasing
        assert (hist["liked"] == (hist["rating"] >= 4)).all()


class TestRecommend:
    def test_returns_ranked_unseen_titles(self, recommender):
        recs = recommender.recommend(1, k=10, n_candidates=200)
        assert 1 <= len(recs) <= 10
        assert all(r.title for r in recs)
        assert all(r.retrieval_rank >= 1 for r in recs)
        # scores are sorted descending (ranker output)
        scores = [r.ranker_score for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_excludes_already_rated_by_default(self, recommender):
        seen = set(recommender.history.loc[recommender.history["user_id"] == 1, "movie_id"])
        recs = recommender.recommend(1, k=25, n_candidates=300, exclude_seen=True)
        assert seen.isdisjoint({r.movie_id for r in recs})

    def test_exclude_seen_false_can_surface_seen_movies(self, recommender):
        with_seen = recommender.recommend(1, k=25, n_candidates=300, exclude_seen=False)
        without = recommender.recommend(1, k=25, n_candidates=300, exclude_seen=True)
        assert {r.movie_id for r in with_seen} != {r.movie_id for r in without}

    def test_k_caps_the_result_size(self, recommender):
        assert len(recommender.recommend(1, k=5, n_candidates=200)) == 5

    def test_unknown_user_raises_keyerror(self, recommender):
        with pytest.raises(KeyError):
            recommender.recommend(10_000)


class TestFeatureParity:
    def test_feature_frame_has_every_ranking_feature_and_no_nan(self, recommender):
        candidates = recommender._retrieve(1, 150)
        frame = recommender._build_features(1, candidates)
        missing = [c for c in RANKING_FEATURES if c not in frame.columns]
        assert not missing
        block = frame[RANKING_FEATURES].to_numpy(dtype=float)
        assert np.isfinite(block).all()

    def test_retrieve_scores_are_descending_probabilities(self, recommender):
        candidates = recommender._retrieve(1, 100)
        assert candidates["retrieval_rank"].tolist() == list(range(1, 101))
        s = candidates["candidate_score"].to_numpy()
        assert ((s > 0) & (s < 1)).all()
        assert (np.diff(s) <= 1e-9).all()  # non-increasing with rank
