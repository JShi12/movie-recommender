"""Tests for ranking.features: ranking metrics, leakage-safe history, genre affinity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ranking.features import (
    BASE_RANKING_FEATURES,
    add_genre_affinity,
    add_historical_observed_features,
    finalize_features,
    ndcg_at_k,
    recall_at_k,
    user_genre_preferences,
)

LOG2_3 = float(np.log2(3))  # discount for rank 2 == 1/log2(3)


class TestNdcgAtK:
    def test_perfect_ranking_scores_one(self):
        assert ndcg_at_k([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], k=4) == pytest.approx(1.0)

    def test_worst_single_swap(self):
        # One positive, ranked second -> DCG = 1/log2(3), IDCG = 1.
        assert ndcg_at_k([1, 0], [0.1, 0.9], k=2) == pytest.approx(1.0 / LOG2_3)

    def test_no_positives_returns_zero(self):
        assert ndcg_at_k([0, 0, 0], [0.3, 0.2, 0.1], k=3) == 0.0

    def test_k_truncates(self):
        # Only the top-1 counts; the positive sits at rank 3 -> DCG 0.
        assert ndcg_at_k([0, 0, 1], [0.9, 0.8, 0.7], k=1) == 0.0


class TestRecallAtK:
    def test_fraction_of_positives_in_top_k(self):
        assert recall_at_k([1, 1, 0, 0, 1], [5, 1, 4, 3, 2], k=2) == pytest.approx(1 / 3)

    def test_all_positives_recovered(self):
        assert recall_at_k([1, 0, 1], [0.2, 0.1, 0.3], k=3) == pytest.approx(1.0)

    def test_no_positives_returns_zero(self):
        assert recall_at_k([0, 0], [0.1, 0.2], k=2) == 0.0


class TestAddHistoricalObservedFeatures:
    def test_before_features_exclude_the_current_row(self, observed_history):
        out = add_historical_observed_features(observed_history)
        user1 = out[out["user_id"] == 1].sort_values("timestamp")

        # user 1 ratings in time order: 5, 3, 4, 2
        counts = user1["user_rating_count_before"].tolist()
        assert counts == [0, 1, 2, 3]  # never counts itself

        avg_before = user1["user_avg_rating_before"].tolist()
        # row 1 has no history -> filled with the global mean, not 5.
        assert avg_before[0] != pytest.approx(5.0)
        assert avg_before[1] == pytest.approx(5.0)  # mean(5)
        assert avg_before[2] == pytest.approx(4.0)  # mean(5, 3)
        assert avg_before[3] == pytest.approx(4.0)  # mean(5, 3, 4)

    def test_like_rate_before_is_causal(self, observed_history):
        out = add_historical_observed_features(observed_history)
        user1 = out[out["user_id"] == 1].sort_values("timestamp")
        # _is_like over user 1 (rating >= 4): 1, 0, 1, 0
        assert user1["user_like_rate_before"].tolist()[1:] == pytest.approx([1.0, 0.5, 2 / 3])

    def test_no_nulls_left_in_filled_columns(self, observed_history):
        out = add_historical_observed_features(observed_history)
        for col in (
            "user_avg_rating_before",
            "user_like_rate_before",
            "movie_avg_rating_before",
            "movie_like_rate_before",
            "user_activity_gap_log",
        ):
            assert out[col].notna().all()

    def test_row_count_preserved(self, observed_history):
        out = add_historical_observed_features(observed_history)
        assert len(out) == len(observed_history)


class TestGenreAffinity:
    def test_user_genre_preferences_normalise_to_one(self, observed_history):
        prefs = user_genre_preferences(observed_history)
        pref_cols = [c for c in prefs.columns if c.startswith("user_pref_")]
        totals = prefs[pref_cols].sum(axis=1)
        assert totals.tolist() == pytest.approx([1.0, 1.0])
        row1 = prefs[prefs["user_id"] == 1].iloc[0]
        assert row1["user_pref_Action"] == pytest.approx(2 / 3)
        assert row1["user_pref_Comedy"] == pytest.approx(1 / 3)

    def test_affinity_is_pref_dot_movie_genres(self, observed_history):
        out = add_genre_affinity(observed_history.copy(), observed_history)
        user1 = out[out["user_id"] == 1].set_index("movie_id")["user_genre_affinity"]
        # user 1: pref Action=2/3, Comedy=1/3
        assert user1.loc[10] == pytest.approx(2 / 3)  # Action only
        assert user1.loc[12] == pytest.approx(1.0)  # Action + Comedy
        assert user1.loc[13] == pytest.approx(0.0)  # Drama, no affinity

    def test_pref_columns_are_dropped_after_use(self, observed_history):
        out = add_genre_affinity(observed_history.copy(), observed_history)
        assert not [c for c in out.columns if c.startswith("user_pref_")]


class TestFinalizeFeatures:
    def _prepared(self, observed_history: pd.DataFrame) -> pd.DataFrame:
        frame = add_historical_observed_features(observed_history)
        return finalize_features(frame, observed_history)

    def test_produces_all_base_features_without_nan_or_inf(self, observed_history):
        out = self._prepared(observed_history)
        block = out[BASE_RANKING_FEATURES].to_numpy(dtype=float)
        assert np.isfinite(block).all()

    def test_user_minus_movie_avg_is_the_difference(self, observed_history):
        out = self._prepared(observed_history)
        expected = out["user_avg_rating_before"] - out["movie_avg_rating_before"]
        pd.testing.assert_series_equal(out["user_avg_minus_movie_avg"], expected, check_names=False)

    def test_infinities_are_replaced(self, observed_history):
        frame = add_historical_observed_features(observed_history)
        frame.loc[frame.index[0], "user_avg_rating_before"] = np.inf
        out = finalize_features(frame, observed_history)
        assert np.isfinite(out["user_avg_rating_before"]).all()
