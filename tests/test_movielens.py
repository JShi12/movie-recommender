"""Tests for shared.movielens: loading, temporal split, genre flattening."""

from __future__ import annotations

import pandas as pd
import pytest

from shared.movielens import (
    GENRE_COLUMNS,
    SplitFractions,
    active_genres,
    load_movielens_100k,
    time_based_split,
)


def _chronological(n: int) -> pd.DataFrame:
    """n rows with strictly increasing, unique timestamps."""
    return pd.DataFrame(
        {
            "user_id": range(n),
            "movie_id": range(n),
            "rating": [4] * n,
            "timestamp": range(1_000, 1_000 + n),
        }
    )


class TestSplitFractions:
    def test_test_fraction_is_remainder(self):
        assert SplitFractions(0.8, 0.1).test == pytest.approx(0.1)

    def test_defaults_sum_to_one(self):
        f = SplitFractions()
        assert f.train + f.validation + f.test == pytest.approx(1.0)


class TestTimeBasedSplit:
    def test_sizes_and_order(self):
        train, val, test = time_based_split(_chronological(100), SplitFractions(0.8, 0.1))
        assert (len(train), len(val), len(test)) == (80, 10, 10)
        # Concatenation reproduces the whole, still time-sorted.
        combined = pd.concat([train, val, test])
        assert combined["timestamp"].is_monotonic_increasing

    def test_boundaries_do_not_overlap(self):
        train, val, test = time_based_split(_chronological(50), SplitFractions(0.8, 0.1))
        assert train["timestamp"].max() < val["timestamp"].min()
        assert val["timestamp"].max() < test["timestamp"].min()

    def test_input_row_order_does_not_matter(self):
        shuffled = _chronological(30).sample(frac=1, random_state=1)
        train, val, test = time_based_split(shuffled, SplitFractions(0.8, 0.1))
        assert list(train["timestamp"]) == sorted(train["timestamp"])
        assert train["timestamp"].max() < test["timestamp"].min()

    def test_rejects_nonpositive_fraction(self):
        with pytest.raises(ValueError, match="Invalid split fractions"):
            time_based_split(_chronological(10), SplitFractions(1.0, 0.0))

    def test_clean_chronological_data_passes_leakage_guards(self):
        # Strictly increasing timestamps -> no raise, blocks strictly ordered.
        train, val, test = time_based_split(_chronological(40), SplitFractions(0.8, 0.1))
        assert train["timestamp"].max() < val["timestamp"].min() < test["timestamp"].min()

    def test_boundary_timestamp_never_spans_two_partitions(self):
        # Timestamp 25 straddles the nominal 90% (val/test) row boundary; the
        # whole group must land in val, not be split across val and test.
        ts = list(range(24)) + [24, 24, 25, 25, 26, 26]
        frame = pd.DataFrame(
            {
                "user_id": range(30),
                "movie_id": range(30),
                "rating": [4] * 30,
                "timestamp": ts,
            }
        )
        train, val, test = time_based_split(frame, SplitFractions(0.8, 0.1))
        assert train["timestamp"].max() < val["timestamp"].min()
        assert val["timestamp"].max() < test["timestamp"].min()
        # No timestamp appears in more than one partition.
        assert not (set(train["timestamp"]) & set(val["timestamp"]))
        assert not (set(val["timestamp"]) & set(test["timestamp"]))
        assert 25 in set(val["timestamp"]) and 25 not in set(test["timestamp"])

    def test_raises_when_timestamps_too_concentrated_to_split(self):
        flat = pd.DataFrame(
            {
                "user_id": range(20),
                "movie_id": range(20),
                "rating": [4] * 20,
                "timestamp": [42] * 20,
            }
        )
        with pytest.raises(ValueError, match="empty (validation|test) partition"):
            time_based_split(flat, SplitFractions(0.8, 0.1))


class TestActiveGenres:
    def test_joins_active_flags(self):
        row = pd.Series(dict.fromkeys(GENRE_COLUMNS, 0) | {"Action": 1, "Comedy": 1})
        assert active_genres(row) == "Action|Comedy"

    def test_falls_back_to_unknown(self):
        row = pd.Series(dict.fromkeys(GENRE_COLUMNS, 0))
        assert active_genres(row) == "unknown"


class TestLoadMovielens100k:
    def test_shapes_and_columns(self, mini_raw_dir):
        ratings, users, movies = load_movielens_100k(mini_raw_dir)
        assert list(ratings.columns) == ["user_id", "movie_id", "rating", "timestamp"]
        assert len(ratings) == 25
        assert len(users) == 5
        assert {"age", "gender", "occupation"}.issubset(users.columns)
        assert set(GENRE_COLUMNS).issubset(movies.columns)
        assert len(movies) == 6
