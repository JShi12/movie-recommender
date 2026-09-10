"""Tests for shared.feature_tables joining and label derivation."""

from __future__ import annotations

import pandas as pd

from shared.feature_tables import (
    load_joined_movielens,
    movie_feature_table,
    user_feature_table,
)
from shared.movielens import GENRE_COLUMNS


class TestLoadJoinedMovielens:
    def test_label_rule(self, mini_raw_dir):
        data = load_joined_movielens(mini_raw_dir)
        assert (data.loc[data["rating"] >= 4, "label"] == 1).all()
        assert (data.loc[data["rating"] <= 2, "label"] == 0).all()
        assert data.loc[data["rating"] == 3, "label"].isna().all()

    def test_joins_user_and_movie_columns(self, mini_raw_dir):
        data = load_joined_movielens(mini_raw_dir)
        assert len(data) == 25  # one row per rating, nothing dropped by the join
        for col in ("age", "gender", "occupation", "title", "genres", "release_year"):
            assert col in data.columns
        assert data[["age", "gender", "occupation"]].notna().all().all()

    def test_release_year_backfilled_when_missing(self, mini_raw_dir):
        # Fixture movie 6 has an empty release_date; year must still be numeric.
        data = load_joined_movielens(mini_raw_dir)
        assert data["release_year"].notna().all()

    def test_genres_string_is_pipe_joined_active_flags(self, mini_raw_dir):
        data = load_joined_movielens(mini_raw_dir)
        toy_story = data[data["movie_id"] == 1].iloc[0]
        assert set(toy_story["genres"].split("|")) == {"Animation", "Children", "Comedy"}


class TestFeatureTables:
    def test_movie_feature_table_is_one_row_per_movie(self, mini_raw_dir):
        movies = movie_feature_table(mini_raw_dir)
        assert movies["movie_id"].is_unique
        assert len(movies) == 6
        assert set(GENRE_COLUMNS).issubset(movies.columns)

    def test_user_feature_table_is_one_row_per_user(self, mini_raw_dir):
        users = user_feature_table(mini_raw_dir)
        assert users["user_id"].is_unique
        assert len(users) == 5
        assert {"age", "gender", "occupation"}.issubset(users.columns)
        assert pd.api.types.is_integer_dtype(users["age"])
