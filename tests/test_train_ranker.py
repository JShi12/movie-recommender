"""Tests for ranking.training.train_ranker data-shaping helpers."""

from __future__ import annotations

import pandas as pd
import pytest

try:
    from ranking.training.train_ranker import group_counts, sorted_for_ranking
except (ImportError, OSError) as exc:  # pragma: no cover - environment guard
    # lightgbm needs the OpenMP runtime (libomp); skip locally when it is absent.
    pytest.skip(f"lightgbm unavailable: {exc}", allow_module_level=True)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [2, 1, 1, 2, 1],
            "candidate_score": [0.1, 0.9, 0.3, 0.7, 0.5],
            "label": [0, 1, 0, 1, 0],
        }
    )


class TestGroupCounts:
    def test_counts_rows_per_user_in_user_order(self):
        assert group_counts(_frame()) == [3, 2]  # user 1 has 3 rows, user 2 has 2

    def test_group_counts_sum_to_row_total(self):
        frame = _frame()
        assert sum(group_counts(frame)) == len(frame)


class TestSortedForRanking:
    def test_sorted_by_user_then_score_desc(self):
        out = sorted_for_ranking(_frame())
        assert out["user_id"].tolist() == [1, 1, 1, 2, 2]
        # within user 1, candidate_score descending
        u1 = out[out["user_id"] == 1]["candidate_score"].tolist()
        assert u1 == sorted(u1, reverse=True)

    def test_groups_are_contiguous_after_sorting(self):
        ordered = sorted_for_ranking(_frame()).reset_index(drop=True)
        # each user's rows form one unbroken block, matching group_counts order
        blocks = [
            len(list(g))
            for _, g in ordered.groupby((ordered["user_id"] != ordered["user_id"].shift()).cumsum())
        ]
        assert blocks == group_counts(_frame())
