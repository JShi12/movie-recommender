"""Tests for the ranker bless/push threshold gates."""

from __future__ import annotations

import json

import pytest

from ranking.training.push_ranker import (
    _metric,
    latest_pushed_metrics_file,
    pushed_version_dirs,
    validate_ndcg_improvement,
    validate_thresholds,
)

PASSING = {"ndcg@10": 0.20, "recall@100": 0.42}


class TestMetric:
    def test_reads_and_casts(self):
        assert _metric({"ndcg@10": "0.5"}, "ndcg@10") == 0.5

    def test_missing_raises(self):
        with pytest.raises(ValueError, match="missing or null"):
            _metric({}, "ndcg@10")

    def test_none_raises(self):
        with pytest.raises(ValueError, match="missing or null"):
            _metric({"ndcg@10": None}, "ndcg@10")


class TestValidateThresholds:
    def test_passes_when_metrics_meet_floor(self):
        assert validate_thresholds(PASSING, 0.15, 0.40) is None

    def test_passes_on_exact_equality(self):
        validate_thresholds(PASSING, 0.20, 0.42)

    def test_fails_below_ndcg_floor(self):
        with pytest.raises(RuntimeError, match="ndcg@10"):
            validate_thresholds(PASSING, 0.25, 0.40)

    def test_failure_message_lists_every_breach(self):
        with pytest.raises(RuntimeError) as exc:
            validate_thresholds(PASSING, 0.25, 0.50)
        assert "ndcg@10" in str(exc.value) and "recall@100" in str(exc.value)

    def test_missing_metric_raises_value_error(self):
        with pytest.raises(ValueError):
            validate_thresholds({"ndcg@10": 0.2}, 0.1, 0.1)


class TestValidateNdcgImprovement:
    def test_no_baseline_is_a_noop(self):
        assert validate_ndcg_improvement(PASSING, None) == (None, None)

    def test_returns_positive_delta_when_improved(self, tmp_path):
        baseline = tmp_path / "end_to_end_metrics.json"
        baseline.write_text(json.dumps({"ndcg@10": 0.18}))
        prev, delta = validate_ndcg_improvement(PASSING, baseline)
        assert prev == pytest.approx(0.18)
        assert delta == pytest.approx(0.02)

    def test_raises_when_not_better_than_baseline(self, tmp_path):
        baseline = tmp_path / "end_to_end_metrics.json"
        baseline.write_text(json.dumps({"ndcg@10": 0.20}))
        with pytest.raises(RuntimeError, match="improvement check"):
            validate_ndcg_improvement(PASSING, baseline)


class TestPushedRegistryScan:
    def _version(self, root, name, metrics=None):
        d = root / name
        d.mkdir()
        if metrics is not None:
            (d / "end_to_end_metrics.json").write_text(json.dumps(metrics))
        return d

    def test_only_dirs_with_metrics_count(self, tmp_path):
        self._version(tmp_path, "20260101000000", {"ndcg@10": 0.1})
        self._version(tmp_path, "20260201000000", None)  # no metrics file
        found = {p.name for p in pushed_version_dirs(tmp_path)}
        assert found == {"20260101000000"}

    def test_latest_is_lexicographically_last(self, tmp_path):
        self._version(tmp_path, "20260101000000", {"ndcg@10": 0.1})
        self._version(tmp_path, "20260501120000", {"ndcg@10": 0.3})
        latest = latest_pushed_metrics_file(tmp_path)
        assert latest is not None
        assert json.loads(latest.read_text())["ndcg@10"] == 0.3

    def test_missing_registry_returns_none(self, tmp_path):
        assert latest_pushed_metrics_file(tmp_path / "does-not-exist") is None
