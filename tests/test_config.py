"""Tests for shared.config helpers."""

from __future__ import annotations

from pathlib import Path

from shared.config import PROJECT_ROOT, relpath


def test_relpath_is_repo_relative_and_posix():
    p = PROJECT_ROOT / "artifacts" / "ranker" / "metrics.json"
    assert relpath(p) == "artifacts/ranker/metrics.json"


def test_relpath_accepts_str():
    assert relpath(str(PROJECT_ROOT / "data" / "x.parquet")) == "data/x.parquet"


def test_relpath_falls_back_to_absolute_when_outside_repo(tmp_path: Path):
    outside = tmp_path / "elsewhere" / "model"
    assert relpath(outside) == outside.resolve().as_posix()
