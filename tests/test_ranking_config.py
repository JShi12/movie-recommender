"""Tests for ranking.config helpers."""

from __future__ import annotations

import pytest

from ranking.config import latest_numeric_subdir


class TestLatestNumericSubdir:
    def test_picks_highest_numeric_directory(self, tmp_path):
        for name in ("1", "2", "10", "9"):
            (tmp_path / name).mkdir()
        (tmp_path / "not-a-run").mkdir()
        (tmp_path / "3").write_text("a file, not a dir")
        assert latest_numeric_subdir(tmp_path).name == "10"

    def test_raises_when_no_numeric_subdirs(self, tmp_path):
        (tmp_path / "latest").mkdir()
        with pytest.raises(FileNotFoundError, match="No numeric run directories"):
            latest_numeric_subdir(tmp_path)
