"""Проверка, что пример из README не устарел.

Документация, которая не запускается в CI, гниёт первой, поэтому сквозной
сценарий из ``examples/walkthrough.py`` прогоняется как обычный тест.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import walkthrough


def test_walkthrough_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    clean = walkthrough.run(tmp_path, receipts=300, seed=3)
    assert not clean.empty
    assert (tmp_path / "report.md").exists()

    printed = capsys.readouterr().out
    for marker in ("После очистки", "Выручка", "RFM-сегменты", "Средняя кривая удержания"):
        assert marker in printed, marker


def test_walkthrough_main_returns_zero(tmp_path: Path) -> None:
    assert walkthrough.main(["--outdir", str(tmp_path), "--receipts", "200"]) == 0
