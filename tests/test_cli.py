"""Тесты командного интерфейса."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ofd_analytics import schema as s
from ofd_analytics.cli import build_parser, main


def test_parser_requires_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_generate_writes_canonical_columns(tmp_path: Path) -> None:
    target = tmp_path / "raw.csv"
    assert main(["generate", "-o", str(target), "--receipts", "50", "--customers", "10"]) == 0
    frame = pd.read_csv(target)
    assert list(frame.columns) == list(s.COLUMN_ORDER)
    assert frame[s.RECEIPT_ID].nunique() == 50


def test_generate_ofd_style_writes_russian_headers(tmp_path: Path) -> None:
    target = tmp_path / "ofd.csv"
    assert (
        main(["generate", "-o", str(target), "--receipts", "20", "--customers", "5", "--ofd-style"])
        == 0
    )
    columns = pd.read_csv(target).columns
    assert "Номер ФД" in columns
    assert "Признак расчёта" in columns


def test_full_chain_generate_clean_analyze(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = tmp_path / "raw.csv"
    clean = tmp_path / "clean.csv"
    outdir = tmp_path / "reports"

    assert (
        main(
            [
                "generate",
                "-o",
                str(raw),
                "--receipts",
                "300",
                "--customers",
                "40",
                "--messy",
                "--ofd-style",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "clean",
                "-i",
                str(raw),
                "-o",
                str(clean),
                "--period",
                "2025-01-01",
                "2025-12-31",
                "--report",
                str(tmp_path / "cleaning.md"),
            ]
        )
        == 0
    )
    cleaned = pd.read_csv(clean)
    assert list(cleaned.columns) == s.ordered_columns(cleaned.columns)
    assert (tmp_path / "cleaning.md").read_text(encoding="utf-8").startswith("### Очистка данных")

    assert main(["analyze", "-i", str(clean), "-o", str(outdir)]) == 0
    assert (outdir / "report.md").exists()
    assert (outdir / "rfm_segments.csv").exists()
    assert "Отчёт:" in capsys.readouterr().out


def test_analyze_can_skip_cleaning(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    main(["generate", "-o", str(raw), "--receipts", "60", "--customers", "10"])
    assert main(["analyze", "-i", str(raw), "-o", str(tmp_path / "out"), "--skip-clean"]) == 0


def test_demo_runs_end_to_end(tmp_path: Path) -> None:
    assert main(["demo", "-o", str(tmp_path), "--receipts", "200", "--customers", "30"]) == 0
    for name in ("raw_ofd_export.csv", "clean_receipts.csv", "cleaning_report.md", "report.md"):
        assert (tmp_path / name).exists(), name


def test_missing_input_file_returns_error_code(tmp_path: Path) -> None:
    code = main(["clean", "-i", str(tmp_path / "нет.csv"), "-o", str(tmp_path / "out.csv")])
    assert code == 2


def test_unrecognised_export_returns_error_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "strange.csv"
    source.write_text("Колонка А;Колонка Б\n1;2\n", encoding="utf-8")
    assert main(["clean", "-i", str(source), "-o", str(tmp_path / "out.csv")]) == 2
    assert "Ошибка" in capsys.readouterr().err


def test_empty_file_returns_error_code(tmp_path: Path) -> None:
    source = tmp_path / "empty.csv"
    source.write_text("", encoding="utf-8")
    assert main(["clean", "-i", str(source), "-o", str(tmp_path / "out.csv")]) == 2
