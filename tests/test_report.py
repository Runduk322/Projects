"""Тесты сборки итогового отчёта."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ofd_analytics import schema as s
from ofd_analytics.report import _md_table, _money, _percent, analyze


def test_money_and_percent_formatting() -> None:
    assert _money(1401448.17) == "1 401 448,17 ₽"
    assert _money(-200) == "-200,00 ₽"
    assert _money("нет") == "нет"
    assert _percent(0.5963) == "59.6%"
    assert _percent(None) == "None"


def test_md_table_renders_header_and_rows() -> None:
    table = _md_table(pd.DataFrame({"a": [1, 2], "b": ["x", None]}))
    lines = table.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 1 | x |"
    assert lines[3] == "| 2 |  |", "пропуск отображается пустой ячейкой"
    assert _md_table(pd.DataFrame()) == "_нет данных_"


def test_md_table_respects_limit() -> None:
    table = _md_table(pd.DataFrame({"a": range(10)}), limit=3)
    assert len(table.splitlines()) == 5, "шапка, разделитель и три строки"


def test_analyze_produces_all_sections(tiny: pd.DataFrame) -> None:
    result = analyze(tiny)
    for heading in (
        "# Аналитика чеков ОФД",
        "## Показатели периода",
        "## Динамика по месяцам",
        "## Профиль недели",
        "## Торговые точки",
        "## ABC/XYZ ассортимента",
        "## Что покупают вместе",
        "## RFM-сегменты",
        "## Удержание по когортам",
    ):
        assert heading in result.markdown, heading
    assert result.notes == []
    assert result.kpi["revenue"] == pytest.approx(430.0)


def test_analyze_exposes_tables(tiny: pd.DataFrame) -> None:
    result = analyze(tiny)
    assert {"monthly_revenue", "abc_xyz", "basket_rules", "rfm", "cohort_retention"} <= set(
        result.tables
    )
    assert not result.tables["rfm"].empty


def test_month_column_is_human_readable(tiny: pd.DataFrame) -> None:
    assert "| 2025-01 |" in result_markdown(tiny)


def result_markdown(frame: pd.DataFrame) -> str:
    return analyze(frame).markdown


def test_analyze_notes_missing_customer_column(tiny: pd.DataFrame) -> None:
    result = analyze(tiny.drop(columns=[s.CUSTOMER_ID]))
    joined = " ".join(result.notes)
    assert "RFM пропущен" in joined
    assert "Когортный анализ пропущен" in joined
    assert "## RFM-сегменты" not in result.markdown
    assert "## Что не посчитано" in result.markdown


def test_analyze_survives_returns_only_data(tiny: pd.DataFrame) -> None:
    """Выручка отрицательная: ABC неприменим, но отчёт обязан собраться."""
    returns_only = tiny[tiny[s.OPERATION] == s.OP_RETURN].reset_index(drop=True)
    result = analyze(returns_only)
    joined = " ".join(result.notes)
    assert "ABC/XYZ пропущен" in joined
    assert "связок выше порога не найдено" in result.markdown
    assert result.kpi["revenue"] == pytest.approx(-200.0)


def test_save_writes_report_and_tables(tiny: pd.DataFrame, tmp_path: Path) -> None:
    files = analyze(tiny).save(tmp_path)
    assert files[0].name == "report.md"
    assert files[0].read_text(encoding="utf-8").startswith("# Аналитика чеков ОФД")
    names = {path.name for path in files}
    assert "rfm_segments.csv" in names
    assert "cohort_retention.csv" in names
    assert all(path.exists() for path in files)


def test_save_creates_missing_directory(tiny: pd.DataFrame, tmp_path: Path) -> None:
    target = tmp_path / "nested" / "reports"
    files = analyze(tiny).save(target)
    assert target.is_dir()
    assert len(files) > 1


def test_top_n_limits_tables(generated: pd.DataFrame) -> None:
    markdown = analyze(generated, top_n=3).markdown
    section = markdown.split("## Что покупают вместе")[1].split("\n## ")[0]
    rows = [line for line in section.splitlines() if line.startswith("| ") and "---" not in line]
    assert len(rows) == 4, "шапка плюс ровно три правила"
