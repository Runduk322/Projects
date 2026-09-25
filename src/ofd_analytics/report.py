"""Сборка сводного аналитического отчёта по чековым данным.

Модуль не считает ничего нового — он собирает результаты аналитических модулей
в один Markdown-документ и набор CSV-таблиц. Markdown выбран сознательно: он
читается в терминале, в GitHub и в Notion без конвертации.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import schema as s
from .analytics import abc as abc_module
from .analytics import basket, cohorts, rfm, timeseries
from .clean import MissingColumnsError
from .features import kpi_summary
from .io import write_receipts

#: Сколько строк показывать в топах отчёта.
TOP_N = 10


def _money(value: object) -> str:
    """Отформатировать сумму как деньги: ``1 401 448,17 ₽``."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    formatted = f"{number:,.2f}".replace(",", " ").replace(".", ",")
    return f"{formatted} ₽"


def _percent(value: object, digits: int = 1) -> str:
    """Отформатировать долю как процент."""
    try:
        return f"{float(value) * 100:.{digits}f}%"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def _md_table(frame: pd.DataFrame, limit: int | None = None) -> str:
    """Отрисовать DataFrame Markdown-таблицей без внешних зависимостей."""
    if frame.empty:
        return "_нет данных_"
    shown = frame.head(limit) if limit else frame
    headers = [str(column) for column in shown.columns]
    rows = [["" if pd.isna(cell) else str(cell) for cell in row] for row in shown.to_numpy()]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


@dataclass
class AnalysisResult:
    """Результат анализа: показатели, таблицы и готовый Markdown-отчёт."""

    kpi: dict[str, object]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    markdown: str = ""
    notes: list[str] = field(default_factory=list)

    def save(self, outdir: str | Path) -> list[Path]:
        """Сохранить отчёт и все таблицы в каталог. Вернуть список файлов."""
        directory = Path(outdir)
        directory.mkdir(parents=True, exist_ok=True)
        written = [directory / "report.md"]
        written[0].write_text(self.markdown, encoding="utf-8")
        for name, table in self.tables.items():
            if not table.empty:
                index = table.index.name is not None or isinstance(table.index, pd.PeriodIndex)
                written.append(
                    write_receipts(
                        table.reset_index() if index else table, directory / f"{name}.csv"
                    )
                )
        return written


def _kpi_section(kpi: dict[str, object]) -> str:
    """Заголовочный блок с показателями периода."""
    lines = ["## Показатели периода", ""]
    period = f"{kpi.get('period_start')} — {kpi.get('period_end')}"
    lines.append(f"- период: **{period}**")
    lines.append(f"- чеков: **{kpi.get('receipts', 0):,}**".replace(",", " "))
    lines.append(f"- позиций: **{kpi.get('rows', 0):,}**".replace(",", " "))
    lines.append(f"- чистая выручка: **{_money(kpi.get('revenue', 0))}**")
    lines.append(f"- средний чек: **{_money(kpi.get('avg_ticket', 0))}**")
    lines.append(f"- товаров в чеке: **{kpi.get('items_per_receipt', 0)}**")
    lines.append(
        f"- возвраты: **{_money(kpi.get('returns_amount', 0))}** "
        f"({kpi.get('returns_receipts', 0)} чеков)"
    )
    if "customers" in kpi:
        lines.append(f"- покупателей опознано: **{kpi['customers']:,}**".replace(",", " "))
        lines.append(
            f"- доля чеков с покупателем: **{_percent(kpi.get('identified_receipts_share', 0))}**"
        )
    if "stores" in kpi:
        lines.append(f"- торговых точек: **{kpi['stores']}**")
    return "\n".join(lines)


def analyze(
    frame: pd.DataFrame,
    asof: str | pd.Timestamp | None = None,
    top_n: int = TOP_N,
) -> AnalysisResult:
    """Посчитать весь набор метрик и собрать отчёт.

    Args:
        frame: очищенная таблица позиций чеков.
        asof: дата отсчёта для RFM; по умолчанию последний чек в данных.
        top_n: сколько строк показывать в топах.

    Returns:
        :class:`AnalysisResult` с Markdown-отчётом и таблицами для выгрузки.
        Разделы, для которых не хватает данных (например, RFM без
        ``customer_id``), пропускаются, а причина попадает в ``notes``.
    """
    kpi = kpi_summary(frame)
    result = AnalysisResult(kpi=kpi)
    parts = ["# Аналитика чеков ОФД", "", _kpi_section(kpi)]

    monthly = timeseries.add_growth(timeseries.revenue_by(frame, "ME"))
    daily = timeseries.add_moving_average(timeseries.revenue_by(frame, "D"))
    weekday = timeseries.weekday_profile(frame)
    hourly = timeseries.hour_profile(frame)
    result.tables |= {
        "monthly_revenue": monthly,
        "daily_revenue": daily,
        "weekday_profile": weekday,
        "hour_profile": hourly,
    }
    monthly_display = monthly.reset_index().rename(columns={s.RECEIPT_DT: "month"})
    monthly_display["month"] = monthly_display["month"].dt.strftime("%Y-%m")
    parts += ["", "## Динамика по месяцам", "", _md_table(monthly_display)]
    parts += ["", "## Профиль недели", "", _md_table(weekday)]
    best_hours = hourly.nlargest(5, "receipts")[["hour", "receipts", "revenue", "avg_ticket"]]
    parts += ["", "## Часы наибольшего потока", "", _md_table(best_hours)]

    if s.STORE_ID in frame:
        stores = timeseries.group_profile(frame, s.STORE_ID)
        result.tables["store_profile"] = stores
        parts += ["", "## Торговые точки", "", _md_table(stores)]

    try:
        combined = abc_module.abc_xyz(frame)
    except ValueError as error:
        result.notes.append(f"ABC/XYZ пропущен: {error}")
    else:
        result.tables["abc_xyz"] = combined
        result.tables["abc_xyz_matrix"] = abc_module.abc_xyz_matrix(combined)
        top_items = combined.head(top_n)[
            [s.ITEM_NAME, s.LINE_SUM, "share", "cum_share", "abc", "xyz", "class"]
        ]
        distribution = (
            combined.groupby("abc")
            .agg(items=(s.ITEM_NAME, "nunique"), revenue=(s.LINE_SUM, "sum"))
            .reset_index()
        )
        distribution["revenue_share"] = (
            distribution["revenue"] / distribution["revenue"].sum()
        ).round(4)
        parts += ["", "## ABC/XYZ ассортимента", "", _md_table(distribution)]
        parts += ["", f"Топ-{top_n} товаров по вкладу в выручку:", "", _md_table(top_items)]

    rules = basket.pair_rules(frame, top=top_n)
    result.tables["basket_rules"] = rules
    parts += ["", "## Что покупают вместе", ""]
    if rules.empty:
        parts.append("_связок выше порога не найдено_")
    else:
        parts.append(
            _md_table(
                rules[["antecedent", "consequent", "receipts", "support", "confidence", "lift"]]
            )
        )

    try:
        scored = rfm.rfm(frame, asof=asof)
    except MissingColumnsError as error:
        result.notes.append(f"RFM пропущен: {error}")
    else:
        if scored.empty:
            result.notes.append("RFM пропущен: нет чеков с известным покупателем")
        else:
            segments = rfm.segment_summary(scored)
            result.tables |= {"rfm": scored, "rfm_segments": segments}
            parts += ["", "## RFM-сегменты", "", _md_table(segments)]

    try:
        matrix = cohorts.cohort_matrix(frame)
    except MissingColumnsError as error:
        result.notes.append(f"Когортный анализ пропущен: {error}")
    else:
        if matrix.empty:
            result.notes.append("Когортный анализ пропущен: нет чеков с известным покупателем")
        else:
            rates = cohorts.retention_rates(matrix)
            curve = cohorts.retention_curve(rates)
            result.tables |= {
                "cohort_customers": matrix,
                "cohort_retention": rates,
                "cohort_sizes": cohorts.cohort_sizes(frame),
            }
            parts += ["", "## Удержание по когортам", "", _md_table(rates.reset_index())]
            readable = ", ".join(
                f"M{offset}: {_percent(value)}" for offset, value in curve.head(7).items()
            )
            parts += ["", f"Средняя кривая удержания — {readable}"]

    if result.notes:
        parts += ["", "## Что не посчитано", ""] + [f"- {note}" for note in result.notes]

    result.markdown = "\n".join(parts) + "\n"
    return result
