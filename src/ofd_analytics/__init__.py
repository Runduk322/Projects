"""Аналитика чековых данных ОФД.

Пайплайн из четырёх шагов, каждый — отдельный модуль:

1. :mod:`ofd_analytics.io` — прочитать выгрузку (CSV в cp1251 с ``;`` тоже поймёт);
2. :mod:`ofd_analytics.clean` — привести к канонической схеме и почистить, с отчётом;
3. :mod:`ofd_analytics.analytics` — RFM, ABC/XYZ, анализ корзины, когорты, динамика;
4. :mod:`ofd_analytics.report` — собрать отчёт и таблицы.

Быстрый старт на синтетических данных::

    from ofd_analytics import generate_receipts, clean_receipts, analyze

    raw = generate_receipts(n_receipts=5000, seed=42)
    clean, report = clean_receipts(raw)
    print(report.to_markdown())
    print(analyze(clean).markdown)
"""

from __future__ import annotations

from . import schema
from .clean import (
    CleaningReport,
    MissingColumnsError,
    clean_receipts,
    normalize_columns,
    require_customers,
    sales_only,
)
from .features import kpi_summary, receipt_totals
from .generate import generate_receipts, to_ofd_export
from .io import read_receipts, write_receipts
from .report import AnalysisResult, analyze

__version__ = "0.1.0"

__all__ = [
    "AnalysisResult",
    "CleaningReport",
    "MissingColumnsError",
    "__version__",
    "analyze",
    "clean_receipts",
    "generate_receipts",
    "kpi_summary",
    "normalize_columns",
    "read_receipts",
    "receipt_totals",
    "require_customers",
    "sales_only",
    "schema",
    "to_ofd_export",
    "write_receipts",
]
