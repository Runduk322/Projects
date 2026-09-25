"""Пайплайн шаг за шагом: от выгрузки ОФД до готовых метрик.

Запуск::

    python examples/walkthrough.py --outdir build/walkthrough

Файл намеренно написан как линейный скрипт, а не как набор функций: его удобно
читать сверху вниз и переносить кусками в свой ноутбук. Если нужен ноутбук,
конвертируйте: ``jupytext --to notebook examples/walkthrough.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ofd_analytics import clean_receipts, generate_receipts, kpi_summary
from ofd_analytics import schema as s
from ofd_analytics.analytics import abc, basket, cohorts, rfm, timeseries
from ofd_analytics.generate import to_ofd_export
from ofd_analytics.report import analyze


def run(outdir: Path, receipts: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Пройти весь путь и вернуть очищенную таблицу."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Шаг 1. Выгрузка. Здесь она синтетическая; на своих данных замените строку на
    #   raw = read_receipts("выгрузка_ОФД.csv")
    raw = to_ofd_export(generate_receipts(n_receipts=receipts, seed=seed, messy=True))
    print(f"Сырая выгрузка: {raw.shape[0]} строк, колонки как у ОФД")
    print(f"  {list(raw.columns)[:4]} ...")

    # Шаг 2. Очистка. Отчёт показывает, что именно изменилось, — его стоит читать
    # каждый раз: расхождение сумм и дубли строк встречаются почти в любой выгрузке.
    clean, cleaning = clean_receipts(raw, period=("2025-01-01", "2025-12-31"), asof="2026-01-01")
    print(f"\nПосле очистки: {cleaning.rows_in} → {cleaning.rows_out} строк")
    for step in cleaning.steps:
        print(f"  {step.name}: {step.affected}")

    # Шаг 3. Показатели периода — то, что спрашивают первым.
    kpi = kpi_summary(clean)
    print(f"\nВыручка: {kpi['revenue']:,.2f} ₽".replace(",", " "))
    print(f"Средний чек: {kpi['avg_ticket']} ₽, товаров в чеке: {kpi['items_per_receipt']}")
    print(
        f"Покупателей опознано: {kpi['customers']} ({kpi['identified_receipts_share']:.0%} чеков)"
    )

    # Шаг 4. Динамика. Дневной ряд сглаживаем, месячный смотрим с приростом.
    monthly = timeseries.add_growth(timeseries.revenue_by(clean, "ME"))
    print("\nМесячная динамика (выручка, прирост):")
    for period, row in monthly.iterrows():
        growth = "—" if pd.isna(row["revenue_growth"]) else f"{row['revenue_growth']:+.1%}"
        print(f"  {period:%Y-%m}: {row['revenue']:>12,.0f} ₽  {growth}".replace(",", " "))

    # Шаг 5. Ассортимент: что приносит деньги и что стабильно.
    combined = abc.abc_xyz(clean)
    print("\nТоп-5 товаров по вкладу в выручку:")
    for _, row in combined.head(5).iterrows():
        print(f"  {row[s.ITEM_NAME]:<28} {row['share']:>6.1%}  класс {row['class']}")

    # Шаг 6. Корзина. Смотреть нужно на lift, а не на support.
    rules = basket.pair_rules(clean, top=5)
    print("\nСильнейшие связки:")
    for _, row in rules.iterrows():
        print(
            f"  {row['antecedent']} → {row['consequent']}: "
            f"lift {row['lift']:.2f}, уверенность {row['confidence']:.0%}"
        )

    # Шаг 7. Покупатели: RFM-сегменты.
    scored = rfm.rfm(clean)
    print("\nRFM-сегменты:")
    for _, row in rfm.segment_summary(scored).iterrows():
        print(
            f"  {row['segment']:<18} {row['customers']:>4} чел. "
            f"({row['customers_share']:>5.1%})  выручка {row['revenue_share']:>5.1%}"
        )

    # Шаг 8. Удержание: возвращаются ли покупатели.
    curve = cohorts.retention_curve(cohorts.retention_rates(cohorts.cohort_matrix(clean)))
    readable = ", ".join(f"M{offset}: {value:.0%}" for offset, value in curve.head(6).items())
    print(f"\nСредняя кривая удержания — {readable}")

    # Шаг 9. Всё то же самое одним вызовом, с выгрузкой в файлы.
    files = analyze(clean, asof="2026-01-01").save(outdir)
    print(f"\nОтчёт: {files[0]}")
    print(f"Таблиц выгружено: {len(files) - 1}")
    return clean


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("build/walkthrough"))
    parser.add_argument("--receipts", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    run(args.outdir, receipts=args.receipts, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
