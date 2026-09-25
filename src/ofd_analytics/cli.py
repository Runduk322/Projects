"""Командный интерфейс пайплайна: ``generate``, ``clean``, ``analyze``, ``demo``.

Пример полного прогона на синтетике::

    ofd demo --outdir build/demo

На своих данных::

    ofd clean --input выгрузка.csv --output data/clean.csv --period 2025-01-01 2025-12-31
    ofd analyze --input data/clean.csv --outdir reports/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .clean import MissingColumnsError, clean_receipts, coerce_types, normalize_columns
from .generate import generate_receipts, to_ofd_export
from .io import read_receipts, write_receipts
from .report import analyze


def _add_generate(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("generate", help="сгенерировать синтетическую выгрузку")
    parser.add_argument(
        "--output", "-o", required=True, type=Path, help="куда записать (csv/parquet)"
    )
    parser.add_argument("--receipts", type=int, default=5000, help="число чеков")
    parser.add_argument("--customers", type=int, default=400, help="размер базы покупателей")
    parser.add_argument("--stores", type=int, default=3, help="число торговых точек")
    parser.add_argument("--start", default="2025-01-01", help="начало периода")
    parser.add_argument("--end", default="2025-12-31", help="конец периода")
    parser.add_argument("--seed", type=int, default=42, help="зерно генератора")
    parser.add_argument("--messy", action="store_true", help="добавить типичный мусор выгрузки")
    parser.add_argument(
        "--ofd-style",
        action="store_true",
        help="записать русские заголовки колонок, как в выгрузке ОФД",
    )


def _add_clean(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("clean", help="нормализовать и очистить выгрузку")
    parser.add_argument("--input", "-i", required=True, type=Path, help="файл выгрузки")
    parser.add_argument("--output", "-o", required=True, type=Path, help="куда записать результат")
    parser.add_argument(
        "--period",
        nargs=2,
        metavar=("START", "END"),
        help="оставить только чеки внутри периода",
    )
    parser.add_argument(
        "--sum-strategy",
        choices=("sum", "price"),
        default="sum",
        help="чему верить при расхождении суммы и цены × количества",
    )
    parser.add_argument("--report", type=Path, help="куда записать отчёт об очистке (markdown)")


def _add_analyze(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("analyze", help="посчитать метрики и собрать отчёт")
    parser.add_argument("--input", "-i", required=True, type=Path, help="очищенные данные")
    parser.add_argument("--outdir", "-o", required=True, type=Path, help="каталог для отчёта")
    parser.add_argument("--asof", help="дата отсчёта для RFM (по умолчанию последний чек)")
    parser.add_argument("--top", type=int, default=10, help="размер топов в отчёте")
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="не отсеивать строки: только привести колонки и типы к схеме",
    )


def _add_demo(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "demo", help="полный прогон на синтетике: генерация → очистка → отчёт"
    )
    parser.add_argument("--outdir", "-o", required=True, type=Path, help="каталог для результатов")
    parser.add_argument("--receipts", type=int, default=5000, help="число чеков")
    parser.add_argument("--customers", type=int, default=400, help="размер базы покупателей")
    parser.add_argument("--seed", type=int, default=42, help="зерно генератора")


def build_parser() -> argparse.ArgumentParser:
    """Собрать парсер аргументов со всеми подкомандами."""
    parser = argparse.ArgumentParser(
        prog="ofd",
        description="Аналитика чековых данных ОФД: очистка, RFM, ABC/XYZ, корзина, когорты",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_generate(subparsers)
    _add_clean(subparsers)
    _add_analyze(subparsers)
    _add_demo(subparsers)
    return parser


def _run_generate(args: argparse.Namespace) -> int:
    frame = generate_receipts(
        n_receipts=args.receipts,
        n_customers=args.customers,
        n_stores=args.stores,
        start=args.start,
        end=args.end,
        seed=args.seed,
        messy=args.messy,
    )
    if args.ofd_style:
        frame = to_ofd_export(frame)
    path = write_receipts(frame, args.output)
    print(f"Записано строк: {len(frame):,}".replace(",", " "))
    print(f"Файл: {path}")
    return 0


def _run_clean(args: argparse.Namespace) -> int:
    raw = read_receipts(args.input)
    period = tuple(args.period) if args.period else None
    frame, report = clean_receipts(raw, period=period, sum_strategy=args.sum_strategy)
    path = write_receipts(frame, args.output)
    print(report.to_markdown())
    print(f"\nФайл: {path}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report.to_markdown() + "\n", encoding="utf-8")
        print(f"Отчёт об очистке: {args.report}")
    return 0


def _run_analyze(args: argparse.Namespace) -> int:
    frame = read_receipts(args.input)
    if args.skip_clean:
        # Схему и типы привести всё равно нужно: из CSV всё приходит текстом.
        frame, _ = normalize_columns(frame)
        frame = coerce_types(frame)
        print(f"Очистка пропущена, строк: {len(frame)}")
    else:
        frame, report = clean_receipts(frame)
        print(f"Очистка: {report.rows_in} → {report.rows_out} строк")
    result = analyze(frame, asof=args.asof, top_n=args.top)
    files = result.save(args.outdir)
    print(f"Отчёт: {files[0]}")
    print(f"Таблиц выгружено: {len(files) - 1}")
    for note in result.notes:
        print(f"Предупреждение: {note}")
    return 0


def _run_demo(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = to_ofd_export(
        generate_receipts(
            n_receipts=args.receipts,
            n_customers=args.customers,
            seed=args.seed,
            messy=True,
        )
    )
    raw_path = write_receipts(raw, outdir / "raw_ofd_export.csv")
    print(f"1/3 Сгенерирована выгрузка ОФД: {raw_path} ({len(raw):,} строк)".replace(",", " "))

    frame, report = clean_receipts(raw, period=("2025-01-01", "2025-12-31"), asof="2026-01-01")
    write_receipts(frame, outdir / "clean_receipts.csv")
    (outdir / "cleaning_report.md").write_text(report.to_markdown() + "\n", encoding="utf-8")
    print(f"2/3 Очищено: {report.rows_in} → {report.rows_out} строк, отчёт рядом с данными")

    result = analyze(frame, asof="2026-01-01")
    files = result.save(outdir)
    print(f"3/3 Отчёт собран: {files[0]}")
    print(f"    Таблиц: {len(files) - 1}, выручка: {result.kpi.get('revenue')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI. Возвращает код выхода процесса."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "generate": _run_generate,
        "clean": _run_clean,
        "analyze": _run_analyze,
        "demo": _run_demo,
    }
    try:
        return handlers[args.command](args)
    except (MissingColumnsError, FileNotFoundError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    except pd.errors.EmptyDataError:
        print("Ошибка: входной файл пуст", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
