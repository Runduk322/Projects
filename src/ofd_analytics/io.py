"""Чтение и запись чековых таблиц.

Выгрузки ОФД приходят в CSV, который редко бывает «правильным»: разделитель
чаще ``;``, кодировка бывает cp1251, а дробная часть — через запятую. Здесь это
определяется автоматически, чтобы файл от бухгалтерии читался без ручной правки.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

#: Кодировки в порядке проверки: BOM-utf8 из Excel, обычный utf-8, cp1251.
ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1251")

#: Разделители-кандидаты для CSV.
DELIMITERS: str = ";,\t|"


def sniff_csv_dialect(path: str | Path, sample_bytes: int = 64_000) -> tuple[str, str]:
    """Определить кодировку и разделитель CSV-файла.

    Returns:
        Пара ``(encoding, delimiter)``.

    Raises:
        UnicodeDecodeError: если файл не читается ни в одной из :data:`ENCODINGS`.
    """
    raw = Path(path).read_bytes()[:sample_bytes]
    for encoding in ENCODINGS:
        try:
            sample = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        lines = sample.splitlines()
        if not lines:
            # Пустой файл: разделитель не важен, ошибку поднимет уже pandas.
            return encoding, ","
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=DELIMITERS).delimiter
        except csv.Error:
            # Одна колонка или слишком короткий файл — разделитель не важен.
            delimiter = ";" if ";" in lines[0] else ","
        return encoding, delimiter
    raise UnicodeDecodeError("ofd", raw, 0, 1, f"не удалось прочитать в {ENCODINGS}")


def read_receipts(path: str | Path, **kwargs: object) -> pd.DataFrame:
    """Прочитать таблицу позиций чеков из CSV или Parquet.

    Для CSV кодировка, разделитель и десятичный разделитель определяются сами;
    любой явно переданный ``kwargs`` перебивает автоопределение.
    """
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, **kwargs)  # type: ignore[arg-type]

    encoding, delimiter = sniff_csv_dialect(path)
    options: dict[str, object] = {
        "encoding": encoding,
        "sep": delimiter,
        # Всё читаем как текст: в выгрузках встречаются «1 234,56», «20%» и
        # пустые строки. Числа разбирает ofd_analytics.clean.to_number.
        "dtype": str,
    }
    options.update(kwargs)
    frame = pd.read_csv(path, **options)  # type: ignore[arg-type]
    return frame.rename(columns=str.strip)


def write_receipts(frame: pd.DataFrame, path: str | Path, **kwargs: object) -> Path:
    """Записать таблицу в CSV или Parquet (формат — по расширению). Вернуть путь."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False, **kwargs)  # type: ignore[arg-type]
        return path
    options: dict[str, object] = {"index": False, "encoding": "utf-8-sig"}
    options.update(kwargs)
    frame.to_csv(path, **options)  # type: ignore[arg-type]
    return path
