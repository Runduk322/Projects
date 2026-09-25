"""Тесты чтения выгрузок: кодировки, разделители, форматы."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ofd_analytics.io import read_receipts, sniff_csv_dialect, write_receipts


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1251"])
@pytest.mark.parametrize("delimiter", [";", ",", "\t"])
def test_sniff_detects_dialect(tmp_path: Path, encoding: str, delimiter: str) -> None:
    path = tmp_path / "export.csv"
    frame = pd.DataFrame(
        {"Номер ФД": ["R1", "R2"], "Наименование товара": ["Хлеб", "Масло"], "Кол-во": [1, 2]}
    )
    frame.to_csv(path, sep=delimiter, index=False, encoding=encoding)

    detected_encoding, detected_delimiter = sniff_csv_dialect(path)
    assert detected_delimiter == delimiter
    # utf-8 без BOM читается как utf-8-sig — важно лишь, что кириллица не побилась.
    assert pd.read_csv(path, sep=delimiter, encoding=detected_encoding).shape == (2, 3)


def test_read_receipts_reads_cp1251_semicolon(tmp_path: Path) -> None:
    path = tmp_path / "ofd.csv"
    path.write_text("Номер ФД;Наименование товара;Сумма\nR1;Хлеб;50,5\n", encoding="cp1251")
    frame = read_receipts(path)
    assert list(frame.columns) == ["Номер ФД", "Наименование товара", "Сумма"]
    assert frame.loc[0, "Наименование товара"] == "Хлеб"
    # Числа остаются текстом: их разбирает clean.to_number.
    assert frame.loc[0, "Сумма"] == "50,5"


def test_read_receipts_strips_header_spaces(tmp_path: Path) -> None:
    path = tmp_path / "spaces.csv"
    path.write_text(" Номер ФД ;  Сумма \nR1;10\n", encoding="utf-8")
    assert list(read_receipts(path).columns) == ["Номер ФД", "Сумма"]


def test_single_column_file_does_not_crash(tmp_path: Path) -> None:
    path = tmp_path / "one.csv"
    path.write_text("Наименование\nХлеб\n", encoding="utf-8")
    assert read_receipts(path).shape == (1, 1)


def test_write_receipts_creates_parent_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deep" / "out.csv"
    written = write_receipts(pd.DataFrame({"a": [1]}), target)
    assert written == target
    assert target.exists()


def test_csv_round_trip_keeps_cyrillic(tmp_path: Path) -> None:
    frame = pd.DataFrame({"Наименование товара": ["Кофе латте 0.4"], "Кол-во": ["1"]})
    path = write_receipts(frame, tmp_path / "rt.csv")
    pd.testing.assert_frame_equal(read_receipts(path), frame)


def test_parquet_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="parquet требует pyarrow")
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = write_receipts(frame, tmp_path / "data.parquet")
    pd.testing.assert_frame_equal(read_receipts(path), frame)


def test_unreadable_encoding_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.csv"
    # 0x98 не определён в cp1251, а BOM-байты ломают utf-8 — не читается ни одной кодировкой.
    path.write_bytes(b"\xff\xfe\x98\x98header")
    with pytest.raises(UnicodeDecodeError):
        sniff_csv_dialect(path)
