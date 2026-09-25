"""Тесты канонической схемы и разбора заголовков выгрузок."""

from __future__ import annotations

import pytest

from ofd_analytics import schema as s


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Дата и  Время ", "дата и время"),
        ("Признак расчёта", "признак расчета"),
        ('"Наименование товара"', "наименование товара"),
        ("КОЛ-ВО", "кол-во"),
        ("Сумма, руб.", "сумма руб"),
    ],
)
def test_normalize_key(raw: str, expected: str) -> None:
    assert s.normalize_key(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Номер ФД", s.RECEIPT_ID),
        ("дата и время", s.RECEIPT_DT),
        ("Кол-во", s.QTY),
        ("Признак расчёта", s.OPERATION),
        ("Телефон покупателя", s.CUSTOMER_ID),
        ("Товарная группа", s.CATEGORY),
        ("Сумма, руб.", s.LINE_SUM),
        ("Цена, руб.", s.PRICE),
    ],
)
def test_resolve_column_known(raw: str, expected: str) -> None:
    assert s.resolve_column(raw) == expected


def test_resolve_column_unknown() -> None:
    assert s.resolve_column("ИНН поставщика") is None


def test_every_canonical_column_has_alias() -> None:
    """У каждой колонки схемы есть хотя бы один синоним — иначе её не распознать."""
    covered = set(s.COLUMN_ALIASES.values())
    assert covered >= set(s.ALL_COLUMNS)


def test_canonical_names_resolve_to_themselves() -> None:
    """Уже канонические заголовки должны проходить нормализацию без изменений."""
    for column in s.ALL_COLUMNS:
        assert s.resolve_column(column) == column


def test_missing_required_keeps_schema_order() -> None:
    assert s.missing_required([s.QTY]) == [s.RECEIPT_ID, s.RECEIPT_DT, s.ITEM_NAME, s.PRICE]
    assert s.missing_required(s.ALL_COLUMNS) == []


def test_ordered_columns_puts_unknown_last() -> None:
    columns = ["ИНН", s.QTY, s.RECEIPT_ID]
    assert s.ordered_columns(columns) == [s.RECEIPT_ID, s.QTY, "ИНН"]


def test_operation_and_payment_aliases_cover_ffd_wording() -> None:
    assert s.OPERATION_ALIASES[s.normalize_key("Возврат прихода")] == s.OP_RETURN
    assert s.OPERATION_ALIASES[s.normalize_key("Приход")] == s.OP_SALE
    assert s.PAYMENT_ALIASES[s.normalize_key("Безналичными")] == s.PAY_CARD
    assert s.PAYMENT_ALIASES[s.normalize_key("Наличными")] == s.PAY_CASH
