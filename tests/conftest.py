"""Общие фикстуры тестов.

``tiny`` — шесть строк, все ожидаемые значения по которым посчитаны руками и
зафиксированы в тестах. ``generated`` — большой синтетический набор: на нём
проверяются свойства, а не конкретные числа.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ofd_analytics import schema as s
from ofd_analytics.clean import clean_receipts
from ofd_analytics.generate import generate_receipts

#: Колонки фикстуры ``tiny`` в каноническом порядке.
TINY_COLUMNS = list(s.COLUMN_ORDER)

#: Четыре чека: два у покупателя C1, один у C2 и один анонимный возврат.
TINY_ROWS = [
    (
        "R1",
        "2025-01-06 10:00:00",
        "S01",
        "C1",
        "Хлеб",
        "Бакалея",
        1.0,
        50.0,
        50.0,
        20.0,
        "card",
        "sale",
    ),
    (
        "R1",
        "2025-01-06 10:00:00",
        "S01",
        "C1",
        "Масло",
        "Молочное",
        1.0,
        200.0,
        200.0,
        20.0,
        "card",
        "sale",
    ),
    (
        "R2",
        "2025-01-07 12:00:00",
        "S01",
        "C1",
        "Хлеб",
        "Бакалея",
        2.0,
        50.0,
        100.0,
        20.0,
        "cash",
        "sale",
    ),
    (
        "R3",
        "2025-02-10 18:00:00",
        "S02",
        "C2",
        "Масло",
        "Молочное",
        1.0,
        200.0,
        200.0,
        20.0,
        "card",
        "sale",
    ),
    (
        "R3",
        "2025-02-10 18:00:00",
        "S02",
        "C2",
        "Молоко",
        "Молочное",
        1.0,
        80.0,
        80.0,
        20.0,
        "card",
        "sale",
    ),
    (
        "R4",
        "2025-02-11 09:00:00",
        "S02",
        None,
        "Масло",
        "Молочное",
        -1.0,
        200.0,
        -200.0,
        20.0,
        "card",
        "return",
    ),
]


@pytest.fixture
def tiny() -> pd.DataFrame:
    """Маленькая уже очищенная таблица с известными наперёд метриками."""
    frame = pd.DataFrame(TINY_ROWS, columns=TINY_COLUMNS)
    frame[s.RECEIPT_DT] = pd.to_datetime(frame[s.RECEIPT_DT])
    for column in (s.RECEIPT_ID, s.STORE_ID, s.CUSTOMER_ID, s.ITEM_NAME, s.CATEGORY):
        frame[column] = frame[column].astype("string")
    return frame


@pytest.fixture(scope="session")
def generated() -> pd.DataFrame:
    """Синтетические данные среднего размера, уже очищенные."""
    frame, _ = clean_receipts(generate_receipts(n_receipts=1200, n_customers=120, seed=17))
    return frame


@pytest.fixture(scope="session")
def messy_export() -> pd.DataFrame:
    """Грязная выгрузка с русскими заголовками — вход для очистки."""
    from ofd_analytics.generate import to_ofd_export

    return to_ofd_export(generate_receipts(n_receipts=600, n_customers=80, seed=23, messy=True))
