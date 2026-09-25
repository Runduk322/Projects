"""Тесты генератора синтетических чеков."""

from __future__ import annotations

import pandas as pd
import pytest

from ofd_analytics import schema as s
from ofd_analytics.generate import generate_receipts, to_ofd_export


def test_schema_and_size() -> None:
    frame = generate_receipts(n_receipts=200, n_customers=40, seed=1)
    assert list(frame.columns) == list(s.COLUMN_ORDER)
    assert frame[s.RECEIPT_ID].nunique() == 200
    assert len(frame) >= 200, "в каждом чеке хотя бы одна позиция"
    assert frame[s.LINE_SUM].gt(0).all()


def test_same_seed_gives_identical_data() -> None:
    first = generate_receipts(n_receipts=150, n_customers=30, seed=99)
    second = generate_receipts(n_receipts=150, n_customers=30, seed=99)
    pd.testing.assert_frame_equal(first, second)


def test_different_seed_gives_different_data() -> None:
    first = generate_receipts(n_receipts=150, n_customers=30, seed=1)
    second = generate_receipts(n_receipts=150, n_customers=30, seed=2)
    assert not first.equals(second)


def test_period_respected() -> None:
    frame = generate_receipts(n_receipts=100, n_customers=20, start="2024-03-01", end="2024-03-31")
    assert frame[s.RECEIPT_DT].min() >= pd.Timestamp("2024-03-01")
    assert frame[s.RECEIPT_DT].max() <= pd.Timestamp("2024-04-01")


def test_line_sum_matches_price_times_qty() -> None:
    frame = generate_receipts(n_receipts=120, n_customers=20, seed=4)
    expected = (frame[s.QTY] * frame[s.PRICE]).round(2)
    pd.testing.assert_series_equal(frame[s.LINE_SUM], expected, check_names=False)


def test_identified_share_is_respected() -> None:
    receipts = generate_receipts(n_receipts=800, n_customers=150, seed=5, identified_share=0.0)
    assert receipts[s.CUSTOMER_ID].isna().all()


def test_returns_are_flagged_not_negative_in_raw_data() -> None:
    """В выгрузке ОФД возврат помечен признаком расчёта, а суммы положительные."""
    frame = generate_receipts(n_receipts=500, n_customers=50, seed=6, return_share=0.5)
    returns = frame[frame[s.OPERATION] == s.OP_RETURN]
    assert not returns.empty
    assert returns[s.LINE_SUM].gt(0).all()


def test_messy_mode_adds_problems() -> None:
    clean = generate_receipts(n_receipts=300, n_customers=40, seed=7)
    messy = generate_receipts(n_receipts=300, n_customers=40, seed=7, messy=True)
    assert len(messy) > len(clean), "дубли строк"
    assert messy.duplicated().any()
    assert messy[s.PRICE].isna().any()
    assert messy[s.QTY].le(0).any()
    assert (messy[s.VAT_RATE].astype("string") == "20%").any()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_receipts": 0}, "n_receipts"),
        ({"n_customers": 0}, "n_customers"),
        ({"identified_share": 1.5}, "identified_share"),
        ({"return_share": -0.1}, "return_share"),
    ],
)
def test_invalid_arguments_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        generate_receipts(**kwargs)  # type: ignore[arg-type]


def test_end_before_start_rejected() -> None:
    with pytest.raises(ValueError, match="Пустой период"):
        generate_receipts(n_receipts=10, n_customers=5, start="2025-05-01", end="2025-04-01")


def test_ofd_export_headers_are_recognised_by_schema() -> None:
    """Главная связка: то, что выдаёт экспорт, обязано распознаваться схемой."""
    export = to_ofd_export(generate_receipts(n_receipts=30, n_customers=10, seed=8))
    for column in export.columns:
        assert s.resolve_column(column) is not None, column
    assert "Приход" in set(export["Признак расчёта"])
