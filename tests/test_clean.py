"""Тесты нормализации и очистки — самой ответственной части пайплайна."""

from __future__ import annotations

import pandas as pd
import pytest

from ofd_analytics import schema as s
from ofd_analytics.clean import (
    MissingColumnsError,
    canonicalize_item_names,
    clean_receipts,
    coerce_types,
    normalize_columns,
    parse_datetime,
    require_customers,
    sales_only,
    to_number,
)


def test_normalize_columns_renames_known_and_reports_unknown() -> None:
    raw = pd.DataFrame(columns=["Номер ФД", "Дата и время", "Кол-во", "ИНН поставщика"])
    frame, unknown = normalize_columns(raw)
    assert list(frame.columns) == [s.RECEIPT_ID, s.RECEIPT_DT, s.QTY, "ИНН поставщика"]
    assert unknown == ["ИНН поставщика"]


def test_normalize_columns_rejects_ambiguous_mapping() -> None:
    raw = pd.DataFrame(columns=["Сумма", "Стоимость"])
    with pytest.raises(ValueError, match="обе отображаются"):
        normalize_columns(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1 234,56", 1234.56),
        ("20%", 20.0),
        ("99 руб.", 99.0),
        ("1 000", 1000.0),
        ("50.5", 50.5),
        ("", None),
        ("—", None),
    ],
)
def test_to_number(raw: str, expected: float | None) -> None:
    result = to_number(pd.Series([raw])).iloc[0]
    if expected is None:
        assert pd.isna(result)
    else:
        assert result == pytest.approx(expected)


def test_to_number_passes_through_numeric() -> None:
    pd.testing.assert_series_equal(
        to_number(pd.Series([1, 2])), pd.Series([1.0, 2.0]), check_names=False
    )


def test_canonicalize_item_names_collapses_variants() -> None:
    names = pd.Series(["Кофе латте", "  КОФЕ ЛАТТЕ ", "кофе  латте", "Кофе латте", "Хлеб"])
    result = canonicalize_item_names(names)
    assert result.iloc[:4].nunique() == 1
    assert result.iloc[0] == "Кофе латте", "представитель — самое частое написание"
    assert result.iloc[4] == "Хлеб"


def test_canonicalize_item_names_keeps_missing() -> None:
    result = canonicalize_item_names(pd.Series(["Хлеб", None]))
    assert result.iloc[0] == "Хлеб"
    assert pd.isna(result.iloc[1])


def test_coerce_types_maps_ffd_wording() -> None:
    raw = pd.DataFrame(
        {
            s.RECEIPT_DT: ["01.02.2025 10:00:00"],
            s.OPERATION: ["Возврат прихода"],
            s.PAYMENT_TYPE: ["Безналичными"],
            s.QTY: ["2,5"],
            s.CUSTOMER_ID: ["  "],
        }
    )
    frame = coerce_types(raw)
    assert frame.loc[0, s.OPERATION] == s.OP_RETURN
    assert frame.loc[0, s.PAYMENT_TYPE] == s.PAY_CARD
    assert frame.loc[0, s.QTY] == pytest.approx(2.5)
    assert pd.isna(frame.loc[0, s.CUSTOMER_ID])
    assert frame.loc[0, s.RECEIPT_DT] == pd.Timestamp("2025-02-01 10:00:00")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # ISO однозначен и должен остаться как есть.
        ("2025-03-06 18:30:00", "2025-03-06 18:30:00"),
        ("2025-03-06", "2025-03-06"),
        ("2025-03-06T18:30:00", "2025-03-06 18:30:00"),
        # Российский формат с точками читается как день-месяц-год.
        ("06.03.2025 18:30:00", "2025-03-06 18:30:00"),
        ("01.02.2025 10:00:00", "2025-02-01 10:00:00"),
        ("31.12.2025", "2025-12-31"),
        ("06/03/2025", "2025-03-06"),
    ],
)
def test_parse_datetime_does_not_swap_day_and_month(raw: str, expected: str) -> None:
    """Регресс: разбор с dayfirst не должен ломать ISO-даты с днём не больше 12."""
    assert parse_datetime(pd.Series([raw])).iloc[0] == pd.Timestamp(expected)


def test_parse_datetime_handles_empty_and_garbage() -> None:
    parsed = parse_datetime(pd.Series(["", None, "не дата", "2025-01-01"]))
    assert parsed.isna().sum() == 3
    assert parsed.iloc[3] == pd.Timestamp("2025-01-01")


def test_parse_datetime_passes_through_datetime_dtype() -> None:
    series = pd.to_datetime(pd.Series(["2025-03-06 18:30:00"]))
    pd.testing.assert_series_equal(parse_datetime(series), series)


def test_missing_required_columns_raise() -> None:
    raw = pd.DataFrame({"Номер ФД": ["R1"], "Сумма": [100]})
    with pytest.raises(MissingColumnsError, match="receipt_dt"):
        clean_receipts(raw)


def _minimal(**overrides: object) -> pd.DataFrame:
    """Одна строка канонической схемы с возможностью переопределить поля."""
    row: dict[str, object] = {
        s.RECEIPT_ID: "R1",
        s.RECEIPT_DT: "2025-05-05 10:00:00",
        s.ITEM_NAME: "Хлеб",
        s.QTY: 2.0,
        s.PRICE: 50.0,
        s.LINE_SUM: 100.0,
        s.OPERATION: s.OP_SALE,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_duplicates_removed_and_counted() -> None:
    raw = pd.concat([_minimal(), _minimal()], ignore_index=True)
    frame, report = clean_receipts(raw, asof="2025-06-01")
    assert len(frame) == 1
    steps = {step.name: step.affected for step in report.steps}
    assert steps["удалены полные дубли строк"] == 1


def test_missing_line_sum_restored_from_price() -> None:
    frame, report = clean_receipts(_minimal(line_sum=None), asof="2025-06-01")
    assert frame.loc[0, s.LINE_SUM] == pytest.approx(100.0)
    assert any("восстановлена сумма" in step.name for step in report.steps)


def test_missing_price_restored_from_line_sum() -> None:
    frame, _ = clean_receipts(_minimal(price=None), asof="2025-06-01")
    assert frame.loc[0, s.PRICE] == pytest.approx(50.0)


def test_sum_strategy_sum_trusts_the_paid_amount() -> None:
    """Скидка живёт в сумме позиции, поэтому по умолчанию пересчитывается цена."""
    frame, report = clean_receipts(_minimal(line_sum=80.0), asof="2025-06-01")
    assert frame.loc[0, s.LINE_SUM] == pytest.approx(80.0)
    assert frame.loc[0, s.PRICE] == pytest.approx(40.0)
    assert any("расхождение суммы" in step.name for step in report.steps)


def test_sum_strategy_price_recomputes_the_sum() -> None:
    frame, _ = clean_receipts(_minimal(line_sum=80.0), sum_strategy="price", asof="2025-06-01")
    assert frame.loc[0, s.LINE_SUM] == pytest.approx(100.0)
    assert frame.loc[0, s.PRICE] == pytest.approx(50.0)


@pytest.mark.parametrize("qty", [0.0, -1.0, None])
def test_non_positive_qty_dropped(qty: float | None) -> None:
    frame, report = clean_receipts(_minimal(qty=qty), asof="2025-06-01")
    assert frame.empty
    assert any("непозитивным количеством" in step.name for step in report.steps)


def test_impossible_dates_dropped() -> None:
    raw = pd.concat(
        [_minimal(), _minimal(receipt_dt="2005-01-01 10:00:00"), _minimal(receipt_dt="2030-01-01")],
        ignore_index=True,
    )
    frame, report = clean_receipts(raw, asof="2025-06-01")
    assert len(frame) == 1
    steps = {step.name: step.affected for step in report.steps}
    assert steps["удалены невозможные даты"] == 2


def test_period_filter_includes_whole_last_day() -> None:
    raw = pd.concat(
        [_minimal(receipt_dt="2025-05-31 23:59:00"), _minimal(receipt_dt="2025-06-01 00:10:00")],
        ignore_index=True,
    )
    frame, _ = clean_receipts(raw, period=("2025-05-01", "2025-05-31"), asof="2025-07-01")
    assert len(frame) == 1
    assert frame.loc[0, s.RECEIPT_DT] == pd.Timestamp("2025-05-31 23:59:00")


def test_returns_become_negative() -> None:
    frame, _ = clean_receipts(_minimal(operation="Возврат прихода"), asof="2025-06-01")
    assert frame.loc[0, s.QTY] < 0
    assert frame.loc[0, s.LINE_SUM] < 0


def test_missing_operation_column_defaults_to_sale() -> None:
    raw = _minimal().drop(columns=[s.OPERATION])
    frame, report = clean_receipts(raw, asof="2025-06-01")
    assert frame.loc[0, s.OPERATION] == s.OP_SALE
    assert any("признак операции" in step.name for step in report.steps)


def test_missing_line_sum_column_is_computed() -> None:
    raw = _minimal().drop(columns=[s.LINE_SUM])
    frame, report = clean_receipts(raw, asof="2025-06-01")
    assert frame.loc[0, s.LINE_SUM] == pytest.approx(100.0)
    assert any("добавлена сумма позиции" in step.name for step in report.steps)


def test_output_columns_are_ordered() -> None:
    frame, _ = clean_receipts(_minimal(), asof="2025-06-01")
    assert list(frame.columns) == s.ordered_columns(frame.columns)


def test_report_markdown_mentions_counts() -> None:
    raw = pd.concat([_minimal(), _minimal()], ignore_index=True)
    _, report = clean_receipts(raw, asof="2025-06-01")
    text = report.to_markdown()
    assert "Очистка данных" in text
    assert "строк на входе" in text
    assert report.rows_dropped == 1
    assert list(report.as_frame().columns) == ["step", "rows", "note"]


def test_full_pipeline_on_messy_export(messy_export: pd.DataFrame) -> None:
    frame, report = clean_receipts(
        messy_export, period=("2025-01-01", "2025-12-31"), asof="2026-01-01"
    )
    assert not frame.empty
    assert report.rows_out < report.rows_in
    assert list(frame.columns) == s.ordered_columns(frame.columns)
    assert not frame.duplicated().any()
    assert frame[s.QTY].ne(0).all()
    assert frame[s.PRICE].notna().all()
    assert set(frame[s.OPERATION].unique()) <= {s.OP_SALE, s.OP_RETURN}
    assert set(frame[s.PAYMENT_TYPE].dropna().unique()) <= {s.PAY_CASH, s.PAY_CARD, s.PAY_PREPAID}
    assert frame[s.VAT_RATE].dropna().eq(20.0).all()
    returns = frame[frame[s.OPERATION] == s.OP_RETURN]
    assert returns[s.LINE_SUM].lt(0).all()


def test_require_customers_filters_anonymous(tiny: pd.DataFrame) -> None:
    identified = require_customers(tiny)
    assert set(identified[s.CUSTOMER_ID]) == {"C1", "C2"}
    assert len(identified) == 5


def test_require_customers_without_column_raises(tiny: pd.DataFrame) -> None:
    with pytest.raises(MissingColumnsError, match="customer_id"):
        require_customers(tiny.drop(columns=[s.CUSTOMER_ID]))


def test_sales_only_drops_returns(tiny: pd.DataFrame) -> None:
    assert len(sales_only(tiny)) == 5
    assert sales_only(tiny.drop(columns=[s.OPERATION])).shape[0] == 6
