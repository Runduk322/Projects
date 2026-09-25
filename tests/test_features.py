"""Тесты перехода между уровнями позиции и чека."""

from __future__ import annotations

import pandas as pd
import pytest

from ofd_analytics import schema as s
from ofd_analytics.features import kpi_summary, receipt_totals


def test_receipt_totals_collapses_lines(tiny: pd.DataFrame) -> None:
    totals = receipt_totals(tiny)
    assert list(totals[s.RECEIPT_ID]) == ["R1", "R2", "R3", "R4"], "отсортировано по времени"
    r1 = totals.set_index(s.RECEIPT_ID).loc["R1"]
    assert r1["lines"] == 2
    assert r1["items"] == pytest.approx(2.0)
    assert r1["categories"] == 2
    assert r1["total"] == pytest.approx(250.0)
    assert r1[s.STORE_ID] == "S01"
    assert r1[s.CUSTOMER_ID] == "C1"


def test_receipt_totals_keeps_returns_negative(tiny: pd.DataFrame) -> None:
    totals = receipt_totals(tiny).set_index(s.RECEIPT_ID)
    assert totals.loc["R4", "total"] == pytest.approx(-200.0)
    assert totals.loc["R4", s.OPERATION] == s.OP_RETURN


def test_receipt_totals_on_empty_frame() -> None:
    totals = receipt_totals(pd.DataFrame(columns=list(s.COLUMN_ORDER)))
    assert totals.empty
    assert "total" in totals.columns


def test_kpi_summary_exact_numbers(tiny: pd.DataFrame) -> None:
    kpi = kpi_summary(tiny)
    assert kpi["rows"] == 6
    assert kpi["receipts"] == 4
    assert kpi["sales_receipts"] == 3
    assert kpi["returns_receipts"] == 1
    assert kpi["revenue"] == pytest.approx(430.0)
    assert kpi["returns_amount"] == pytest.approx(-200.0)
    assert kpi["avg_ticket"] == pytest.approx(210.0), "630 рублей продаж на 3 чека"
    assert kpi["items_per_receipt"] == pytest.approx(2.0)
    assert kpi["lines_per_receipt"] == pytest.approx(1.67)
    assert kpi["unique_items"] == 3
    assert kpi["stores"] == 2
    assert kpi["customers"] == 2
    assert kpi["identified_receipts_share"] == pytest.approx(0.75)


def test_kpi_summary_on_empty_frame() -> None:
    kpi = kpi_summary(pd.DataFrame(columns=list(s.COLUMN_ORDER)))
    assert kpi == {"rows": 0, "receipts": 0, "revenue": 0.0}


def test_kpi_summary_without_optional_columns(tiny: pd.DataFrame) -> None:
    reduced = tiny.drop(columns=[s.STORE_ID, s.CUSTOMER_ID, s.CATEGORY])
    kpi = kpi_summary(reduced)
    assert "stores" not in kpi
    assert "customers" not in kpi
    assert kpi["revenue"] == pytest.approx(430.0)
