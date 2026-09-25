"""Агрегаты над позициями чеков: уровень чека и сводные показатели.

Чековые данные приходят построчно, а почти все вопросы бизнеса задаются на
уровне чека («средний чек», «сколько позиций в корзине») или всего периода.
Здесь один раз описан переход между уровнями, чтобы остальные модули не
пересчитывали его каждый по-своему.
"""

from __future__ import annotations

import pandas as pd

from . import schema as s


def receipt_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Свернуть позиции в чеки: сумма, количество позиций и товаров, категории.

    Returns:
        DataFrame с одной строкой на чек. Реквизиты чека (время, точка,
        покупатель) берутся из первой позиции — внутри чека они одинаковы.
    """
    if frame.empty:
        return pd.DataFrame(
            columns=[
                s.RECEIPT_ID,
                s.RECEIPT_DT,
                s.STORE_ID,
                s.CUSTOMER_ID,
                s.PAYMENT_TYPE,
                s.OPERATION,
                "lines",
                "items",
                "categories",
                "total",
            ]
        )

    aggregations: dict[str, tuple[str, str]] = {
        s.RECEIPT_DT: (s.RECEIPT_DT, "min"),
        "lines": (s.ITEM_NAME, "size"),
        "items": (s.QTY, "sum"),
        "total": (s.LINE_SUM, "sum"),
    }
    for column in (s.STORE_ID, s.CUSTOMER_ID, s.PAYMENT_TYPE, s.OPERATION):
        if column in frame:
            aggregations[column] = (column, "first")
    if s.CATEGORY in frame:
        aggregations["categories"] = (s.CATEGORY, "nunique")

    totals = frame.groupby(s.RECEIPT_ID, dropna=False).agg(**aggregations).reset_index()
    ordered = [s.RECEIPT_ID, s.RECEIPT_DT] + [
        column
        for column in (s.STORE_ID, s.CUSTOMER_ID, s.PAYMENT_TYPE, s.OPERATION)
        if column in totals
    ]
    ordered += [column for column in ("lines", "items", "categories", "total") if column in totals]
    return totals[ordered].sort_values(s.RECEIPT_DT).reset_index(drop=True)


def kpi_summary(frame: pd.DataFrame) -> dict[str, object]:
    """Посчитать заголовочные показатели периода.

    Возвраты учитываются со знаком минус, поэтому ``revenue`` — это чистая
    выручка, а ``returns_amount`` показывает, сколько денег вернули отдельно.
    """
    if frame.empty:
        return {"rows": 0, "receipts": 0, "revenue": 0.0}

    receipts = receipt_totals(frame)
    sales = receipts[
        receipts.get(s.OPERATION, pd.Series("sale", index=receipts.index)) == s.OP_SALE
    ]
    returns = receipts[
        receipts.get(s.OPERATION, pd.Series("sale", index=receipts.index)) == s.OP_RETURN
    ]

    revenue = float(frame[s.LINE_SUM].sum())
    summary: dict[str, object] = {
        "rows": len(frame),
        "receipts": len(receipts),
        "period_start": frame[s.RECEIPT_DT].min(),
        "period_end": frame[s.RECEIPT_DT].max(),
        "revenue": round(revenue, 2),
        "sales_receipts": len(sales),
        "returns_receipts": len(returns),
        "returns_amount": round(float(returns["total"].sum()), 2),
        "avg_ticket": round(float(sales["total"].mean()), 2) if len(sales) else 0.0,
        "items_per_receipt": round(float(sales["items"].mean()), 2) if len(sales) else 0.0,
        "lines_per_receipt": round(float(sales["lines"].mean()), 2) if len(sales) else 0.0,
        "unique_items": int(frame[s.ITEM_NAME].nunique()),
    }
    if s.STORE_ID in frame:
        summary["stores"] = int(frame[s.STORE_ID].nunique())
    if s.CUSTOMER_ID in frame:
        known = frame[s.CUSTOMER_ID].notna()
        summary["customers"] = int(frame.loc[known, s.CUSTOMER_ID].nunique())
        identified = receipts[s.CUSTOMER_ID].notna().mean() if len(receipts) else 0.0
        summary["identified_receipts_share"] = round(float(identified), 4)
    return summary
