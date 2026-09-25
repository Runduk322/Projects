"""Когортный анализ удержания покупателей.

Покупатели группируются по месяцу первой покупки (когорте), а затем считается,
какая доля когорты вернулась через 1, 2, 3 ... месяца. Это единственный способ
отличить «выручка растёт, потому что база возвращается» от «выручка растёт,
потому что мы всё время покупаем новых».

Требует ``customer_id``: чеки без покупателя в расчёт не попадают.
"""

from __future__ import annotations

import pandas as pd

from .. import schema as s
from ..clean import require_customers, sales_only


def _period_offsets(periods: pd.Series, cohorts: pd.Series) -> pd.Series:
    """Номер периода внутри когорты: 0 — месяц первой покупки, 1 — следующий."""
    return ((periods - cohorts).map(lambda offset: offset.n)).astype("int64")


def cohort_frame(frame: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """Разметить чеки когортой покупателя и номером периода с момента прихода."""
    identified = require_customers(sales_only(frame))
    if identified.empty:
        return pd.DataFrame(columns=[s.CUSTOMER_ID, "cohort", "period", "offset", s.LINE_SUM])

    marked = identified.copy()
    marked["period"] = marked[s.RECEIPT_DT].dt.to_period(freq)
    marked["cohort"] = marked.groupby(s.CUSTOMER_ID)["period"].transform("min")
    marked["offset"] = _period_offsets(marked["period"], marked["cohort"])
    return marked


def cohort_matrix(frame: pd.DataFrame, freq: str = "M", value: str = "customers") -> pd.DataFrame:
    """Построить когортную матрицу: когорты по строкам, номер периода по столбцам.

    Args:
        frame: очищенные позиции чеков.
        freq: размер когорты, по умолчанию месяц.
        value: ``"customers"`` — число вернувшихся покупателей,
            ``"revenue"`` — их выручка, ``"receipts"`` — число чеков.

    Returns:
        Матрица со строками-когортами. Правый верхний треугольник заполнен
        нулями: у поздних когорт ещё не было поздних периодов.
    """
    marked = cohort_frame(frame, freq=freq)
    if marked.empty:
        return pd.DataFrame()

    aggregations = {
        "customers": (s.CUSTOMER_ID, "nunique"),
        "receipts": (s.RECEIPT_ID, "nunique"),
        "revenue": (s.LINE_SUM, "sum"),
    }
    if value not in aggregations:
        raise ValueError(f"value должен быть одним из {sorted(aggregations)}")
    column, how = aggregations[value]

    matrix = (
        marked.groupby(["cohort", "offset"])
        .agg(metric=(column, how))
        .reset_index()
        .pivot(index="cohort", columns="offset", values="metric")
    )
    matrix.index = matrix.index.astype("string")
    matrix.columns.name = "offset"
    return matrix.sort_index().round(2)


def retention_rates(matrix: pd.DataFrame) -> pd.DataFrame:
    """Перевести когортную матрицу в доли удержания от размера когорты."""
    if matrix.empty:
        return matrix
    base = matrix.iloc[:, 0]
    return matrix.div(base.where(base != 0), axis=0).astype("float64").round(4)


def retention_curve(rates: pd.DataFrame) -> pd.Series:
    """Средняя кривая удержания по всем когортам — одна строка вместо матрицы.

    Усреднение идёт по имеющимся наблюдениям, поэтому «хвост» кривой опирается
    на меньшее число когорт, чем начало: это нормально, но об этом надо помнить.
    """
    if rates.empty:
        return pd.Series(dtype="float64")
    return rates.mean(axis=0, skipna=True).round(4)


def cohort_sizes(frame: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """Размер когорт: сколько новых покупателей пришло в каждом периоде."""
    marked = cohort_frame(frame, freq=freq)
    if marked.empty:
        return pd.DataFrame(columns=["cohort", "customers"])
    new_customers = marked[marked["offset"] == 0]
    sizes = (
        new_customers.groupby("cohort")[s.CUSTOMER_ID]
        .nunique()
        .reset_index(name="customers")
        .sort_values("cohort")
    )
    sizes["cohort"] = sizes["cohort"].astype("string")
    return sizes.reset_index(drop=True)
