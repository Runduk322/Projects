"""Динамика продаж: ряды по периодам и профили сезонности.

Все расчёты идут от уровня чека, а не позиции: иначе «число чеков» превратится в
число строк, а средний чек — в среднюю цену позиции. Это самая частая ошибка
при переходе от SQL-выгрузки к pandas.
"""

from __future__ import annotations

import pandas as pd

from .. import schema as s
from ..features import receipt_totals

#: Короткие имена дней недели в порядке ``Timestamp.dayofweek``.
WEEKDAY_NAMES: tuple[str, ...] = ("Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс")


def revenue_by(frame: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Построить ряд выручки и чеков по периодам.

    Args:
        frame: очищенные позиции чеков.
        freq: период в нотации pandas: ``"D"`` — день, ``"W"`` — неделя,
            ``"ME"`` — месяц, ``"QE"`` — квартал.

    Returns:
        DataFrame с индексом-датой периода и колонками ``revenue``, ``receipts``,
        ``items``, ``avg_ticket``. Пропущенные периоды заполняются нулями, чтобы
        график не «склеивал» дни без продаж.
    """
    receipts = receipt_totals(frame)
    if receipts.empty:
        return pd.DataFrame(columns=["revenue", "receipts", "items", "avg_ticket"])

    indexed = receipts.set_index(s.RECEIPT_DT).sort_index()
    series = indexed.resample(freq).agg(
        revenue=("total", "sum"),
        receipts=(s.RECEIPT_ID, "nunique"),
        items=("items", "sum"),
    )
    counted = series["receipts"]
    series["avg_ticket"] = (series["revenue"] / counted.where(counted > 0)).astype("float64")
    series = series.fillna({"revenue": 0.0, "receipts": 0, "items": 0.0, "avg_ticket": 0.0})
    return series.round(2)


def add_moving_average(
    series: pd.DataFrame, window: int = 7, column: str = "revenue"
) -> pd.DataFrame:
    """Добавить скользящее среднее — сглаживает недельную пилу в дневном ряде."""
    if window < 2:
        raise ValueError("window должен быть не меньше 2")
    result = series.copy()
    result[f"{column}_ma{window}"] = result[column].rolling(window, min_periods=1).mean().round(2)
    return result


def add_growth(series: pd.DataFrame, periods: int = 1, column: str = "revenue") -> pd.DataFrame:
    """Добавить прирост к предыдущему периоду в долях (``0.05`` = +5%)."""
    result = series.copy()
    previous = result[column].shift(periods)
    growth = (result[column] - previous) / previous.where(previous != 0)
    result[f"{column}_growth"] = growth.astype("float64").round(4)
    return result


def weekday_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Профиль недели: сколько чеков и выручки приходится на каждый день.

    ``index_to_mean`` — отношение средней выручки дня к средней по неделе: 1.2
    значит, что день на 20% сильнее обычного.
    """
    receipts = receipt_totals(frame)
    if receipts.empty:
        return pd.DataFrame(columns=["weekday", "receipts", "revenue", "avg_ticket"])

    weekday = receipts[s.RECEIPT_DT].dt.dayofweek
    profile = (
        receipts.groupby(weekday)
        .agg(receipts=(s.RECEIPT_ID, "nunique"), revenue=("total", "sum"))
        .reindex(range(7), fill_value=0)
    )
    profile.insert(0, "weekday", list(WEEKDAY_NAMES))
    counted = profile["receipts"]
    profile["avg_ticket"] = (profile["revenue"] / counted.where(counted > 0)).astype("float64")
    mean_revenue = profile["revenue"].mean()
    profile["index_to_mean"] = profile["revenue"] / mean_revenue if mean_revenue else 0.0
    return profile.round({"revenue": 2, "avg_ticket": 2, "index_to_mean": 3}).reset_index(drop=True)


def hour_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Профиль дня по часам: где пики потока и где проваливается средний чек."""
    receipts = receipt_totals(frame)
    if receipts.empty:
        return pd.DataFrame(columns=["hour", "receipts", "revenue", "avg_ticket"])

    hour = receipts[s.RECEIPT_DT].dt.hour
    profile = (
        receipts.groupby(hour)
        .agg(receipts=(s.RECEIPT_ID, "nunique"), revenue=("total", "sum"))
        .reset_index(names="hour")
    )
    counted = profile["receipts"]
    profile["avg_ticket"] = (profile["revenue"] / counted.where(counted > 0)).astype("float64")
    profile["share"] = (counted / counted.sum()).round(4)
    return profile.round({"revenue": 2, "avg_ticket": 2})


def group_profile(frame: pd.DataFrame, by: str = s.STORE_ID) -> pd.DataFrame:
    """Сравнить срезы между собой: точки, формы оплаты, категории."""
    if by not in frame:
        raise KeyError(f"В таблице нет колонки {by!r}")
    receipts = receipt_totals(frame)
    if receipts.empty or by not in receipts:
        grouped = frame.groupby(by, dropna=False)[s.LINE_SUM].sum().reset_index(name="revenue")
        return grouped.round(2)

    profile = (
        receipts.groupby(by, dropna=False)
        .agg(receipts=(s.RECEIPT_ID, "nunique"), revenue=("total", "sum"), items=("items", "sum"))
        .reset_index()
    )
    counted = profile["receipts"]
    profile["avg_ticket"] = (profile["revenue"] / counted.where(counted > 0)).astype("float64")
    profile["revenue_share"] = (profile["revenue"] / profile["revenue"].sum()).round(4)
    return (
        profile.round({"revenue": 2, "items": 2, "avg_ticket": 2})
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
