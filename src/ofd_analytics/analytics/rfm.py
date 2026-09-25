"""RFM-сегментация покупателей.

Recency — сколько дней назад покупал, Frequency — сколько чеков, Monetary —
сколько денег принёс. Каждая метрика переводится в оценку от 1 до 5 по
квантилям, дальше пара (R, F) раскладывает базу на понятные сегменты.

Оценка считается по ранговым квантилям, а не через ``pd.qcut``: у реальных
данных распределение частоты покупок дискретное и вырожденное (у половины базы
ровно один чек), на таких данных ``qcut`` падает или отдаёт пустые корзины.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .. import schema as s
from ..clean import require_customers

#: Сегменты в порядке проверки: первое подошедшее правило и даёт имя сегмента.
#: Правила покрывают все 25 комбинаций (R, F) — «Прочие» не должны появляться.
SEGMENT_RULES: tuple[tuple[str, Callable[[int, int], bool]], ...] = (
    ("Чемпионы", lambda r, f: r >= 4 and f >= 4),
    ("Лояльные", lambda r, f: r >= 3 and f >= 3),
    ("Нельзя потерять", lambda r, f: r <= 2 and f >= 4),
    ("В зоне риска", lambda r, f: r <= 2 and f == 3),
    ("Перспективные", lambda r, f: r >= 4 and f <= 2),
    ("Требуют внимания", lambda r, f: r == 3 and f <= 2),
    ("Спящие", lambda r, f: r == 2 and f <= 2),
    ("Потерянные", lambda r, f: r == 1 and f <= 2),
)


def quantile_score(values: pd.Series, bins: int = 5, ascending: bool = True) -> pd.Series:
    """Перевести метрику в оценку ``1..bins`` по ранговым квантилям.

    Args:
        values: числовая метрика по покупателям.
        bins: сколько градаций (классический RFM использует 5).
        ascending: ``True`` — больше значит лучше (частота, деньги);
            ``False`` — больше значит хуже (давность покупки).
    """
    if bins < 2:
        raise ValueError("bins должен быть не меньше 2")
    if values.empty:
        return pd.Series(dtype="int64", index=values.index)
    ranks = values.rank(method="average", pct=True, ascending=ascending)
    scores = np.ceil(ranks * bins)
    return scores.clip(lower=1, upper=bins).astype("int64")


def rfm_table(frame: pd.DataFrame, asof: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """Посчитать сырые R, F, M по каждому покупателю.

    Args:
        frame: очищенные позиции чеков; строки без ``customer_id`` отбрасываются.
        asof: дата, от которой считается давность. По умолчанию — последний чек
            во всех данных, включая анонимные: это конец периода выгрузки, и
            отчёт не «старится» от того, что его открыли позже.

    Returns:
        DataFrame по покупателям: ``recency`` (дни), ``frequency`` (чеки),
        ``monetary`` (чистая сумма), ``first_purchase``, ``last_purchase``,
        ``avg_ticket``.

    Давность считается в календарных днях между датами, без учёта времени, —
    как ``DATEDIFF(day, ...)`` в SQL. Иначе покупка вчера вечером и покупка
    вчера утром дали бы разную давность при одном и том же «сегодня».
    """
    identified = require_customers(frame)
    if identified.empty:
        return pd.DataFrame(
            columns=[
                s.CUSTOMER_ID,
                "recency",
                "frequency",
                "monetary",
                "first_purchase",
                "last_purchase",
                "avg_ticket",
            ]
        )

    horizon = pd.Timestamp(asof) if asof is not None else frame[s.RECEIPT_DT].max()
    grouped = identified.groupby(s.CUSTOMER_ID, dropna=True)
    table = grouped.agg(
        frequency=(s.RECEIPT_ID, "nunique"),
        monetary=(s.LINE_SUM, "sum"),
        first_purchase=(s.RECEIPT_DT, "min"),
        last_purchase=(s.RECEIPT_DT, "max"),
    )
    table["recency"] = (horizon.normalize() - table["last_purchase"].dt.normalize()).dt.days
    table["monetary"] = table["monetary"].round(2)
    table["avg_ticket"] = (table["monetary"] / table["frequency"]).round(2)
    columns = [
        "recency",
        "frequency",
        "monetary",
        "first_purchase",
        "last_purchase",
        "avg_ticket",
    ]
    return table[columns].reset_index()


def add_scores(table: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Добавить к таблице RFM оценки ``r``, ``f``, ``m`` и их комбинацию."""
    scored = table.copy()
    scored["r"] = quantile_score(scored["recency"], bins=bins, ascending=False)
    scored["f"] = quantile_score(scored["frequency"], bins=bins, ascending=True)
    scored["m"] = quantile_score(scored["monetary"], bins=bins, ascending=True)
    scored["rfm"] = (
        scored["r"].astype("string") + scored["f"].astype("string") + scored["m"].astype("string")
    )
    scored["rfm_sum"] = scored[["r", "f", "m"]].sum(axis=1)
    return scored


def assign_segments(scored: pd.DataFrame) -> pd.DataFrame:
    """Проставить сегмент по правилам :data:`SEGMENT_RULES`."""
    result = scored.copy()

    def segment_of(row: pd.Series) -> str:
        r, f = int(row["r"]), int(row["f"])
        for name, matches in SEGMENT_RULES:
            if matches(r, f):
                return name
        return "Прочие"

    if result.empty:
        result["segment"] = pd.Series(dtype="object")
        return result
    result["segment"] = result.apply(segment_of, axis=1)
    return result


def rfm(
    frame: pd.DataFrame,
    asof: str | pd.Timestamp | None = None,
    bins: int = 5,
) -> pd.DataFrame:
    """Полный RFM: метрики, оценки и сегменты одним вызовом."""
    return assign_segments(add_scores(rfm_table(frame, asof=asof), bins=bins))


def segment_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """Свод по сегментам: сколько покупателей, сколько денег и какие метрики.

    Отсортировано по выручке: сверху сегменты, за которые действительно стоит
    бороться, а не самые многочисленные.
    """
    if scored.empty or "segment" not in scored:
        return pd.DataFrame(
            columns=["segment", "customers", "customers_share", "revenue", "revenue_share"]
        )
    summary = (
        scored.groupby("segment")
        .agg(
            customers=(s.CUSTOMER_ID, "nunique"),
            revenue=("monetary", "sum"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .reset_index()
    )
    summary["customers_share"] = (summary["customers"] / summary["customers"].sum()).round(4)
    summary["revenue_share"] = (summary["revenue"] / summary["revenue"].sum()).round(4)
    for column in ("revenue", "avg_recency", "avg_frequency", "avg_monetary"):
        summary[column] = summary[column].round(2)
    columns = [
        "segment",
        "customers",
        "customers_share",
        "revenue",
        "revenue_share",
        "avg_recency",
        "avg_frequency",
        "avg_monetary",
    ]
    return summary[columns].sort_values("revenue", ascending=False).reset_index(drop=True)
