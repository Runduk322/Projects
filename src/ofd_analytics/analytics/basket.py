"""Анализ корзины: какие товары покупают вместе.

Считаются классические метрики ассоциативных правил для пар товаров:

* **support** — доля чеков, в которых встретилась пара;
* **confidence** — вероятность купить B, если уже взяли A;
* **lift** — во сколько раз пара встречается чаще, чем при независимом выборе.

Ориентироваться нужно на lift: у молока и хлеба высокий support просто потому,
что их берут все, а вместе они попадают в чек не чаще случайного.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import pandas as pd

from .. import schema as s
from ..clean import sales_only


def receipt_baskets(frame: pd.DataFrame, min_items: int = 1) -> pd.Series:
    """Собрать корзины: по чеку — множество уникальных наименований.

    Возвраты исключаются: возврат товара не говорит о совместной покупке.
    """
    sales = sales_only(frame)
    if sales.empty:
        return pd.Series(dtype="object")
    baskets = sales.groupby(s.RECEIPT_ID)[s.ITEM_NAME].agg(lambda items: frozenset(items.dropna()))
    return baskets[baskets.map(len) >= min_items]


def item_support(frame: pd.DataFrame) -> pd.DataFrame:
    """Доля чеков, в которых встречается каждый товар."""
    baskets = receipt_baskets(frame)
    if baskets.empty:
        return pd.DataFrame(columns=[s.ITEM_NAME, "receipts", "support"])
    counts = Counter(item for basket in baskets for item in basket)
    support = pd.DataFrame({s.ITEM_NAME: list(counts.keys()), "receipts": list(counts.values())})
    support["support"] = (support["receipts"] / len(baskets)).round(6)
    return support.sort_values("receipts", ascending=False).reset_index(drop=True)


def pair_rules(
    frame: pd.DataFrame,
    min_support: float = 0.005,
    min_item_support: float = 0.01,
    min_confidence: float = 0.0,
    min_lift: float = 1.0,
    top: int | None = None,
) -> pd.DataFrame:
    """Построить ассоциативные правила для пар товаров.

    Args:
        frame: очищенные позиции чеков.
        min_support: минимальная доля чеков с парой — отсекает случайные совпадения.
        min_item_support: товары реже этой доли чеков не рассматриваются; это
            держит число пар под контролем на широком ассортименте.
        min_confidence: минимальная уверенность правила.
        min_lift: минимальный lift; значение 1.0 оставляет только связи сильнее
            случайных.
        top: сколько правил вернуть (по убыванию lift).

    Returns:
        DataFrame с правилами ``antecedent -> consequent``. Каждая пара
        присутствует в двух направлениях: support у них одинаковый, а confidence
        разный, и решение зависит от того, что чем «дотягивать».
    """
    baskets = receipt_baskets(frame, min_items=1)
    total = len(baskets)
    if total == 0:
        return pd.DataFrame(
            columns=[
                "antecedent",
                "consequent",
                "receipts",
                "support",
                "support_antecedent",
                "support_consequent",
                "confidence",
                "lift",
            ]
        )

    item_counts = Counter(item for basket in baskets for item in basket)
    frequent = {item for item, count in item_counts.items() if count / total >= min_item_support}

    pair_counts: Counter[tuple[str, str]] = Counter()
    for basket in baskets:
        interesting = sorted(basket & frequent)
        if len(interesting) > 1:
            pair_counts.update(combinations(interesting, 2))

    rows: list[dict[str, object]] = []
    for (left, right), count in pair_counts.items():
        support = count / total
        if support < min_support:
            continue
        support_left = item_counts[left] / total
        support_right = item_counts[right] / total
        lift = support / (support_left * support_right)
        for antecedent, consequent, support_a, support_b in (
            (left, right, support_left, support_right),
            (right, left, support_right, support_left),
        ):
            rows.append(
                {
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "receipts": count,
                    "support": round(support, 6),
                    "support_antecedent": round(support_a, 6),
                    "support_consequent": round(support_b, 6),
                    "confidence": round(support / support_a, 6),
                    "lift": round(lift, 4),
                }
            )

    rules = pd.DataFrame(rows)
    if rules.empty:
        return rules
    rules = rules[(rules["lift"] >= min_lift) & (rules["confidence"] >= min_confidence)]
    rules = rules.sort_values(["lift", "support"], ascending=False).reset_index(drop=True)
    return rules.head(top) if top else rules
