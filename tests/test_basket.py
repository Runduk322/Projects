"""Тесты анализа корзины."""

from __future__ import annotations

import pandas as pd
import pytest

from ofd_analytics import schema as s
from ofd_analytics.analytics.basket import item_support, pair_rules, receipt_baskets


def test_baskets_exclude_returns(tiny: pd.DataFrame) -> None:
    baskets = receipt_baskets(tiny)
    assert set(baskets.index) == {"R1", "R2", "R3"}, "возвратный чек R4 не корзина"
    assert baskets["R1"] == frozenset({"Хлеб", "Масло"})


def test_min_items_filters_single_item_receipts(tiny: pd.DataFrame) -> None:
    assert set(receipt_baskets(tiny, min_items=2).index) == {"R1", "R3"}


def test_item_support_exact(tiny: pd.DataFrame) -> None:
    support = item_support(tiny).set_index(s.ITEM_NAME)
    assert support.loc["Хлеб", "receipts"] == 2
    assert support.loc["Хлеб", "support"] == pytest.approx(2 / 3, abs=1e-6)
    assert support.loc["Молоко", "support"] == pytest.approx(1 / 3, abs=1e-6)


def test_pair_rules_exact_metrics(tiny: pd.DataFrame) -> None:
    """Из трёх корзин Масло+Молоко встречаются вместе один раз: lift = 1.5."""
    rules = pair_rules(tiny, min_support=0.0, min_lift=0.0).set_index(["antecedent", "consequent"])
    assert len(rules) == 4, "две пары в двух направлениях"

    milk_butter = rules.loc[("Масло", "Молоко")]
    assert milk_butter["support"] == pytest.approx(1 / 3, abs=1e-6)
    assert milk_butter["confidence"] == pytest.approx(0.5, abs=1e-6)
    assert milk_butter["lift"] == pytest.approx(1.5, abs=1e-4)

    reverse = rules.loc[("Молоко", "Масло")]
    assert reverse["support"] == pytest.approx(1 / 3, abs=1e-6)
    assert reverse["confidence"] == pytest.approx(1.0, abs=1e-6), "молоко всегда с маслом"
    assert reverse["lift"] == pytest.approx(1.5, abs=1e-4), "lift симметричен"

    assert rules.loc[("Хлеб", "Масло"), "lift"] == pytest.approx(0.75, abs=1e-4)


def test_default_min_lift_keeps_only_real_connections(tiny: pd.DataFrame) -> None:
    rules = pair_rules(tiny, min_support=0.0)
    pairs = set(zip(rules["antecedent"], rules["consequent"], strict=True))
    assert pairs == {("Масло", "Молоко"), ("Молоко", "Масло")}


def test_min_support_filters_rare_pairs(tiny: pd.DataFrame) -> None:
    assert pair_rules(tiny, min_support=0.9).empty


def test_min_item_support_filters_rare_items(tiny: pd.DataFrame) -> None:
    """Молоко есть в одной корзине из трёх — порог 0.5 его исключает."""
    rules = pair_rules(tiny, min_support=0.0, min_lift=0.0, min_item_support=0.5)
    assert set(rules["antecedent"]) <= {"Хлеб", "Масло"}


def test_empty_frame_returns_typed_empty_result() -> None:
    rules = pair_rules(pd.DataFrame(columns=list(s.COLUMN_ORDER)))
    assert rules.empty
    assert "lift" in rules.columns


def test_rules_recover_planted_affinities(generated: pd.DataFrame) -> None:
    """Генератор кладёт в корзину связки — анализ обязан их найти."""
    rules = pair_rules(generated, min_support=0.003).set_index(["antecedent", "consequent"])
    assert ("Кофе латте 0.4", "Круассан с миндалем") in rules.index
    assert rules.loc[("Кофе латте 0.4", "Круассан с миндалем"), "lift"] > 1.5
    assert ("Пиво светлое 0.5", "Чипсы паприка 90г") in rules.index
    assert rules.loc[("Пиво светлое 0.5", "Чипсы паприка 90г"), "lift"] > 1.5


def test_rules_are_sorted_and_limited(generated: pd.DataFrame) -> None:
    rules = pair_rules(generated, top=5)
    assert len(rules) == 5
    assert rules["lift"].is_monotonic_decreasing
    assert (rules["lift"] >= 1.0).all()


def test_confidence_never_exceeds_one(generated: pd.DataFrame) -> None:
    rules = pair_rules(generated, min_support=0.0, min_lift=0.0)
    assert rules["confidence"].le(1.0).all()
    assert rules["support"].le(1.0).all()
