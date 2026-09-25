"""Тесты RFM-сегментации."""

from __future__ import annotations

import itertools

import pandas as pd
import pytest

from ofd_analytics import schema as s
from ofd_analytics.analytics.rfm import (
    SEGMENT_RULES,
    add_scores,
    assign_segments,
    quantile_score,
    rfm,
    rfm_table,
    segment_summary,
)
from ofd_analytics.clean import MissingColumnsError


def test_quantile_score_bounds_and_direction() -> None:
    values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ascending = quantile_score(values, bins=5, ascending=True)
    assert ascending.min() == 1
    assert ascending.max() == 5
    assert ascending.is_monotonic_increasing
    descending = quantile_score(values, bins=5, ascending=False)
    assert descending.is_monotonic_decreasing


def test_quantile_score_survives_degenerate_distribution() -> None:
    """У половины базы ровно одна покупка — на таких данных qcut падает, ранги нет."""
    values = pd.Series([1] * 50 + [2] * 10)
    scores = quantile_score(values, bins=5)
    assert set(scores.unique()) <= {1, 2, 3, 4, 5}
    assert scores[values == 2].min() >= scores[values == 1].max()


def test_quantile_score_rejects_small_bins() -> None:
    with pytest.raises(ValueError, match="bins"):
        quantile_score(pd.Series([1.0, 2.0]), bins=1)


def test_rfm_table_exact_numbers(tiny: pd.DataFrame) -> None:
    table = rfm_table(tiny, asof="2025-02-28").set_index(s.CUSTOMER_ID)
    assert table.loc["C1", "recency"] == 52
    assert table.loc["C1", "frequency"] == 2
    assert table.loc["C1", "monetary"] == pytest.approx(350.0)
    assert table.loc["C1", "avg_ticket"] == pytest.approx(175.0)
    assert table.loc["C2", "recency"] == 18
    assert table.loc["C2", "frequency"] == 1
    assert table.loc["C2", "monetary"] == pytest.approx(280.0)
    assert "C1" in table.index and len(table) == 2, "анонимный чек не попадает в RFM"


def test_rfm_table_defaults_asof_to_last_receipt(tiny: pd.DataFrame) -> None:
    table = rfm_table(tiny).set_index(s.CUSTOMER_ID)
    # Последний чек во всех данных — 2025-02-11 (анонимный возврат), значит для
    # C2 с покупкой 2025-02-10 давность равна одному календарному дню.
    assert table.loc["C2", "recency"] == 1
    assert table.loc["C1", "recency"] == 35


def test_rfm_requires_customer_column(tiny: pd.DataFrame) -> None:
    with pytest.raises(MissingColumnsError):
        rfm_table(tiny.drop(columns=[s.CUSTOMER_ID]))


def test_segment_rules_cover_all_score_pairs() -> None:
    """Все 25 комбинаций (R, F) должны попадать в именованный сегмент."""
    for r, f in itertools.product(range(1, 6), repeat=2):
        assert any(matches(r, f) for _, matches in SEGMENT_RULES), (r, f)


def test_assign_segments_uses_first_matching_rule() -> None:
    scored = pd.DataFrame({"r": [5, 1, 5], "f": [5, 1, 1], "m": [5, 1, 3]})
    segments = assign_segments(scored)["segment"]
    assert list(segments) == ["Чемпионы", "Потерянные", "Перспективные"]


def test_rfm_pipeline_on_generated_data(generated: pd.DataFrame) -> None:
    scored = rfm(generated)
    assert not scored.empty
    assert set(scored.columns) >= {"r", "f", "m", "rfm", "rfm_sum", "segment"}
    assert scored["r"].between(1, 5).all()
    assert scored["rfm"].str.len().eq(3).all()
    assert "Прочие" not in set(scored["segment"]), "правила покрывают все комбинации"
    assert scored[s.CUSTOMER_ID].is_unique


def test_champions_are_the_best_customers(generated: pd.DataFrame) -> None:
    """Содержательная проверка: чемпионы покупают чаще и свежее потерянных."""
    scored = rfm(generated)
    champions = scored[scored["segment"] == "Чемпионы"]
    lost = scored[scored["segment"] == "Потерянные"]
    assert champions["frequency"].mean() > lost["frequency"].mean()
    assert champions["recency"].mean() < lost["recency"].mean()
    assert champions["monetary"].mean() > lost["monetary"].mean()


def test_segment_summary_shares_sum_to_one(generated: pd.DataFrame) -> None:
    summary = segment_summary(rfm(generated))
    assert summary["customers_share"].sum() == pytest.approx(1.0, abs=1e-3)
    assert summary["revenue_share"].sum() == pytest.approx(1.0, abs=1e-3)
    assert summary["revenue"].is_monotonic_decreasing


def test_empty_input_returns_empty_frames() -> None:
    empty = pd.DataFrame(columns=list(s.COLUMN_ORDER))
    table = rfm_table(empty)
    assert table.empty
    assert segment_summary(assign_segments(add_scores(table))).empty
