"""Тесты ABC- и XYZ-анализа."""

from __future__ import annotations

import pandas as pd
import pytest

from ofd_analytics import schema as s
from ofd_analytics.analytics.abc import abc_analysis, abc_xyz, abc_xyz_matrix, xyz_analysis


def test_abc_exact_classes_and_shares(tiny: pd.DataFrame) -> None:
    """Масло 200, Хлеб 150, Молоко 80 из 430 рублей — классы A, A, B."""
    result = abc_analysis(tiny).set_index(s.ITEM_NAME)
    assert list(result.index) == ["Масло", "Хлеб", "Молоко"]
    assert result.loc["Масло", "share"] == pytest.approx(200 / 430, abs=1e-6)
    assert result.loc["Хлеб", "cum_share"] == pytest.approx(350 / 430, abs=1e-6)
    assert list(result["abc"]) == ["A", "A", "B"]
    assert list(result["rank"]) == [1, 2, 3]


def test_abc_boundary_item_belongs_to_the_class_it_closes() -> None:
    """Товар, которым доля переваливает за 80%, ещё относится к A."""
    frame = pd.DataFrame(
        {
            s.ITEM_NAME: ["a", "b", "c", "d"],
            s.LINE_SUM: [70.0, 15.0, 10.0, 5.0],
            s.RECEIPT_DT: pd.to_datetime(["2025-01-01"] * 4),
            s.QTY: [1.0] * 4,
        }
    )
    result = abc_analysis(frame).set_index(s.ITEM_NAME)
    assert result.loc["a", "abc"] == "A"
    assert result.loc["b", "abc"] == "A", "накопленная доля 85% — переваливает границу"
    assert result.loc["c", "abc"] == "B"
    assert result.loc["d", "abc"] == "C"


def test_abc_can_group_by_category(tiny: pd.DataFrame) -> None:
    result = abc_analysis(tiny, by=s.CATEGORY)
    assert set(result[s.CATEGORY]) == {"Молочное", "Бакалея"}
    assert result[s.LINE_SUM].sum() == pytest.approx(430.0)


def test_abc_rejects_non_positive_total(tiny: pd.DataFrame) -> None:
    returns_only = tiny[tiny[s.OPERATION] == s.OP_RETURN]
    with pytest.raises(ValueError, match="неположительно"):
        abc_analysis(returns_only)


def test_abc_rejects_bad_thresholds(tiny: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="thresholds"):
        abc_analysis(tiny, thresholds=(0.95, 0.80))


def test_abc_on_empty_frame() -> None:
    result = abc_analysis(pd.DataFrame(columns=list(s.COLUMN_ORDER)))
    assert result.empty


def test_xyz_uses_zero_for_periods_without_sales() -> None:
    """Товар, проданный раз в год, не должен выглядеть стабильным."""
    dates = pd.to_datetime(["2025-01-05", "2025-02-05", "2025-03-05"])
    frame = pd.DataFrame(
        {
            s.ITEM_NAME: ["ровный", "ровный", "ровный", "разовый"],
            s.RECEIPT_DT: [*dates, pd.Timestamp("2025-02-05")],
            s.QTY: [10.0, 10.0, 10.0, 30.0],
            s.LINE_SUM: [100.0, 100.0, 100.0, 300.0],
        }
    )
    result = xyz_analysis(frame).set_index(s.ITEM_NAME)
    assert result.loc["ровный", "cv"] == pytest.approx(0.0)
    assert result.loc["ровный", "xyz"] == "X"
    assert result.loc["разовый", "cv"] > 1.0
    assert result.loc["разовый", "xyz"] == "Z"


def test_xyz_rejects_bad_thresholds(tiny: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="thresholds"):
        xyz_analysis(tiny, thresholds=(0.5, 0.1))


def test_abc_xyz_combines_into_one_class(generated: pd.DataFrame) -> None:
    combined = abc_xyz(generated)
    assert set(combined["abc"]) <= {"A", "B", "C"}
    assert set(combined["xyz"]) <= {"X", "Y", "Z"}
    assert combined["class"].str.len().eq(2).all()
    assert combined[s.ITEM_NAME].is_unique


def test_abc_xyz_matrix_is_three_by_three(generated: pd.DataFrame) -> None:
    matrix = abc_xyz_matrix(abc_xyz(generated))
    assert list(matrix.index) == ["A", "B", "C"]
    assert list(matrix.columns) == ["X", "Y", "Z"]
    assert matrix.to_numpy().sum() == pytest.approx(generated[s.LINE_SUM].sum(), rel=1e-6)


def test_abc_classes_are_ordered_by_contribution(generated: pd.DataFrame) -> None:
    """Классы обязаны идти по убыванию вклада, а доли — попадать в границы.

    Сколько именно позиций окажется в классе A, зависит от ассортимента: на ровном
    спросе, как в генераторе, это больше половины позиций, и это не ошибка метода.
    """
    result = abc_analysis(generated)
    shares = result.groupby("abc")["share"].sum()
    assert shares["A"] >= 0.80, "класс A по определению закрывает первые 80% выручки"
    assert shares["C"] <= 0.05
    means = result.groupby("abc")[s.LINE_SUM].mean()
    assert means["A"] > means["B"] > means["C"]
