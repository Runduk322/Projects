"""Тесты когортного анализа удержания."""

from __future__ import annotations

import pandas as pd
import pytest

from ofd_analytics import schema as s
from ofd_analytics.analytics.cohorts import (
    cohort_frame,
    cohort_matrix,
    cohort_sizes,
    retention_curve,
    retention_rates,
)
from ofd_analytics.clean import MissingColumnsError

#: A приходит в январе и возвращается в феврале и апреле, B — только в январе,
#: C приходит в феврале и возвращается в марте. Плюс анонимный чек и возврат,
#: которые в расчёт попадать не должны.
COHORT_ROWS = [
    ("R1", "2025-01-10", "A", 100.0, "sale"),
    ("R2", "2025-02-15", "A", 100.0, "sale"),
    ("R3", "2025-04-20", "A", 100.0, "sale"),
    ("R4", "2025-01-20", "B", 100.0, "sale"),
    ("R5", "2025-02-05", "C", 100.0, "sale"),
    ("R6", "2025-03-10", "C", 100.0, "sale"),
    ("R7", "2025-03-11", None, 100.0, "sale"),
    ("R8", "2025-03-12", "B", -100.0, "return"),
]


@pytest.fixture
def cohort_data() -> pd.DataFrame:
    frame = pd.DataFrame(
        COHORT_ROWS,
        columns=[s.RECEIPT_ID, s.RECEIPT_DT, s.CUSTOMER_ID, s.LINE_SUM, s.OPERATION],
    )
    frame[s.RECEIPT_DT] = pd.to_datetime(frame[s.RECEIPT_DT])
    frame[s.ITEM_NAME] = "Товар"
    frame[s.QTY] = 1.0
    return frame


def test_cohort_frame_marks_cohort_and_offset(cohort_data: pd.DataFrame) -> None:
    marked = cohort_frame(cohort_data).set_index(s.RECEIPT_ID)
    assert str(marked.loc["R1", "cohort"]) == "2025-01"
    assert marked.loc["R1", "offset"] == 0
    assert marked.loc["R2", "offset"] == 1, "февраль — второй месяц когорты A"
    assert marked.loc["R3", "offset"] == 3, "апрель — четвёртый"
    assert str(marked.loc["R5", "cohort"]) == "2025-02"
    assert "R7" not in marked.index, "анонимный чек исключён"
    assert "R8" not in marked.index, "возврат исключён"


def test_cohort_matrix_exact(cohort_data: pd.DataFrame) -> None:
    matrix = cohort_matrix(cohort_data)
    assert list(matrix.index) == ["2025-01", "2025-02"]
    assert list(matrix.columns) == [0, 1, 3]
    assert matrix.loc["2025-01", 0] == 2, "A и B пришли в январе"
    assert matrix.loc["2025-01", 1] == 1, "вернулся только A"
    assert matrix.loc["2025-01", 3] == 1
    assert matrix.loc["2025-02", 0] == 1
    assert matrix.loc["2025-02", 1] == 1
    assert pd.isna(matrix.loc["2025-02", 3]), "у февральской когорты ещё не было этого месяца"


def test_retention_rates_are_relative_to_cohort_size(cohort_data: pd.DataFrame) -> None:
    rates = retention_rates(cohort_matrix(cohort_data))
    assert rates.loc["2025-01", 0] == pytest.approx(1.0)
    assert rates.loc["2025-01", 1] == pytest.approx(0.5)
    assert rates.loc["2025-02", 1] == pytest.approx(1.0)


def test_retention_curve_averages_cohorts(cohort_data: pd.DataFrame) -> None:
    curve = retention_curve(retention_rates(cohort_matrix(cohort_data)))
    assert curve[0] == pytest.approx(1.0)
    assert curve[1] == pytest.approx(0.75), "среднее из 0.5 и 1.0"
    assert curve[3] == pytest.approx(0.5), "только январская когорта дожила"


def test_revenue_matrix(cohort_data: pd.DataFrame) -> None:
    matrix = cohort_matrix(cohort_data, value="revenue")
    assert matrix.loc["2025-01", 0] == pytest.approx(200.0)
    assert matrix.loc["2025-02", 1] == pytest.approx(100.0)


def test_unknown_value_rejected(cohort_data: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="value"):
        cohort_matrix(cohort_data, value="прибыль")


def test_cohort_sizes(cohort_data: pd.DataFrame) -> None:
    sizes = cohort_sizes(cohort_data).set_index("cohort")
    assert sizes.loc["2025-01", "customers"] == 2
    assert sizes.loc["2025-02", "customers"] == 1


def test_without_customer_column_raises(cohort_data: pd.DataFrame) -> None:
    with pytest.raises(MissingColumnsError, match="customer_id"):
        cohort_matrix(cohort_data.drop(columns=[s.CUSTOMER_ID]))


def test_anonymous_only_data_gives_empty_matrix(cohort_data: pd.DataFrame) -> None:
    anonymous = cohort_data.assign(**{s.CUSTOMER_ID: None})
    assert cohort_matrix(anonymous).empty
    assert retention_rates(cohort_matrix(anonymous)).empty
    assert retention_curve(retention_rates(cohort_matrix(anonymous))).empty


def test_retention_decays_on_generated_data(generated: pd.DataFrame) -> None:
    """Содержательная проверка: удержание падает, а не растёт."""
    curve = retention_curve(retention_rates(cohort_matrix(generated)))
    assert curve[0] == pytest.approx(1.0)
    assert curve[1] < 1.0
    assert curve[3] < curve[1]
