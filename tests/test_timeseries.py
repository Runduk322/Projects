"""Тесты рядов и профилей сезонности."""

from __future__ import annotations

import pandas as pd
import pytest

from ofd_analytics import schema as s
from ofd_analytics.analytics.timeseries import (
    add_growth,
    add_moving_average,
    group_profile,
    hour_profile,
    revenue_by,
    weekday_profile,
)


def test_daily_series_fills_gaps(tiny: pd.DataFrame) -> None:
    """С 6 января по 11 февраля 37 дней, и дни без продаж должны быть нулями."""
    series = revenue_by(tiny, "D")
    assert len(series) == 37
    assert series.loc["2025-01-06", "revenue"] == pytest.approx(250.0)
    assert series.loc["2025-01-06", "receipts"] == 1
    assert series.loc["2025-01-08", "revenue"] == pytest.approx(0.0)
    assert series.loc["2025-01-08", "receipts"] == 0
    assert series.loc["2025-01-08", "avg_ticket"] == pytest.approx(0.0)


def test_monthly_series_counts_receipts_not_lines(tiny: pd.DataFrame) -> None:
    series = revenue_by(tiny, "ME")
    assert len(series) == 2
    january, february = series.iloc[0], series.iloc[1]
    assert january["revenue"] == pytest.approx(350.0)
    assert january["receipts"] == 2, "два чека, а не три строки"
    assert january["avg_ticket"] == pytest.approx(175.0)
    assert february["revenue"] == pytest.approx(80.0), "280 продаж минус 200 возврата"
    assert february["receipts"] == 2


def test_growth_is_relative_to_previous_period(tiny: pd.DataFrame) -> None:
    series = add_growth(revenue_by(tiny, "ME"))
    assert pd.isna(series.iloc[0]["revenue_growth"])
    assert series.iloc[1]["revenue_growth"] == pytest.approx((80 - 350) / 350, abs=1e-4)


def test_growth_survives_zero_baseline(tiny: pd.DataFrame) -> None:
    series = add_growth(revenue_by(tiny, "D"))
    assert series["revenue_growth"].dtype == "float64"
    assert series["revenue_growth"].notna().any()


def test_moving_average_smooths(tiny: pd.DataFrame) -> None:
    series = add_moving_average(revenue_by(tiny, "D"), window=2)
    assert "revenue_ma2" in series.columns
    assert series.iloc[0]["revenue_ma2"] == pytest.approx(250.0), "min_periods=1 в начале ряда"
    assert series.iloc[1]["revenue_ma2"] == pytest.approx((250 + 100) / 2)


def test_moving_average_rejects_short_window(tiny: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="window"):
        add_moving_average(revenue_by(tiny, "D"), window=1)


def test_weekday_profile_covers_all_seven_days(tiny: pd.DataFrame) -> None:
    profile = weekday_profile(tiny)
    assert list(profile["weekday"]) == ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    monday = profile.iloc[0]
    assert monday["receipts"] == 2, "6 января и 10 февраля — понедельники"
    assert monday["revenue"] == pytest.approx(530.0)
    tuesday = profile.iloc[1]
    assert tuesday["revenue"] == pytest.approx(-100.0), "100 продаж минус 200 возврата"
    assert profile.loc[2, "receipts"] == 0


def test_hour_profile_shares_sum_to_one(tiny: pd.DataFrame) -> None:
    profile = hour_profile(tiny)
    assert set(profile["hour"]) == {9, 10, 12, 18}
    assert profile["share"].sum() == pytest.approx(1.0, abs=1e-6)


def test_group_profile_by_store(tiny: pd.DataFrame) -> None:
    profile = group_profile(tiny, s.STORE_ID).set_index(s.STORE_ID)
    assert profile.loc["S01", "revenue"] == pytest.approx(350.0)
    assert profile.loc["S01", "receipts"] == 2
    assert profile.loc["S02", "revenue"] == pytest.approx(80.0)
    assert profile["revenue_share"].sum() == pytest.approx(1.0, abs=1e-6)


def test_group_profile_rejects_unknown_column(tiny: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="нет колонки"):
        group_profile(tiny, "нет_такой")


def test_empty_frame_gives_empty_series() -> None:
    empty = pd.DataFrame(columns=list(s.COLUMN_ORDER))
    assert revenue_by(empty).empty
    assert weekday_profile(empty).empty
    assert hour_profile(empty).empty


def test_generated_data_has_weekend_peak(generated: pd.DataFrame) -> None:
    """Содержательная проверка: генератор закладывает всплеск в субботу."""
    profile = weekday_profile(generated).set_index("weekday")
    assert profile.loc["Сб", "revenue"] > profile.loc["Вт", "revenue"]
    assert profile.loc["Сб", "index_to_mean"] > 1.0
