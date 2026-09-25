"""Аналитические расчёты над очищенными чеками.

Каждый модуль отвечает на свой вопрос бизнеса:

* :mod:`~ofd_analytics.analytics.rfm` — кто наши покупатели и кого мы теряем;
* :mod:`~ofd_analytics.analytics.abc` — что приносит выручку и что стабильно;
* :mod:`~ofd_analytics.analytics.basket` — что покупают вместе;
* :mod:`~ofd_analytics.analytics.timeseries` — как продажи меняются во времени;
* :mod:`~ofd_analytics.analytics.cohorts` — возвращаются ли покупатели.
"""

from __future__ import annotations

from . import abc, basket, cohorts, rfm, timeseries

__all__ = ["abc", "basket", "cohorts", "rfm", "timeseries"]
