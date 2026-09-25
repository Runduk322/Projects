"""Генератор синтетических чековых данных, похожих на выгрузку ОФД.

Нужен, чтобы пайплайн и тесты работали без реальных данных. Данные не случайны
насквозь: в них заложены сезонность по дням недели и часам, жизненный цикл
покупателя (кто-то уходит через месяц, кто-то остаётся) и товарные связки —
именно это потом находят RFM, когортный анализ и анализ корзины.

Реальную выгрузку ОФД подставляют вместо генератора: см. :mod:`ofd_analytics.clean`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import schema as s


@dataclass(frozen=True)
class CatalogItem:
    """Позиция товарного справочника."""

    name: str
    category: str
    price: float
    weight: float


#: Справочник магазина «у дома» с кофейной точкой.
CATALOG: tuple[CatalogItem, ...] = (
    CatalogItem("Кофе латте 0.4", "Кофе", 240.0, 9.0),
    CatalogItem("Кофе американо 0.3", "Кофе", 190.0, 6.0),
    CatalogItem("Эспрессо", "Кофе", 130.0, 3.5),
    CatalogItem("Капучино 0.3", "Кофе", 220.0, 7.0),
    CatalogItem("Круассан с миндалем", "Выпечка", 160.0, 5.0),
    CatalogItem("Сырник творожный", "Выпечка", 120.0, 3.0),
    CatalogItem("Печенье овсяное", "Выпечка", 70.0, 2.5),
    CatalogItem("Булочка с корицей", "Выпечка", 110.0, 2.0),
    CatalogItem("Молоко 2.5% 0.9л", "Молочное", 89.0, 6.5),
    CatalogItem("Сметана 15% 300г", "Молочное", 99.0, 2.5),
    CatalogItem("Масло сливочное 180г", "Молочное", 219.0, 2.0),
    CatalogItem("Йогурт питьевой 0.29л", "Молочное", 79.0, 3.0),
    CatalogItem("Сыр российский 200г", "Молочное", 249.0, 2.0),
    CatalogItem("Вода без газа 0.5л", "Напитки", 55.0, 5.5),
    CatalogItem("Сок апельсиновый 1л", "Напитки", 149.0, 2.5),
    CatalogItem("Пиво светлое 0.5", "Напитки", 99.0, 4.5),
    CatalogItem("Энергетик 0.45", "Напитки", 129.0, 2.0),
    CatalogItem("Чипсы паприка 90г", "Снеки", 129.0, 3.5),
    CatalogItem("Орешки соленые 100г", "Снеки", 149.0, 2.0),
    CatalogItem("Шоколад молочный 90г", "Снеки", 119.0, 3.0),
    CatalogItem("Хлеб бородинский", "Бакалея", 65.0, 5.0),
    CatalogItem("Хлопья кукурузные 300г", "Бакалея", 139.0, 1.8),
    CatalogItem("Макароны спагетти 450г", "Бакалея", 89.0, 2.2),
    CatalogItem("Рис длиннозерный 900г", "Бакалея", 129.0, 1.5),
    CatalogItem("Пельмени домашние 800г", "Готовая еда", 379.0, 2.5),
    CatalogItem("Салат цезарь 200г", "Готовая еда", 219.0, 2.0),
    CatalogItem("Пакет майка", "Хозтовары", 8.0, 8.0),
    CatalogItem("Салфетки бумажные", "Хозтовары", 59.0, 1.2),
)

#: Товарные связки: к товару-«локомотиву» с некоторой вероятностью добавляется спутник.
#: Именно из них анализ корзины достаёт lift > 1.
AFFINITIES: dict[str, tuple[tuple[str, float], ...]] = {
    "Кофе латте 0.4": (("Круассан с миндалем", 0.45), ("Сырник творожный", 0.18)),
    "Капучино 0.3": (("Булочка с корицей", 0.35),),
    "Эспрессо": (("Печенье овсяное", 0.30),),
    "Пиво светлое 0.5": (("Чипсы паприка 90г", 0.50), ("Орешки соленые 100г", 0.25)),
    "Хлеб бородинский": (("Масло сливочное 180г", 0.28),),
    "Молоко 2.5% 0.9л": (("Хлопья кукурузные 300г", 0.22),),
    "Пельмени домашние 800г": (("Сметана 15% 300г", 0.40),),
}

#: Понедельник .. воскресенье. Выходные в рознице выше, вторник — дно недели.
WEEKDAY_WEIGHTS = np.array([0.90, 0.82, 0.95, 1.00, 1.28, 1.45, 1.15])

#: Часы работы и профиль потока: утренний кофе и вечерний заход после работы.
HOURS = np.arange(7, 23)
HOUR_WEIGHTS = np.array(
    [0.6, 1.5, 1.8, 1.2, 0.9, 1.1, 1.2, 0.9, 0.8, 1.0, 1.6, 1.9, 1.5, 1.0, 0.7, 0.4]
)

DEFAULT_START = "2025-01-01"
DEFAULT_END = "2025-12-31"


def _normalized(weights: np.ndarray) -> np.ndarray:
    return weights / weights.sum()


def _sample_datetimes(
    rng: np.random.Generator,
    n: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    trend: float = 0.25,
) -> pd.DatetimeIndex:
    """Сгенерировать ``n`` моментов чеков с недельной и внутридневной сезонностью.

    ``trend`` — относительный рост потока от начала к концу периода (0.25 = +25%).
    """
    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    if len(days) == 0:
        raise ValueError("Пустой период: end раньше start")

    ramp = np.linspace(0.0, trend, num=len(days))
    day_weights = _normalized(WEEKDAY_WEIGHTS[days.dayofweek.to_numpy()] * (1.0 + ramp))

    day_index = rng.choice(len(days), size=n, p=day_weights)
    hours = rng.choice(HOURS, size=n, p=_normalized(HOUR_WEIGHTS))
    minutes = rng.integers(0, 60, size=n)
    seconds = rng.integers(0, 60, size=n)

    moments = (
        days.to_numpy()[day_index]
        + (hours * 3_600_000_000_000).astype("timedelta64[ns]")
        + (minutes * 60_000_000_000).astype("timedelta64[ns]")
        + (seconds * 1_000_000_000).astype("timedelta64[ns]")
    )
    return pd.DatetimeIndex(np.sort(moments))


def _customer_windows(
    rng: np.random.Generator,
    n_customers: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Окна активности покупателей: дата первого чека, дата ухода и интенсивность.

    Из этих окон получаются осмысленные когорты: покупатели приходят в разные
    месяцы, а срок жизни распределён экспоненциально, поэтому retention падает.
    """
    span_days = max((end - start).days, 1)
    first_offset = rng.uniform(0, span_days * 0.9, size=n_customers)
    lifetime_days = rng.exponential(scale=110.0, size=n_customers) + 5.0
    intensity = rng.gamma(shape=2.0, scale=1.0, size=n_customers) + 0.15

    first_seen = pd.Series(start + pd.to_timedelta(first_offset, unit="D"))
    last_seen = (first_seen + pd.to_timedelta(lifetime_days, unit="D")).clip(upper=end)
    return pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n_customers)],
            "first_seen": first_seen,
            "last_seen": last_seen,
            "intensity": intensity,
        }
    )


def _assign_customers(
    rng: np.random.Generator,
    moments: pd.DatetimeIndex,
    customers: pd.DataFrame,
    identified_share: float,
) -> np.ndarray:
    """Привязать чеки к покупателям так, чтобы уважались окна их активности.

    Часть чеков остаётся анонимной — как в реальном ОФД, где покупатель известен
    только по чекам с привязанным телефоном или картой лояльности.
    """
    first = customers["first_seen"].to_numpy()
    last = customers["last_seen"].to_numpy()
    intensity = customers["intensity"].to_numpy()
    ids = customers["customer_id"].to_numpy()

    assigned = np.full(len(moments), None, dtype=object)
    wants_id = rng.random(len(moments)) < identified_share

    for position, moment in enumerate(moments.to_numpy()):
        if not wants_id[position]:
            continue
        eligible = (first <= moment) & (moment <= last)
        if not eligible.any():
            continue
        weights = _normalized(intensity[eligible])
        assigned[position] = rng.choice(ids[eligible], p=weights)
    return assigned


def _build_basket(rng: np.random.Generator, names: np.ndarray, weights: np.ndarray) -> list[str]:
    """Собрать список товаров одного чека с учётом связок."""
    size = int(min(1 + rng.poisson(1.6), 8))
    basket: list[str] = []

    lead = str(rng.choice(names, p=weights))
    basket.append(lead)
    for companion, probability in AFFINITIES.get(lead, ()):
        if len(basket) < size and rng.random() < probability:
            basket.append(companion)

    while len(basket) < size:
        candidate = str(rng.choice(names, p=weights))
        if candidate not in basket:
            basket.append(candidate)
    return basket


def generate_receipts(
    n_receipts: int = 5000,
    n_customers: int = 400,
    start: str | pd.Timestamp = DEFAULT_START,
    end: str | pd.Timestamp = DEFAULT_END,
    n_stores: int = 3,
    identified_share: float = 0.6,
    return_share: float = 0.02,
    seed: int = 42,
    messy: bool = False,
) -> pd.DataFrame:
    """Сгенерировать таблицу позиций чеков в канонической схеме.

    Args:
        n_receipts: сколько чеков выпустить.
        n_customers: размер базы покупателей, к которым привязываются чеки.
        start, end: границы периода включительно.
        n_stores: число торговых точек.
        identified_share: доля чеков с известным ``customer_id``.
        return_share: доля чеков-возвратов.
        seed: зерно генератора, полностью определяет результат.
        messy: добавить типичный «грязный» слой (дубли, пропуски, расхождения
            суммы), чтобы было что чистить в :func:`ofd_analytics.clean.clean_receipts`.

    Returns:
        DataFrame, одна строка — одна позиция чека. Возвраты помечены в
        ``operation``, но количество и сумма положительные — как в выгрузке ОФД.
    """
    if n_receipts <= 0:
        raise ValueError("n_receipts должен быть положительным")
    if n_customers <= 0:
        raise ValueError("n_customers должен быть положительным")
    if not 0.0 <= identified_share <= 1.0:
        raise ValueError("identified_share должен лежать в [0, 1]")
    if not 0.0 <= return_share <= 1.0:
        raise ValueError("return_share должен лежать в [0, 1]")

    rng = np.random.default_rng(seed)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    moments = _sample_datetimes(rng, n_receipts, start_ts, end_ts)
    customers = _customer_windows(rng, n_customers, start_ts, end_ts)
    customer_ids = _assign_customers(rng, moments, customers, identified_share)

    names = np.array([item.name for item in CATALOG])
    weights = _normalized(np.array([item.weight for item in CATALOG]))
    by_name = {item.name: item for item in CATALOG}

    stores = [f"S{i + 1:02d}" for i in range(n_stores)]
    store_weights = _normalized(np.arange(n_stores, 0, -1, dtype=float) + 1.0)
    payments = np.array([s.PAY_CARD, s.PAY_CASH])
    payment_weights = np.array([0.78, 0.22])

    rows: list[dict[str, object]] = []
    for position, moment in enumerate(moments):
        receipt_id = f"R{position + 1:07d}"
        store_id = str(rng.choice(stores, p=store_weights))
        payment_type = str(rng.choice(payments, p=payment_weights))
        operation = s.OP_RETURN if rng.random() < return_share else s.OP_SALE

        for item_name in _build_basket(rng, names, weights):
            item = by_name[item_name]
            qty = 1.0 + float(rng.poisson(0.35))
            price = round(item.price * float(rng.normal(1.0, 0.02)), 2)
            rows.append(
                {
                    s.RECEIPT_ID: receipt_id,
                    s.RECEIPT_DT: moment,
                    s.STORE_ID: store_id,
                    s.CUSTOMER_ID: customer_ids[position],
                    s.ITEM_NAME: item.name,
                    s.CATEGORY: item.category,
                    s.QTY: qty,
                    s.PRICE: price,
                    s.LINE_SUM: round(qty * price, 2),
                    s.VAT_RATE: 20.0,
                    s.PAYMENT_TYPE: payment_type,
                    s.OPERATION: operation,
                }
            )

    frame = pd.DataFrame(rows, columns=list(s.COLUMN_ORDER))
    frame[s.RECEIPT_DT] = pd.to_datetime(frame[s.RECEIPT_DT])
    if messy:
        frame = _make_messy(rng, frame)
    return frame.reset_index(drop=True)


def _make_messy(rng: np.random.Generator, frame: pd.DataFrame) -> pd.DataFrame:
    """Испортить данные так, как их портит реальная выгрузка.

    Каждый вид порчи отрабатывается конкретным шагом очистки, поэтому список
    здесь и список шагов в :func:`ofd_analytics.clean.clean_receipts` зеркальны.
    """
    frame = frame.copy()
    size = len(frame)

    # 1. Разнобой в написании наименований: регистр и лишние пробелы.
    noisy_names = rng.choice(size, size=max(size // 12, 1), replace=False)
    frame.loc[noisy_names, s.ITEM_NAME] = [
        f"  {name.upper()} " if index % 2 else f"{name.lower()}  "
        for index, name in enumerate(frame.loc[noisy_names, s.ITEM_NAME])
    ]

    # 2. Пропуски в цене там, где сумма есть, и наоборот — восстановимо из пары.
    no_price = rng.choice(size, size=max(size // 25, 1), replace=False)
    frame.loc[no_price, s.PRICE] = np.nan
    no_sum = rng.choice(size, size=max(size // 25, 1), replace=False)
    frame.loc[no_sum, s.LINE_SUM] = np.nan

    # 3. Сумма не сходится с ценой × количеством (скидки, потерянные при выгрузке).
    mismatched = rng.choice(size, size=max(size // 30, 1), replace=False)
    frame.loc[mismatched, s.LINE_SUM] = (frame.loc[mismatched, s.LINE_SUM].fillna(0.0) * 0.8).round(
        2
    )

    # 4. Нулевое и отрицательное количество в продажах.
    broken_qty = rng.choice(size, size=max(size // 60, 1), replace=False)
    frame.loc[broken_qty, s.QTY] = rng.choice([0.0, -1.0], size=len(broken_qty))

    # 5. Ставка НДС строкой с процентом.
    as_text = rng.choice(size, size=max(size // 15, 1), replace=False)
    frame[s.VAT_RATE] = frame[s.VAT_RATE].astype(object)
    frame.loc[as_text, s.VAT_RATE] = "20%"

    # 6. Пустые строки вместо отсутствующего покупателя.
    frame[s.CUSTOMER_ID] = frame[s.CUSTOMER_ID].fillna("")

    # 7. Опечатка в годе — чек улетает за пределы периода.
    if size > 10:
        typos = rng.choice(size, size=3, replace=False)
        frame.loc[typos, s.RECEIPT_DT] = frame.loc[typos, s.RECEIPT_DT] - pd.DateOffset(years=10)

    # 8. Полные дубли строк — классика выгрузок, склеенных из нескольких файлов.
    duplicated = rng.choice(size, size=max(size // 50, 1), replace=False)
    frame = pd.concat([frame, frame.iloc[duplicated]], ignore_index=True)

    return frame.sample(frac=1.0, random_state=int(rng.integers(0, 10_000))).reset_index(drop=True)


#: Каноническое имя -> заголовок, похожий на выгрузку ОФД.
OFD_EXPORT_HEADERS: dict[str, str] = {
    s.RECEIPT_ID: "Номер ФД",
    s.RECEIPT_DT: "Дата и время",
    s.STORE_ID: "Торговая точка",
    s.CUSTOMER_ID: "Телефон покупателя",
    s.ITEM_NAME: "Наименование товара",
    s.CATEGORY: "Товарная группа",
    s.QTY: "Кол-во",
    s.PRICE: "Цена",
    s.LINE_SUM: "Сумма",
    s.VAT_RATE: "Ставка НДС",
    s.PAYMENT_TYPE: "Форма оплаты",
    s.OPERATION: "Признак расчёта",
}

_RU_OPERATION = {s.OP_SALE: "Приход", s.OP_RETURN: "Возврат прихода"}
_RU_PAYMENT = {s.PAY_CARD: "Безналичными", s.PAY_CASH: "Наличными", s.PAY_PREPAID: "Предоплата"}


def to_ofd_export(frame: pd.DataFrame) -> pd.DataFrame:
    """Переименовать канонические колонки в «как у ОФД» — для демо нормализации."""
    export = frame.copy()
    if s.OPERATION in export:
        export[s.OPERATION] = export[s.OPERATION].map(_RU_OPERATION).fillna(export[s.OPERATION])
    if s.PAYMENT_TYPE in export:
        export[s.PAYMENT_TYPE] = (
            export[s.PAYMENT_TYPE].map(_RU_PAYMENT).fillna(export[s.PAYMENT_TYPE])
        )
    return export.rename(columns=OFD_EXPORT_HEADERS)
