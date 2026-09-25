"""Каноническая схема чековых данных ОФД и приведение к ней внешних выгрузок.

Выгрузки разных ОФД (Платформа, Такском, Первый ОФД, ...) отличаются
названиями колонок, но набор сущностей один и тот же: чек, позиция в чеке,
торговая точка, время, сумма. Здесь описано единое внутреннее представление,
к которому приводится всё остальное, — дальше весь пакет работает только с ним.

Гранулярность таблицы — **одна строка = одна позиция в чеке** (товарная строка).
"""

from __future__ import annotations

import re

# --- Канонические имена колонок -------------------------------------------------

RECEIPT_ID = "receipt_id"
RECEIPT_DT = "receipt_dt"
STORE_ID = "store_id"
CUSTOMER_ID = "customer_id"
ITEM_NAME = "item_name"
CATEGORY = "category"
QTY = "qty"
PRICE = "price"
LINE_SUM = "line_sum"
VAT_RATE = "vat_rate"
PAYMENT_TYPE = "payment_type"
OPERATION = "operation"

#: Без этих колонок работать нельзя — это минимум для любой чековой аналитики.
REQUIRED_COLUMNS: tuple[str, ...] = (
    RECEIPT_ID,
    RECEIPT_DT,
    ITEM_NAME,
    QTY,
    PRICE,
)

#: Эти колонки желательны, но их отсутствие не ломает пайплайн.
#: Часть метрик без них недоступна (например, RFM и когорты без customer_id).
OPTIONAL_COLUMNS: tuple[str, ...] = (
    STORE_ID,
    CUSTOMER_ID,
    CATEGORY,
    LINE_SUM,
    VAT_RATE,
    PAYMENT_TYPE,
    OPERATION,
)

ALL_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

#: Порядок колонок для человека: сначала «где и когда», потом «что и на сколько».
COLUMN_ORDER: tuple[str, ...] = (
    RECEIPT_ID,
    RECEIPT_DT,
    STORE_ID,
    CUSTOMER_ID,
    ITEM_NAME,
    CATEGORY,
    QTY,
    PRICE,
    LINE_SUM,
    VAT_RATE,
    PAYMENT_TYPE,
    OPERATION,
)


def ordered_columns(columns: object) -> list[str]:
    """Отсортировать колонки по :data:`COLUMN_ORDER`, неизвестные — в конец."""
    known = [column for column in COLUMN_ORDER if column in set(columns)]
    extra = [column for column in columns if column not in set(COLUMN_ORDER)]
    return known + extra


# --- Значения категориальных полей ----------------------------------------------

OP_SALE = "sale"
OP_RETURN = "return"

PAY_CASH = "cash"
PAY_CARD = "card"
PAY_PREPAID = "prepaid"

#: Значения признака расчёта из ФФД и их канонические аналоги.
OPERATION_ALIASES: dict[str, str] = {
    "приход": OP_SALE,
    "продажа": OP_SALE,
    "sale": OP_SALE,
    "1": OP_SALE,
    "возврат прихода": OP_RETURN,
    "возврат": OP_RETURN,
    "return": OP_RETURN,
    "refund": OP_RETURN,
    "2": OP_RETURN,
}

PAYMENT_ALIASES: dict[str, str] = {
    "наличные": PAY_CASH,
    "наличными": PAY_CASH,
    "нал": PAY_CASH,
    "cash": PAY_CASH,
    "безналичные": PAY_CARD,
    "безналичными": PAY_CARD,
    "безнал": PAY_CARD,
    "карта": PAY_CARD,
    "картой": PAY_CARD,
    "card": PAY_CARD,
    "электронными": PAY_CARD,
    "предоплата": PAY_PREPAID,
    "предоплатой": PAY_PREPAID,
    "аванс": PAY_PREPAID,
    "prepaid": PAY_PREPAID,
}

# --- Синонимы названий колонок в выгрузках ОФД -----------------------------------

#: Ключ — нормализованное имя колонки из выгрузки, значение — каноническое имя.
#: Нормализация описана в :func:`normalize_key`, поэтому ключи здесь в нижнем
#: регистре, без «ё» и лишних пробелов.
COLUMN_ALIASES: dict[str, str] = {
    # чек
    "receipt_id": RECEIPT_ID,
    "номер чека": RECEIPT_ID,
    "чек": RECEIPT_ID,
    "фд": RECEIPT_ID,
    "номер фд": RECEIPT_ID,
    "фискальный документ": RECEIPT_ID,
    "номер фискального документа": RECEIPT_ID,
    "id чека": RECEIPT_ID,
    # дата и время
    "receipt_dt": RECEIPT_DT,
    "дата": RECEIPT_DT,
    "дата чека": RECEIPT_DT,
    "дата и время": RECEIPT_DT,
    "дата время": RECEIPT_DT,
    "datetime": RECEIPT_DT,
    "date": RECEIPT_DT,
    # торговая точка
    "store_id": STORE_ID,
    "магазин": STORE_ID,
    "торговая точка": STORE_ID,
    "точка продаж": STORE_ID,
    "адрес": STORE_ID,
    "адрес расчетов": STORE_ID,
    "место расчетов": STORE_ID,
    "ккт": STORE_ID,
    "номер ккт": STORE_ID,
    # покупатель
    "customer_id": CUSTOMER_ID,
    "покупатель": CUSTOMER_ID,
    "клиент": CUSTOMER_ID,
    "телефон": CUSTOMER_ID,
    "телефон покупателя": CUSTOMER_ID,
    "email": CUSTOMER_ID,
    "телефон или электронный адрес покупателя": CUSTOMER_ID,
    "карта лояльности": CUSTOMER_ID,
    # позиция
    "item_name": ITEM_NAME,
    "наименование": ITEM_NAME,
    "наименование товара": ITEM_NAME,
    "товар": ITEM_NAME,
    "предмет расчета": ITEM_NAME,
    "название позиции": ITEM_NAME,
    "category": CATEGORY,
    "категория": CATEGORY,
    "группа товаров": CATEGORY,
    "товарная группа": CATEGORY,
    # количество и деньги
    "qty": QTY,
    "количество": QTY,
    "кол-во": QTY,
    "колво": QTY,
    "price": PRICE,
    "цена": PRICE,
    "цена руб": PRICE,
    "цена за единицу": PRICE,
    "цена единицы": PRICE,
    "line_sum": LINE_SUM,
    "сумма": LINE_SUM,
    "сумма руб": LINE_SUM,
    "стоимость": LINE_SUM,
    "сумма позиции": LINE_SUM,
    "итого по позиции": LINE_SUM,
    # налоги и оплата
    "vat_rate": VAT_RATE,
    "ндс": VAT_RATE,
    "ставка ндс": VAT_RATE,
    "payment_type": PAYMENT_TYPE,
    "тип оплаты": PAYMENT_TYPE,
    "форма оплаты": PAYMENT_TYPE,
    "способ оплаты": PAYMENT_TYPE,
    # тип операции
    "operation": OPERATION,
    "операция": OPERATION,
    "тип операции": OPERATION,
    "признак расчета": OPERATION,
}

_SPACES = re.compile(r"\s+")
_PUNCT = re.compile(r"[\"'«».,;:()\[\]]")


def normalize_key(name: object) -> str:
    """Привести имя колонки (или категориальное значение) к виду для сравнения.

    Регистр, «ё», кавычки, знаки препинания и двойные пробелы в выгрузках
    непредсказуемы, поэтому сравниваем не сырые строки, а их нормализованный вид.

    >>> normalize_key("  Дата и  Время ")
    'дата и время'
    >>> normalize_key("Признак расчёта")
    'признак расчета'
    """
    text = str(name).strip().lower().replace("ё", "е")
    text = _PUNCT.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def resolve_column(name: object) -> str | None:
    """Вернуть каноническое имя для колонки выгрузки или ``None``, если неизвестна."""
    return COLUMN_ALIASES.get(normalize_key(name))


def missing_required(columns: object) -> list[str]:
    """Вернуть обязательные колонки, которых нет в ``columns`` (порядок схемы)."""
    present = set(columns)
    return [column for column in REQUIRED_COLUMNS if column not in present]
