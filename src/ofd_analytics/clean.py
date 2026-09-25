"""Приведение выгрузки к канонической схеме и очистка чековых данных.

Модуль решает ровно одну задачу: превратить «файл, который дали» в таблицу,
которой можно доверять в расчётах. Каждый шаг очистки считает, сколько строк он
затронул, и попадает в :class:`CleaningReport` — без отчёта очистка превращается
в необъяснимую потерю выручки.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from . import schema as s

#: Раньше этой даты чеков в ОФД быть не может: онлайн-кассы по 54-ФЗ
#: заработали в 2016 году. Всё, что раньше, — опечатка в годе.
MIN_VALID_DT = pd.Timestamp("2016-01-01")

_NUMBER_TRASH = re.compile(r"[\s ₽%]|руб\.?", flags=re.IGNORECASE)
_SPACES = re.compile(r"\s+")


class MissingColumnsError(ValueError):
    """В таблице нет обязательных колонок схемы."""


@dataclass(frozen=True)
class CleaningStep:
    """Один шаг очистки: что сделали и сколько строк это затронуло."""

    name: str
    affected: int
    note: str = ""


@dataclass
class CleaningReport:
    """Итог очистки: сколько строк пришло, сколько осталось и что происходило."""

    rows_in: int
    rows_out: int = 0
    steps: list[CleaningStep] = field(default_factory=list)
    unknown_columns: list[str] = field(default_factory=list)

    def add(self, name: str, affected: int, note: str = "") -> None:
        """Записать шаг, если он что-то затронул."""
        if affected:
            self.steps.append(CleaningStep(name, int(affected), note))

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_out

    def as_frame(self) -> pd.DataFrame:
        """Шаги очистки таблицей — удобно печатать и сохранять рядом с данными."""
        return pd.DataFrame(
            [{"step": step.name, "rows": step.affected, "note": step.note} for step in self.steps],
            columns=["step", "rows", "note"],
        )

    def to_markdown(self) -> str:
        """Отчёт об очистке в Markdown."""
        share = self.rows_dropped / self.rows_in * 100 if self.rows_in else 0.0
        lines = [
            "### Очистка данных",
            "",
            f"- строк на входе: **{self.rows_in:,}**".replace(",", " "),
            f"- строк на выходе: **{self.rows_out:,}**".replace(",", " "),
            f"- удалено: **{self.rows_dropped:,}** ({share:.2f}%)".replace(",", " "),
        ]
        if self.unknown_columns:
            extra = ", ".join(self.unknown_columns)
            lines.append(f"- колонки вне схемы (оставлены как есть): {extra}")
        if self.steps:
            lines += ["", "| шаг | строк | комментарий |", "| --- | ---: | --- |"]
            lines += [f"| {step.name} | {step.affected} | {step.note} |" for step in self.steps]
        return "\n".join(lines)


def normalize_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Переименовать колонки выгрузки в канонические.

    Колонки, которых нет в :data:`ofd_analytics.schema.COLUMN_ALIASES`, остаются
    с исходными именами и возвращаются вторым элементом — их видно в отчёте.

    Raises:
        ValueError: если две разные колонки претендуют на одно каноническое имя.
    """
    mapping: dict[object, str] = {}
    unknown: list[str] = []
    for column in frame.columns:
        canonical = s.resolve_column(column)
        if canonical is None:
            unknown.append(str(column))
        elif canonical in mapping.values():
            taken = next(key for key, value in mapping.items() if value == canonical)
            raise ValueError(
                f"Колонки {taken!r} и {column!r} обе отображаются в {canonical!r} — "
                "уберите лишнюю или переименуйте вручную"
            )
        else:
            mapping[column] = canonical
    return frame.rename(columns=mapping), unknown


def to_number(series: pd.Series) -> pd.Series:
    """Разобрать числа из текста: ``"1 234,56"``, ``"20%"``, ``"99 руб."``, ``""``.

    Нераспознанное становится ``NaN`` — это видно в отчёте, в отличие от нуля.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64")
    text = series.astype("string").str.strip()
    text = text.str.replace(_NUMBER_TRASH, "", regex=True).str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce").astype("float64")


def parse_datetime(series: pd.Series) -> pd.Series:
    """Разобрать дату и время чека, не перепутав день с месяцем.

    Разбор идёт в два прохода, и порядок здесь принципиален:

    1. сначала ISO (``2025-03-06 18:30``) — такие строки однозначны;
    2. остаток разбирается с ``dayfirst=True``, потому что российская выгрузка
       пишет ``06.03.2025``.

    Один проход с ``dayfirst=True`` не годится: он молча переставит день и месяц
    в ISO-датах, где день не больше 12, — самая дорогая ошибка в чековых данных,
    потому что данные остаются правдоподобными.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    text = series.astype("string").str.strip()
    parsed = pd.to_datetime(text, errors="coerce", format="ISO8601")
    remaining = parsed.isna() & text.notna() & text.ne("")
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(
            text[remaining], errors="coerce", format="mixed", dayfirst=True
        )
    return parsed


def _map_aliases(series: pd.Series, aliases: dict[str, str]) -> pd.Series:
    """Привести значения категориального поля к каноническим."""
    keys = series.astype("string").map(s.normalize_key, na_action="ignore")
    return keys.map(aliases, na_action="ignore").fillna(keys)


def canonicalize_item_names(series: pd.Series) -> pd.Series:
    """Свести написания наименований к одному варианту на товар.

    ``" КОФЕ ЛАТТЕ 0.4 "``, ``"кофе латте 0.4"`` и ``"Кофе латте 0.4"`` — это один
    товар. Представителем группы становится самое частое исходное написание,
    поэтому нормализация не портит читаемость отчётов.
    """
    trimmed = series.astype("string").str.strip().str.replace(_SPACES, " ", regex=True)
    keys = trimmed.map(s.normalize_key, na_action="ignore")
    known = trimmed.notna()
    if not known.any():
        return trimmed
    representative = trimmed[known].groupby(keys[known]).agg(lambda group: group.mode().iat[0])
    return keys.map(representative).astype("string").fillna(trimmed)


def coerce_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Привести типы канонических колонок: даты, числа, категориальные значения."""
    frame = frame.copy()

    if s.RECEIPT_DT in frame:
        frame[s.RECEIPT_DT] = parse_datetime(frame[s.RECEIPT_DT])
    for column in (s.QTY, s.PRICE, s.LINE_SUM, s.VAT_RATE):
        if column in frame:
            frame[column] = to_number(frame[column])
    for column in (s.RECEIPT_ID, s.STORE_ID, s.CUSTOMER_ID):
        if column in frame:
            text = frame[column].astype("string").str.strip()
            frame[column] = text.replace("", pd.NA)
    if s.OPERATION in frame:
        frame[s.OPERATION] = _map_aliases(frame[s.OPERATION], s.OPERATION_ALIASES).fillna(s.OP_SALE)
    if s.PAYMENT_TYPE in frame:
        frame[s.PAYMENT_TYPE] = _map_aliases(frame[s.PAYMENT_TYPE], s.PAYMENT_ALIASES)
    return frame


def clean_receipts(
    frame: pd.DataFrame,
    *,
    period: tuple[str | pd.Timestamp, str | pd.Timestamp] | None = None,
    sum_strategy: Literal["sum", "price"] = "sum",
    sum_tolerance: float = 0.01,
    asof: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Очистить таблицу позиций чеков и вернуть её вместе с отчётом.

    Args:
        frame: сырая таблица — уже канонические колонки или выгрузка ОФД.
        period: если задан, строки вне ``[начало, конец]`` удаляются.
        sum_strategy: чему верить при расхождении ``line_sum`` и ``qty × price``.
            ``"sum"`` (по умолчанию) считает главной сумму позиции — в фискальном
            документе это фактически оплаченные деньги, включая скидку, — и
            пересчитывает цену. ``"price"`` делает наоборот.
        sum_tolerance: допустимое расхождение в рублях, ниже него ничего не правим.
        asof: «сегодня» для отсечения чеков из будущего; по умолчанию текущий момент.

    Returns:
        Кортеж ``(очищенная таблица, отчёт)``. Возвраты в результате имеют
        отрицательные ``qty`` и ``line_sum``, поэтому суммы можно складывать
        напрямую, не вычитая возвраты отдельно.

    Raises:
        MissingColumnsError: если после нормализации нет обязательных колонок.
    """
    report = CleaningReport(rows_in=len(frame))

    frame, unknown = normalize_columns(frame)
    report.unknown_columns = unknown

    absent = s.missing_required(frame.columns)
    if absent:
        raise MissingColumnsError(
            f"Нет обязательных колонок: {', '.join(absent)}. "
            f"Известные синонимы см. в ofd_analytics.schema.COLUMN_ALIASES"
        )

    frame = coerce_types(frame)

    names_before = frame[s.ITEM_NAME].astype("string")
    frame[s.ITEM_NAME] = canonicalize_item_names(frame[s.ITEM_NAME])
    report.add(
        "унифицированы наименования",
        int((names_before.fillna("") != frame[s.ITEM_NAME].fillna("")).sum()),
        "регистр и пробелы",
    )

    duplicates = int(frame.duplicated().sum())
    frame = frame.drop_duplicates()
    report.add("удалены полные дубли строк", duplicates)

    has_sum = s.LINE_SUM in frame
    if has_sum:
        restorable = frame[s.LINE_SUM].isna() & frame[s.PRICE].notna() & frame[s.QTY].notna()
        frame.loc[restorable, s.LINE_SUM] = (
            frame.loc[restorable, s.PRICE] * frame.loc[restorable, s.QTY]
        ).round(2)
        report.add("восстановлена сумма позиции", int(restorable.sum()), "price × qty")

        recoverable_price = frame[s.PRICE].isna() & frame[s.LINE_SUM].notna() & frame[s.QTY].ne(0)
        frame.loc[recoverable_price, s.PRICE] = (
            frame.loc[recoverable_price, s.LINE_SUM] / frame.loc[recoverable_price, s.QTY]
        ).round(2)
        report.add("восстановлена цена", int(recoverable_price.sum()), "line_sum / qty")

        expected = frame[s.PRICE] * frame[s.QTY]
        mismatch = (frame[s.LINE_SUM] - expected).abs() > sum_tolerance
        mismatch &= frame[s.LINE_SUM].notna() & expected.notna()
        if sum_strategy == "sum":
            repairable = mismatch & frame[s.QTY].ne(0)
            frame.loc[repairable, s.PRICE] = (
                frame.loc[repairable, s.LINE_SUM] / frame.loc[repairable, s.QTY]
            ).round(4)
            note = "сумма главная, пересчитана цена"
        else:
            frame.loc[mismatch, s.LINE_SUM] = expected[mismatch].round(2)
            note = "цена главная, пересчитана сумма"
        report.add("устранено расхождение суммы", int(mismatch.sum()), note)
    else:
        frame[s.LINE_SUM] = (frame[s.PRICE] * frame[s.QTY]).round(2)
        report.add("добавлена сумма позиции", len(frame), "в выгрузке не было колонки")

    bad_qty = frame[s.QTY].isna() | (frame[s.QTY] <= 0)
    frame = frame[~bad_qty]
    report.add("удалены строки с непозитивным количеством", int(bad_qty.sum()))

    incomplete = frame[s.RECEIPT_DT].isna() | frame[s.LINE_SUM].isna() | frame[s.ITEM_NAME].isna()
    frame = frame[~incomplete]
    report.add("удалены строки с незаполненным ключевым полем", int(incomplete.sum()))

    horizon = pd.Timestamp(asof) if asof is not None else pd.Timestamp.now()
    impossible = (frame[s.RECEIPT_DT] < MIN_VALID_DT) | (frame[s.RECEIPT_DT] > horizon)
    frame = frame[~impossible]
    report.add(
        "удалены невозможные даты",
        int(impossible.sum()),
        f"вне [{MIN_VALID_DT.date()}, {horizon.date()}]",
    )

    if period is not None:
        start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
        if end == end.normalize():
            end = end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        outside = (frame[s.RECEIPT_DT] < start) | (frame[s.RECEIPT_DT] > end)
        frame = frame[~outside]
        report.add(
            "удалены строки вне периода", int(outside.sum()), f"{start.date()}..{end.date()}"
        )

    if s.OPERATION in frame:
        returns = frame[s.OPERATION].eq(s.OP_RETURN)
        for column in (s.QTY, s.LINE_SUM):
            frame.loc[returns, column] = -frame.loc[returns, column].abs()
        report.add("возвраты переведены в минус", int(returns.sum()), "qty и line_sum")
    else:
        frame[s.OPERATION] = s.OP_SALE
        report.add("добавлен признак операции", len(frame), f"все строки как {s.OP_SALE}")

    frame = frame[s.ordered_columns(frame.columns)].reset_index(drop=True)
    frame[s.QTY] = frame[s.QTY].astype("float64")
    frame[s.LINE_SUM] = frame[s.LINE_SUM].astype("float64")
    report.rows_out = len(frame)
    return frame, report


def require_customers(frame: pd.DataFrame) -> pd.DataFrame:
    """Оставить только строки с известным покупателем.

    RFM и когорты без ``customer_id`` посчитать нельзя, а в чеках ОФД покупатель
    известен далеко не всегда — поэтому отсев выделен в явный шаг.

    Raises:
        MissingColumnsError: если колонки ``customer_id`` нет вовсе.
    """
    if s.CUSTOMER_ID not in frame:
        raise MissingColumnsError(
            "Нет колонки customer_id: RFM и когортный анализ недоступны. "
            "В выгрузке ОФД покупатель есть только у чеков с телефоном, "
            "email или картой лояльности"
        )
    identified = frame[frame[s.CUSTOMER_ID].notna() & frame[s.CUSTOMER_ID].astype("string").ne("")]
    return identified.reset_index(drop=True)


def sales_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Оставить только продажи, без возвратов."""
    if s.OPERATION not in frame:
        return frame.reset_index(drop=True)
    return frame[frame[s.OPERATION].eq(s.OP_SALE)].reset_index(drop=True)
