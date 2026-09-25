"""ABC- и XYZ-анализ ассортимента.

ABC отвечает на вопрос «что приносит деньги»: товары сортируются по вкладу в
выручку, первые 80% попадают в класс A, следующие 15% — в B, остаток — в C.
XYZ отвечает на вопрос «на что можно опираться в планировании»: чем ровнее
спрос по месяцам, тем ниже коэффициент вариации и тем ближе товар к классу X.

Вместе они дают матрицу 3×3: AX — то, на чём стоит бизнес, CZ — кандидаты на
вывод из ассортимента.
"""

from __future__ import annotations

import pandas as pd

from .. import schema as s

#: Границы накопленной доли выручки для классов A и B.
ABC_THRESHOLDS: tuple[float, float] = (0.80, 0.95)

#: Границы коэффициента вариации для классов X и Y.
XYZ_THRESHOLDS: tuple[float, float] = (0.10, 0.25)


def abc_analysis(
    frame: pd.DataFrame,
    by: str = s.ITEM_NAME,
    value: str = s.LINE_SUM,
    thresholds: tuple[float, float] = ABC_THRESHOLDS,
) -> pd.DataFrame:
    """Разложить ассортимент по классам A, B, C по вкладу в выручку.

    Args:
        frame: очищенные позиции чеков. Возвраты учитываются со знаком минус,
            поэтому в основе лежит чистая выручка.
        by: по какому признаку группировать — товар, категория, точка.
        value: какую метрику накапливать.
        thresholds: границы накопленной доли для A и B.

    Returns:
        DataFrame, отсортированный по убыванию вклада: ``value``, ``share``,
        ``cum_share``, ``abc``, ``rank``.

    Raises:
        ValueError: если суммарное значение метрики неположительно — накопленную
            долю в этом случае считать бессмысленно.
    """
    low, high = thresholds
    if not 0 < low < high < 1:
        raise ValueError("thresholds должны быть возрастающими долями внутри (0, 1)")
    if frame.empty:
        return pd.DataFrame(columns=[by, value, "share", "cum_share", "abc", "rank"])

    grouped = frame.groupby(by, dropna=False)[value].sum().sort_values(ascending=False)
    total = float(grouped.sum())
    if total <= 0:
        raise ValueError(f"Суммарное значение {value!r} неположительно — ABC-анализ неприменим")

    result = grouped.reset_index()
    result["share"] = (result[value] / total).round(6)
    result["cum_share"] = result["share"].cumsum().round(6)
    # Класс определяется накопленной долей на самом товаре включительно:
    # товар, которым доля переваливает за 80%, ещё относится к A.
    previous = result["cum_share"].shift(1, fill_value=0.0)
    result["abc"] = "C"
    result.loc[previous < high, "abc"] = "B"
    result.loc[previous < low, "abc"] = "A"
    result[value] = result[value].round(2)
    result["rank"] = range(1, len(result) + 1)
    return result


def xyz_analysis(
    frame: pd.DataFrame,
    by: str = s.ITEM_NAME,
    value: str = s.QTY,
    freq: str = "M",
    thresholds: tuple[float, float] = XYZ_THRESHOLDS,
) -> pd.DataFrame:
    """Разложить ассортимент по стабильности спроса: классы X, Y, Z.

    Периоды без продаж заполняются нулями — иначе товар, проданный дважды в год,
    выглядел бы стабильнее, чем товар, продающийся каждый день.

    Args:
        frame: очищенные позиции чеков.
        by: признак группировки.
        value: метрика спроса, по умолчанию количество.
        freq: период агрегации в нотации периодов pandas (``"M"`` — месяц).
        thresholds: границы коэффициента вариации для X и Y.

    Returns:
        DataFrame: ``mean``, ``std``, ``cv`` (коэффициент вариации), ``xyz``.
    """
    low, high = thresholds
    if not 0 < low < high:
        raise ValueError("thresholds должны быть возрастающими положительными числами")
    if frame.empty:
        return pd.DataFrame(columns=[by, "mean", "std", "cv", "xyz"])

    periods = frame[s.RECEIPT_DT].dt.to_period(freq)
    demand = frame.groupby([by, periods], dropna=False)[value].sum().unstack(fill_value=0.0)
    full_range = pd.period_range(periods.min(), periods.max(), freq=freq)
    demand = demand.reindex(columns=full_range, fill_value=0.0)

    mean = demand.mean(axis=1)
    std = demand.std(axis=1, ddof=0)
    cv = (std / mean.where(mean != 0)).astype("float64")

    result = pd.DataFrame(
        {by: demand.index, "mean": mean.round(3).to_numpy(), "std": std.round(3).to_numpy()}
    )
    result["cv"] = cv.round(4).to_numpy()
    result["xyz"] = "Z"
    result.loc[result["cv"] <= high, "xyz"] = "Y"
    result.loc[result["cv"] <= low, "xyz"] = "X"
    result.loc[result["cv"].isna(), "xyz"] = "Z"
    return result.sort_values("cv").reset_index(drop=True)


def abc_xyz(
    frame: pd.DataFrame,
    by: str = s.ITEM_NAME,
    freq: str = "M",
) -> pd.DataFrame:
    """Совместить ABC и XYZ в одну матрицу и вернуть сводный класс, например ``"AX"``."""
    abc = abc_analysis(frame, by=by)
    xyz = xyz_analysis(frame, by=by, freq=freq)
    merged = abc.merge(xyz, on=by, how="left")
    merged["xyz"] = merged["xyz"].fillna("Z")
    merged["class"] = merged["abc"].astype("string") + merged["xyz"].astype("string")
    return merged


def abc_xyz_matrix(combined: pd.DataFrame, value: str = s.LINE_SUM) -> pd.DataFrame:
    """Свернуть результат :func:`abc_xyz` в таблицу 3×3 с выручкой в клетках."""
    if combined.empty:
        return pd.DataFrame()
    matrix = combined.pivot_table(
        index="abc", columns="xyz", values=value, aggfunc="sum", fill_value=0.0
    )
    return matrix.reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0.0).round(2)
