"""Checks to run on your own data before building a ``Dataset``.

pylfit reads pandas' missing values (``NaN``, ``None``, ``pd.NA``,
``NaT``) as unknown values, and nothing else. Real data often writes a
missing value some other way: a ``"?"``, an empty string, a ``"null"``,
or a zero in a measurement that can't be zero. pylfit would read those
as ordinary values and learn rules on them. ``find_missing_stand_ins``
points them out, so you can decide.

Why it works this way: :doc:`/explanation/about-data`.
"""

import datetime
from collections.abc import Iterable, Set
from typing import Literal

import numpy as np
import pandas as pd

#: The strings ``find_missing_stand_ins`` reports by default, compared
#: case-insensitively and with surrounding spaces ignored. deepchecks'
#: ``MixedNulls`` check looks for ``"none"``, ``"null"``, ``"nan"``,
#: ``"na"`` and ``""``; this list adds ``"?"`` (the UCI datasets'
#: marker), ``"-"``, ``"missing"`` and ``"unknown"``, and spells out
#: ``"n/a"``, since punctuation is compared as written here.
DEFAULT_STAND_INS: tuple[str, ...] = (
    "?",
    "",
    "-",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "missing",
    "unknown",
)

#: The ``issue`` column of ``find_missing_stand_ins``'s report: a
#: stand-in string, or a zero far from the rest of its column.
Issue = Literal["stand_in", "far_zero"]

# The fewest distinct non-zero values a column needs for its zeros to be
# checked: with fewer, quartiles say nothing, and a column of two
# values, such as 0/1, is Boolean-like, where a zero is half the data.
_MIN_NONZERO_VALUES = 3

# Values on no number line, whatever ``pandas.to_numeric`` makes of them:
# it would read a complex number's real part, so 7j as zero, and a date
# or a duration as its count of nanoseconds, so 1970-01-01 as zero.
_NOT_NUMBERS = (
    complex,
    np.complexfloating,
    datetime.date,
    datetime.timedelta,
    np.datetime64,
    np.timedelta64,
)


def _check_collection(name: str, argument: object) -> None:
    """Refuses a single string where a collection of names belongs.

    A ``str`` is iterable, so ``"a"`` would otherwise read as one name
    per character.

    Args:
        name: The argument's name, for the message.
        argument: The argument passed.

    Raises:
        TypeError: If ``argument`` is a ``str``.
    """
    if isinstance(argument, str):
        raise TypeError(
            f"{name} must be a collection, such as [{argument!r}], not a single string"
        )


def _check_frame(dataframe: object) -> None:
    """Refuses anything but a ``DataFrame`` where one belongs.

    Args:
        dataframe: The argument passed.

    Raises:
        TypeError: If ``dataframe`` is not a ``pandas.DataFrame``.
    """
    if not isinstance(dataframe, pd.DataFrame):
        hint = (
            "; pass df[[column]] or series.to_frame()"
            if isinstance(dataframe, pd.Series)
            else ""
        )
        raise TypeError(f"expected a DataFrame, got {type(dataframe).__name__}{hint}")


def find_missing_stand_ins(
    dataframe: pd.DataFrame,
    *,
    stand_ins: Iterable[str] = DEFAULT_STAND_INS,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Finds values that may stand for a missing value, for you to check.

    Two kinds of value are reported:

    - A string in ``stand_ins``, compared case-insensitively with
      surrounding spaces ignored: ``"?"``, ``""``, ``"null"``,
      ``"unknown"``...
    - A zero far from the rest of a numeric-looking column, such as a
      blood pressure of ``0`` among values from 60 to 180: a common way
      to record a measurement nobody took. A column looks numeric when
      every known value that isn't a stand-in reads as a finite number,
      ``"0"`` included, so a table read with ``dtype=str`` is checked
      too; a ``True`` or ``False``, a date or a duration is not a number
      here.
      "Far" is outside Tukey's fences of the column's non-zero values,
      ``Q1 - 1.5 * IQR`` to ``Q3 + 1.5 * IQR``, the usual rule for an
      outlier [Tukey1977]_. A column with fewer than three distinct
      non-zero values is skipped: it is Boolean-like, such as
      ``0``/``1``, or too small for quartiles. So is one where zeros are
      more than half the rows checked: a column mostly zero, such as an
      amount most rows don't have, has zero as its usual value, not as
      an outlier.

    Nothing is changed. A reported value may be a real one: a vote of
    ``"?"`` meaning "neither", an answer ``"unknown"``. If it stands for
    a value nobody recorded, replace it with ``None`` before building
    the ``Dataset``. If the absence itself carries information, keep it
    as a value of its own, such as ``"not measured"``. ``NaN``,
    ``None``, ``pd.NA`` and ``NaT`` are not reported: pylfit already
    reads them as unknown.

    Args:
        dataframe: The data to check.
        stand_ins: The strings to report.
        columns: The columns to check, in this order, each once however
            often it is named. ``None`` checks every column. A ``set``
            is refused, as its order changes between runs.

    Returns:
        One row per finding, with columns ``column``, ``value`` (as
        found in the data), ``count`` (its number of rows) and
        ``issue``: ``"stand_in"`` for a string in ``stand_ins``,
        ``"far_zero"`` for a zero far from the rest of its column. Empty
        when nothing looks suspect.

    Raises:
        ValueError: If ``columns`` names a column ``dataframe`` lacks
            (as ``Dataset`` reports one), or a column to check appears
            more than once in ``dataframe``.
        TypeError: If ``dataframe`` is not a ``DataFrame``, ``columns``
            or ``stand_ins`` is a single string or not iterable,
            ``columns`` is a set, or ``stand_ins`` holds anything but
            strings.

    References:
        .. [Tukey1977] Tukey, J. W. (1977). *Exploratory Data Analysis*.
           Addison-Wesley. ISBN 978-0-201-07616-5.

    Examples:
        >>> import pandas as pd
        >>> from pylfit import find_missing_stand_ins
        >>> df = pd.DataFrame(
        ...     {
        ...         "smoker": ["yes", "no", "?", "no", "yes", "NULL"],
        ...         "pressure": [120, 135, 0, 110, 160, 140],
        ...     }
        ... )
        >>> report = find_missing_stand_ins(df)
        >>> for column, value, count, issue in report.itertuples(index=False):
        ...     print(f"{column}: {value!r} x{count} ({issue})")
        smoker: '?' x1 (stand_in)
        smoker: 'NULL' x1 (stand_in)
        pressure: 0 x1 (far_zero)
    """
    _check_frame(dataframe)
    _check_collection("stand_ins", stand_ins)
    _check_collection("columns", columns)
    if isinstance(columns, Set):
        raise TypeError(
            f"columns must have an order, not be a {type(columns).__name__}, "
            "whose order changes between runs: pass sorted(...) or a list"
        )
    stand_ins = list(stand_ins)
    others = [stand_in for stand_in in stand_ins if not isinstance(stand_in, str)]
    if others:
        raise TypeError(f"stand_ins must hold strings, got {others!r}")
    wanted = {stand_in.strip().lower() for stand_in in stand_ins}
    selected = list(dict.fromkeys(dataframe.columns if columns is None else columns))
    missing = [column for column in selected if column not in dataframe.columns]
    if missing:
        raise ValueError(f"columns not in the dataframe: {missing}")
    repeated = set(dataframe.columns[dataframe.columns.duplicated()])
    duplicates = [column for column in selected if column in repeated]
    if duplicates:
        raise ValueError(
            f"duplicate column names in the dataframe: {duplicates}. "
            "Each column to check must appear once."
        )

    findings: list[tuple[object, object, int, Issue]] = []
    for column in selected:
        values = dataframe[column]
        # Each test below runs once per distinct value, not once per
        # row.
        counts = values.value_counts(sort=False)
        # A categorical column also counts its unused categories, as
        # zeros.
        counts = counts[counts > 0]
        if pd.api.types.is_numeric_dtype(values):
            is_stand_in = np.zeros(len(counts), dtype=bool)
        else:
            is_stand_in = np.array(
                [
                    isinstance(value, str) and value.strip().lower() in wanted
                    for value in counts.index
                ],
                dtype=bool,
            )
        for value, count in _sorted(counts[is_stand_in]):
            findings.append((column, value, count, "stand_in"))
        for value, count in _far_zeros(counts[~is_stand_in], values):
            findings.append((column, value, count, "far_zero"))
    return _report(findings, dataframe.columns.dtype)


def _report(
    findings: list[tuple[object, object, int, Issue]],
    column_dtype: np.dtype | pd.api.extensions.ExtensionDtype,
) -> pd.DataFrame:
    """Builds the report with the same column types, empty or not.

    Args:
        findings: One ``(column, value, count, issue)`` per finding.
        column_dtype: The dtype of the checked dataframe's column
            labels.

    Returns:
        The report, columns ``column``, ``value``, ``count``, ``issue``.
    """
    column, value, count, issue = (
        zip(*findings, strict=True) if findings else ((), (), (), ())
    )
    return pd.DataFrame(
        {
            "column": pd.Series(column, dtype=column_dtype),
            "value": pd.Series(value, dtype=object),
            "count": pd.Series(count, dtype="int64"),
            # "str": what pandas makes of text, object before pandas 3.
            "issue": pd.Series(issue, dtype="str"),
        }
    )


def _sorted(counts: pd.Series) -> list[tuple[object, int]]:
    """Lists each value with its count, sorted by its text.

    Args:
        counts: Values (the index) with their number of rows.

    Returns:
        Each value with its count.
    """
    # A NumPy scalar, as a nullable column gives, as its Python value:
    # 0 rather than np.int64(0) in the report and the warning.
    return sorted(
        (
            (value.item() if isinstance(value, np.generic) else value, int(count))
            for value, count in counts.items()
        ),
        key=lambda pair: str(pair[0]),
    )


def _far_zeros(counts: pd.Series, values: pd.Series) -> list[tuple[object, int]]:
    """Finds the zeros of a numeric-looking column outside Tukey's fences.

    Args:
        counts: A column's distinct known values (the index), stand-ins
            left out, with their number of rows.
        values: The whole column.

    Returns:
        Each way a far zero is written (``0``, ``"0"``, ``"0.0"``...)
        with its count; empty if the column doesn't look numeric, has
        too few distinct non-zero values, is mostly zeros, or its zeros
        are not far.
    """
    if counts.empty:
        return []
    distinct = pd.Series(counts.index.to_numpy(dtype=object))
    if any(isinstance(value, _NOT_NUMBERS) for value in distinct):
        return []
    numbers = pd.to_numeric(distinct, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numbers).all():
        return []
    zero = numbers == 0
    nonzero = numbers[~zero]
    if not zero.any() or len(np.unique(nonzero)) < _MIN_NONZERO_VALUES:
        return []
    if 2 * counts.to_numpy()[zero].sum() > counts.sum():
        return []  # mostly zeros: zero is the column's usual value
    # Each non-zero number as many times as it occurs: the same
    # quartiles as over the rows, interpolated linearly as pandas does.
    rows = np.repeat(nonzero, counts.to_numpy()[~zero])
    q1, q3 = np.quantile(rows, [0.25, 0.75])
    spread = 1.5 * (q3 - q1)
    if q1 - spread <= 0 <= q3 + spread:
        return []
    # to_numeric reads False as 0, and value_counts merges False with 0
    # (they are equal), so a bool is looked for in the rows themselves:
    # only here, about to report, where it is rarely needed.
    if not pd.api.types.is_numeric_dtype(values) and any(
        isinstance(value, (bool, np.bool_)) for value in values
    ):
        return []
    return _sorted(counts[zero])
