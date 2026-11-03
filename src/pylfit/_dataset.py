"""Dataset representation for pylfit.

A Dataset is the boundary between raw user data (typically a
``pandas.DataFrame``) and the rest of the framework. It validates and
coerces feature/target columns to ``str`` once, up front, so that
downstream components (for example, ``Rule.matches``) can trust every
state they receive without re-checking it themselves.

Why it works this way: :doc:`/explanation/about-data`.
"""

import datetime
import decimal
import numbers
from collections.abc import Iterable, Iterator, Set
from typing import TypeGuard
from warnings import warn

import numpy as np
import pandas as pd

from pylfit._checks import _check_collection, _check_frame, find_missing_stand_ins
from pylfit._exceptions import ConflictWarning, PylfitWarning

# How many (negative, positive) feature comparisons
# `_could_be_seen_with` materializes at once: about 16 MB of booleans
# per chunk, so a large dataset is checked in slices rather than in one
# allocation of every pair.
_PAIR_CHUNK = 1 << 24

# How many findings the stand-in warning lists; the rest are counted.
_STAND_INS_SHOWN = 5

# From this share of distinct values among a column's known values, the
# max_categories warning calls the column an identifier or a precise
# measurement.
_ID_SHARE = 0.9


def _is_float(value: object) -> TypeGuard[float | np.floating]:
    """Tells whether a value is a Python or NumPy float."""
    return isinstance(value, float | np.floating)


#: Value types an ``object`` or ``category`` column may not hold: their
#: text is not one value per category (``Decimal("1.5")`` and
#: ``Decimal("1.50")`` are equal, ``bytes`` read as ``"b'x'"`` or fail
#: to decode), or they are dates and durations, as the column types
#: refused too.
_UNSUPPORTED_VALUES = (
    bytes,
    bytearray,
    np.bytes_,
    decimal.Decimal,
    complex,
    np.complexfloating,
    datetime.date,
    datetime.time,
    datetime.timedelta,
    np.datetime64,
    np.timedelta64,
)


def _as_text(value: object) -> str:
    """Writes a value as ``Dataset`` stores it.

    Args:
        value: A known value.

    Returns:
        ``str(value)``, except a whole float, which gives its integer
        text (``1.0`` → ``"1"``), as ``Dataset`` reads a float column of
        whole numbers.
    """
    if _is_float(value) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _check_dataset(dataset: object) -> None:
    """Refuses anything but a ``Dataset`` where one belongs.

    Args:
        dataset: The argument passed.

    Raises:
        TypeError: If ``dataset`` is not a ``Dataset``.
    """
    if not isinstance(dataset, Dataset):
        hint = (
            "; wrap it first: Dataset(df, features=..., targets=...)"
            if isinstance(dataset, pd.DataFrame)
            else ""
        )
        raise TypeError(f"expected a Dataset, got {type(dataset).__name__}{hint}")


def _check_flag(value: object, name: str) -> None:
    """Refuses anything but a ``bool`` for an on/off option.

    Without it, ``"False"`` or ``"no"`` would turn the option on, being
    truthy, and a property reading it back would return the string.

    Args:
        value: The argument passed.
        name: The argument's name, for the error message.

    Raises:
        TypeError: If ``value`` is neither a ``bool`` nor a NumPy bool.
    """
    if not isinstance(value, bool | np.bool_):
        raise TypeError(
            f"{name} must be True or False, got {value!r} ({type(value).__name__})"
        )


class Dataset:
    """A classification dataset backed by a pandas DataFrame.

    Supports multiple target variables, for example, for multi-label
    classification.

    Only categorical-compatible columns are accepted: ``bool``,
    ``str``/``object``, ``category`` and integer columns. Any of them
    warns if it holds more than ``max_categories`` distinct values: an
    integer one is likely continuous data, such as an age or a count,
    and each value of a text or category one is a category of its own,
    which the rules grow with. Float values are rejected, in a float
    column or among the values of an ``object`` or ``category`` one,
    since exact string equality on floats is essentially never
    meaningful; discretize them first. The one exception is a float that
    is a whole number: that is how pandas stores an integer column with
    a missing value, so it is read as an integer (``1.0`` becomes
    ``"1"``) whatever its column's type. Complex, date and duration
    columns are rejected too, and so are an ``object`` or ``category``
    column's ``bytes``, ``Decimal``, complex, date, time and duration
    values; convert them to ``str`` (decode ``bytes``) or to bins first.

    A missing value (``NaN``, ``None``, ``pd.NA`` or ``NaT``) is read as
    unknown: a value the variable had but that was not recorded. An
    unknown is never part of a domain and never satisfies a rule's
    condition. Every state this class hands out holds every feature,
    with ``None`` for an unknown one, so a feature state can be partly
    known. A row whose target is unknown is left out of that target's
    positives and negatives only; it still counts for the dataset's
    other targets.

    Attributes:
        features: The names of the feature (input) variables. Read-only;
            each access returns a new ``list``.
        targets: The names of the target (output) variables. Read-only;
            each access returns a new ``list``.
        conflicting_targets: The targets, in ``targets`` order, for
            which two rows with the same feature state have different
            values. Only rows where every feature and that target are
            known are compared: an unknown could hide a difference
            between two states. Empty, so false, when no target
            conflicts. Conflicts are expected, not an error (for
            example, after discretizing an originally-deterministic
            dataset), but they warn unless ``assume_nondeterministic``
            is set. Read-only; each access returns a new ``list``.
        assume_nondeterministic: Whether the system behind the data is
            taken to be non-deterministic, as passed to the constructor.
            Read-only: ``negatives`` caches results that depend on it.

    Examples:
        >>> import pandas as pd
        >>> from pylfit import Dataset
        >>> df = pd.DataFrame(
        ...     {
        ...         "a": ["0", "0", "1", "1"],
        ...         "b": ["0", "1", "0", "1"],
        ...         "y": ["0", "1", "1", "1"],
        ...     }
        ... )
        >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
        >>> dataset
        <Dataset: 4 rows, features ['a', 'b'], targets ['y']>
        >>> len(dataset)
        4
        >>> sorted(dataset.domain("y"))
        ['0', '1']
    """

    # --- Validation ---

    @staticmethod
    def check_values(dataframe: pd.DataFrame, *, max_categories: int = 5) -> None:
        """Checks that every column holds categorical-compatible values.

        Checks every column of ``dataframe`` as given; a caller wanting
        only a subset checked should narrow ``dataframe`` down first.
        Missing values (``NaN``, ``None``, ``pd.NA``, ``NaT``) are
        allowed: they are unknowns.

        Args:
            dataframe: The DataFrame to check.
            max_categories: Max distinct values a column may hold
                before a warning is raised. Low by default, as in
                ``Dataset``.

        Raises:
            TypeError: If ``dataframe`` is not a ``DataFrame``, or
                ``max_categories`` is a ``bool`` or not an integer (a
                NumPy integer counts as one).
            ValueError: If ``max_categories`` is negative, a column name
                appears more than once in ``dataframe``, a column is
                complex, a date or a duration or holds a value of such a
                type (or ``bytes`` or a ``Decimal``), or a column holds
                a known float value that is not a whole number, in that
                order.

        Examples:
            >>> import pandas as pd
            >>> from pylfit import Dataset
            >>> Dataset.check_values(pd.DataFrame({"a": ["0", "1"], "n": [1, 2]}))
            >>> Dataset.check_values(pd.DataFrame({"n": [1.0, None]}))
            >>> floats = pd.DataFrame({"age": [23.5, 41.0]})
            >>> Dataset.check_values(floats)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: float columns are not supported (not categorical): ['age']. ...
        """
        _check_frame(dataframe)
        Dataset._check_max_categories(max_categories)
        repeated = list(
            dict.fromkeys(dataframe.columns[dataframe.columns.duplicated()])
        )
        if repeated:
            raise ValueError(
                f"duplicate column names in the dataframe: {repeated}. "
                "Rename or drop the copies first."
            )
        Dataset._check_values(dataframe, max_categories)

    @staticmethod
    def _check_max_categories(max_categories: object) -> None:
        """Refuses a ``max_categories`` that isn't a non-negative integer.

        Args:
            max_categories: The value passed.

        Raises:
            TypeError: If it is a ``bool`` or not an integer.
            ValueError: If it is negative.
        """
        if isinstance(max_categories, bool) or not isinstance(
            max_categories, numbers.Integral
        ):
            raise TypeError(
                f"max_categories must be an int, got {max_categories!r} "
                f"({type(max_categories).__name__})"
            )
        if max_categories < 0:
            raise ValueError(
                f"max_categories must be non-negative, got {max_categories!r}"
            )

    @staticmethod
    def _check_values(dataframe: pd.DataFrame, max_categories: int) -> None:
        """Does ``check_values``' work, for it and for ``__init__``.

        Both call it directly, so the warning's ``stacklevel`` points at
        the user's line from either.

        Args:
            dataframe: The DataFrame to check.
            max_categories: As in ``check_values``.
        """
        Dataset._check_floats(dataframe)
        for column in dataframe.columns:
            values = dataframe[column]
            try:
                nunique = values.nunique()
            except TypeError:  # an unhashable value, such as a list
                nunique = values.dropna().astype(str).nunique()
            if nunique <= max_categories:
                continue
            if nunique >= _ID_SHARE * values.count():
                hint = (
                    "Nearly one value per row: an identifier or a precise "
                    "measurement, which no rule can generalize from; drop "
                    "or bin it."
                )
            elif pd.api.types.is_integer_dtype(values) or (
                pd.api.types.is_float_dtype(values)
            ):
                hint = (
                    "It may be continuous data rather than a true category; "
                    "consider discretizing it."
                )
            else:
                hint = (
                    "Each value is a category of its own, and the rules grow "
                    "with them; group rare values, or raise max_categories if "
                    "they are all needed."
                )
            warn(
                f"Column '{column}' has {nunique} distinct values "
                f"(> max_categories={max_categories}). {hint}",
                PylfitWarning,
                stacklevel=3,
            )

    @staticmethod
    def _check_floats(
        dataframe: pd.DataFrame, *, before: str = "building the Dataset"
    ) -> None:
        """Refuses a float column holding a known value that is not whole.

        The half of ``check_values`` that ``predict`` applies too; the
        count warning is for building a dataset only.

        Args:
            dataframe: The DataFrame to check.
            before: What the user is doing, for the message's advice to
                convert "before" it.

        Raises:
            ValueError: If a column is complex, a date or a duration, or
                holds a value of such a type (or ``bytes`` or a
                ``Decimal``); then if a column holds a known float value
                that is not a whole number.
        """
        kinds = [
            column
            for column in dataframe.columns
            if pd.api.types.is_complex_dtype(dataframe[column])
            or pd.api.types.is_datetime64_any_dtype(dataframe[column])
            or pd.api.types.is_timedelta64_dtype(dataframe[column])
            or isinstance(dataframe[column].dtype, pd.PeriodDtype)
        ]
        if kinds:
            raise ValueError(
                f"complex, date and duration columns are not supported (not "
                f"categorical): {kinds}. Convert them to str, or into bins, "
                f"before {before}."
            )
        odd = {
            column: type(value).__name__
            for column in dataframe.columns
            for value in Dataset._present_values(dataframe[column])
            if isinstance(value, _UNSUPPORTED_VALUES)
        }
        if odd:
            listed = ", ".join(f"{column!r} ({kind})" for column, kind in odd.items())
            raise ValueError(
                f"columns holding values with no single text are not supported: "
                f"{listed}. Convert them to str (decode bytes), or into bins, "
                f"before {before}."
            )
        floats = [
            column
            for column in dataframe.columns
            if not Dataset._is_whole(Dataset._float_values(dataframe[column]))
        ]
        if floats:
            raise ValueError(
                f"float columns are not supported (not categorical): {floats}. "
                f"Discretize them (e.g. into bins) before {before}."
            )

    @staticmethod
    def _float_values(column: pd.Series) -> pd.Series:
        """Gives the float values a column holds, whatever its type.

        Args:
            column: Any column.

        Returns:
            A float column's known values; an ``object`` or ``category``
            column's known values that are floats; otherwise nothing.
        """
        if pd.api.types.is_float_dtype(column):
            return column.dropna().astype(float)
        floats = [
            value for value in Dataset._present_values(column) if _is_float(value)
        ]
        return pd.Series(floats, dtype=float)

    @staticmethod
    def _present_values(column: pd.Series) -> list[object]:
        """Gives an ``object`` or ``category`` column's distinct known values.

        The values present, not a category column's categories: a
        category no row holds, as filtering rows leaves, is no value.

        Args:
            column: Any column.

        Returns:
            Its distinct known values, or nothing for another column
            type.
        """
        if not (
            isinstance(column.dtype, pd.CategoricalDtype) or column.dtype == object
        ):
            return []
        known = column.dropna()
        try:
            return list(known.unique())
        except TypeError:  # an unhashable value, such as a list
            return list(known)

    @staticmethod
    def _is_whole(known: pd.Series) -> bool:
        """Tells whether every value of a float series is a whole number.

        Args:
            known: Float values, none missing.

        Returns:
            True if every value is finite, has no fractional part and
            fits a 64-bit integer (vacuously true for no value).
        """
        return bool(
            np.isfinite(known).all()
            and (known % 1 == 0).all()
            and (known.abs() < 2**63).all()
        )

    @staticmethod
    def _warn_stand_ins(dataframe: pd.DataFrame) -> None:
        """Warns about values that may stand for a missing one, if any.

        Run on the columns as passed, before ``_as_str``, so that a zero
        is still a number. Called from ``__init__`` only: ``stacklevel``
        points at the line building the ``Dataset``.

        Args:
            dataframe: The feature and target columns.
        """
        report = find_missing_stand_ins(dataframe)
        if report.empty:
            return
        listed = report.head(_STAND_INS_SHOWN).itertuples(index=False)
        shown = ", ".join(
            f"{column} {value!r} x{count} ({issue})"
            for column, value, count, issue in listed
        )
        if len(report) > _STAND_INS_SHOWN:
            shown += f", and {len(report) - _STAND_INS_SHOWN} more"
        values = "1 value may" if len(report) == 1 else f"{len(report)} values may"
        warn(
            f"{values} stand for a missing value: {shown}. Replace them with "
            "None if unknown; find_missing_stand_ins(df[features + targets]) "
            "gives the full report; pass check_stand_ins=False to silence this.",
            PylfitWarning,
            stacklevel=3,
        )

    @staticmethod
    def _as_str(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Converts every column to ``str``, keeping missing values missing.

        A whole-number float column, as ``check_values`` lets through,
        goes through pandas' nullable integers first, so ``1.0`` becomes
        ``"1"``, the same string an integer column gives; so does a
        whole float among the values of an ``object`` or ``category``
        column.

        Args:
            dataframe: Columns already accepted by ``check_values``.

        Returns:
            The same table with ``str`` values, and ``NaN`` where a
            value was missing (``NaN`` rather than ``None`` because that
            is what pandas' ``str`` columns store).
        """

        def convert(column: pd.Series) -> pd.Series:
            known = column.notna()
            if isinstance(column.dtype, pd.CategoricalDtype):
                # pandas 3 writes a category column with a missing value
                # through floats: 1 as "1.0", the interval (0, 30] as
                # "(0.0, 30.0]". As objects, each keeps its own text.
                column = column.astype(object)
            if pd.api.types.is_float_dtype(column):
                column = column.astype("Int64")
            elif column.dtype == object and len(Dataset._float_values(column)):
                column = column.map(_as_text, na_action="ignore")
            column = column.astype(str).where(known)
            # One object per value, whatever the input held: `astype(str)`
            # on an integer column makes a new `str` per cell, and
            # comparing values is what fitting spends its time on. On
            # 32,768 rows and 15 features, `PRIDE` fits 18% faster from
            # 0/1 integers, 2.1 times faster from strings made per cell;
            # a table whose cells already share costs about 0.05 s more.
            codes, uniques = pd.factorize(column)
            values = np.append(np.asarray(uniques, dtype=object), np.nan)
            return pd.Series(values[codes], index=column.index, dtype=column.dtype)

        return dataframe.apply(convert)

    def __init__(
        self,
        dataframe: pd.DataFrame,
        features: Iterable[str],
        targets: Iterable[str],
        *,
        max_categories: int = 5,
        assume_nondeterministic: bool = False,
        check_stand_ins: bool = True,
    ) -> None:
        """Initializes a dataset from a DataFrame.

        Checks run in this order, each failing before the next runs:
        ``dataframe`` not a ``DataFrame``, ``features`` or ``targets`` a
        single string or a set, a name that is not a ``str``, either
        empty, ``max_categories`` not a non-negative ``int``, a flag
        that is not a ``bool``, missing columns, repeated column names,
        a column named twice in ``dataframe``, then the column checks of
        ``check_values``. A column with more than ``max_categories``
        distinct values then warns rather than raising, and so do values
        that may stand for a missing one (as ``find_missing_stand_ins``
        reports them) when ``check_stand_ins`` is on, and conflicting
        rows when ``assume_nondeterministic`` is off.

        Args:
            dataframe: The raw data, containing at least the columns
                listed in ``features`` and ``targets``. ``NaN``,
                ``None``, ``pd.NA`` and ``NaT`` are read as unknown
                values.
            features: The names of the feature (input) variables: a
                ``list``, a ``tuple``, a ``pandas.Index``... A ``set``
                is refused: its order changes between runs, and so would
                the model's text and saved file.
            targets: The names of the target (output) variables, in the
                same forms.
            max_categories: Max distinct values a feature or target may
                hold before a warning. Low by default, since each value
                multiplies the rules ``GULA`` considers, and the many
                rules on a measurement's values are hard to read: raise
                it for a column known to be a category, such as a
                region code.
            assume_nondeterministic: Set it when the system behind the
                data can give several target values for the same feature
                state. It is a statement about the system that the rows
                can't prove: an unknown can hide whether two rows share
                a state. It changes ``negatives`` only on data with
                unknown feature values, and silences the conflict
                warning.
            check_stand_ins: Whether to warn about values in
                ``features`` and ``targets`` that may stand for a
                missing value, such as a ``"?"`` or a zero far from the
                rest of its column, with ``find_missing_stand_ins``'s
                defaults. Turn it off when they are real values; call
                ``find_missing_stand_ins`` for the full report or other
                settings.

        Raises:
            TypeError: If ``dataframe`` is not a ``DataFrame``,
                ``features`` or ``targets`` is a single string, a set or
                not iterable,
                a name in them is not a ``str``, ``max_categories`` is a
                ``bool`` or not an integer, or
                ``assume_nondeterministic`` or ``check_stand_ins`` is
                not a ``bool``.
            ValueError: If ``features`` or ``targets`` is empty,
                ``max_categories`` is negative, a column is missing, a
                column name is repeated within or across ``features``
                and ``targets``, a column to use appears twice in
                ``dataframe``, or a column fails ``check_values``.

        Warns:
            PylfitWarning: If a column has more than ``max_categories``
                distinct values, or, when ``check_stand_ins`` is on, a
                value may stand for a missing one.
            ConflictWarning: If rows with the same features have
                different values of a target and
                ``assume_nondeterministic`` is off.
        """
        _check_frame(dataframe)
        for name, names in (("features", features), ("targets", targets)):
            _check_collection(name, names)
            if isinstance(names, Set):
                raise TypeError(
                    f"{name} must have an order, not be a {type(names).__name__}, "
                    "whose order changes between runs: pass sorted(...) or a "
                    "list such as [c for c in df.columns if ...]"
                )
        features = list(features)
        targets = list(targets)
        named = [name for name in features + targets if not isinstance(name, str)]
        if named:
            raise TypeError(
                f"feature and target names must be str, got {named!r}: rule "
                "conditions name variables by str. A frame read with "
                "header=None has integer names; rename them first, for "
                "example df.columns = df.columns.astype(str)."
            )
        if not features:
            raise ValueError(
                "features must be non-empty: a dataset with no features has "
                "nothing for a rule body to condition on"
            )
        if not targets:
            raise ValueError(
                "targets must be non-empty: a dataset with no targets has "
                "nothing to learn"
            )
        Dataset._check_max_categories(max_categories)
        _check_flag(assume_nondeterministic, "assume_nondeterministic")
        _check_flag(check_stand_ins, "check_stand_ins")
        # Plain str: a NumPy string (a str subclass) would show as
        # np.str_(...) in every repr and prediction.
        features = [str(name) for name in features]
        targets = [str(name) for name in targets]

        columns = features + targets
        missing = [column for column in columns if column not in dataframe.columns]
        if missing:
            raise ValueError(f"columns not in the dataframe: {missing}")

        duplicates = {column for column in columns if columns.count(column) > 1}
        if duplicates:
            raise ValueError(
                "duplicate names in features and targets: "
                f"{sorted(duplicates)}. A column must appear at most once, "
                "and cannot be both a feature and a target."
            )
        repeated = set(dataframe.columns[dataframe.columns.duplicated()])
        twice = [column for column in columns if column in repeated]
        if twice:
            raise ValueError(
                f"duplicate column names in the dataframe: {twice}. "
                "Rename or drop the copies first."
            )

        Dataset._check_values(dataframe[columns], max_categories)
        if check_stand_ins:
            Dataset._warn_stand_ins(dataframe[columns])

        self._features = list(features)
        self._targets = list(targets)
        self._assume_nondeterministic = bool(assume_nondeterministic)
        # Unknowns stay NaN here, so that pandas' vectorized comparisons
        # treat them as never equal to a value; every state handed out
        # converts them to None (see `_states`).
        self._data = Dataset._as_str(dataframe[columns]).reset_index(drop=True)
        # Positions everywhere inside; the caller's labels only come
        # back out through `to_dataframe`, so its rows join back to
        # their table.
        self._index = dataframe.index.copy()
        self._has_unknowns = bool(self._data[self._features].isna().any().any())

        complete = self._data[self._features].notna().all(axis=1)
        self._conflicting_targets: list[str] = [
            target
            for target in self._targets
            if (
                self._data[complete]
                .groupby(self._features)[target]
                .nunique()  # an unknown target is not a value
                > 1
            ).any()
        ]
        # `negatives` sets aside every state seen with the head value,
        # an unknown in a state read as part of it. Where no state is
        # seen with two target values that check can't exclude
        # anything, so it is skipped. This is wider than
        # `conflicting_targets`, which only compares fully known states.
        state_values = self._data.groupby(self._features, dropna=False)[
            self._targets
        ].nunique()
        self._states_conflict: dict[str, bool] = {
            target: bool((state_values[target] > 1).any()) for target in self._targets
        }

        conflicting = self._conflicting_targets
        if conflicting and not assume_nondeterministic:
            warn(
                f"Rows with the same feature state have different values for "
                f"target(s) {conflicting}. Usual causes: noise, a feature "
                "that matters but is not in the data, or a system that is "
                "non-deterministic. Such a state stays possible for each of "
                "its values. Noise also costs accuracy beyond the conflicts: "
                "a mislabelled row seen once is learned as a true example. If "
                "the system itself is non-deterministic, pass "
                "assume_nondeterministic=True, which also silences this warning.",
                ConflictWarning,
                stacklevel=2,
            )

        # `positives`/`negatives` only depend on `_data`, which never
        # changes after this point, so results are cached per (target,
        # value) for the lifetime of this Dataset — callers
        # (algorithms, model scoring) typically re-derive the same
        # head's positives/negatives many times. Cached as plain `dict`
        # rows rather than `pandas.Series`: cheaper to build and lighter
        # to hold, and `Rule.matches` only ever needs `.get`. Each row's
        # `dict` is built once, in `_rows`, and every head's list holds
        # that same object: a copy per head, about 450 bytes per row and
        # head on 15 features, held 444 MB on 32,768 rows against 14 MB
        # shared.
        self._rows_cache: list[dict[str, str | None]] | None = None
        self._positives_cache: dict[tuple[str, str], list[dict[str, str | None]]] = {}
        self._negatives_cache: dict[tuple[str, str], list[dict[str, str | None]]] = {}
        # Each head's negatives once per distinct state, for the
        # learners: deduplicating builds a key per row, a third of a
        # warm `PRIDE` fit on 32,768 rows (0.38 of 1.14 s) when redone
        # on every fit.
        self._distinct_negatives_cache: dict[
            tuple[str, str], list[dict[str, str | None]]
        ] = {}

        # `domain` only depends on `_data` too, same as
        # positives/negatives above — cached per variable for the same
        # reason. Unlike those, a caller (e.g.
        # `GULA._least_specializations`) typically re-derives the same
        # variable's domain many times per negative example, so an
        # uncached `.unique()` rescans the whole column every single
        # call.
        self._domain_cache: dict[str, list[str]] = {}

    # --- Read-only views ---

    @property
    def features(self) -> list[str]:
        """The names of the feature variables, as a new ``list``."""
        return list(self._features)

    @property
    def targets(self) -> list[str]:
        """The names of the target variables, as a new ``list``."""
        return list(self._targets)

    @property
    def conflicting_targets(self) -> list[str]:
        """The targets whose known rows conflict, as a new ``list``."""
        return list(self._conflicting_targets)

    @property
    def assume_nondeterministic(self) -> bool:
        """Whether the system is taken to be non-deterministic."""
        return self._assume_nondeterministic

    # --- Observers ---

    def __repr__(self) -> str:
        """A one-line summary: row count, features and targets."""
        rows = f"{len(self._data)} row" + ("" if len(self._data) == 1 else "s")
        return (
            f"<{type(self).__name__}: {rows}, "
            f"features {self._features}, targets {self._targets}>"
        )

    def __getstate__(self) -> dict[str, object]:
        """Pickles the dataset without its row caches.

        ``n_jobs`` sends the dataset to every worker process, one after
        another. After a fit the cached rows are several times the size
        of the table itself (9 MB against 2 MB on 32,768 rows and 15
        variables), and each worker rebuilds only those of its own
        heads, in parallel with the others.

        Returns:
            The instance's attributes, the row, positive and negative
            caches emptied.
        """
        state = self.__dict__.copy()
        state["_rows_cache"] = None
        state["_positives_cache"] = {}
        state["_negatives_cache"] = {}
        state["_distinct_negatives_cache"] = {}
        return state

    def __len__(self) -> int:
        """Returns the number of rows, unknown targets included."""
        return len(self._data)

    def __iter__(self) -> Iterator[pd.Series]:
        """Iterates over the dataset's rows, in the DataFrame's row order.

        Each row whole: every feature and target together, in row
        order, a row whose target is unknown included. ``positives``
        gives the same rows one target value at a time, features only.

        Yields:
            Each row, as an ``object`` ``pandas.Series`` named by its
            position, over every feature and target column: a ``str``
            value, or ``None`` where the value is unknown. A caller
            wanting just one half slices it, for example
            ``row[dataset.features]``/``row[dataset.targets]``; a rule
            body reads only the features it names, so an unsliced row
            is still a valid ``Rule.matches`` state.
        """
        # Converted once for the whole frame, not row by row. A
        # frame-wide `where(..., None)` would turn the columns back into
        # pandas' string dtype; one object array keeps each row
        # `object`. A copy: the array may be a read-only view, or the
        # stored data.
        values = self._data.to_numpy(dtype=object, copy=True)
        values[pd.isna(values)] = None
        for position, row in enumerate(values):
            yield pd.Series(row, index=self._data.columns, name=position, dtype=object)

    def domain(self, variable: str) -> list[str]:
        """Returns the distinct values held by a feature or target column.

        Args:
            variable: The name of a feature or target variable.

        Returns:
            The distinct ``str`` values in the column, sorted, as a new
            ``list``. An unknown is not a value, so it is never in it.

        Raises:
            ValueError: If ``variable`` is not a feature or target of
                this dataset.
        """
        if variable not in self._features and variable not in self._targets:
            raise ValueError(f"{variable!r} is not a feature or target of this dataset")
        if variable not in self._domain_cache:
            self._domain_cache[variable] = sorted(
                set(self._data[variable].dropna().unique())
            )
        return list(self._domain_cache[variable])

    def domains(self, variables: Iterable[str] | None = None) -> dict[str, list[str]]:
        """Returns the distinct values held by several variables.

        The several-variable form of ``domain``, shaped for assigning to
        a model's ``domain``.

        Args:
            variables: The feature or target variables to look up.
                Defaults to every feature and target of this dataset.

        Returns:
            Each variable mapped to its distinct values, as in
            ``domain``.

        Raises:
            TypeError: If ``variables`` is a single string.
            ValueError: If ``variables`` names a variable that is not a
                feature or target of this dataset.

        Examples:
            >>> import pandas as pd
            >>> from pylfit import Dataset
            >>> df = pd.DataFrame(
            ...     {
            ...         "a": ["0", "0", "1", "1"],
            ...         "b": ["0", "1", "0", "1"],
            ...         "y": ["0", "1", "1", "1"],
            ...     }
            ... )
            >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
            >>> dataset.domains()
            {'a': ['0', '1'], 'b': ['0', '1'], 'y': ['0', '1']}
        """
        _check_collection("variables", variables)
        if variables is None:
            variables = self._features + self._targets
        return {variable: self.domain(variable) for variable in variables}

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the table this dataset learns from, as a new DataFrame.

        The feature and target columns, in that order, with every value
        as pylfit reads it: a ``str`` (``1`` and ``1.0`` both ``"1"``,
        ``True`` as ``"True"``), or ``NaN`` where it is unknown. The
        rows keep the index labels of the DataFrame the dataset was
        built from, so they join back to it, for example to a column
        such as an ID that the dataset left out. Iterating the dataset
        names its rows by position instead.

        Write it to a file with pandas, such as ``to_csv``; changing the
        result leaves the dataset as it was.

        Returns:
            A copy of the dataset's table.

        Examples:
            >>> import pandas as pd
            >>> from pylfit import Dataset
            >>> df = pd.DataFrame(
            ...     {"id": ["p1", "p2", "p3"], "a": [1.0, None, 0.0], "y": [1, 0, 1]},
            ...     index=[10, 11, 12],
            ... )
            >>> dataset = Dataset(df, features=["a"], targets=["y"])
            >>> dataset.to_dataframe()
                  a  y
            10    1  1
            11  NaN  0
            12    0  1
            >>> dataset.to_dataframe().join(df["id"])["id"].tolist()
            ['p1', 'p2', 'p3']
        """
        frame = self._data.copy()
        frame.index = self._index.copy()
        return frame

    def negatives(self, target: str, value: str) -> list[dict[str, str | None]]:
        """Returns the feature states where a target never holds a value.

        A row is a negative when its target is known and differs from
        ``value``. A feature state seen with ``target == value`` in any
        row is left out entirely, even from the rows where the target
        has another value: on data with conflicts the state is possible
        for both. Here an unknown counts as part of the state, so two
        rows are the same state only when the same features are unknown
        in both.

        With ``assume_nondeterministic``, a row is also left out when
        its state *could* be one seen with ``value``: when it agrees
        with some row of ``value`` on every feature known in both. On a
        non-deterministic system an unknown can hide such a state, and
        keeping the row would rule out a value the system can take.
        Without unknowns this changes nothing. On a deterministic system
        keep it off: a row whose target differs is a true negative
        whatever its unknowns hide, and leaving it out makes rules too
        general. The check compares rows in pairs, so its cost grows
        with the number of rows times the number of distinct states seen
        with ``value``.

        Args:
            target: The target variable to check.
            value: The value ``target`` must never hold, across every
                row sharing a feature state, for that state to be
                included.

        Returns:
            One ``dict`` per row whose feature state is never seen with
            ``target == value``, mapping each of ``self.features`` to
            its value, or to ``None`` where it is unknown. Each call
            gives new dicts: changing one changes nothing in the
            dataset.

        Raises:
            ValueError: If ``target`` is not a target of this dataset.
            TypeError: If ``value`` is not a ``str``.

        Examples:
            >>> import pandas as pd
            >>> from pylfit import Dataset
            >>> df = pd.DataFrame(
            ...     {
            ...         "a": ["0", "0", "1", "1"],
            ...         "b": ["0", "1", "0", "1"],
            ...         "y": ["0", "1", "1", "1"],
            ...     }
            ... )
            >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
            >>> dataset.negatives("y", "1")
            [{'a': '0', 'b': '0'}]

            An unknown shows as ``None``. Here ``b=0`` could be the
            state ``a=0, b=0`` seen with ``y=1``, so only the assumption
            of a non-deterministic system leaves it out:

            >>> df = pd.DataFrame(
            ...     {"a": ["0", None], "b": ["0", "0"], "y": ["1", "0"]}
            ... )
            >>> Dataset(df, features=["a", "b"], targets=["y"]).negatives("y", "1")
            [{'a': None, 'b': '0'}]
            >>> Dataset(
            ...     df, features=["a", "b"], targets=["y"], assume_nondeterministic=True
            ... ).negatives("y", "1")
            []
        """
        return [dict(state) for state in self._negative_states(target, value)]

    def _negative_states(self, target: str, value: str) -> list[dict[str, str | None]]:
        """``negatives`` without the copy, for the algorithms and models.

        The dicts are the cached ones: a caller must not change them.

        Args:
            target: As in ``negatives``.
            value: As in ``negatives``.

        Returns:
            A new list of the cached states.
        """
        self._check_head(target, value)
        key = (target, value)
        if key not in self._negatives_cache:
            column = self._data[target]
            value_observed = column.eq(value)
            if self._states_conflict[target]:
                value_observed = value_observed.groupby(
                    [self._data[feature] for feature in self._features], dropna=False
                ).transform("any")
            negative = column.notna() & ~value_observed
            if self._assume_nondeterministic and self._has_unknowns:
                negative &= ~self._could_be_seen_with(column.eq(value), negative)
            self._negatives_cache[key] = self._states(negative)
        return list(self._negatives_cache[key])

    def _distinct_negative_states(
        self, target: str, value: str
    ) -> list[dict[str, str | None]]:
        """``_negative_states`` with each repeated state dropped.

        A learner would process a repeated state again for nothing,
        since every rule it keeps is already consistent with it. The
        first occurrence is kept, so the order is the negatives' own.
        When nothing repeats, the cache holds the negatives' own list
        rather than a copy of it.

        Args:
            target: As in ``negatives``.
            value: As in ``negatives``.

        Returns:
            A new list of the cached states, each distinct state once.
        """
        key = (target, value)
        if key not in self._distinct_negatives_cache:
            negatives = self._negative_states(target, value)
            seen: set[tuple[str | None, ...]] = set()
            distinct = []
            for negative in negatives:
                state = tuple(negative[feature] for feature in self._features)
                if state not in seen:
                    seen.add(state)
                    distinct.append(negative)
            self._distinct_negatives_cache[key] = (
                distinct
                if len(distinct) < len(negatives)
                else self._negatives_cache[key]
            )
        return list(self._distinct_negatives_cache[key])

    def _could_be_seen_with(
        self, positive: pd.Series, candidate: pd.Series
    ) -> pd.Series:
        """Marks the candidate rows whose state could equal a positive's.

        Two states could be equal when they agree on every feature known
        in both. Only pairs where one side has an unknown need checking:
        two fully known states are equal or not, and ``negatives``
        already sets aside the equal ones.

        Args:
            positive: Per row, whether it is a positive of the head.
            candidate: Per row, whether it is still a negative.

        Returns:
            Per row, True for a candidate that could be the same state
            as some positive.
        """
        features = self._data[self._features]
        # Integer codes per column, -1 for an unknown: one code array
        # compares every feature of a pair at once.
        codes = np.column_stack(
            [pd.factorize(features[feature])[0] for feature in self._features]
        )
        partial = (codes < 0).any(axis=1)
        result = np.zeros(len(codes), dtype=bool)
        positive_mask = positive.to_numpy()
        candidate_mask = candidate.to_numpy()
        for rows, others in (
            (candidate_mask & partial, positive_mask),
            (candidate_mask & ~partial, positive_mask & partial),
        ):
            if not rows.any() or not others.any():
                continue
            row_index = np.flatnonzero(rows)
            other_codes = np.unique(codes[others], axis=0)
            step = max(1, _PAIR_CHUNK // (len(other_codes) * len(self._features)))
            for start in range(0, len(row_index), step):
                chunk = codes[row_index[start : start + step]][:, None, :]
                agree = (
                    (chunk == other_codes[None, :, :])
                    | (chunk < 0)
                    | (other_codes[None, :, :] < 0)
                ).all(axis=2)
                result[row_index[start : start + step]] = agree.any(axis=1)
        return pd.Series(result, index=positive.index)

    def positives(self, target: str, value: str) -> list[dict[str, str | None]]:
        """Returns the feature states where a target holds a value.

        Args:
            target: The target variable to check.
            value: The value ``target`` must hold for a row to be
                included.

        Returns:
            One ``dict`` per row whose ``target`` column equals
            ``value``, mapping each of ``self.features`` to its value,
            or to ``None`` where it is unknown. Each call gives new
            dicts: changing one changes nothing in the dataset.

        Raises:
            ValueError: If ``target`` is not a target of this dataset.
            TypeError: If ``value`` is not a ``str``.

        Examples:
            >>> import pandas as pd
            >>> from pylfit import Dataset
            >>> df = pd.DataFrame(
            ...     {
            ...         "a": ["0", "0", "1", "1"],
            ...         "b": ["0", "1", "0", "1"],
            ...         "y": ["0", "1", "1", "1"],
            ...     }
            ... )
            >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
            >>> dataset.positives("y", "0")
            [{'a': '0', 'b': '0'}]
        """
        return [dict(state) for state in self._positive_states(target, value)]

    def _positive_states(self, target: str, value: str) -> list[dict[str, str | None]]:
        """``positives`` without the copy, for the algorithms and models.

        The dicts are the cached ones: a caller must not change them.

        Args:
            target: As in ``positives``.
            value: As in ``positives``.

        Returns:
            A new list of the cached states.
        """
        self._check_head(target, value)
        key = (target, value)
        if key not in self._positives_cache:
            self._positives_cache[key] = self._states(self._data[target].eq(value))
        return list(self._positives_cache[key])

    def _check_head(self, target: str, value: str) -> None:
        """Checks the head asked of ``positives`` or ``negatives``.

        A value absent from the domain is accepted: no row holds it. A
        value that isn't a ``str`` would match no row either, silently,
        since every value is stored as a ``str``, so it is refused.

        Args:
            target: The target variable asked for.
            value: The value asked for.

        Raises:
            ValueError: If ``target`` is not a target of this dataset.
            TypeError: If ``value`` is not a ``str``.
        """
        if target not in self._targets:
            raise ValueError(f"{target!r} is not a target of this dataset")
        if not isinstance(value, str):
            raise TypeError(
                f"value for {target!r} is {value!r} ({type(value).__name__}), "
                "not a str: pylfit compares values as strings. Convert it "
                "with str()."
            )

    def _states(self, mask: pd.Series) -> list[dict[str, str | None]]:
        """Returns the selected rows' feature states, unknowns as ``None``.

        Args:
            mask: Per row, whether to include it.

        Returns:
            One ``dict`` per selected row, in row order, mapping every
            feature to its value or to ``None``: the ones ``_rows``
            holds, shared with every other head's list.
        """
        rows = self._rows()
        return [rows[index] for index in np.flatnonzero(mask.to_numpy(dtype=bool))]

    def _rows(self) -> list[dict[str, str | None]]:
        """Returns every row's feature state, built on the first call.

        Returns:
            One ``dict`` per row, in row order, mapping every feature to
            its value or to ``None``. The cached list itself: a caller
            must not change it or its dicts.
        """
        if self._rows_cache is None:
            rows = self._data[self._features]
            if self._has_unknowns:
                rows = rows.astype(object).where(rows.notna(), None)
            # One `zip` over each column's list rather than
            # `to_dict("records")`, which boxes every cell: about 2.5
            # times faster on a large table.
            columns = [rows[feature].tolist() for feature in self._features]
            self._rows_cache = [
                dict(zip(self._features, values, strict=True))
                for values in zip(*columns, strict=True)
            ]
        return self._rows_cache
