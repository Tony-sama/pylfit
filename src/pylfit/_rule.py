"""Logic rule representation for pylfit.

A rule is a Horn-clause-like structure of the form::

    target(value) :- var_1(val_1), var_2(val_2), ...

where the head is the conclusion and the body is a conjunction of
conditions. A rule holds (matches) in a given state of the system when
all its conditions are satisfied by that state.

Why it works this way: :doc:`/explanation/about-rules`.
"""

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, cast

import numpy as np
import pandas as pd


class _Gettable(Protocol):
    """Any object with a mapping's ``get``, for ``State``."""

    def get(self, key: str, default: Any = ..., /) -> Any: ...


#: The type of a system state, which maps each variable to its ``str``
#: value, or to ``None`` where the value is unknown. A ``dict`` works,
#: as does any other mapping, a ``pandas.Series`` such as a row of a
#: ``DataFrame``, or any object with a mapping's ``get``. ``Any`` keys,
#: since ``Mapping`` is invariant in its key: ``Mapping[str, ...]``
#: would refuse the ``dict[Hashable, Any]`` of ``row.to_dict()``.
State: TypeAlias = Mapping[Any, object] | pd.Series | _Gettable

# Default for `state.get` in `Rule.matches`: tells a missing key from
# a key whose value is None (an unknown).
_ABSENT = object()
# pandas' missing-value markers, bound once for that same hot loop.
_NA = pd.NA
_NAT = pd.NaT


def _is_unknown(value: object) -> bool:
    """Tells whether a state value is unknown, as ``Dataset`` reads it.

    ``None``, ``NaN``, ``NA`` and ``NaT`` are tested first, since they
    are what a ``dict`` or a ``DataFrame`` row usually holds; any other
    value ``pandas.isna`` calls missing (numpy's ``NaT``,
    ``Decimal("NaN")``) is unknown too, as ``Dataset`` stores it.

    Args:
        value: One value of a state.

    Returns:
        True for every marker of a missing value.
    """
    if value is None or value is _NA or value is _NAT:
        return True
    if isinstance(value, str):
        return False
    if isinstance(value, float | np.floating):
        return bool(value != value)
    missing = pd.isna(cast(Any, value))  # an array for a list: not a marker
    return isinstance(missing, bool | np.bool_) and bool(missing)


def _absent_message(variable: str) -> str:
    """The ``KeyError`` message for a state missing a tested variable."""
    return (
        f"state has no entry for {variable!r}, which the rule "
        f"tests. Write {variable!r}: None for an unknown value."
    )


def _not_str_message(variable: str, observed: object) -> str:
    """The ``TypeError`` message for a known value that isn't a ``str``."""
    return (
        f"state value for {variable!r} is {observed!r} "
        f"({type(observed).__name__}), not a str: pylfit "
        f"compares values as strings. Convert it with str(), "
        f"or pass a DataFrame to predict, which converts for you."
    )


def _condition_problem(variable: object, value: object) -> str | None:
    """The ``TypeError`` message for a condition that isn't ``str`` to ``str``.

    A value that isn't a ``str`` would never match a state, whose values
    are ``str``.

    Args:
        variable: The variable the condition tests.
        value: The value it requires.

    Returns:
        The message, or ``None`` if both are ``str``.
    """
    if not isinstance(variable, str):
        return f"body variable {variable!r} ({type(variable).__name__}) is not a str"
    if not isinstance(value, str):
        return (
            f"body value for {variable!r} is {value!r} "
            f"({type(value).__name__}), not a str: pylfit compares "
            "values as strings. Convert it with str()."
        )
    return None


# --- from_string grammar ---
#
# "variable(value)" atoms joined by ":-", e.g.
# "y(1) :- a(0), c(1)." — also to_string's own output format, so
# from_string(rule.to_string()) round-trips. A variable is everything
# before an atom's first "(", and a value everything up to the atom's
# last ")", so names like "a_t+1" or "petal width" and values like
# pandas.cut's "(0, 40]" parse. Body atoms are split on ")," — the
# one sequence a value can't contain. Two atoms with the comma left out
# would read as one condition whose value holds ") ... (", and a doubled
# comma as a variable starting with ",": both are refused instead of
# guessed at (_MISSING_COMMA_RE). Only equality is expressed,
# matching Rule's invariant; other rule notations (CN2's "IF ... THEN",
# BoolNet-style infix, etc.) are deliberately not parsed here.
_BODY_SEPARATOR_RE = re.compile(r"\)\s*,")
# DOTALL: with the two atoms on separate lines, the comma is as missing.
_MISSING_COMMA_RE = re.compile(r"\).*\(", re.DOTALL)


class Rule:
    """A logic rule composed of a head (conclusion) and a body (conditions).

    A rule is immutable and hashable, like a ``frozenset``, so models
    can hold it in a ``set``. ``with_condition`` and
    ``without_condition`` return a new rule instead of changing this
    one. Rules sort by their text, so ``sorted(rules)`` lists them as
    ``to_string`` writes them.

    Attributes:
        head: The conclusion of the rule, as a ``(variable, value)``
            pair.
        body: The conditions of the rule, as a read-only mapping of
            variable to required value, sorted by variable. At most one
            condition per variable.

    Examples:
        >>> rule = Rule(head=("y", "1"), body={"a": "1"})
        >>> rule.to_string()
        'y(1) :- a(1).'
        >>> rule
        Rule(head=('y', '1'), body={'a': '1'})
    """

    # No __dict__: with read-only properties, nothing can be assigned.
    __slots__ = ("_body", "_hash", "_head")

    def __init__(
        self,
        head: tuple[str, str],
        body: Mapping[str, str] | None = None,
    ) -> None:
        """Initializes a rule.

        Args:
            head: The conclusion of the rule, as a ``(variable, value)``
                pair.
            body: The conditions of the rule, as a mapping of variable
                to required value. Copied, so changing it later doesn't
                change the rule. Defaults to an empty body.

        Raises:
            TypeError: If ``head`` is not a tuple of two ``str``, or
                ``body`` is not a mapping of ``str`` to ``str``.
        """
        if not (
            isinstance(head, tuple)
            and len(head) == 2
            and isinstance(head[0], str)
            and isinstance(head[1], str)
        ):
            raise TypeError(
                f"head is {head!r} ({type(head).__name__}), not a "
                "(variable, value) tuple of two str"
            )
        if body is not None and not isinstance(body, Mapping):
            raise TypeError(
                f"body is {body!r} ({type(body).__name__}), not a mapping "
                "of variable to value"
            )
        conditions = dict(body) if body is not None else {}
        for variable, value in conditions.items():
            problem = _condition_problem(variable, value)
            if problem is not None:
                raise TypeError(problem)
        # Plain str, so a numpy string (a str subclass) doesn't show as
        # np.str_(...) in repr.
        self._set(
            (str(head[0]), str(head[1])),
            {str(variable): str(value) for variable, value in conditions.items()},
        )

    @classmethod
    def _derived(cls, head: tuple[str, str], conditions: dict[str, str]) -> "Rule":
        """A rule from parts already checked, without checking them again.

        For ``with_condition`` and ``without_condition``, which GULA and
        PRIDE call in their inner loops; the checks in ``__init__`` cost
        them up to 12% of a fit on random tables of 6 to 8 variables.
        """
        rule = cls.__new__(cls)
        rule._set(head, conditions)
        return rule

    def _set(self, head: tuple[str, str], conditions: dict[str, str]) -> None:
        """Fills the slots from a checked head and a private dict."""
        self._head = head
        # Sorted by variable, so equal rules scan their conditions in
        # the same order in `matches` and raise on the same missing
        # variable.
        items = tuple(sorted(conditions.items()))
        # A view over a private copy: readable, not writable.
        self._body = MappingProxyType(dict(items))
        # Computed once: models and GULA hash the same rule many times.
        self._hash = hash((head, items))

    @property
    def head(self) -> tuple[str, str]:
        """The conclusion of the rule, as a ``(variable, value)`` pair."""
        return self._head

    @property
    def body(self) -> Mapping[str, str]:
        """The conditions of the rule, as a read-only mapping.

        Variable to required value, sorted by variable.
        ``dict(rule.body)`` gives a plain, writable copy.
        """
        return self._body

    def __reduce__(self) -> tuple[type["Rule"], tuple[tuple[str, str], dict[str, str]]]:
        """Pickles the rule as its head and body (worker processes need it).

        ``MappingProxyType`` can't be pickled, so the body travels as a
        plain ``dict`` and ``__init__`` rebuilds the rest.
        """
        return (type(self), (self._head, dict(self._body)))

    # --- Factory methods ---

    @classmethod
    def from_string(cls, text: str) -> "Rule":
        """Parses a rule from its Horn-clause string.

        The head and each condition are written ``variable(value)``,
        joined by ``:-`` and optionally ended with a period, for example
        ``"y(1) :- a(0), c(1)."``. A rule with an empty body is written
        ``"y(1) :- ."``. No other rule notation is accepted.

        This is the format ``to_string`` writes, so
        ``Rule.from_string(rule.to_string()) == rule`` for names such as
        ``a_t+1`` and values such as ``(0, 40]`` or ``[0, 40)``. It
        doesn't hold when:

        - a variable name is empty, contains ``(``, or a ``)`` followed
          by a ``,`` (spaces between them included), or starts with
          ``,``;
        - a value contains a ``)`` followed by a ``,`` (spaces between
          them included), or a ``)`` later followed by ``(``;
        - a name or value starts or ends with whitespace (a space, a
          tab: it is dropped);
        - the head's variable or value contains ``:-``.

        Args:
            text: The rule's Horn-clause string.

        Returns:
            The parsed ``Rule``.

        Raises:
            TypeError: If ``text`` is not a ``str``.
            ValueError: If ``text`` isn't valid Horn-clause syntax, has
                two conditions on the same variable, or looks like it is
                missing a comma between conditions or has one too many.

        Examples:
            >>> rule = Rule.from_string("y(1) :- a(0), c(1).")
            >>> rule.head, dict(rule.body)
            (('y', '1'), {'a': '0', 'c': '1'})
            >>> dict(Rule.from_string("sick(yes) :- age((40, 100]), a_t+1(0).").body)
            {'a_t+1': '0', 'age': '(40, 100]'}
        """
        if not isinstance(text, str):
            raise TypeError(f"text is {text!r} ({type(text).__name__}), not a str")
        stripped = text.strip()
        if not stripped:
            raise ValueError("cannot parse a rule from an empty string")
        stripped = stripped.removesuffix(".")
        head_str, sep, body_str = stripped.partition(":-")
        if not sep:
            raise ValueError(f"missing ':-' in rule: {text!r}")
        head = cls._parse_horn_atom(head_str)
        body: dict[str, str] = {}
        if body_str.strip():
            pieces = _BODY_SEPARATOR_RE.split(body_str)
            # The split eats each separator's ")": put it back on every
            # atom but the last, which keeps its own.
            atoms = [piece + ")" for piece in pieces[:-1]] + pieces[-1:]
            for atom in atoms:
                variable, value = cls._parse_horn_atom(atom)
                if variable in body:
                    raise ValueError(
                        f"two conditions on {variable!r} in rule: {text!r}"
                    )
                body[variable] = value
        return cls(head, body)

    @staticmethod
    def _parse_horn_atom(atom_str: str) -> tuple[str, str]:
        """Parses a single ``variable(value)`` atom.

        The variable is everything before the first ``(``, the value
        everything between it and the closing ``)``, both stripped.
        """
        atom = atom_str.strip()
        variable, paren, rest = atom.partition("(")
        variable = variable.strip()
        if not paren or not variable or not rest.endswith(")"):
            raise ValueError(f"malformed atom (expected 'variable(value)'): {atom!r}")
        value = rest[:-1].strip()
        if variable.startswith(","):
            raise ValueError(f"empty condition (two commas in a row?): {atom!r}")
        if _MISSING_COMMA_RE.search(value):
            raise ValueError(
                f"{atom!r} reads as one condition, {variable!r} = {value!r}: "
                "is a ',' missing between two conditions?"
            )
        return variable, value

    # --- Observers ---

    def size(self) -> int:
        """Returns the number of conditions in the body.

        Returns:
            The number of conditions in the rule's body.
        """
        return len(self._body)

    def has_condition(self, variable: str) -> bool:
        """Checks whether the rule has a condition on a variable.

        Args:
            variable: The variable to check.

        Returns:
            True if the rule has a condition on ``variable``, False
            otherwise.

        Examples:
            >>> rule = Rule.from_string("y(1) :- a(1).")
            >>> rule.has_condition("a"), rule.has_condition("b")
            (True, False)
        """
        return variable in self._body

    def matches(self, state: State) -> bool:
        """Checks whether the rule's conditions all hold in a given state.

        Values are compared as strings, as ``Dataset`` stores them. An
        unknown value, written ``None`` (or ``NaN``, as a ``DataFrame``
        row holds it), never matches a condition. A variable the rule
        tests must be present in ``state``, even when its value is
        unknown: a missing key is far more often a typo or a forgotten
        column than an intended unknown, so it raises. A value that is
        not a ``str`` raises too, rather than silently not matching. The
        conditions are checked in variable order and the scan stops at
        the first that fails, so a wrong value after that one goes
        unnoticed.

        Args:
            state: A mapping of variable to ``str`` value describing a
                state of the system, ``None`` for an unknown value. Any
                object with a ``.get`` method works, including a
                ``dict`` or a ``pandas.Series`` (for example, a row of a
                DataFrame read with ``dtype=str``).

        Returns:
            True if every condition of the rule is satisfied by
            ``state``, False otherwise.

        Raises:
            KeyError: If ``state`` has no entry for a variable one of
                the rule's conditions tests (conditions are checked in
                variable order, and the scan stops at the first that
                fails).
            TypeError: If a value ``state`` holds for one of the rule's
                conditions is neither a ``str`` nor unknown (checked in
                the same order, with the same stop).

        Examples:
            >>> rule = Rule.from_string("y(1) :- a(1).")
            >>> rule.matches({"a": "1", "b": "0"})
            True
            >>> rule.matches({"a": "0", "b": "0"})
            False
            >>> rule.matches({"a": None, "b": "0"})
            False
            >>> rule.matches({"b": "0"})  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            KeyError: "state has no entry for 'a', which the rule tests. ..."
            >>> rule.matches({"a": 1})  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            TypeError: state value for 'a' is 1 (int), not a str: ...
        """
        for variable, value in self._body.items():
            observed = state.get(variable, _ABSENT)
            # The type comes first: comparing pandas' NA gives NA, and a
            # numpy array gives an array, neither of them an answer.
            if isinstance(observed, str):
                if observed != value:
                    return False
            elif observed is _ABSENT:
                raise KeyError(_absent_message(variable))
            elif _is_unknown(observed):
                return False
            else:
                raise TypeError(_not_str_message(variable, observed))
        return True

    def is_more_general_than(self, other: "Rule") -> bool:
        """Checks whether this rule is more general than another rule.

        A rule is more general than another with the same head when its
        conditions are a subset of the other's: it matches every state
        the other matches. The rule-learning literature calls this
        subsumption. Unlike subsumption, the test is strict, as ``<``
        between sets: no rule is more general than itself, or than a
        rule equal to it.

        Args:
            other: The rule to compare against.

        Returns:
            True if this rule is more general than ``other``, False
            otherwise. Always False if the two rules have different
            heads.

        Raises:
            TypeError: If ``other`` is not a ``Rule``.

        Examples:
            >>> general = Rule.from_string("y(1) :- a(1).")
            >>> specific = Rule.from_string("y(1) :- a(1), b(0).")
            >>> general.is_more_general_than(specific)
            True
            >>> general.is_more_general_than(general)
            False
        """
        # A subset test on the items views runs in C; GULA's domination
        # check calls this millions of times, so the type is checked
        # only once the comparison has failed.
        try:
            return (
                self._head == other._head and self._body.items() < other._body.items()
            )
        except AttributeError:  # a Rule always has both: other is not one
            raise TypeError(
                f"other is {other!r} ({type(other).__name__}), not a Rule"
            ) from None

    # --- Derived rules ---

    def with_condition(self, variable: str, value: str) -> "Rule":
        """Returns this rule with a condition added or replaced.

        Args:
            variable: The variable the condition applies to.
            value: The value required on ``variable``.

        Returns:
            A new rule with the same head, whose body also requires
            ``value`` on ``variable``.

        Raises:
            TypeError: If ``variable`` or ``value`` is not a ``str``.

        Examples:
            >>> rule = Rule.from_string("y(1) :- a(1).")
            >>> print(rule.with_condition("b", "0"))
            y(1) :- a(1), b(0).
            >>> print(rule)
            y(1) :- a(1).
        """
        problem = _condition_problem(variable, value)
        if problem is not None:
            raise TypeError(problem)
        return type(self)._derived(
            self._head, {**self._body, str(variable): str(value)}
        )

    def without_condition(self, variable: str) -> "Rule":
        """Returns this rule without its condition on a variable, if any.

        Args:
            variable: The variable whose condition is left out.

        Returns:
            A new rule with the same head and every other condition; an
            equal rule if there was no condition on ``variable``.

        Examples:
            >>> print(Rule.from_string("y(1) :- a(1), b(0).").without_condition("a"))
            y(1) :- b(0).
        """
        return type(self)._derived(
            self._head,
            {other: value for other, value in self._body.items() if other != variable},
        )

    # --- String representation ---

    def to_string(self) -> str:
        """Converts the rule to a readable string representation.

        Returns:
            A string of the form
            ``"target(value) :- var_1(val_1), ..."``, with the
            conditions sorted by variable. A rule with an empty body is
            written ``"target(value) :- ."``.

        Examples:
            >>> Rule(head=("y", "1"), body={"a": "0", "c": "1"}).to_string()
            'y(1) :- a(0), c(1).'
        """
        head_str = f"{self._head[0]}({self._head[1]})"
        # `_body` is stored sorted by variable.
        conditions = [f"{variable}({value})" for variable, value in self._body.items()]
        return f"{head_str} :- {', '.join(conditions)}."

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        """The constructor call that rebuilds the rule; ``str`` reads better.

        Conditions sorted by variable, as in ``to_string``.
        """
        return f"Rule(head={self._head!r}, body={dict(self._body)!r})"

    # --- Comparison operators ---

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rule):
            # Lets the other operand decide, as Python's data model
            # asks; `==` still ends up False for an unrelated type.
            return NotImplemented
        return self._head == other._head and self._body == other._body

    def __hash__(self) -> int:
        return self._hash

    def _sort_key(self) -> tuple[str, tuple[str, str], tuple[tuple[str, str], ...]]:
        """Orders by text, then by head and body.

        Head and body tell apart two rules that share a text, as a
        variable named with ``", "`` and ``"("`` can make them.

        Returns:
            The key ``sorted`` compares.
        """
        return (self.to_string(), self._head, tuple(self._body.items()))

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Rule):
            return NotImplemented
        return self._sort_key() < other._sort_key()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Rule):
            return NotImplemented
        return self._sort_key() <= other._sort_key()

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Rule):
            return NotImplemented
        return self._sort_key() > other._sort_key()

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Rule):
            return NotImplemented
        return self._sort_key() >= other._sort_key()
