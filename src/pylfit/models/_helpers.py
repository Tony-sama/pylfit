"""Helper functions shared by pylfit's model classes.

Plain functions, not methods: ``RuleSetModel`` and, by inheritance,
``WeightedRuleSetModel`` call them with their own ``features``,
``targets`` and ``domain``. ``pylfit_version`` also gives
``pylfit.__version__``.
"""

from collections.abc import Iterable, Mapping
from importlib import metadata
from typing import Literal, cast, get_args

import pandas as pd

from pylfit._dataset import Dataset
from pylfit._rule import (
    _ABSENT,
    Rule,
    State,
    _absent_message,
    _is_unknown,
    _not_str_message,
)


def pylfit_version() -> str:
    """Returns the installed ``pylfit`` version, or ``"unknown"``.

    ``pylfit.__version__`` and the ``pylfit_version`` a saved model
    records both come from here. A source checkout with no installed
    distribution metadata gets the placeholder rather than an error,
    so importing pylfit and ``save`` still work; ``from_file`` never
    checks the recorded version.
    """
    try:
        return metadata.version("pylfit")
    except metadata.PackageNotFoundError:
        return "unknown"


def check_columns(
    dataframe: pd.DataFrame, features: Iterable[str], targets: Iterable[str]
) -> None:
    """Validates that ``dataframe`` has exactly ``features`` as its columns.

    Four independent problems are checked for, and — unlike a
    fail-on-first-problem check — all that apply are reported
    together in one message:

    - A feature column missing from ``dataframe``.
    - A column present in ``dataframe`` that is one of ``targets``
      rather than a feature — called out separately, since it's the
      mistake most likely to be a genuine leftover-target bug rather
      than a typo.
    - Any other column present in ``dataframe`` that is neither a
      feature nor a target.
    - A column name given more than once, whose copies would go
      unchecked.

    A column whose label is not a ``str`` but reads as a missing
    feature, such as the ``int`` ``0`` for the feature ``"0"``, is
    named as such in the message.

    Args:
        dataframe: The dataframe to check.
        features: The feature columns ``dataframe`` must have, exactly.
        targets: The model's target variables — used only to give a
            more specific message when one leaks into the input.

    Raises:
        ValueError: If any of the four problems above apply, with
            every applicable one listed by column name (sorted for a
            stable message) in a single combined message.
    """
    columns: set[object] = set(dataframe.columns)
    features = set(features)
    targets = set(targets)

    # Sorted by their text: a frame's labels may mix types, which don't
    # sort together.
    missing_features = sorted(map(str, features - columns))
    extra_columns = columns - features
    leaked_targets = sorted(map(str, extra_columns & targets))
    unexpected_columns = sorted(map(str, extra_columns - targets))
    repeated = sorted(map(str, set(dataframe.columns[dataframe.columns.duplicated()])))
    mistyped = sorted(
        (str(label), type(label).__name__)
        for label in extra_columns
        if not isinstance(label, str) and str(label) in features - columns
    )

    if not (missing_features or leaked_targets or unexpected_columns or repeated):
        return

    problems: list[str] = []
    if repeated:
        problems.append("column(s) given more than once: " + ", ".join(repeated))
    if missing_features:
        problems.append(
            "missing required feature column(s): " + ", ".join(missing_features)
        )
    if leaked_targets:
        problems.append(
            "target column(s) must not be included in prediction input, "
            "since the model predicts these rather than reading them: "
            + ", ".join(leaked_targets)
        )
    if unexpected_columns:
        problems.append(
            "unexpected column(s), not part of the trained features: "
            + ", ".join(unexpected_columns)
        )
    if mistyped:
        problems.append(
            "column label(s) "
            + ", ".join(f"{text} ({kind})" for text, kind in mistyped)
            + " name a feature but are not str; convert the labels with "
            "dataframe.columns.astype(str)"
        )
    raise ValueError("; ".join(problems))


#: How a prediction treats a state with unknown feature values; see
#: ``RuleSetModel.predict_candidates``.
Unknowns = Literal["fallback", "strict"]
_UNKNOWNS: tuple[str, ...] = get_args(Unknowns)


def check_unknowns(unknowns: object) -> None:
    """Rejects an ``unknowns`` argument that is not one of the known modes.

    Args:
        unknowns: A prediction method's argument of the same name.

    Raises:
        ValueError: If ``unknowns`` is not ``"fallback"`` or
            ``"strict"``.
    """
    if unknowns not in _UNKNOWNS:
        raise ValueError(f"unknowns must be one of {_UNKNOWNS!r}, got {unknowns!r}")


def match_weight(
    rule: Rule,
    state: State,
    frequencies: Mapping[str, Mapping[str, float]] | None = None,
) -> float:
    """Returns the probability that a rule matches a partly unknown state.

    Each condition on a known value either holds or rules the match
    out. Each condition on an unknown value holds with the share its
    value has among the training rows' known values of that feature,
    the features taken as independent. Without ``frequencies``, or for
    a feature absent from it, a condition on an unknown weighs ``1``:
    the match is merely possible.

    Args:
        rule: The rule to weigh.
        state: A feature state, ``None`` (or ``NaN``) where unknown.
        frequencies: Per feature, each value's share among the known
            values of the training rows.

    Returns:
        ``1.0`` when the rule matches the known values outright, ``0.0``
        when a known value contradicts it, and the product of its
        unknown conditions' weights otherwise.

    Raises:
        KeyError: If ``state`` lacks a variable the rule tests, as in
            ``Rule.matches``.
        TypeError: If a known value the rule tests is not a ``str``, as
            in ``Rule.matches``.
    """
    weight = 1.0
    for variable, value in rule.body.items():
        observed = state.get(variable, _ABSENT)
        if observed is _ABSENT:
            raise KeyError(_absent_message(variable))
        if _is_unknown(observed):
            if frequencies is not None and variable in frequencies:
                weight *= frequencies[variable].get(value, 0.0)
            continue
        if observed != value:
            if not isinstance(observed, str):
                raise TypeError(_not_str_message(variable, observed))
            return 0.0
    return weight


#: How to fix an out-of-domain value, appended to its error.
_WIDEN_DOMAIN = (
    ". If the value is valid, give the model every value first, for "
    "example model.domain = full_dataset.domains(), with full_dataset "
    "built from all your data"
)


def check_domains(
    dataframe: pd.DataFrame,
    features: Iterable[str],
    domain: Mapping[str, set[str]],
) -> None:
    """Validates that every feature value is in the model's domain.

    Only meaningful once ``check_columns`` has already confirmed the
    column set is exactly ``features`` — this function assumes that
    and doesn't re-check it. Values are compared as ``str``:
    ``dataframe_states`` passes the frame ``Dataset._as_str`` made,
    the same values the rules are then matched against.

    A value outside the domain is always a hard failure, with no
    opt-out: it can never satisfy a condition on its feature, so
    letting it through would only turn a clear, upfront error into a
    prediction silently made without every rule that tests that
    feature. A missing value is unknown, not out of the domain, and is
    not checked.

    Args:
        dataframe: The dataframe to check, already confirmed by
            ``check_columns`` to have exactly ``features`` as its
            columns.
        features: The feature columns to check, in the order to report
            them in.
        domain: Per variable, the values the model allows — only the
            entries for ``features`` are consulted.

    Raises:
        ValueError: If any feature column contains a value outside
            ``domain[feature]``, with every offending feature listed
            alongside its offending values (sorted, deduplicated), all
            in a single combined message.
    """
    problems: list[str] = []
    for feature in features:
        allowed = domain[feature]
        values = set(dataframe[feature].dropna().astype(str))
        out_of_domain = sorted(values - allowed)
        if out_of_domain:
            problems.append(
                f"out-of-domain value(s) for feature {feature!r}: "
                + ", ".join(out_of_domain)
            )
    if problems:
        raise ValueError("; ".join(problems) + _WIDEN_DOMAIN)


def check_state(
    state: State,
    features: Iterable[str],
    domain: Mapping[str, set[str]],
) -> None:
    """Validates one state given as a mapping, as ``dataframe_states`` a row.

    The single-state forms of the prediction methods go through it, so
    a ``dict``'s values are held to the same checks as a ``DataFrame``
    row's: every feature present, every known value a ``str`` in its
    feature's domain. An unknown value (``None``, ``NaN``) is not
    checked. Unlike a ``DataFrame``, which may hold only feature
    columns, a mapping's keys that are not features are ignored, so a
    full row with its target columns is accepted.

    Args:
        state: A mapping of feature to value, such as a ``dict`` or a
            ``pandas.Series``.
        features: The model's feature variables, in the order to report
            them in.
        domain: Per variable, the values the model allows.

    Raises:
        KeyError: If ``state`` lacks a feature, with every missing one
            named.
        TypeError: If ``state`` is a ``DataFrame`` or not a mapping, or
            a known value is not a ``str``.
        ValueError: If a known value is outside its feature's domain,
            with every offending feature listed in a single combined
            message.
    """
    if isinstance(state, pd.DataFrame):
        raise TypeError(
            "a state is one mapping of feature to value, got a DataFrame; "
            "predict and predict_proba take a DataFrame of states"
        )
    if not callable(getattr(state, "get", None)):
        raise TypeError(
            f"a state is a mapping of feature to value, such as a dict, got "
            f"{type(state).__name__}; for several states, pass a DataFrame "
            "to predict or predict_proba"
        )
    missing: list[str] = []
    out_of_domain: list[str] = []
    for feature in features:
        observed = state.get(feature, _ABSENT)
        if observed is _ABSENT:
            missing.append(feature)
        elif _is_unknown(observed):
            continue
        elif not isinstance(observed, str):
            raise TypeError(_not_str_message(feature, observed))
        elif observed not in domain[feature]:
            out_of_domain.append(
                f"out-of-domain value(s) for feature {feature!r}: {observed}"
            )
    if missing:
        raise KeyError(
            "state has no entry for feature(s) "
            + ", ".join(repr(feature) for feature in missing)
            + ". Write "
            + ", ".join(f"{feature!r}: None" for feature in missing)
            + " for an unknown value."
        )
    if out_of_domain:
        raise ValueError("; ".join(out_of_domain) + _WIDEN_DOMAIN)


def dataframe_states(
    dataframe: pd.DataFrame,
    features: list[str],
    targets: Iterable[str],
    domain: Mapping[str, set[str]],
) -> list[dict[str, str | None]]:
    """Validates prediction input and converts it to one state per row.

    The one conversion every model's ``DataFrame`` entry point goes
    through, so all of them accept and reject the same input:
    ``check_columns``, the float refusal of ``Dataset.check_values``,
    then ``check_domains`` on the values as ``str``. The count warning
    of ``check_values`` is left out: the domain check already refuses
    every value the model never saw.

    Args:
        dataframe: One row per state, with exactly ``features`` as
            columns.
        features: The model's feature variables, in column order.
        targets: The model's target variables, refused as columns.
        domain: Per variable, the values the model allows.

    Returns:
        One ``dict`` of feature to ``str`` value per row, in row order,
        with ``None`` for a missing value.

    Raises:
        ValueError: If ``dataframe`` is missing a feature column, has a
            target column, any other extra column or a repeated one; if
            a column is complex, a date or a duration, or holds a value
            of such a type, ``bytes`` or a ``Decimal``, or a float that
            is not a whole number; or if a value is outside its
            feature's domain.
    """
    check_columns(dataframe, features, targets)
    Dataset._check_floats(dataframe[features], before="predicting")
    frame = Dataset._as_str(dataframe[features])
    check_domains(frame, features, domain)
    frame = frame.astype(object).where(frame.notna(), None)
    return cast(list[dict[str, str | None]], frame.to_dict("records"))


def validate_domain(
    features: Iterable[str],
    targets: Iterable[str],
    value: Mapping[str, Iterable[str]],
) -> dict[str, set[str]]:
    """Validates a candidate domain mapping and normalizes it to sets.

    Shared by every concrete model whose ``domain`` property must cover
    exactly ``features + targets``, such as the setter of
    ``RuleSetModel.domain``.

    A ``str`` where the values should be is refused rather than split
    into its characters, and so is a value that is not a ``str``: rules
    and states are compared as strings, so an ``int`` value would make
    every matching value look out of the domain.

    Args:
        features: The model's feature variables.
        targets: The model's target variables.
        value: A mapping from variable name to an iterable of allowed
            ``str`` values. Must have exactly one entry per variable in
            ``features`` and ``targets`` combined — no missing, no
            extra variable.

    Returns:
        A new mapping with every value normalized to a ``set``,
        regardless of what iterable was handed in for it.

    Raises:
        ValueError: If ``value``'s keys don't exactly match ``features
            + targets``, with every missing and/or extra variable
            named in the message, or a variable's values are a single
            ``str``, as every other collection argument refuses one.
        TypeError: If ``value`` is not a mapping, or a variable's values
            include a value that is not a ``str``, with every offending
            variable named in a single combined message.
    """
    if not isinstance(value, Mapping):
        raise TypeError(
            f"domain must be a mapping of variable to values, got "
            f"{type(value).__name__}"
        )
    expected = set(features) | set(targets)
    given = set(value.keys())
    # Sorted by their text, shown as repr: an extra key may be any type.
    missing = sorted(map(repr, expected - given))
    extra = sorted(map(repr, given - expected))
    if missing or extra:
        problems: list[str] = []
        if missing:
            problems.append("missing variable(s): " + ", ".join(missing))
        if extra:
            problems.append("unexpected variable(s): " + ", ".join(extra))
        raise ValueError(
            "domain must have exactly one entry per feature/target; "
            + "; ".join(problems)
        )
    single = [
        f"{variable!r} is given the str {values!r}, not a collection of "
        f"values; write [{values!r}] for a single value"
        for variable, values in value.items()
        if isinstance(values, str)
    ]
    if single:
        raise ValueError("domain values must be collections; " + "; ".join(single))
    domain: dict[str, set[str]] = {}
    type_problems: list[str] = []
    for variable, values in value.items():
        domain[variable] = set(values)
        not_str = sorted(repr(v) for v in domain[variable] if not isinstance(v, str))
        if not_str:
            type_problems.append(
                f"{variable!r} has non-str value(s) {', '.join(not_str)}; "
                f"pylfit compares values as strings, convert them with str()"
            )
    if type_problems:
        raise TypeError("domain values must be str; " + "; ".join(type_problems))
    return domain


def check_rules(
    rules: Iterable[Rule],
    features: Iterable[str],
    targets: Iterable[str],
    domain: Mapping[str, set[str]],
) -> None:
    """Validates that every rule uses only the model's variables and values.

    A rule whose head is not a target, whose body names a non-feature,
    or that uses a value outside its variable's domain could never match
    or never be predicted: it would sit in the model doing nothing,
    usually because of a misspelling.

    Args:
        rules: The rules to check.
        features: The model's feature variables.
        targets: The model's target variables.
        domain: Per variable, the values the model allows.

    Raises:
        ValueError: If any rule names an unknown variable or value, with
            every offending rule listed in a single combined message.
    """
    features = set(features)
    targets = set(targets)
    problems: list[tuple[Rule, str]] = []
    for rule in rules:
        target, value = rule.head
        if target not in targets:
            problems.append((rule, f"{rule}: head {target!r} is not a target"))
        elif value not in domain[target]:
            problems.append((rule, f"{rule}: {value!r} is not a value of {target!r}"))
        for feature, condition in sorted(rule.body.items()):
            if feature not in features:
                problems.append((rule, f"{rule}: {feature!r} is not a feature"))
            elif condition not in domain[feature]:
                problems.append(
                    (rule, f"{rule}: {condition!r} is not a value of {feature!r}")
                )
    if problems:
        # By rule, as `rules` is often a set, whose order varies between
        # runs; a stable sort keeps each rule's own problems in order.
        problems.sort(key=lambda problem: problem[0]._sort_key())
        raise ValueError(
            "rule(s) outside the model: "
            + "; ".join(message for _, message in problems)
        )
