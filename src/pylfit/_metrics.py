"""Metrics for pylfit: of predictions, and of rule sets.

The prediction metrics are plain functions over predictions you have
already computed, in the ``(y_true, y_pred)`` shape of
``sklearn.metrics``: split off the target columns, call the model's
``predict`` on the features, then pass both here. Of the rule-set
metrics, ``coverage`` takes rules and the ``Dataset`` to measure them
on, and ``rule_recovery`` compares learned rules with known true ones.
"""

from collections.abc import Iterable, Iterator, Mapping, Sequence, Sized
from statistics import fmean
from typing import Any, cast

import pandas as pd

from pylfit._checks import _check_collection
from pylfit._dataset import Dataset, _as_text, _check_dataset
from pylfit._rule import Rule, _is_unknown


def accuracy_score(
    y_true: Sequence[Mapping[Any, object]] | pd.DataFrame,
    y_pred: Sequence[Mapping[str, object]] | pd.DataFrame,
    target: str,
) -> float:
    """Computes the accuracy of one target across paired rows.

    A ``None`` prediction counts as incorrect; it is not left out of the
    total. A row whose actual value is unknown (``NaN`` or ``None``) is
    left out: there is nothing to compare with, and ``Dataset`` leaves
    such a row out of that target's positives and negatives too.

    Actual and predicted values are compared as strings, the way
    ``Dataset`` stores them, so an integer column scores the same as its
    text, and so does a whole float such as ``1.0``: pandas stores an
    integer column with a missing value as floats. Predictions from
    another classifier, as integers, score as pylfit's own would.

    Args:
        y_true: The actual values: a ``DataFrame`` with a ``target``
            column, such as ``test_df[targets]``, or one mapping per row
            holding at least ``target``, such as
            ``test_df[targets].to_dict("records")``. Values may be of
            any type ``Dataset`` accepts.
        y_pred: One mapping per row, in the order of ``y_true``, as
            ``RuleSetModel.predict`` returns: target mapped to its
            predicted value, or ``None``. A ``DataFrame`` of the
            predictions, such as ``pd.DataFrame(model.predict(...))``,
            works too. Values are converted as those of ``y_true`` are,
            and an unknown one counts as incorrect, like ``None``.
        target: The target to score.

    Returns:
        The fraction of rows with a known actual value where the
        predicted value equals it.

    Raises:
        ValueError: If ``y_true`` and ``y_pred`` have different lengths,
            either is empty, either is a ``DataFrame`` without a
            ``target`` column, or no row's actual value is known.
        KeyError: If a row of ``y_true`` or ``y_pred`` lacks ``target``.
        TypeError: If ``y_true`` or ``y_pred`` is neither a
            ``DataFrame`` nor a sequence of rows, or a row is not a
            mapping, such as a label where ``{target: label}`` belongs.

    Examples:
        >>> y_true = [{"y": "1"}, {"y": "0"}]
        >>> y_pred = [{"y": "1"}, {"y": "1"}]
        >>> accuracy_score(y_true, y_pred, "y")
        0.5

        The actual values can stay in their ``DataFrame``; other columns
        are ignored:

        >>> import pandas as pd
        >>> test = pd.DataFrame({"a": ["0", "1"], "y": [1, 0]})
        >>> accuracy_score(test, y_pred, "y")
        0.5
    """
    _check_rows(y_true, "y_true")
    _check_rows(y_pred, "y_pred")
    if isinstance(y_true, pd.DataFrame):
        y_true = _records_as_text(y_true, [target], "y_true")
    pred_rows = _rows(y_pred, [target])
    if len(y_true) != len(pred_rows):
        raise ValueError(
            "y_true and y_pred must be the same length, got "
            f"{len(y_true)} and {len(pred_rows)}"
        )
    if len(y_true) == 0:
        raise ValueError("y_true/y_pred must be non-empty")

    correct = known = 0
    for row, (actual, predicted) in enumerate(zip(y_true, pred_rows, strict=True)):
        for name, mapping in (("y_true", actual), ("y_pred", predicted)):
            if not isinstance(mapping, Mapping | pd.Series):
                raise TypeError(
                    f"{name} row {row} is {mapping!r}, not a mapping of "
                    f"target to value: pass rows such as {{{target!r}: ...}}"
                )
            if target not in mapping:
                raise KeyError(f"{name} row {row} has no {target!r}")
        value = actual[target]
        if _is_unknown(value):
            continue
        known += 1
        guess = predicted[target]
        correct += not _is_unknown(guess) and _as_text(guess) == _as_text(value)
    if not known:
        raise ValueError(f"no row of y_true has a known value for {target!r}")
    return correct / known


def _check_rows(rows: object, name: str) -> None:
    """Refuses a ``y_true`` or ``y_pred`` that is not a table of rows.

    Args:
        rows: The argument passed.
        name: ``"y_true"`` or ``"y_pred"``, for the error message.

    Raises:
        TypeError: If ``rows`` is a ``str``, a one-pass iterator, or has
            no length. A ``Series`` of labels gets through, for the row
            check to name its first label.
    """
    if isinstance(rows, pd.DataFrame):
        return
    if isinstance(rows, str | bytes | Iterator) or not isinstance(rows, Sized):
        raise TypeError(
            f"{name} must be a DataFrame or a list of one mapping per row, "
            f"such as [{{'y': '1'}}], got {type(rows).__name__}"
        )


def _rows(
    y_pred: Sequence[Mapping[str, object]] | pd.DataFrame,
    targets: Sequence[str],
) -> Sequence[Mapping[str, object]]:
    """Gives the predictions as one mapping per row, from either form.

    Args:
        y_pred: The predictions, as passed to ``accuracy_score``.
        targets: The targets scored.

    Returns:
        ``y_pred`` itself, or a ``DataFrame``'s ``targets`` columns as
        rows of text, as for ``y_true``.
    """
    if isinstance(y_pred, pd.DataFrame):
        return _records_as_text(y_pred, targets, "y_pred")
    return y_pred


def _records_as_text(
    frame: pd.DataFrame, targets: Sequence[str], name: str
) -> list[dict[Any, object]]:
    """Turns the target columns of a ``DataFrame`` into rows of text.

    Each column is converted as ``Dataset`` converts it, so its type
    decides how a whole float reads.

    Args:
        frame: The actual or predicted values, as passed as ``y_true``
            or ``y_pred``.
        targets: The targets to keep, at least one.
        name: ``"y_true"`` or ``"y_pred"``, for the error message.

    Returns:
        One mapping per row, of target to ``str`` value or ``NaN``.

    Raises:
        ValueError: If ``frame`` lacks a column of ``targets``.
    """
    # dict.fromkeys: a target listed twice would select its column
    # twice.
    columns = list(dict.fromkeys(targets))
    missing = [target for target in columns if target not in frame.columns]
    if missing:
        raise ValueError(f"{name} has no column {missing[0]!r}")
    converted = frame[columns].apply(_column_as_text)
    records: list[dict[Any, object]] = converted.to_dict("records")
    return records


def _column_as_text(column: pd.Series) -> pd.Series:
    """Converts one target column to ``str`` as ``Dataset`` does.

    Value by value, so a float ``Dataset`` would refuse (a fraction, or
    one too large for an integer) is written with ``str`` instead of
    failing here.

    Args:
        column: A target column of ``y_true``.

    Returns:
        The column as ``str``, ``NaN`` where a value is missing.
    """
    known = column.notna()
    return column.astype(object).map(_as_text, na_action="ignore").where(known)


def accuracy_by_target(
    y_true: Sequence[Mapping[Any, object]] | pd.DataFrame,
    y_pred: Sequence[Mapping[str, object]] | pd.DataFrame,
    targets: Sequence[str],
) -> dict[str, float]:
    """Computes the accuracy of every target in ``targets``.

    One score per target, since the targets of a multi-target model are
    predicted independently.

    Args:
        y_true: As in ``accuracy_score``.
        y_pred: As in ``accuracy_score``.
        targets: The targets to score, for example ``model.targets``.

    Returns:
        A mapping of target name to its accuracy.

    Raises:
        ValueError: As in ``accuracy_score``.
        KeyError: As in ``accuracy_score``.
        TypeError: If ``targets`` is a single string or a one-pass
            iterator, which the
            conversion of a ``DataFrame`` would use up, or as in
            ``accuracy_score``.

    Examples:
        >>> y_true = [{"y": "1", "z": "0"}, {"y": "0", "z": "0"}]
        >>> y_pred = [{"y": "1", "z": "0"}, {"y": "1", "z": None}]
        >>> accuracy_by_target(y_true, y_pred, ["y", "z"])
        {'y': 0.5, 'z': 0.5}
    """
    _check_targets(targets)
    _check_rows(y_true, "y_true")
    _check_rows(y_pred, "y_pred")
    if isinstance(y_true, pd.DataFrame):
        # Once, not once per target.
        y_true = _records_as_text(y_true, targets, "y_true")
    pred_rows = _rows(y_pred, targets)
    return {target: accuracy_score(y_true, pred_rows, target) for target in targets}


def _check_targets(targets: object) -> None:
    """Refuses an ``accuracy_by_target`` ``targets`` that would score wrong.

    Args:
        targets: The argument passed.

    Raises:
        TypeError: If ``targets`` is a single string, which would be
            read as one target per character, or a one-pass iterator,
            which converting
            a ``DataFrame`` would use up before any target is scored.
    """
    _check_collection("targets", targets)
    if isinstance(targets, Iterator):
        raise TypeError(
            "targets must be a collection, such as ['y'], not a one-pass "
            "iterator; pass list(...) of it"
        )


def coverage(dataset: Dataset, rules: Iterable[Rule]) -> dict[tuple[str, str], float]:
    """Computes, per head, the share of its positive rows some rule matches.

    A row counts for a head ``(target, value)`` when its target holds
    ``value``, and is covered when a rule concluding that head matches
    it. A rule set learned without restrictions from complete data
    covers every row. ``min_support``, ``allowed_conditions`` and
    unknown values can leave rows uncovered: a rule never matches a row
    whose value is unknown for a feature it tests. Use it to see what a
    higher ``min_support`` costs.

    Args:
        dataset: The dataset whose rows are counted. It comes first, as
            in a model's constructor.
        rules: The rules to measure, for example ``model.rules`` or an
            algorithm's output.

    Returns:
        Each head of ``dataset``, every value of every target, mapped to
        its covered share, from ``0.0`` to ``1.0``. A head no rule
        concludes scores ``0.0``. A rule whose head value ``dataset``
        never holds, such as one learned on other rows, counts for no
        head.

    Raises:
        TypeError: If ``dataset`` is not a ``Dataset``, ``rules`` is a
            single ``str``, or an element of ``rules`` is not a
            ``Rule``.
        ValueError: If a rule concludes on a variable that is not a
            target of ``dataset``, or tests one that is not a feature of
            it: the rules were learned on another dataset.

    Examples:
        >>> import pandas as pd
        >>> from pylfit import Dataset, GULA
        >>> df = pd.DataFrame({"a": ["0", "0", "0", "1"], "y": ["0", "0", "0", "1"]})
        >>> dataset = Dataset(df, features=["a"], targets=["y"])
        >>> coverage(dataset, GULA.fit(dataset, min_support=2))
        {('y', '0'): 1.0, ('y', '1'): 0.0}
    """
    _check_dataset(dataset)
    targets = set(dataset.targets)
    features = set(dataset.features)
    by_head: dict[tuple[str, str], list[Rule]] = {}
    for rule in _rule_list(rules, "rules"):
        if rule.head[0] not in targets:
            raise ValueError(
                f"rule {rule} concludes on {rule.head[0]!r}, which is not "
                "a target of the dataset"
            )
        unknown = sorted(set(rule.body) - features)
        if unknown:
            raise ValueError(
                f"rule {rule} tests {unknown[0]!r}, which is not a feature "
                "of the dataset"
            )
        by_head.setdefault(rule.head, []).append(rule)
    shares = {}
    for target in dataset.targets:
        for value in dataset.domain(target):
            positives = dataset._positive_states(target, value)
            head_rules = by_head.get((target, value), [])
            covered = sum(
                1
                for state in positives
                if any(rule.matches(state) for rule in head_rules)
            )
            shares[(target, value)] = covered / len(positives)
    return shares


def _rule_list(rules: Iterable[Rule], name: str) -> list[Rule]:
    """Checks that ``rules`` is a collection of ``Rule`` and lists it.

    Args:
        rules: The argument passed.
        name: Its name, for the error message.

    Returns:
        The rules, in the order given.

    Raises:
        TypeError: If ``rules`` is a single ``str``, or an element of it
            is not a ``Rule``.
    """
    given: object = rules  # checked as the caller may pass it
    if isinstance(given, str):
        raise TypeError(
            f"{name} must be a collection of Rule, not a str; read a rule "
            "with Rule.from_string and pass [rule]"
        )
    listed = []
    # As objects: the hint says Rule, but a list of rule strings is a
    # likely mistake.
    for rule in cast(Iterable[object], rules):
        if not isinstance(rule, Rule):
            hint = "; read it with Rule.from_string" if isinstance(rule, str) else ""
            raise TypeError(
                f"{name} holds {rule!r} ({type(rule).__name__}), not a Rule{hint}"
            )
        listed.append(rule)
    return listed


#: The relations a learned rule can have to a true rule of its head,
#: best first. A rule takes the best it has with any rule on the other
#: side; with none, a learned rule is ``"spurious"`` and a true rule
#: ``"missing"``.
_RELATIONS = ("exact", "too_specific", "too_general", "partial")


def rule_recovery(
    true_rules: Iterable[Rule],
    learned_rules: Iterable[Rule],
    states: Sequence[Mapping[str, object]] | pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Measures how well learned rules recover a known set of true rules.

    Each rule is compared with the rules of the same head on the other
    side, and takes the best relation it has with any of them:

    - ``"exact"``: the same rule.
    - ``"too_specific"``: a true rule is more general than the learned
      one. The learned rule is correct, but matches fewer states.
    - ``"too_general"``: the learned rule is more general than a true
      one. It is wrong: a true rule is minimal, so anything more general
      matches a state where the head value does not hold.
    - ``"partial"``: neither is more general, but some state matches
      both: no feature has a different value in each.
    - ``"spurious"``, for a learned rule, or ``"missing"``, for a true
      one: no rule on the other side is related in any of these ways.

    The names are those of named-entity scoring (MUC, SemEval 2013),
    with ``"too_specific"`` and ``"too_general"`` from inductive logic
    programming. As in the biclustering scores of Prelić et al.
    [Prelic2006]_, both sides are scored, and several rules may share
    one partner.

    A rule with the right conditions and the wrong value is
    ``"spurious"`` or ``"partial"``: only rules of the same head are
    compared.

    ``true_rules`` must be every minimal rule of the system, as ``GULA``
    or ``BruteForce`` learn from its complete table. A learned rule is
    then correct on that table exactly when it is ``"exact"`` or
    ``"too_specific"``; that share is the ``"precision"``. With a
    hand-written subset, a correct rule may be scored as wrong.

    With ``states``, the comparison is made at each state, between the
    rules that match it: for each target, the learned rules that vote
    there against the true rules that conclude there. Every share is
    averaged over each (state, target) pair, so rules that match many
    test states weigh more. At a state, every learned rule shares that
    state with every true rule of its head, so ``"partial"`` there means
    the right value from the wrong rule, and ``"spurious"`` a vote for a
    value the system never gives there. A state no learned rule matches
    scores 0 on precision and recall.

    Every learned rule is compared with every true rule of its head,
    once without ``states`` and once per state with them: on thousands
    of rules of one head this takes seconds.

    Args:
        true_rules: The rules to recover, every minimal rule of the
            system.
        learned_rules: The rules to measure, for example an algorithm's
            output or ``model.rules``.
        states: Feature states to compare at, for example the test rows:
            a ``DataFrame`` or one mapping per row. Every feature the
            rules test must have a known value. ``None`` compares the
            whole sets, each rule counting once.

    Returns:
        A dictionary with ``"precision"``, the share of learned rules
        that are ``"exact"`` or ``"too_specific"``; ``"recall"``, the
        same share of true rules; ``"learned"``, each category's share
        of learned rules, ``"exact"`` to ``"spurious"``; and ``"true"``,
        each category's share of true rules, ``"exact"`` to
        ``"missing"``. Each side's shares sum to 1, except the learned
        side's, all 0, when no learned rule is given or matches.

    Raises:
        TypeError: If ``true_rules`` or ``learned_rules`` is a single
            ``str`` or holds something other than a ``Rule``; if
            ``states`` is neither a ``DataFrame`` nor a sequence of
            rows, or a row is not a mapping; or, as ``Rule.matches``
            raises, if a mapping row holds a value a rule tests that is
            neither a ``str`` nor unknown.
        ValueError: If ``true_rules`` or ``states`` is empty; if a
            learned rule concludes on a variable no true rule concludes
            on; if ``states`` is a ``DataFrame`` without a column a rule
            tests; if a state has an unknown value for a feature a rule
            tests; or if no true rule on some target matches a state,
            which complete true rules never allow.
        KeyError: If a row of ``states`` lacks a feature a rule tests.

    See Also:
        coverage: The share of each head's rows the rules match.

    References:
        .. [Prelic2006] Prelić, A., Bleuler, S., Zimmermann, P., Wille,
           A., Bühlmann, P., Gruissem, W., Hennig, L., Thiele, L. &
           Zitzler, E. (2006). A systematic comparison and evaluation of
           biclustering methods for gene expression data.
           Bioinformatics, 22(9), 1122-1129.
           https://doi.org/10.1093/bioinformatics/btl060

    Examples:
        The true rules of ``y = a or b``, from its complete table, and
        rules learned without the state ``a=1, b=1``:

        >>> import pandas as pd
        >>> from pylfit import Dataset, GULA
        >>> table = pd.DataFrame(
        ...     {
        ...         "a": ["0", "0", "1", "1"],
        ...         "b": ["0", "1", "0", "1"],
        ...         "y": ["0", "1", "1", "1"],
        ...     }
        ... )
        >>> full = Dataset(table, features=["a", "b"], targets=["y"])
        >>> train = Dataset(table.iloc[:3], features=["a", "b"], targets=["y"])
        >>> truth = GULA.fit(full)
        >>> learned = GULA.fit(train)
        >>> [str(rule) for rule in learned if rule not in truth]
        ['y(0) :- a(1), b(1).']
        >>> scores = rule_recovery(truth, learned)
        >>> scores["precision"], scores["recall"], scores["learned"]["spurious"]
        (0.75, 1.0, 0.25)

        At the unseen state, that rule votes for a value ``y`` never
        takes there:

        >>> unseen = table.iloc[3:][["a", "b"]]
        >>> round(rule_recovery(truth, learned, unseen)["learned"]["spurious"], 3)
        0.333
        >>> rule_recovery(truth, GULA.fit(train, min_support=1), unseen)["precision"]
        1.0
    """
    truth = _rule_list(true_rules, "true_rules")
    learned = _rule_list(learned_rules, "learned_rules")
    if not truth:
        raise ValueError("true_rules must be non-empty")
    targets = sorted({rule.head[0] for rule in truth})
    for rule in learned:
        if rule.head[0] not in targets:
            raise ValueError(
                f"learned rule {rule} concludes on {rule.head[0]!r}, which no "
                "true rule concludes on"
            )
    if states is None:
        return _recovery(truth, learned)

    tested = sorted({variable for rule in truth + learned for variable in rule.body})
    rows = _states(states, tested)
    true_by_target = {
        target: [r for r in truth if r.head[0] == target] for target in targets
    }
    learned_by_target = {
        target: [r for r in learned if r.head[0] == target] for target in targets
    }
    scores = []
    for row, state in enumerate(rows):
        for target in targets:
            true_here = [r for r in true_by_target[target] if r.matches(state)]
            if not true_here:
                raise ValueError(
                    f"no true rule on {target!r} matches states row {row}: "
                    "true_rules must be every minimal rule of the system"
                )
            learned_here = [r for r in learned_by_target[target] if r.matches(state)]
            scores.append(_recovery(true_here, learned_here))
    return {
        "precision": fmean(score["precision"] for score in scores),
        "recall": fmean(score["recall"] for score in scores),
        **{
            side: {
                name: fmean(score[side][name] for score in scores)
                for name in scores[0][side]
            }
            for side in ("learned", "true")
        },
    }


def _states(
    states: Sequence[Mapping[str, object]] | pd.DataFrame, tested: Sequence[str]
) -> Sequence[Mapping[str, object]]:
    """Checks ``rule_recovery``'s ``states`` and gives them as rows of text.

    Args:
        states: The argument passed.
        tested: Every feature a rule tests.

    Returns:
        One mapping per state; a ``DataFrame``'s ``tested`` columns
        converted as ``Dataset`` converts them.

    Raises:
        TypeError: If ``states`` is not a table of rows, or a row is not
            a mapping.
        ValueError: If ``states`` is empty, a ``DataFrame`` lacks a
            column of ``tested``, or a state's value for one of
            ``tested`` is unknown.
        KeyError: If a row lacks a feature of ``tested``.
    """
    _check_rows(states, "states")
    rows: Sequence[Mapping[Any, object]]
    if not isinstance(states, pd.DataFrame):
        rows = states
    elif tested:
        rows = _records_as_text(states, tested, "states")
    else:  # with no feature tested, every row still counts, as an empty state
        rows = [{} for _ in range(len(states))]
    if len(rows) == 0:
        raise ValueError("states must be non-empty")
    for row, state in enumerate(rows):
        if not isinstance(state, Mapping | pd.Series):
            raise TypeError(
                f"states row {row} is {state!r}, not a mapping of feature to value"
            )
        for feature in tested:
            if feature not in state:
                raise KeyError(f"states row {row} has no {feature!r}")
            if _is_unknown(state[feature]):
                raise ValueError(
                    f"states row {row} has an unknown value for {feature!r}, "
                    "which a rule tests: no rule would match it"
                )
    return rows


def _recovery(truth: Sequence[Rule], learned: Sequence[Rule]) -> dict[str, Any]:
    """``rule_recovery`` between two sets of rules, without states.

    Args:
        truth: The true rules, at least one.
        learned: The learned rules, possibly none.

    Returns:
        The dictionary ``rule_recovery`` returns.
    """
    true_by_head: dict[tuple[str, str], list[Rule]] = {}
    for rule in truth:
        true_by_head.setdefault(rule.head, []).append(rule)
    learned_by_head: dict[tuple[str, str], list[Rule]] = {}
    for rule in learned:
        learned_by_head.setdefault(rule.head, []).append(rule)
    learned_counts = dict.fromkeys((*_RELATIONS, "spurious"), 0)
    for rule in learned:
        best = min(
            (_relation(rule, true) for true in true_by_head.get(rule.head, ())),
            default=len(_RELATIONS),
        )
        learned_counts[(*_RELATIONS, "spurious")[best]] += 1
    true_counts = dict.fromkeys((*_RELATIONS, "missing"), 0)
    for rule in truth:
        best = min(
            (_relation(other, rule) for other in learned_by_head.get(rule.head, ())),
            default=len(_RELATIONS),
        )
        true_counts[(*_RELATIONS, "missing")[best]] += 1
    learned_shares = {
        name: count / len(learned) if learned else 0.0
        for name, count in learned_counts.items()
    }
    true_shares = {name: count / len(truth) for name, count in true_counts.items()}
    return {
        "precision": learned_shares["exact"] + learned_shares["too_specific"],
        "recall": true_shares["exact"] + true_shares["too_specific"],
        "learned": learned_shares,
        "true": true_shares,
    }


def _relation(learned: Rule, true: Rule) -> int:
    """Ranks how a learned rule relates to a true rule of the same head.

    Args:
        learned: The learned rule.
        true: The true rule.

    Returns:
        The index in ``_RELATIONS`` of the relation, or
        ``len(_RELATIONS)`` when no state matches both rules.
    """
    if learned == true:
        return 0
    if true.is_more_general_than(learned):
        return 1
    if learned.is_more_general_than(true):
        return 2
    # Some state matches both unless a feature has two different values.
    body = true.body
    if all(
        body.get(feature, value) == value for feature, value in learned.body.items()
    ):
        return 3
    return len(_RELATIONS)
