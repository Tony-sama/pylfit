"""GULA (General Usage LFIT Algorithm), pylfit's main rule learner.

For each target value, ``GULA`` starts from the rule with no conditions
and specializes a rule only when it matches a negative example (a
feature state the target never takes that value in), one condition at
a time.

On an unrestricted run, the learned rule set is:

- correct: no rule matches a negative example of its own head.
- complete: for each target, every combination of the features'
  domain values is matched by at least one rule, whatever the unknowns
  in the data.
- minimal: dropping any condition from a rule makes it match a
  negative example.

That is the same rule set ``BruteForce`` finds by exhaustive
enumeration, reached by a much cheaper search; the worst case is still
exponential in the number of features and the size of their domains.
The proofs are in [Ribeiro2022]_.

``heads`` learns only some of the (target, value) pairs, with the same
rules for each as a full run. ``allowed_conditions``, ``min_support``
and ``min_possible_support`` restrict the search. The rules stay
correct, but the set is no longer complete: a feature state only a
discarded rule would have matched is left unmatched.
``negative_order`` changes the runtime, never the rules.

Unknown feature values (``None`` in ``Dataset``'s states) need nothing
special: an unknown never satisfies a condition, so a negative with
an unknown feature is excluded by a condition on any value of that
feature, and a positive with one is matched only by rules that don't
test that feature. The rules are those the unknowns leave consistent,
and none of them ever tests for an unknown. Every unknown lets more
rules survive, so learning slows down as unknowns grow; on many
features with many unknowns ``PRIDE`` is the faster choice.

A positive row that no known feature separates from some negative can
be matched by no consistent rule; ``fit`` warns when unknowns, or the
options above, leave positive rows uncovered.

Why it works this way: :doc:`/explanation/about-algorithms`.

References:
    .. [Ribeiro2022] Ribeiro, T., Folschette, M., Magnin, M. & Inoue, K.
       (2022). Learning any memory-less discrete semantics for dynamical
       systems represented by logic programs. Machine Learning, 111(10),
       3593-3670. https://doi.org/10.1007/s10994-021-06105-4
"""

import time
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from functools import partial
from typing import Literal, get_args

from pylfit._dataset import Dataset, _check_dataset
from pylfit._progress import duration
from pylfit._rule import Rule
from pylfit.algorithms._algorithm import (
    Algorithm,
    AllowedConditions,
    Heads,
    Uncovered,
)

#: A ``(feature, value)`` condition.
_Condition = tuple[str, str]

#: A kept rule, before it is a ``Rule``: its conditions as bits (one bit
#: per condition this head can use), the features they test as bits (one
#: bit per feature), the conditions themselves, then the positives (bit
#: ``i`` for its head's ``positives[i]``) it matches exactly, then those
#: it possibly matches (an unknown counted as satisfying any condition)
#: — each of the last two ``None`` when its threshold, ``min_support``
#: or ``min_possible_support``, is unset and that support isn't tracked
#: at all.
_Tracked = tuple[int, int, tuple[_Condition, ...], int | None, int | None]

#: A feature, its bit, and the conditions on it a rule may use, each
#: with its bit.
_Usable = tuple[str, int, list[tuple[_Condition, int]]]

#: The orders ``fit`` can process negatives in; see its docstring.
NegativeOrder = Literal["similarity", "lexicographic", "unordered"]


class GULA(Algorithm):
    """Rule learner based on incremental specialization against negatives."""

    _NEGATIVE_ORDERS: tuple[str, ...] = get_args(NegativeOrder)

    @staticmethod
    def fit(
        dataset: Dataset,
        *,
        verbose: bool = False,
        heads: Heads | None = None,
        allowed_conditions: AllowedConditions | None = None,
        min_support: int | float | None = None,
        min_possible_support: int | float | None = None,
        negative_order: NegativeOrder = "similarity",
        warn_uncovered: bool = True,
        n_jobs: int | None = 1,
    ) -> list[Rule]:
        """Learns rules from a dataset.

        Args:
            dataset: The dataset to learn from.
            verbose: If True, prints progress for each (target, value)
                pair to stderr: negative count, rules kept, elapsed
                time.
            heads: The (target, value) pairs to learn rules for, each a
                ``(target, value)`` pair or a bare target name (every
                value of that target). Each head's rules are the same as
                in a full run. ``None`` learns every head. A model built
                from a subset predicts by a random draw wherever no rule
                matches, so learn every head of a target to predict it.
            allowed_conditions: Background knowledge. Maps a head
                ``(target, value)`` to the conditions its rules may use,
                each a ``(feature, value)`` pair or a bare feature name
                (every value of that feature). A head absent from the
                mapping, or mapped to ``None``, is unrestricted, and
                ``None`` for the whole argument restricts nothing. A
                head left out of ``heads`` is still checked against
                ``dataset``.
            min_support: Drops rules matching fewer positive rows than
                this, counting a repeated state once per row: an ``int``
                count, or a ``float`` in ``(0, 1]`` read as a fraction
                of the head's positive rows. The rules kept are exactly
                the unrestricted run's rules that meet the threshold.
                ``None`` keeps every rule, including minimal consistent
                rules that match no training row. ``RuleSetModel`` makes
                their value a candidate wherever they match, like any
                other rule; ``WeightedRuleSetModel`` gives them no
                weight. ``1`` drops them. A positive row with an unknown
                value for a feature the rule tests does not count.
            min_possible_support: As ``min_support``, but counting the
                positive rows a rule could match: an unknown value
                counts as satisfying any condition. A rule is dropped
                when fewer positive rows than the threshold could match
                it, so on data with unknowns a rule whose support the
                unknowns hide is kept; at ``1``, only the rules no row
                could match go. Without unknowns it is the same as
                ``min_support``. Both can be set: a rule is kept when it
                meets both.
            negative_order: The order negatives are processed in:
                ``"similarity"`` (most typical negatives first),
                ``"lexicographic"`` (by feature values), or
                ``"unordered"`` (``Dataset.negatives``' own order).
                Changes the runtime, never the learned rules.
            warn_uncovered: If True, warns when some positive rows are
                matched by no learned rule of their head, giving their
                share and why: unknown values that no known feature
                separates from a negative, or one of the options above.
                With ``verbose``, lists each row. Checked only when it
                can happen: with unknown feature values,
                ``allowed_conditions``, ``min_support`` or
                ``min_possible_support``.
            n_jobs: The number of processes heads are learned in at
                once: ``1`` (default) or ``None`` learns them one after
                another in this process, ``-1`` uses every CPU, ``-2``
                all but one, as in scikit-learn; ``0`` raises
                ``ValueError``, and any other value that is not an
                integer ``TypeError``. The rules are the same whatever
                the value. No more processes are started than there are
                heads to learn, nor more than 61 on Windows, the most it
                allows. Starting processes takes time, so it pays off
                only when a fit takes seconds or more; see
                :doc:`/explanation/about-algorithms`. With ``verbose``
                and more than one process, prints a line per head as it
                finishes, instead of each head's progress. A script
                fitting with ``n_jobs`` other than 1 must do so under
                ``if __name__ == "__main__":``, or the fit fails: on
                macOS, on Windows and on Linux from Python 3.14, each
                worker process imports the script again.

        Returns:
            The learned rules, sorted by text, without duplicates. Wrap
            them in a model to predict, for example
            ``RuleSetModel(dataset, rules)``.

        Raises:
            ValueError: If ``heads`` holds an entry that is neither a
                target name nor a
                ``(target, value)`` pair, or names a target or value
                ``dataset`` lacks; if ``negative_order`` is not one of
                the three orders; if ``min_support`` or
                ``min_possible_support`` is a negative ``int`` or a
                ``float`` outside ``(0, 1]``; if ``allowed_conditions``
                has a key that is not a ``(target, value)`` pair, names
                a head that is not a target value of ``dataset``, a
                variable that is not a feature, or a value outside its
                feature's domain, gives a condition that is neither a
                feature name nor a ``(feature, value)`` pair, or gives a
                head a single string or a mapping instead of a
                collection of conditions; or if ``n_jobs`` is ``0``.
            TypeError: If ``dataset`` is not a ``Dataset``; if
                ``min_support`` or ``min_possible_support`` is neither
                an integer nor a float (``bool`` is refused); if
                ``heads`` is a single string, a mapping or not iterable;
                if ``allowed_conditions`` is not a mapping, or gives a
                head conditions that are neither ``None`` nor iterable;
                if a value in ``heads`` or ``allowed_conditions`` is not
                a ``str``; if ``verbose`` or ``warn_uncovered`` is not a
                ``bool``; or if ``n_jobs`` is neither an integer nor
                ``None``.
            ~concurrent.futures.process.BrokenProcessPool: If a worker
                process dies or can't start, as when a script fitting
                with ``n_jobs`` other than 1 lacks the
                ``if __name__ == "__main__":`` guard (see ``n_jobs``).

        See Also:
            BruteForce: The exhaustive learner ``GULA`` is tested
                against.
            PRIDE: Much faster on large data; learns a subset of these
                rules, complete only over the rows present.

        Examples:
            >>> import pandas as pd
            >>> from pylfit import Dataset, GULA
            >>> df = pd.DataFrame(
            ...     {
            ...         "a": ["0", "0", "1", "1"],
            ...         "b": ["0", "1", "0", "1"],
            ...         "y": ["0", "1", "1", "1"],
            ...     }
            ... )
            >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
            >>> rules = GULA.fit(dataset)
            >>> [rule.to_string() for rule in rules]
            ['y(0) :- a(0), b(0).', 'y(1) :- a(1).', 'y(1) :- b(1).']

            On a full truth table, the rules for ``y=1`` say everything;
            ``heads`` skips ``y=0``:

            >>> rules = GULA.fit(dataset, heads=[("y", "1")])
            >>> [rule.to_string() for rule in rules]
            ['y(1) :- a(1).', 'y(1) :- b(1).']

            With unknown values, ``None`` in the ``DataFrame``: the only
            row with ``y=1`` has ``b`` unknown, so no rule for ``y=1``
            matches it for sure, but one could:

            >>> df = pd.DataFrame(
            ...     {
            ...         "a": ["0", "0", None, "1"],
            ...         "b": ["1", None, "0", "0"],
            ...         "y": ["0", "1", "0", "0"],
            ...     }
            ... )
            >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
            >>> one = [("y", "1")]
            >>> [r.to_string() for r in GULA.fit(dataset, heads=one)]
            ['y(1) :- a(0), b(0).', 'y(1) :- a(1), b(1).']
            >>> GULA.fit(dataset, heads=one, min_support=1)
            []
            >>> rules = GULA.fit(dataset, heads=one, min_possible_support=1)
            >>> [rule.to_string() for rule in rules]
            ['y(1) :- a(0), b(0).']
        """
        # First, so a DataFrame passed with options gets the same error.
        _check_dataset(dataset)
        if negative_order not in GULA._NEGATIVE_ORDERS:
            raise ValueError(
                f"negative_order must be one of {GULA._NEGATIVE_ORDERS!r}, "
                f"got {negative_order!r}"
            )
        if min_support is not None:
            GULA._check_min_support(min_support)
        if min_possible_support is not None:
            GULA._check_min_support(min_possible_support, "min_possible_support")
        if allowed_conditions is not None:
            allowed_conditions = GULA._check_allowed_conditions(
                dataset, allowed_conditions
            )
        learn = partial(
            GULA._learn_head,
            allowed_conditions=allowed_conditions,
            min_support=min_support,
            negative_order=negative_order,
            min_possible_support=min_possible_support,
            warn_uncovered=warn_uncovered,
        )
        return GULA._fit_heads(dataset, learn, heads, verbose, warn_uncovered, n_jobs)

    @staticmethod
    def _learn_head(
        dataset: Dataset,
        target: str,
        value: str,
        verbose: bool,
        *,
        allowed_conditions: AllowedConditions | None,
        min_support: int | float | None,
        negative_order: NegativeOrder,
        min_possible_support: int | float | None,
        warn_uncovered: bool,
    ) -> tuple[list[Rule], list[Uncovered]]:
        """Learns one head for ``fit``'s shared loop, with its uncovered rows.

        Args:
            dataset: The dataset to learn from.
            target: The head's target.
            value: The head's value.
            verbose: If True, prints this head's progress to stderr.
            allowed_conditions: ``fit``'s argument, already validated.
            min_support: ``fit``'s argument, already validated.
            negative_order: ``fit``'s argument, already validated.
            min_possible_support: ``fit``'s argument, already validated.
            warn_uncovered: Whether to look for uncovered rows at all.

        Returns:
            The head's rules, and its positive rows none of them
            matches, each with why; the rows are only looked for when
            they can exist: with unknown values or a restricting option.
        """
        allowed = (
            None
            if allowed_conditions is None
            else allowed_conditions.get((target, value))
        )
        rules = GULA._fit_target_value(
            dataset,
            target,
            value,
            verbose=verbose,
            allowed_conditions=allowed,
            min_support=min_support,
            negative_order=negative_order,
            min_possible_support=min_possible_support,
        )
        restrictions = [
            name
            for name, option in (
                ("allowed_conditions", allowed),
                ("min_support", min_support),
                ("min_possible_support", min_possible_support),
            )
            if option is not None
        ]
        uncovered = (
            GULA._uncovered(dataset, target, value, rules, restrictions)
            if warn_uncovered and (restrictions or dataset._has_unknowns)
            else []
        )
        return rules, uncovered

    @staticmethod
    def _fit_target_value(  # noqa: C901
        dataset: Dataset,
        target: str,
        value: str,
        verbose: bool = False,
        allowed_conditions: Collection[str | tuple[str, str]] | None = None,
        min_support: int | float | None = None,
        negative_order: NegativeOrder = "similarity",
        min_possible_support: int | float | None = None,
    ) -> list[Rule]:
        """Learns the minimal consistent rules concluding target=value.

        Starts from the fully general rule and, for each distinct
        negative example in turn, replaces any rule that wrongly matches
        it with its least specializations against that example, keeping
        those no kept rule dominates. The kept rules are always the
        minimal ones for the negatives seen so far, so no minimality
        pass is needed at the end.

        The domination check is narrow. A specialization is a rule ``R``
        that matched the negative ``n`` plus one condition ``c`` that
        ``n`` violates. A kept rule that dominates it doesn't match
        ``n``, yet has only conditions of ``R``, which all agree with
        ``n``, or ``c``: so it contains ``c``. Only kept rules holding
        ``c`` are checked, through an index from each condition to the
        kept rules using it. Two specializations from the same negative
        never dominate each other, since the rules they come from don't,
        so each is checked against the rules kept before that negative.

        The loop builds no ``Rule``: each body is held as the bits of an
        ``int``, one per condition, and each negative as the bits of the
        conditions it satisfies. A rule matches a negative when its bits
        are all among those, and a domination test is one AND.
        Only the rules returned are built, at the end: on the published
        networks most specializations are dominated, and building them
        was half of the fit.

        With ``min_support`` set, support is tracked incrementally:
        every rule carries the positives it matches as the bits of an
        ``int``, and a specialization's are its parent's ANDed with
        those of the condition it adds. Support only shrinks under
        specialization, so a specialization below threshold is dropped
        at once. Dropping it early loses nothing: a rule it would have
        dominated is a specialization of it, with no more support.
        ``min_possible_support`` is tracked the same way, an unknown
        passing every filter; possible support only shrinks under
        specialization too.

        Args:
            dataset: The classification dataset to learn from.
            target: The target variable to conclude on.
            value: The value of ``target`` the learned rules conclude.
            verbose: If True, prints progress for this (target, value)
                pair to stderr.
            allowed_conditions: Optional conditions allowed for this
                head, already resolved out of ``fit``'s mapping — see
                ``fit``. ``None`` means unrestricted.
            min_support: Optional minimum-support filter for this head,
                already validated by ``fit``. ``None`` means
                unrestricted.
            negative_order: Which order negatives are processed in —
                see ``fit``. Assumed already validated by ``fit``.
            min_possible_support: Optional minimum possible support for
                this head, already validated by ``fit``. ``None`` means
                unrestricted.

        Returns:
            The rules consistent with the dataset and not dominated by
            another consistent rule — complete only if
            ``allowed_conditions``, ``min_support`` and
            ``min_possible_support`` are all ``None``.
        """
        start = time.perf_counter()
        # `features` sorted by name so the negatives' order, and with it
        # the runtime, is the same on every run.
        features = sorted(dataset.features)
        negatives = GULA._order_negatives(
            dataset._distinct_negative_states(target, value),
            features,
            negative_order,
            dataset._has_unknowns,
        )
        pool = (
            None
            if allowed_conditions is None
            else GULA._expand_allowed(allowed_conditions, dataset)
        )

        tracking = min_support is not None
        tracking_possible = min_possible_support is not None
        positive_rows = dataset._positive_rows(target, value)
        min_count = (
            GULA._resolve_min_support(min_support, len(positive_rows))
            if min_support is not None
            else 0
        )
        min_possible_count = (
            GULA._resolve_min_support(min_possible_support, len(positive_rows))
            if min_possible_support is not None
            else 0
        )
        matching, unknown = (
            dataset._condition_masks(positive_rows)
            if tracking or tracking_possible
            else ({}, {})
        )
        everyone = (1 << len(positive_rows)) - 1

        conditions = (
            pool
            if pool is not None
            else [
                (feature, domain_value)
                for feature in dataset.features
                for domain_value in dataset.domain(feature)
            ]
        )
        bit = {condition: 1 << i for i, condition in enumerate(conditions)}
        per_feature: dict[str, list[tuple[_Condition, int]]] = {}
        for condition, condition_bit in bit.items():
            per_feature.setdefault(condition[0], []).append((condition, condition_bit))
        usable: list[_Usable] = [
            (feature, 1 << i, per_feature[feature])
            for i, feature in enumerate(dataset.features)
            if feature in per_feature
        ]

        kept: list[_Tracked] = []
        # Each condition mapped to the bodies, as bits, of the kept
        # rules that use it.
        by_condition: dict[tuple[str, str], set[int]] = {}
        if len(positive_rows) >= max(min_count, min_possible_count):
            # The empty body matches every positive, and has no
            # condition to index.
            kept.append(
                (
                    0,
                    0,
                    (),
                    everyone if tracking else None,
                    everyone if tracking_possible else None,
                )
            )

        total = len(negatives)
        progress = GULA._progress(f"{target}={value}: processing negatives", total)
        for step, negative in enumerate(negatives, start=1):
            satisfied, violated = GULA._split_conditions(negative, usable)
            still_consistent: list[_Tracked] = []
            inconsistent: list[_Tracked] = []
            for entry in kept:
                if entry[0] & satisfied == entry[0]:
                    inconsistent.append(entry)
                else:
                    still_consistent.append(entry)
            kept = still_consistent
            for body, _, body_conditions, _, _ in inconsistent:
                for condition in body_conditions:
                    by_condition[condition].discard(body)

            newly_kept: list[_Tracked] = []
            for body, tested, body_conditions, matched, possible in inconsistent:
                for feature_bit, options in violated:
                    # Features the rule already tests are skipped: since
                    # it matches the negative, its condition there
                    # agrees with the negative, so that feature offers
                    # no new way to disagree with it.
                    if tested & feature_bit:
                        continue
                    for condition, condition_bit in options:
                        # Domination first: its cost doesn't grow with
                        # the rows, while the support filters below AND
                        # masks as long as the positives. Both only
                        # skip, so the order changes the time, never the
                        # rules. No kept body equals the candidate's:
                        # that rule would be dominated by the one
                        # specialized, kept alongside it. So the subset
                        # test is a strict one.
                        candidate_body = body | condition_bit
                        if any(
                            other & candidate_body == other
                            for other in by_condition.get(condition, ())
                        ):
                            continue
                        candidate_matched = None
                        if matched is not None:
                            candidate_matched = matched & matching[condition]
                            if candidate_matched.bit_count() < min_count:
                                continue
                        candidate_possible = None
                        if possible is not None:
                            candidate_possible = possible & (
                                matching[condition] | unknown[condition[0]]
                            )
                            if candidate_possible.bit_count() < min_possible_count:
                                continue
                        newly_kept.append(
                            (
                                candidate_body,
                                tested | feature_bit,
                                (*body_conditions, condition),
                                candidate_matched,
                                candidate_possible,
                            )
                        )

            for entry in newly_kept:
                kept.append(entry)
                for condition in entry[2]:
                    by_condition.setdefault(condition, set()).add(entry[0])

            if verbose:
                progress.update(step)

        rules = [
            Rule((target, value), dict(body_conditions))
            for _, _, body_conditions, _, _ in kept
        ]

        if verbose:
            elapsed = time.perf_counter() - start
            GULA._log(
                f"{target}={value}: {total} distinct negatives, {len(rules)} rules "
                f"kept ({duration(elapsed)} total)"
            )

        return rules

    @staticmethod
    def _split_conditions(
        negative: Mapping[str, str | None], usable: list[_Usable]
    ) -> tuple[int, list[tuple[int, list[tuple[_Condition, int]]]]]:
        """Splits a head's conditions by what a negative does to them.

        A condition the negative violates is what a minimal
        specialization adds: added to a rule matching the negative, it
        is the smallest possible change that makes the rule stop
        matching. An unknown value satisfies no condition, so it
        violates every condition on its feature.

        With a pool of allowed conditions, ``usable`` holds only those.
        A rule with nothing usable left is dropped with nothing to
        replace it — expected under restricted search, see ``fit``.

        Args:
            negative: The feature state to split by.
            usable: This head's conditions, per feature.

        Returns:
            The bits of the conditions ``negative`` satisfies, then each
            feature's bit with the conditions on it ``negative``
            violates, each with its bit.
        """
        satisfied = 0
        violated = []
        for feature, feature_bit, options in usable:
            value = negative[feature]
            others = []
            for option in options:
                if option[0][1] == value:
                    satisfied |= option[1]
                else:
                    others.append(option)
            violated.append((feature_bit, others))
        return satisfied, violated

    @staticmethod
    def _order_negatives(
        negatives: Sequence[Mapping[str, str | None]],
        features: list[str],
        negative_order: str,
        unknowns: bool = True,
    ) -> list[Mapping[str, str | None]]:
        """Orders one head's negatives for the specialization loop.

        See ``fit`` for what each ``negative_order`` does. None of them
        changes the learned rule set.

        Args:
            negatives: The head's negatives, each distinct state
                once.
            features: ``sorted(dataset.features)`` — the fixed column
                order both the ``"lexicographic"`` and ``"similarity"``
                orders sort by/tie-break on.
            negative_order: ``"similarity"``, ``"lexicographic"``, or
                ``"unordered"``.
            unknowns: Whether a negative may hold ``None``; ``False``
                allows a faster sort key.

        Returns:
            ``negatives`` in the requested order.
        """
        if negative_order == "unordered":
            return list(negatives)
        if negative_order == "lexicographic":
            return sorted(negatives, key=GULA._state_key(features, unknowns))
        # "similarity": ascending summed Hamming distance to the rest of
        # this head's negatives — `sum_j hamming(i, j)` for negative
        # `i`, computed in `O(n * len(features))` rather than
        # materializing every pair: a feature's contribution to every
        # row's total is `n - count[value]`, that value's disagreement
        # count in that column (`n` includes `i` itself, so this already
        # excludes comparing `i` against itself). Ties broken by the
        # same lexicographic key `"lexicographic"` sorts by, for a fully
        # deterministic order.
        n = len(negatives)
        totals = [0] * n
        for feature in features:
            values = [negative[feature] for negative in negatives]
            counts = Counter(values)
            for i, value in enumerate(values):
                totals[i] += n - counts[value]
        key = GULA._state_key(features, unknowns)
        keys = [key(negative) for negative in negatives]
        order = sorted(range(n), key=lambda i: (totals[i], keys[i]))
        return [negatives[i] for i in order]
