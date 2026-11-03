"""PRIDE, a covering rule learner.

``PRIDE`` covers positive examples one at a time: it picks a positive
no rule covers yet, builds one rule around it by adding conditions
until the rule matches no negative example, drops conditions the rule
doesn't need, then sets aside every positive the new rule covers.
Each condition added is, among those excluding the negative at hand,
the one leaving the rule matching the most positives still left to
cover (not covered by a rule yet, nor tried as a seed); ties go to the
first feature in ``Dataset.features`` order.

The learned rule set is:

- correct: no rule matches a negative example of its own head.
- minimal: no subset of a rule's conditions is consistent either.
- a subset of what ``GULA`` and ``BruteForce`` learn from the same
  data.
- complete only over the rows present in the dataset: a feature
  state absent from the data may be matched by no rule. On data with
  every feature state present, this gap disappears.

Unknown feature values (``None`` in ``Dataset``'s states) never
satisfy a condition, so a rule grows only on its seed's known
features. A positive that no known feature separates from some
negative can be matched by no rule at all, and gets none: its row is
left uncovered, and ``fit`` warns about it. ``heads``,
``allowed_conditions``, ``min_support`` and ``min_possible_support``
work as in ``GULA``; a seed the last three leave without a rule is
reported the same way.

``PRIDE`` takes seeds and negatives in the dataset's row order, so
reordering the rows can change the rules, though not their
correctness or minimality. Sort the ``DataFrame`` first for rules that
don't depend on the order rows came in.

``try_all_conditions=True`` also builds, for each seed, one rule per
seed condition, that condition put in first, and keeps every distinct
one. It usually finds more of ``GULA``'s rules, at several times the
fit time and with more rules. It is not a superset of the default
run: the extra rules cover positives too, so later seeds change, and
a rule the default run learns can be missing.

The full algorithm is in [Ribeiro2021]_.

Why it works this way: :doc:`/explanation/about-algorithms`.

References:
    .. [Ribeiro2021] Ribeiro, T., Folschette, M., Magnin, M. & Inoue, K.
       (2021). Polynomial algorithm for learning from interpretation
       transition. 1st International Joint Conference on Learning and
       Reasoning (IJCLR 2021), late-breaking abstract.
       https://hal.science/hal-03347026
"""

import time
from collections.abc import Mapping, Sequence
from functools import partial

from pylfit._dataset import Dataset, _check_dataset, _check_flag
from pylfit._progress import duration
from pylfit._rule import Rule
from pylfit.algorithms._algorithm import (
    Algorithm,
    AllowedConditions,
    Heads,
    Uncovered,
)

_State = Mapping[str, str | None]


class PRIDE(Algorithm):
    """Rule learner based on covering positive examples one at a time."""

    @staticmethod
    def fit(
        dataset: Dataset,
        *,
        verbose: bool = False,
        heads: Heads | None = None,
        allowed_conditions: AllowedConditions | None = None,
        min_support: int | float | None = None,
        min_possible_support: int | float | None = None,
        try_all_conditions: bool = False,
        warn_uncovered: bool = True,
        n_jobs: int | None = 1,
    ) -> list[Rule]:
        """Learns rules from a dataset.

        Args:
            dataset: The dataset to learn from.
            verbose: If True, prints progress for each (target, value)
                pair to stderr: positive count, rules kept, positives
                left uncovered, elapsed time.
            heads: The (target, value) pairs to learn rules for, as in
                ``GULA.fit``. ``None`` learns every head.
            allowed_conditions: Background knowledge, as in
                ``GULA.fit``: per head, the conditions its rules may
                use. A seed the allowed conditions can't separate from
                some negative gets no rule.
            min_support: The positive rows a rule must match, an ``int``
                count or a ``float`` fraction of the head's positive
                rows, as in ``GULA.fit``. Each seed's plain rule is kept
                if it meets the threshold. Otherwise the rule is
                rebuilt, adding at each step, among the conditions that
                keep the threshold met, the one chosen as above (the
                most positives still to cover, then feature order); a
                seed with no such condition, or whose rebuilt rule still
                falls short, gets no rule. Every rule kept is one of
                ``GULA``'s with the same threshold. ``1`` changes
                nothing, since every rule matches its seed, unlike on
                ``GULA``, where it drops the rules matching no row.
            min_possible_support: As ``min_support``, counting a row as
                a match when each condition holds or falls on an
                unknown, as in ``GULA.fit``; ``1`` changes nothing here
                either. Both can be set.
            try_all_conditions: If True, each seed also gets one rule
                per condition it can take (its known values, within
                ``allowed_conditions``), built with that condition put
                in first and tried last when dropping the unneeded ones;
                every distinct rule is kept. More of the possible rules
                are usually found, at several times the fit time, though
                not always every rule of the default run; see the module
                docstring. Each extra rule meets ``min_support`` and
                ``min_possible_support`` too, or is not kept.
            warn_uncovered: If True, warns when some positive rows are
                matched by no learned rule of their head, giving their
                share and why: unknown values that no known feature
                separates from a negative, or one of the options above.
                With ``verbose``, lists each row. It can only happen
                with unknown feature values or one of those options.
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
            ValueError: If ``heads``, ``allowed_conditions``,
                ``min_support`` or ``min_possible_support`` is invalid,
                as in ``GULA.fit``; or if ``n_jobs`` is ``0``.
            TypeError: If ``dataset``, ``heads``, ``min_support``,
                ``min_possible_support``, ``allowed_conditions``,
                ``verbose``, ``warn_uncovered`` or ``n_jobs`` is of the
                wrong type, as in ``GULA.fit``, or
                ``try_all_conditions`` is not a ``bool``.
            ~concurrent.futures.process.BrokenProcessPool: If a worker
                process dies or can't start, as when a script fitting
                with ``n_jobs`` other than 1 lacks the
                ``if __name__ == "__main__":`` guard (see ``n_jobs``).

        See Also:
            GULA: Learns every rule, of which ``PRIDE``'s are a subset.

        Examples:
            >>> import pandas as pd
            >>> from pylfit import Dataset, PRIDE
            >>> df = pd.DataFrame(
            ...     {
            ...         "a": ["0", "0", "1", "1"],
            ...         "b": ["0", "1", "0", "1"],
            ...         "y": ["0", "1", "1", "1"],
            ...     }
            ... )
            >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
            >>> rules = PRIDE.fit(dataset)
            >>> [rule.to_string() for rule in rules]
            ['y(0) :- a(0), b(0).', 'y(1) :- a(1).', 'y(1) :- b(1).']
        """
        # First, so a DataFrame passed with options gets the same error.
        _check_dataset(dataset)
        _check_flag(try_all_conditions, "try_all_conditions")
        if min_support is not None:
            PRIDE._check_min_support(min_support)
        if min_possible_support is not None:
            PRIDE._check_min_support(min_possible_support, "min_possible_support")
        if allowed_conditions is not None:
            allowed_conditions = PRIDE._check_allowed_conditions(
                dataset, allowed_conditions
            )
        learn = partial(
            PRIDE._learn_head,
            allowed_conditions=allowed_conditions,
            min_support=min_support,
            min_possible_support=min_possible_support,
            try_all_conditions=try_all_conditions,
        )
        return PRIDE._fit_heads(dataset, learn, heads, verbose, warn_uncovered, n_jobs)

    @staticmethod
    def _learn_head(
        dataset: Dataset,
        target: str,
        value: str,
        verbose: bool,
        *,
        allowed_conditions: AllowedConditions | None,
        min_support: int | float | None,
        min_possible_support: int | float | None,
        try_all_conditions: bool,
    ) -> tuple[list[Rule], list[Uncovered]]:
        """Learns one head for ``fit``'s shared loop, with its uncovered rows.

        Args:
            dataset: The dataset to learn from.
            target: The head's target.
            value: The head's value.
            verbose: If True, prints this head's progress to stderr.
            allowed_conditions: ``fit``'s argument, already validated.
            min_support: ``fit``'s argument, already validated.
            min_possible_support: ``fit``'s argument, already validated.
            try_all_conditions: ``fit``'s argument.

        Returns:
            ``_fit_target_value``'s rules and uncovered rows.
        """
        allowed = (
            None
            if allowed_conditions is None
            else allowed_conditions.get((target, value))
        )
        return PRIDE._fit_target_value(
            dataset,
            target,
            value,
            verbose=verbose,
            pool=(
                None
                if allowed is None
                else set(PRIDE._expand_allowed(allowed, dataset))
            ),
            min_support=min_support,
            min_possible_support=min_possible_support,
            try_all_conditions=try_all_conditions,
        )

    @staticmethod
    def _fit_target_value(  # noqa: C901
        dataset: Dataset,
        target: str,
        value: str,
        verbose: bool = False,
        pool: set[tuple[str, str]] | None = None,
        min_support: int | float | None = None,
        min_possible_support: int | float | None = None,
        try_all_conditions: bool = False,
    ) -> tuple[list[Rule], list[Uncovered]]:
        """Learns a correct, minimal cover of target=value's positives.

        Repeatedly seeds a new rule on the first remaining (not yet
        covered) positive example, in row order, builds a consistent,
        minimal rule around it, then drops every remaining positive it
        covers, until none remain. A seed that gets no rule is set aside
        with the reason; it can still be covered by a later seed's rule.
        With ``try_all_conditions``, the seed's extra rules are built
        against the same remaining positives and set aside what they
        cover too.

        Args:
            dataset: The classification dataset to learn from.
            target: The target variable to conclude on.
            value: The value of ``target`` the learned rules conclude.
            verbose: If True, prints progress for this (target, value)
                pair to stderr.
            pool: The ``(feature, value)`` conditions allowed for this
                head, or ``None`` for every condition.
            min_support: ``fit``'s argument, already validated.
            min_possible_support: ``fit``'s argument, already validated.
            try_all_conditions: ``fit``'s argument.

        Returns:
            The rules, each consistent with every negative example and
            minimal, and the positive rows none of them matches, each
            with its head and the reason its seed got no rule.
        """
        start = time.perf_counter()
        features = dataset.features
        positives = dataset._positive_states(target, value)
        negatives = PRIDE._distinct(dataset._negative_states(target, value), features)
        thresholds = (
            0
            if min_support is None
            else PRIDE._resolve_min_support(min_support, len(positives)),
            0
            if min_possible_support is None
            else PRIDE._resolve_min_support(min_possible_support, len(positives)),
        )
        threshold_cause = " or ".join(
            name
            for name, option in (
                ("min_support", min_support),
                ("min_possible_support", min_possible_support),
            )
            if option is not None
        )
        # A seed backs its own rule, so a threshold of 1 always holds;
        # one above the head's positive count never does.
        check_support = max(thresholds) > 1
        impossible = max(thresholds) > len(positives)

        remaining: list[_State] = list(positives)
        rules: list[Rule] = []
        skipped: list[tuple[_State, str]] = []
        # A seed whose plain build fails fails the same way whatever is
        # left to cover (some negative agrees with it on every condition
        # it may take), and so do its extras: its copies are not
        # rebuilt.
        failed: dict[tuple[str | None, ...], str] = {}
        total = len(positives)
        progress = PRIDE._progress(f"{target}={value}: covering positives", total)

        def seed_rule(
            seed: _State, forced: tuple[str, str] | None = None
        ) -> Rule | str:
            """The seed's rule, or why it has none."""
            rule = PRIDE._covering_rule(
                target,
                value,
                seed,
                negatives,
                features,
                pool,
                uncovered=remaining,
                forced=forced,
            )
            if rule is None:
                return (
                    PRIDE._UNKNOWN_CAUSE
                    if PRIDE._inseparable(seed, negatives, features)
                    else "allowed_conditions"
                )
            if impossible:
                return threshold_cause
            if check_support and not PRIDE._meets(rule, positives, thresholds):
                rule = PRIDE._covering_rule(
                    target,
                    value,
                    seed,
                    negatives,
                    features,
                    pool,
                    positives,
                    thresholds,
                    uncovered=remaining,
                    forced=forced,
                )
                # A rebuilt rule always meets the thresholds: each
                # condition added keeps support there, dropping
                # conditions only raises it, and the empty-body rule
                # matches every positive, a count `impossible` has
                # checked. Only `None` falls short.
                if rule is None:
                    return threshold_cause
            return rule

        while remaining:
            seed = remaining.pop(0)
            key = tuple(seed[feature] for feature in features)
            if key in failed:
                skipped.append((seed, failed[key]))
                if verbose:
                    progress.update(total - len(remaining))
                continue
            new: list[Rule] = []
            rule = seed_rule(seed)
            if isinstance(rule, str):
                skipped.append((seed, rule))
                if rule in (PRIDE._UNKNOWN_CAUSE, "allowed_conditions"):
                    failed[key] = rule
            else:
                new.append(rule)
            if try_all_conditions:
                for feature in features:
                    known = seed[feature]
                    if known is None or (
                        pool is not None and (feature, known) not in pool
                    ):
                        continue
                    extra = seed_rule(seed, (feature, known))
                    if not isinstance(extra, str):
                        new.append(extra)
            # Distinct: an extra can equal the default rule. None is in
            # `rules` already, since every one matches the seed, which
            # no earlier rule matches.
            new = list(dict.fromkeys(new))
            rules.extend(new)
            if new:
                remaining = [
                    state
                    for state in remaining
                    if not any(rule.matches(state) for rule in new)
                ]

            if verbose:
                progress.update(total - len(remaining))

        uncovered = [
            ((target, value), seed, cause)
            for seed, cause in skipped
            if not any(rule.matches(seed) for rule in rules)
        ]

        if verbose:
            elapsed = time.perf_counter() - start
            PRIDE._log(
                f"{target}={value}: {total} positives, {len(rules)} rules kept, "
                f"{len(uncovered)} positives uncovered ({duration(elapsed)} total)"
            )

        return rules, uncovered

    @staticmethod
    def _support(rule: Rule, positives: Sequence[_State]) -> tuple[int, int]:
        """Counts the positives a rule matches, exactly and possibly.

        One condition at a time, each pass over the rows the previous
        ones kept.

        Args:
            rule: The rule to count for.
            positives: The head's positive examples.

        Returns:
            The rows it matches, and the rows it could match: each
            condition holds or falls on an unknown.
        """
        exact = possible = positives
        for feature, wanted in rule.body.items():
            exact = [state for state in exact if state[feature] == wanted]
            possible = [state for state in possible if state[feature] in (wanted, None)]
        return len(exact), len(possible)

    @staticmethod
    def _meets(
        rule: Rule, positives: Sequence[_State], thresholds: tuple[int, int]
    ) -> bool:
        """Tells whether a rule meets both support thresholds.

        Args:
            rule: The rule to check.
            positives: The head's positive examples.
            thresholds: The exact and the possible support required.

        Returns:
            True if both counts reach their threshold.
        """
        exact, possible = PRIDE._support(rule, positives)
        return exact >= thresholds[0] and possible >= thresholds[1]

    @staticmethod
    def _covering_rule(  # noqa: C901
        target: str,
        value: str,
        seed: _State,
        negatives: Sequence[_State],
        features: list[str],
        pool: set[tuple[str, str]] | None = None,
        positives: Sequence[_State] | None = None,
        thresholds: tuple[int, int] = (0, 0),
        uncovered: Sequence[_State] = (),
        forced: tuple[str, str] | None = None,
    ) -> Rule | None:
        """Builds one minimal consistent rule around ``seed``.

        Starts from the fully general rule for ``(target, value)`` (or
        the one with only ``forced``) and, each time it matches a
        negative example, adds one condition: a feature where the seed's
        value is known, allowed by ``pool``, and differs from the
        negative's, fixed to the seed's value. Of those, the one leaving
        the rule matching the most of ``uncovered``; ties, and every
        choice when the rule matches none of them, go to the first in
        ``features``. The rule then excludes that negative and still
        matches ``seed``. A feature the rule already constrains never
        qualifies: every condition holds the seed's value, and the rule
        matches the negative, so the two agree there. Then drops every
        condition that turns out not to be necessary (``_minimize``),
        ``forced`` last.

        With ``positives``, a condition also has to keep the rule's
        support, exact and possible, at ``thresholds`` or above.
        Dropping conditions only raises support, so the minimized rule
        keeps it.

        Args:
            target: The target variable to conclude on.
            value: The value of ``target`` the rule concludes.
            seed: The positive example the rule is built around; it
                matches ``seed`` throughout and never stops.
            negatives: The negative examples to stay consistent with.
            features: The features conditions may be added on, tried in
                this order.
            pool: The allowed ``(feature, value)`` conditions, or
                ``None`` for every condition.
            positives: The head's positive examples, to keep support at
                ``thresholds``; ``None`` to ignore support.
            thresholds: The exact and the possible support to keep.
            uncovered: The positives still to cover (not covered by a
                rule yet, nor tried as a seed), which each
                condition is chosen to keep matching.
            forced: A condition on the seed's value, in the rule from
                the start and the last one tried for dropping; ``None``
                for none.

        Returns:
            A rule concluding ``(target, value)``, matching ``seed``,
            consistent with every entry in ``negatives``, and minimal;
            or ``None`` if some negative leaves no condition to add, or
            ``forced`` alone takes support below ``thresholds``.
        """
        # The conditions the rule may add: the seed's known values.
        candidates: list[tuple[str, str]] = []
        for feature in features:
            known = seed[feature]
            if known is not None and (pool is None or (feature, known) in pool):
                candidates.append((feature, known))
        # The positives the rule matches exactly, and possibly, and the
        # uncovered ones it matches.
        exact = possible = list(range(len(positives or ())))
        matched = list(uncovered)

        def supported(feature: str, wanted: str) -> tuple[list[int], list[int]] | None:
            """``exact`` and ``possible`` with the condition, or ``None``.

            ``None`` when either count would drop below ``thresholds``.
            """
            if positives is None:
                return exact, possible
            kept = (
                [i for i in exact if positives[i][feature] == wanted],
                [i for i in possible if positives[i][feature] in (wanted, None)],
            )
            if len(kept[0]) < thresholds[0] or len(kept[1]) < thresholds[1]:
                return None
            return kept

        rule = Rule(head=(target, value))
        if forced is not None:
            kept = supported(*forced)
            if kept is None:
                return None
            exact, possible = kept
            matched = [state for state in matched if state[forced[0]] == forced[1]]
            rule = rule.with_condition(*forced)
        for negative in negatives:
            if not rule.matches(negative):
                continue
            best = None
            best_score = -1
            for feature, wanted in candidates:
                if negative[feature] == wanted:
                    continue
                kept = supported(feature, wanted)
                if kept is None:
                    continue
                score = sum(state[feature] == wanted for state in matched)
                if score > best_score:
                    best, best_score = (feature, wanted, kept), score
                if not matched:
                    break
            if best is None:
                return None
            feature, wanted, (exact, possible) = best
            matched = [state for state in matched if state[feature] == wanted]
            rule = rule.with_condition(feature, wanted)
        return PRIDE._minimize(
            rule, negatives, features, last=None if forced is None else forced[0]
        )

    @staticmethod
    def _minimize(
        rule: Rule,
        negatives: Sequence[_State],
        features: Sequence[str],
        last: str | None = None,
    ) -> Rule:
        """Drops every condition a consistent rule doesn't need.

        Tries dropping each condition in ``features`` order, the one on
        ``last`` at the end, keeping the drop when the rule still
        matches no negative. Not body order: that is alphabetical, so
        renaming the variables would change which redundant condition
        goes. One pass over the conditions is enough: dropping
        conditions only makes the rule more general, so a condition that
        was needed stays needed.

        Most tries are settled without rescanning. A condition some
        negative violates alone (``_needed_conditions``) is needed in
        every rule this loop produces, so it is never tried. And since
        those conditions are never dropped, a negative violating one of
        them stays unmatched: only the negatives meeting all of them are
        rescanned. The result is the same as trying every condition
        against every negative.

        Args:
            rule: A rule matching no entry of ``negatives``.
            negatives: The negative examples the rule must stay
                consistent with.
            features: The order to try the conditions in, every
                variable of ``rule`` included.
            last: A feature whose condition is tried last, or ``None``.

        Returns:
            The rule without the conditions it doesn't need.
        """
        needed = PRIDE._needed_conditions(rule, negatives)
        assert needed is not None, "_minimize needs a consistent rule"
        if len(needed) == len(rule.body):
            return rule
        position = {feature: i for i, feature in enumerate(features)}
        conditions = sorted(
            rule.body.items(), key=lambda c: (c[0] == last, position[c[0]])
        )
        kept = [(feature, value) for feature, value in conditions if feature in needed]
        live = [
            negative
            for negative in negatives
            if all(negative[feature] == value for feature, value in kept)
        ]
        for feature, _ in conditions:
            if feature in needed:
                continue
            shorter = rule.without_condition(feature)
            if not any(shorter.matches(negative) for negative in live):
                rule = shorter
        return rule
