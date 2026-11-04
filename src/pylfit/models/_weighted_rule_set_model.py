"""``WeightedRuleSetModel``: a ``RuleSetModel`` with per-rule support counts.

Each rule carries ``(positives, negatives)`` counts from a dataset, and
``predict_likelihood``, ``predict_proba`` and ``predict`` weigh matching
rules by them.

Build one from an algorithm's output and the dataset it learned from,
``WeightedRuleSetModel(dataset, GULA.fit(dataset))``, or from an
existing model's rules, ``WeightedRuleSetModel(dataset, model.rules)``.

Why it works this way: :doc:`/explanation/about-models`.
"""

import math
import time
from collections.abc import Iterable, Sequence
from typing import Any, cast, overload

import pandas as pd

from pylfit._dataset import Dataset, _check_flag
from pylfit._progress import Progress, duration, log
from pylfit._rule import Rule, State, _is_unknown
from pylfit.models._helpers import (
    Unknowns,
    check_state,
    check_unknowns,
    match_weight,
)
from pylfit.models._rule_set_model import RuleSetModel

#: A value's matching rules' ``(positives, negatives)``: whole counts,
#: or fractional ones when a fallback prediction weighs a rule by its
#: probability of matching.
_RuleScores = Sequence[tuple[float, float]]


def _vote(rule_scores: _RuleScores) -> float:
    """Each rule votes ``p * p / (p + n)``; the votes are summed.

    Support times precision, so a rule's weight grows with how much it
    covers and shrinks with how often it is wrong. With no negatives
    this is exactly ``sum(p)``, an ordering by raw support. A large,
    slightly impure value outranks a tiny clean one.

    Why not a purity measure, such as the Laplace estimate
    ``(p + 1) / (p + n + k)`` on the pooled counts that CN2 ranks its
    rules by ([Clark1991]_, Eq. 1): purity is blind to how much
    evidence there is once ``n > 0``, so a pooled ``(8, 0)`` outranks a
    pooled ``(68, 7)``. Rules with negatives, learned from conflicting
    data or made more general than the data allows, then predict a
    little worse: over ten random 80/20 splits of the UCI breast cancer
    data, ``GULA``'s rules score 0.71 under the Laplace estimate and
    0.73 under this vote. Where no
    rule has negatives, the two pick the same values for a state
    without unknowns, and this vote's ``predict_proba`` shares are the
    better probabilities (a lower Brier score on the UCI heart disease
    data).

    Scaling a rule's counts scales its vote by the same factor, so a
    fallback rule, whose counts are multiplied by its chance of
    matching, votes in proportion to that chance. With no negatives,
    summing per rule equals pooling the counts first.

    References:
        .. [Clark1991] Clark, P. & Boswell, R. (1991). Rule induction
           with CN2: some recent improvements. In Machine Learning —
           EWSL-91, 151-163. https://doi.org/10.1007/BFb0017011
    """
    return math.fsum(p * p / (p + n) for p, n in rule_scores)


def _counts(positives: object, negatives: object) -> tuple[int, int]:
    """Validates a rule's support counts and returns them as a pair.

    Args:
        positives: Rows the body matches whose target has the head
            value.
        negatives: Rows the body matches whose target has another value.

    Returns:
        ``(positives, negatives)``.

    Raises:
        TypeError: If a count is not an ``int`` (a ``bool`` is not).
        ValueError: If a count is negative.
    """
    for name, count in (("positives", positives), ("negatives", negatives)):
        if not isinstance(count, int) or isinstance(count, bool):
            raise TypeError(f"{name} must be an int, got {count!r}")
        if count < 0:
            raise ValueError(f"{name} must be >= 0, got {count}")
    return cast(int, positives), cast(int, negatives)


def _count(rule: Rule, dataset: Dataset) -> tuple[int, int]:
    """Counts a rule's ``(positives, negatives)`` rows in ``dataset``.

    What each count means: ``WeightedRuleSetModel.get_score``.

    Args:
        rule: The rule to count.
        dataset: The dataset to count it in.

    Returns:
        The ``(positives, negatives)`` pair.
    """
    target, value = rule.head
    positives = sum(
        1 for state in dataset._positive_states(target, value) if rule.matches(state)
    )
    # Every body-matching row whose target holds some *other* value.
    # ``Dataset.positives`` partitions the rows by target value and
    # caches each partition per ``(target, value)``, so this reuses that
    # cache and never builds a per-row Series — unlike iterating
    # ``dataset`` itself.
    negatives = sum(
        1
        for other in dataset.domain(target)
        if other != value
        for state in dataset._positive_states(target, other)
        if rule.matches(state)
    )
    return (positives, negatives)


class WeightedRuleSetModel(RuleSetModel):
    """A ``RuleSetModel`` with per-rule support counts.

    The counts turn matches into likelihoods. Every rule of the model
    has them, counted in the dataset the model is built from;
    ``get_score`` reads one. ``features``, ``targets``, ``rules`` and
    ``domain`` are inherited from ``RuleSetModel``. Equality also
    compares the counts and ``value_frequencies``; a model is not
    hashable, as there.

    Attributes:
        value_frequencies: For each feature, each value's share among
            the known values of that feature in the dataset the scores
            came from. The fallback prediction on a state with unknown
            values weighs a rule by them. A feature absent from it
            counts each value as equally likely. A value outside the
            feature's current ``domain``, which no rule can test, is
            left out. Read-only.

    Examples:
        >>> import pandas as pd
        >>> from pylfit import GULA, Dataset, WeightedRuleSetModel
        >>> df = pd.DataFrame(
        ...     {
        ...         "a": ["0", "0", "1", "1"],
        ...         "b": ["0", "1", "0", "1"],
        ...         "y": ["0", "1", "1", "1"],
        ...     }
        ... )
        >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
        >>> model = WeightedRuleSetModel(dataset, GULA.fit(dataset))
        >>> print(model)
        features: ['a', 'b']
        targets: ['y']
        rules:
          y(0) :- a(0), b(0).  (+1/-0)
          y(1) :- a(1).  (+2/-0)
          y(1) :- b(1).  (+2/-0)
    """

    _SCHEMA_VERSION = 2

    def __init__(
        self,
        dataset: Dataset,
        rules: Iterable[Rule],
        *,
        verbose: bool = False,
    ) -> None:
        """Builds a scored model from a dataset and a set of rules.

        Args:
            dataset: The dataset the rules come from and are scored
                against.
            rules: The rules of the model. Duplicates are collapsed.
                Pass an empty iterable for a model without rules; there
                is no default.
            verbose: If True, prints progress to stderr while scoring,
                at most once a second. Scoring runs in this one process.

        Raises:
            TypeError: If ``dataset`` is not a ``Dataset``, ``rules`` is
                not iterable or is a single ``str``, a rule is not a
                ``Rule``, or ``verbose`` is not a ``bool``.
            ValueError: If a rule names a variable or value outside the
                dataset's features, targets and domain.
        """
        super().__init__(dataset, rules)
        _check_flag(verbose, "verbose")
        self._rule_scores: dict[Rule, tuple[int, int]] = {}
        self._value_frequencies: dict[str, dict[str, float]] = {}
        self._score_all_rules(dataset, verbose=verbose)

    # --- Read-only views ---

    @property
    def value_frequencies(self) -> dict[str, dict[str, float]]:
        """Per feature, each value's share among known values, as a copy."""
        return self._shares_in_domain()

    def _shares_in_domain(self) -> dict[str, dict[str, float]]:
        """``value_frequencies`` without the values the domain lacks.

        The domain can be narrowed after scoring; the shares of the
        values it lost stay stored, so widening it again restores them,
        but no view, saved file or comparison includes them.
        """
        return {
            feature: {
                value: share
                for value, share in shares.items()
                if value in self._domain[feature]
            }
            for feature, shares in self._value_frequencies.items()
        }

    # --- Prediction ---

    @overload
    def predict_likelihood(
        self,
        state: pd.DataFrame,
        *,
        unknowns: Unknowns = "fallback",
    ) -> list[dict[str, dict[str, float]]]: ...

    @overload
    def predict_likelihood(
        self,
        state: State,
        *,
        unknowns: Unknowns = "fallback",
    ) -> dict[str, dict[str, float]]: ...

    def predict_likelihood(
        self,
        state: State | pd.DataFrame,
        *,
        unknowns: Unknowns = "fallback",
    ) -> dict[str, dict[str, float]] | list[dict[str, dict[str, float]]]:
        """Predicts a raw, unnormalized likelihood for each target value.

        For each value, each rule that concludes it and matches
        ``state`` votes ``p * p / (p + n)``, support times precision,
        from its positives ``p`` and negatives ``n``, and the votes are
        summed; a rule with no positives doesn't count. A value's
        negatives lower its score in proportion to its support, so a
        large, slightly noisy value can outrank a small clean one. On
        deterministic data, ``BruteForce``, ``GULA`` and ``PRIDE`` learn
        rules with no negatives, and the vote is then the matching
        rules' pooled positives.

        An unknown feature value (``None``) never satisfies a condition,
        so a rule testing it doesn't match. With
        ``unknowns="fallback"``, the default, a target that no rule with
        positives matches that way is predicted from the rules
        that *could* match instead, each weighed by the probability that
        it does: its counts are multiplied by the share, in
        ``value_frequencies``, of each value its conditions on unknowns
        require, the features taken as independent. The weighted counts
        are then voted as above, so a rule that almost surely doesn't
        match adds almost nothing. Matching strictly first keeps every
        prediction a strict match can make, and falls back only where it
        would otherwise give nothing. ``unknowns="strict"`` skips the
        fallback, leaving such a target empty. On a state without
        unknowns both give the same result.

        Args:
            state: One state, as a mapping of feature to ``str`` value
                (``None`` where unknown), where keys that are not
                features are ignored as in ``predict_candidates``; or a
                DataFrame with one row per state, checked and converted
                as ``predict`` checks its input, which refuses a column
                that is not a feature.
            unknowns: ``"fallback"`` or ``"strict"``, as above.

        Returns:
            A mapping of target to (value -> likelihood), values in
            sorted order: unbounded non-negative weights, comparable
            only within a target. A target with no entries had no
            matching rule with positives, nor under ``"fallback"`` one
            that could match. Several entries mean rules for several
            values matched. For a DataFrame, a list of those, one per
            row, in row order.

        Raises:
            KeyError: If ``state`` is a mapping that lacks a feature.
            TypeError: If ``state`` is neither a mapping nor a
                DataFrame, or a mapping with a known value that is not a
                ``str``.
            ValueError: If ``unknowns`` is not a known mode, a value of
                ``state`` is outside its feature's domain, or ``state``
                is a DataFrame that ``predict`` would reject: a missing,
                target or extra column, or a column of a type
                ``predict`` refuses.

        Examples:
            >>> import pandas as pd
            >>> from pylfit import GULA, Dataset, WeightedRuleSetModel
            >>> df = pd.DataFrame(
            ...     {
            ...         "a": ["0", "0", "1", "1"],
            ...         "b": ["0", "1", "0", "1"],
            ...         "y": ["0", "1", "1", "1"],
            ...     }
            ... )
            >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
            >>> model = WeightedRuleSetModel(dataset, GULA.fit(dataset))
            >>> model.predict_likelihood({"a": "1", "b": "1"})
            {'y': {'1': 4.0}}
        """
        check_unknowns(unknowns)
        if isinstance(state, pd.DataFrame):
            return [self._likelihood(row, unknowns) for row in self._states(state)]
        check_state(state, self._features, self._domain)
        return self._likelihood(state, unknowns)

    def _likelihood(
        self, state: State, unknowns: Unknowns
    ) -> dict[str, dict[str, float]]:
        """``predict_likelihood`` on one state already checked."""
        check_unknowns(unknowns)

        predictions: dict[str, dict[str, float]] = {
            target: {} for target in self._targets
        }
        # Per-rule scores kept as a list rather than summed on the way
        # in: the vote is taken per rule, so summing first would erase
        # exactly the information it needs.
        supports: dict[str, dict[str, list[tuple[float, float]]]] = {
            target: {} for target in self._targets
        }

        for rule in self._rules:
            variable, value = rule.head
            if variable not in predictions or not rule.matches(state):
                continue
            positives, negatives = self._rule_scores[rule]
            if positives == 0:
                continue
            supports[variable].setdefault(value, []).append((positives, negatives))

        unmatched = {target for target, found in supports.items() if not found}
        if (
            unknowns == "fallback"
            and unmatched
            and any(_is_unknown(state.get(feature)) for feature in self._features)
        ):
            frequencies = {
                feature: self._value_frequencies.get(feature)
                or {value: 1 / len(values) for value in values}
                for feature, values in (
                    (feature, self._domain.get(feature, set()))
                    for feature in self._features
                )
            }
            for rule in self._rules:
                variable, value = rule.head
                score = self._rule_scores[rule]
                if variable not in unmatched or score[0] == 0:
                    continue
                weight = match_weight(rule, state, frequencies)
                if weight > 0:
                    positives, negatives = score
                    supports[variable].setdefault(value, []).append(
                        (weight * positives, weight * negatives)
                    )

        for variable, value_supports in supports.items():
            # Sorted: values arrive in the order of `rules`, a set,
            # which varies between processes, and `_proba` keeps the
            # value order. The scores need no sorting: `_vote` sums
            # with `math.fsum`, which rounds once whatever the order.
            for value, rule_scores in sorted(value_supports.items()):
                predictions[variable][value] = _vote(rule_scores)

        return predictions

    @overload
    def predict_proba(
        self,
        state: pd.DataFrame,
        *,
        unknowns: Unknowns = "fallback",
    ) -> list[dict[str, dict[str, float]]]: ...

    @overload
    def predict_proba(
        self,
        state: State,
        *,
        unknowns: Unknowns = "fallback",
    ) -> dict[str, dict[str, float]]: ...

    def predict_proba(
        self,
        state: State | pd.DataFrame,
        *,
        unknowns: Unknowns = "fallback",
    ) -> dict[str, dict[str, float]] | list[dict[str, dict[str, float]]]:
        """Predicts a probability distribution over each target's values.

        The evidence of ``predict_likelihood``, normalized to sum to
        ``1`` for each target. A target for which no rule with positives
        matches, nor under ``"fallback"`` could match, is uniform
        over its whole domain, as in
        ``RuleSetModel.predict_candidates``; ``predict_likelihood``
        returns nothing for it.

        The shares rank the values the matching rules conclude against
        each other; they are not calibrated probabilities. A value no
        matching rule concludes gets ``0.0``, so a value whose only
        matching rule is often wrong still gets ``1.0`` when nothing
        else matches: read its rule's score, or ``predict_likelihood``,
        for how much evidence backs it.

        Args:
            state: One state, as a mapping of feature to ``str`` value,
                where keys that are not features are ignored as in
                ``predict_candidates``; or a DataFrame with one row per
                state, checked and converted as ``predict`` checks its
                input, which refuses a column that is not a feature.
            unknowns: ``"fallback"`` or ``"strict"``, passed to
                ``predict_likelihood``.

        Returns:
            A mapping of target to (value -> probability), summing to
            ``1`` for each target, with every value of the target's
            domain, in sorted order, ``0.0`` where there is no evidence.
            Empty only for a target whose domain is empty. For a
            DataFrame, a list of those, one per row, in row order.

        Raises:
            KeyError: If ``state`` is a mapping that lacks a feature.
            TypeError: If ``state`` is neither a mapping nor a
                DataFrame, or a mapping with a known value that is not a
                ``str``.
            ValueError: If ``unknowns`` is not a known mode, a value of
                ``state`` is outside its feature's domain, or ``state``
                is a DataFrame that ``predict`` would reject: a missing,
                target or extra column, or a column of a type
                ``predict`` refuses.

        Examples:
            >>> import pandas as pd
            >>> from pylfit import GULA, Dataset, WeightedRuleSetModel
            >>> df = pd.DataFrame(
            ...     {
            ...         "a": ["0", "0", "1", "1"],
            ...         "b": ["0", "1", "0", "1"],
            ...         "y": ["0", "1", "1", "1"],
            ...     }
            ... )
            >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
            >>> model = WeightedRuleSetModel(dataset, GULA.fit(dataset))
            >>> model.predict_proba({"a": "1", "b": "1"})
            {'y': {'0': 0.0, '1': 1.0}}

            With ``a`` unknown and ``b=1``, ``y(1) :- b(1).`` still
            matches. With ``b=0`` no rule matches for sure, but
            ``y(1) :- a(1).`` and ``y(0) :- a(0), b(0).`` each could,
            and their counts are weighed by the share of ``a``'s value,
            ``0.5`` for each. ``"strict"`` finds no match and returns
            the uniform distribution:

            >>> model.predict_proba({"a": None, "b": "1"})
            {'y': {'0': 0.0, '1': 1.0}}
            >>> proba = model.predict_proba({"a": None, "b": "0"})
            >>> {value: round(p, 2) for value, p in sorted(proba["y"].items())}
            {'0': 0.33, '1': 0.67}
            >>> model.predict_proba({"a": None, "b": "0"}, unknowns="strict")
            {'y': {'0': 0.5, '1': 0.5}}
        """
        return super().predict_proba(state, unknowns=unknowns)

    def _proba(self, state: State, unknowns: Unknowns) -> dict[str, dict[str, float]]:
        """Normalizes ``_likelihood``; see ``predict_proba``."""
        likelihoods = self._likelihood(state, unknowns)
        probabilities: dict[str, dict[str, float]] = {}
        for target, value_likelihoods in likelihoods.items():
            if not value_likelihoods:
                domain_values = sorted(self._domain.get(target, set()))
                if not domain_values:
                    probabilities[target] = {}
                    continue
                uniform = 1 / len(domain_values)
                probabilities[target] = dict.fromkeys(domain_values, uniform)
                continue
            total = math.fsum(value_likelihoods.values())
            if total == 0:  # votes too small for a float, weighed by chance
                uniform = 1 / len(value_likelihoods)
                shares = dict.fromkeys(value_likelihoods, uniform)
            else:
                shares = {
                    value: likelihood / total
                    for value, likelihood in value_likelihoods.items()
                }
            # Every domain value, 0.0 where there is no evidence,
            # sorted.
            probabilities[target] = {
                value: shares.get(value, 0.0)
                for value in sorted(self._domain.get(target, set()))
            }
        return probabilities

    # ``predict`` itself is inherited unchanged from ``RuleSetModel`` —
    # it's written generically against ``_proba``, which is all this
    # class overrides (evidence-weighted here, uniform there).

    # --- Scores ---

    def _score_all_rules(self, dataset: Dataset, verbose: bool = False) -> None:
        """Scores every rule currently in the model against ``dataset``.

        Also captures ``value_frequencies``.
        Called once, from ``__init__``.

        A plain per-rule loop over ``_count``. Both of its terms read
        from ``Dataset.positives``'s internal per-``(target, value)``
        cache (the ``negatives`` term sums the other-value partitions),
        so no per-row ``Series`` is built; what is *not* cached is the
        combined wrong-value count per head, re-summed for every rule
        sharing it.

        Args:
            dataset: The dataset to score every rule in ``self.rules``
                against.
            verbose: If ``True``, prints throttled progress (at most
                once per second, plus a final line) while scoring.
        """
        total = len(self._rules)
        tag = type(self).__name__
        progress = Progress(tag, "scoring rules", total)
        for scored, rule in enumerate(self._rules, start=1):
            self._rule_scores[rule] = _count(rule, dataset)
            if verbose:
                progress.update(scored)

        self._value_frequencies = {
            feature: {
                str(value): float(share)
                for value, share in dataset._data[feature]
                .value_counts(normalize=True)
                .items()
            }
            for feature in self._features
        }

        if verbose:
            elapsed = time.perf_counter() - progress.start
            log(tag, f"{total} rules scored in {duration(elapsed)}")

    def get_score(self, rule: Rule) -> tuple[int, int] | None:
        """Returns a rule's ``(positives, negatives)`` counts, if in the model.

        ``positives`` counts the rows the body matches whose target has
        the rule's head value; ``negatives`` counts the rows the body
        matches whose target has another value. A row repeated in the
        dataset counts once per occurrence. On non-deterministic data
        this differs from ``Dataset.negatives``, which leaves out any
        feature state also seen with the head value: here, every row the
        rule gets wrong counts.

        Counts are exact: a row whose value is unknown for a feature the
        rule tests is not known to match, so it counts for neither side,
        and a row whose target is unknown counts for no rule. Counting
        such a row by its probability of matching, as the fallback
        prediction weighs rules, was tested on Boolean networks and the
        car evaluation data, and did no better overall. The known cost
        of exact counts: when one target value's rows have more unknowns
        than the others', as when a value goes unrecorded because of the
        outcome, that value's rules match fewer rows, so its predictions
        are under-weighted.

        Args:
            rule: The rule to look up.

        Returns:
            The counts, or ``None`` if ``rule`` is not a rule of the
            model.

        Raises:
            TypeError: If ``rule`` is not a ``Rule``.

        Examples:
            >>> import pandas as pd
            >>> from pylfit import Dataset, Rule, WeightedRuleSetModel
            >>> df = pd.DataFrame(
            ...     {
            ...         "a": ["0", "0", "1", "1"],
            ...         "b": ["0", "1", "0", "1"],
            ...         "y": ["0", "1", "1", "1"],
            ...     }
            ... )
            >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
            >>> rule = Rule.from_string("y(1) :- a(1).")
            >>> model = WeightedRuleSetModel(dataset, [rule])
            >>> model.get_score(rule)
            (2, 0)
            >>> model.get_score(Rule.from_string("y(0) :- .")) is None
            True
        """
        if not isinstance(rule, Rule):
            raise TypeError(f"rule must be a Rule, got {type(rule).__name__}")
        return self._rule_scores.get(rule)

    # --- Persistence ---

    def _rule_payload(self, rule: Rule) -> dict[str, Any]:
        """Extends ``RuleSetModel``'s rule payload with a ``score`` key."""
        payload = super()._rule_payload(rule)
        payload["score"] = list(self._rule_scores[rule])
        return payload

    def _extra_save_payload(self) -> dict[str, Any]:
        """Adds ``value_frequencies`` to the saved file."""
        return {
            "value_frequencies": {
                feature: dict(sorted(shares.items()))
                for feature, shares in self._shares_in_domain().items()
            },
        }

    def _init_extra_state(self) -> None:
        """Initializes the scores and ``value_frequencies`` before a load."""
        self._rule_scores = {}
        self._value_frequencies = {}

    def _load_rule(self, entry: dict[str, Any]) -> Rule:
        """Extends ``RuleSetModel``'s rule loading with the stored score.

        Every rule must have one, as in the model; ``_counts`` checks
        it.
        """
        rule = super()._load_rule(entry)
        score = entry["score"]
        if score is None:
            raise ValueError(f"rule {rule} has no score; every rule needs one")
        if not isinstance(score, list) or len(score) != 2:
            raise TypeError(
                f"rule {rule}'s score must be a [positives, negatives] list, "
                f"got {score!r}"
            )
        counts = _counts(*score)
        if self._rule_scores.get(rule, counts) != counts:
            raise ValueError(
                f"rule {rule} is listed twice, with scores "
                f"{list(self._rule_scores[rule])} and {score}"
            )
        self._rule_scores[rule] = counts
        return rule

    def _load_extra(self, payload: dict[str, Any]) -> None:
        """Restores ``value_frequencies`` from a load.

        A file without ``value_frequencies``, written before it existed,
        loads with none: each value then counts as equally likely. A
        share for a value outside the feature's domain, which ``save``
        never writes, is ignored.

        Args:
            payload: The loaded file, its ``domain`` already set.

        Raises:
            TypeError: If ``value_frequencies`` is not a mapping of
                feature to a mapping of value to share.
            ValueError: If it names a variable that is not a feature, or
                gives a share that is not a number in ``[0, 1]``.
        """
        given = payload.get("value_frequencies", {})
        if not isinstance(given, dict) or not all(
            isinstance(shares, dict) for shares in given.values()
        ):
            raise TypeError(
                "file's 'value_frequencies' must map each feature to a "
                f"mapping of value to share, got {given!r}"
            )
        for feature, shares in given.items():
            if feature not in self._features:
                raise ValueError(
                    f"file's value_frequencies names {feature!r}, not a feature"
                )
            for value, share in shares.items():
                if (
                    isinstance(share, bool)
                    or not isinstance(share, int | float)
                    or not 0 <= share <= 1
                ):
                    raise ValueError(
                        f"file's value_frequencies gives {feature}={value} the "
                        f"share {share!r}, not a number in [0, 1]"
                    )
        self._value_frequencies = {
            feature: dict(shares) for feature, shares in given.items()
        }
        self._value_frequencies = self._shares_in_domain()

    # --- Comparison operators ---

    def __eq__(self, other: object) -> bool:
        """Checks whether two models are equal in every attribute.

        Compares ``features``, ``targets``, ``rules``, the scores,
        ``value_frequencies`` and ``domain`` — deliberately more than
        just ``rules``, so two equal models are guaranteed to predict
        identically for any state. Models of different classes are never
        equal.
        """
        if not isinstance(other, RuleSetModel):
            return NotImplemented
        return (
            isinstance(other, WeightedRuleSetModel)
            and super().__eq__(other)
            and self._rule_scores == other._rule_scores
            and self._shares_in_domain() == other._shares_in_domain()
        )

    # --- String representation ---

    def _rule_line(self, rule: Rule) -> str:
        """Extends ``RuleSetModel``'s rule line with its score suffix.

        The suffix is ``(+positives/-negatives)``.
        """
        positives, negatives = self._rule_scores[rule]
        return f"  {rule}  (+{positives}/-{negatives})"
