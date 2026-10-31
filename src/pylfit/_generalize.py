"""``generalize``: rules without the conditions the data does not need.

Each rule loses, one at a time, the condition the data least needs,
until every condition left is needed; a rule none of whose conditions
is needed keeps no condition when no other rule of its head is left,
and is dropped otherwise. Of two rules left where one is more general
than the other, one is dropped.

A condition ``variable=value`` of a rule is needed when, among the rows
matching the rest of the rule, those with ``variable=value`` hold the
head value significantly more often than those without, by a one-sided
Fisher exact test at ``alpha``. This is Webb's test for a productive
rule [Webb2006]_, applied by dropping the condition instead of
rejecting the rule, as C4.5rules' ``-F`` option does [Quinlan1993]_.
Rows are counted as ``Rule.matches`` counts them: an unknown value
matches no condition.

Each rule is generalized on its own, so a rule can come out beside a
broader one of its head; one of the two is then dropped, so that no
rule returned is more general than another of its head. The narrow
rule is kept when, among the broad rule's rows, those it matches hold
the head value significantly more often than the rest, by the same
test with ``alpha`` divided by the number of rules that add as many
conditions to the broad one, on any feature it leaves free, with any
value. This is Webb's layered critical values [Webb2008]_: the learner
chose the narrow rule among all of those, so without the division a
condition that only dodges a few mislabelled rows passes. When the
narrow rule has one condition more, the table is that condition's own
test, which ``alpha`` alone already passed, so only the division can
drop it.

References:
    .. [Webb2006] Webb, G. I. (2006). Discovering significant rules. In
       Proceedings of the 12th ACM SIGKDD International Conference on
       Knowledge Discovery and Data Mining (KDD 2006), 434-443.
       https://doi.org/10.1145/1150402.1150451

    .. [Quinlan1993] Quinlan, J. R. (1993). *C4.5: Programs for Machine
       Learning*. Morgan Kaufmann. ISBN 978-1-55860-238-0.

    .. [Webb2008] Webb, G. I. (2008). Layered critical values: a
       powerful direct-adjustment approach to discovering significant
       patterns. *Machine Learning*, 71(2-3), 307-323.
       https://doi.org/10.1007/s10994-008-5046-x
"""

import itertools
import math
import numbers
from collections.abc import Iterable
from warnings import warn

import numpy as np

from pylfit._dataset import Dataset, _check_dataset
from pylfit._exceptions import PylfitWarning
from pylfit._rule import Rule

#: The smallest ``alpha`` the search for ``max_rules`` tries.
_ALPHA_FLOOR = 1e-12

#: Halvings of the search's ``log(alpha)`` interval: ``alpha`` to within
#: 3% between the floor and 1.
_SEARCH_STEPS = 10


def _log_comb(n: int, k: int) -> float:
    """``log(math.comb(n, k))``, through ``math.lgamma``."""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_greater(a: int, b: int, c: int, d: int) -> float:
    """The one-sided Fisher exact test's p-value for ``[[a, b], [c, d]]``.

    The chance, with the margins fixed, of ``a`` or more in the top left
    cell: a sum of hypergeometric terms, each from ``math.lgamma``. The
    same value as ``scipy.stats.fisher_exact(table,
    alternative="greater")``, without the dependency. When ``a`` is the
    smallest count the margins allow, the sum covers every table and is
    1 exactly; it is returned as such, since the terms' rounding,
    which depends on the platform's ``lgamma`` and ``exp``, can leave
    it just below.

    Args:
        a: Rows with the condition holding the head value.
        b: Rows with the condition holding another value.
        c: Rows without the condition holding the head value.
        d: Rows without the condition holding another value.

    Returns:
        The p-value, at most 1.

    Examples:
        >>> round(fisher_greater(8, 2, 1, 5), 6)
        0.024476
        >>> fisher_greater(0, 3, 3, 0)
        1.0
    """
    row, column, total = a + b, a + c, a + b + c + d
    if a <= max(0, row + column - total):
        return 1.0
    denominator = _log_comb(total, row)
    terms = (
        _log_comb(column, x) + _log_comb(total - column, row - x) - denominator
        for x in range(a, min(row, column) + 1)
    )
    return min(1.0, math.fsum(math.exp(term) for term in terms))


def _bits(rows: np.ndarray) -> int:
    """A boolean array as an ``int`` whose bit ``i`` is ``rows[i]``."""
    return int.from_bytes(np.packbits(rows, bitorder="little").tobytes(), "little")


class _Rows:
    """The dataset's rows as bitmasks, one per condition and head.

    A body's rows are then the AND of its conditions' masks, and a
    count is ``int.bit_count``, rather than ``Rule.matches`` on every
    row for every table.
    """

    def __init__(self, dataset: Dataset) -> None:
        data = dataset._data
        self._data = data
        self._conditions: dict[tuple[str, str], int] = {}
        self._heads: dict[tuple[str, str], tuple[int, int]] = {}
        self._bodies: dict[frozenset[tuple[str, str]], int] = {}
        self._p_values: dict[tuple[int, int, int, int], float] = {}

    def condition(self, variable: str, value: str) -> int:
        """The rows where ``variable`` holds ``value``."""
        key = (variable, value)
        if key not in self._conditions:
            self._conditions[key] = _bits(self._data[variable].eq(value).to_numpy())
        return self._conditions[key]

    def head(self, target: str, value: str) -> tuple[int, int]:
        """The rows holding ``value``, and those holding another known one."""
        key = (target, value)
        if key not in self._heads:
            column = self._data[target]
            positive = _bits(column.eq(value).to_numpy())
            known = _bits(column.notna().to_numpy())
            self._heads[key] = (positive, known & ~positive)
        return self._heads[key]

    def body(self, conditions: frozenset[tuple[str, str]]) -> int:
        """The rows matching every condition; every row when there is none."""
        if conditions not in self._bodies:
            rows = (1 << len(self._data)) - 1
            for variable, value in conditions:
                rows &= self.condition(variable, value)
            self._bodies[conditions] = rows
        return self._bodies[conditions]

    def p_value(self, a: int, b: int, c: int, d: int) -> float:
        """``fisher_greater``, cached: many tables recur across rules."""
        key = (a, b, c, d)
        if key not in self._p_values:
            self._p_values[key] = fisher_greater(a, b, c, d)
        return self._p_values[key]


def _generalize(rule: Rule, rows: _Rows, alpha: float) -> Rule:
    """``rule`` without the conditions the data does not need.

    Each pass tests every condition left and drops the one with the
    highest p-value, unless it is at most ``alpha``. On a tie it drops
    the one whose removal leaves the rule matching the most rows, then
    the first in variable order: when a value matches no row, every
    condition's p-value is 1, and this drops that value's condition
    first rather than whichever variable sorts first. A last condition
    is tested against every row.

    Args:
        rule: The rule to generalize.
        rows: The dataset's rows.
        alpha: The significance level.

    Returns:
        The generalized rule, with an empty body when no condition is
        needed.
    """
    positive, negative = rows.head(*rule.head)
    body = dict(rule.body)
    while body:
        worst, worst_key = "", (-1.0, -1)
        for variable in sorted(body):
            others = frozenset((v, w) for v, w in body.items() if v != variable)
            rest = rows.body(others)
            inside = rest & rows.condition(variable, body[variable])
            outside = rest & ~inside
            p = rows.p_value(
                (inside & positive).bit_count(),
                (inside & negative).bit_count(),
                (outside & positive).bit_count(),
                (outside & negative).bit_count(),
            )
            if (p, rest.bit_count()) > worst_key:
                worst, worst_key = variable, (p, rest.bit_count())
        if worst_key[0] <= alpha:
            return Rule(rule.head, body)
        del body[worst]
    return Rule(rule.head, {})


def generalize(
    rules: Iterable[Rule],
    dataset: Dataset,
    *,
    alpha: float = 0.01,
    max_rules: int | None = None,
) -> list[Rule]:
    """Returns the rules without the conditions ``dataset`` does not need.

    Each rule loses, one at a time, the condition with the highest
    p-value, while that p-value is above ``alpha``. A condition's
    p-value comes from the rows matching the rest of the rule: a
    one-sided Fisher exact test of whether those with the condition hold
    the head value more often than those without. Rules that become the
    same are merged. Of a rule and a broader one of the same head, the
    narrow one is kept only when the broad rule's rows need its extra
    conditions, by the same test at a stricter level (the module's
    description gives it), so that no rule returned is more general
    than another of its head, as in ``GULA``'s rules.

    Rules learned from partial or noisy data carry conditions that only
    dodge a few rows, and some rules hold on the rows seen but not on
    the others; this drops both, and what is left is the true rules or
    generalizations of them. On 27 Boolean network nodes, learned from
    70% of the states with 10% of the training labels flipped, 92% of
    the rules returned are true rules, 6% are a true rule without one
    or two of its conditions, and none is spurious (``rule_recovery``'s
    terms), against 26% spurious among ``GULA``'s ~390 rules before;
    from 30% of the states with no flipped label, 77% and 21%. A
    ``WeightedRuleSetModel`` of them scores 0.975 on the unseen states,
    against 0.984 for ``GULA``'s rules. With 20% of the labels flipped
    58% are true rules and 32% one or two conditions short; with 30%,
    42% and 21%, while 21% overlap a true rule without being a
    generalization of it and 5% are spurious. On splits of the UCI
    heart disease data, ``GULA``'s ~12000 rules become ~27, and a
    ``WeightedRuleSetModel`` of them scores 0.807 held-out, against
    0.815. A lower ``alpha`` keeps fewer conditions and rules.

    A true rule that matches few rows cannot keep its conditions: those
    rows are too few for the test. On the UCI car evaluation data
    (acceptable or better), whose 90 true rules have up to six
    conditions, ``GULA``'s rules from every state are those 90 rules,
    and ``generalize`` returns 15 rules, 4 of them true and 11 each a
    true rule without one or two conditions. Learned from 70% of the
    states with 10% of the labels flipped, its 7 rules are all
    generalizations of true rules, nearly all one or two conditions
    short, none spurious, and score 0.88 held-out against 0.95 for
    ``GULA``'s ~520 rules, a third of them spurious.

    The default ``alpha``, 0.01, was the best level, or as good as any,
    on every data set measured with fewer than about a thousand rows. On
    the 27 network nodes above (358 to 717 training rows), from
    ``PRIDE``'s rules, a lower ``alpha`` never recovered more true
    rules: with 10% of the labels flipped, 92% of the true rules come
    back exactly at 0.01 and 82% at 0.001. On heart disease, any
    ``alpha`` from 0.0001 to 0.2 scores 0.80 to 0.82 held-out (see
    ``max_rules`` below). The default is too high for thousands of rows
    with noisy labels: a learner such as ``PRIDE`` then returns many
    more rules, so more chance conditions pass the test. On network
    nodes of 11 to 15 variables (1434 to 22938 training rows) with 30%
    of the labels flipped, 21% to 71% of the rules returned at 0.01 are
    a true rule or a true rule with extra conditions, against 80% to
    100% at 0.001, with at most 4 points fewer true rules recovered
    exactly. On targets that are nearly constant, the rare value's
    chance rules then win the vote: 0.49 to 0.62 held-out at 0.01, 0.89
    to 0.95 at 0.001, 0.98 to 1.00 at 0.0001. With no label flipped,
    0.001 returns the same rules as 0.01 on those networks, but for the
    nearly constant targets. If the default gives rules that look too
    many or too specific for your data, try ``alpha=0.001`` or lower; if
    rules you expect lose conditions they need, try a higher one.

    A rule none of whose conditions is needed is dropped when another
    rule of its head is left, and kept with no condition otherwise, so
    that no value loses every rule. Where that value is common, such as
    ``0`` on a target that is ``1`` in 3% of rows, dropping all its
    rules would leave the other value's rules to win every vote they
    enter. On the 27 network nodes with up to 30% of the training labels
    flipped, this changes 7 of 675 runs, from ``GULA``'s rules as from
    ``PRIDE``'s: where the rare value's rules won every vote, scoring
    0.27 to 0.57, the rule with no condition now wins them and the
    common value is predicted everywhere, scoring its share of the test
    rows, 0.81 to 0.97. Beside other rules, such a rule would outvote
    them all wherever its value is common, so it is dropped there. With
    ``RuleSetModel``, which does not weigh rules, a rule with no
    condition makes its value possible in every state.

    ``max_rules`` caps the rules returned, every target together, for a
    model of a size you can read: when ``alpha`` keeps more, a search
    returns the rules at the largest lower ``alpha`` that keeps at most
    ``max_rules``. One level serves every target, so the conditions a
    tight cap removes are those with the weakest evidence, on whichever
    target.

    The cap keeps a model short; it is not a remedy for noisy labels,
    which lowering ``alpha`` is. Under noise ``generalize`` already
    returns few rules, a few of them chance ones: on network nodes of 11
    to 13 variables with 30% of the labels flipped, ~6 per node at 0.01,
    so a cap of 20 changes nothing there and one of 10 almost nothing,
    and a cap of 5 leaves 62% of the rules a true rule or a true rule
    with extra conditions, against 75% at ``alpha=0.001``. Every rule
    returned still passes the test, so the cap is not met by cutting
    rules: when even ``alpha=1e-12`` keeps more, those rules are
    returned with a warning. The search generalizes the rules about a
    dozen times, ~2.5 s for ``GULA``'s heart disease rules against 0.3 s
    without a cap. On ten splits of that data, ``alpha`` from 0.0001 to
    0.2 keeps 16 to 98 of ``GULA``'s rules and scores 0.80 to 0.82
    throughout, so the cap costs little accuracy there.

    ``dataset`` can be another split than the one the rules were
    learned from, to test them on rows they were not fitted to. A rule
    value ``dataset`` lacks matches none of its rows, so its condition
    is never needed and is dropped: ``y(1) :- a(1), b(2).`` becomes
    ``y(1) :- a(1).`` where ``b`` is never ``2``. The models, which
    refuse such a rule, accept the result.

    Args:
        rules: The rules to generalize, from any algorithm or written by
            hand.
        dataset: The rows that decide which conditions are needed.
        alpha: The significance level of each test, above 0 and at most
            1; with ``max_rules``, the largest one tried.
        max_rules: The most rules to return, every target together, at
            least 1, or ``None`` for no cap.

    Returns:
        The generalized rules, sorted by text, without duplicates; none
        is more general than another of its head.

    Raises:
        TypeError: If ``rules`` is not iterable, is a single ``str`` or
            holds something other than a ``Rule``, ``dataset`` is not a
            ``Dataset``, ``alpha`` is not a number, or ``max_rules`` is
            not an integer (NumPy numbers count as numbers and
            integers).
        ValueError: If ``alpha`` is not above 0 and at most 1,
            ``max_rules`` is below 1, or a rule names a variable that is
            not one of ``dataset``'s features and targets.

    Warns:
        PylfitWarning: If ``max_rules`` is set and even ``alpha=1e-12``
            keeps more rules than that, every target together.

    Examples:
        >>> import pandas as pd
        >>> from pylfit import Dataset, Rule, generalize
        >>> rows = [{"a": a, "b": b, "y": a} for a in "01" for b in "01"] * 5
        >>> dataset = Dataset(pd.DataFrame(rows), features=["a", "b"], targets=["y"])
        >>> generalize([Rule.from_string("y(1) :- a(1), b(0).")], dataset)
        [Rule(head=('y', '1'), body={'a': '1'})]
    """
    _check_dataset(dataset)
    given: object = rules  # checked as the caller may pass it
    if isinstance(given, str):
        raise TypeError("rules must be a collection of Rule, not a str")
    rules = list(rules)
    for rule in rules:
        if not isinstance(rule, Rule):
            raise TypeError(f"rules must be Rule objects, got {rule!r}")
    level: object = alpha  # checked as the caller may pass it
    if isinstance(level, bool) or not isinstance(level, numbers.Real):
        raise TypeError(f"alpha must be a number, got {alpha!r}")
    alpha = float(level)
    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be above 0 and at most 1, got {level!r}")
    if max_rules is not None:
        cap: object = max_rules  # checked as the caller may pass it
        if isinstance(cap, bool) or not isinstance(cap, numbers.Integral):
            raise TypeError(f"max_rules must be an int, got {max_rules!r}")
        max_rules = int(cap)
        if max_rules < 1:
            raise ValueError(f"max_rules must be at least 1, got {cap!r}")
    features, targets = set(dataset.features), set(dataset.targets)
    for rule in rules:
        if rule.head[0] not in targets or not set(rule.body) <= features:
            raise ValueError(
                f"rule {rule} names a variable that is not one of the "
                "dataset's features and targets"
            )
    rows = _Rows(dataset)
    sizes = {variable: len(dataset.domain(variable)) for variable in dataset.features}
    found = _generalize_all(rules, rows, sizes, alpha)
    if max_rules is None or len(found) <= max_rules:
        return found
    low = min(alpha, _ALPHA_FLOOR)
    best = _generalize_all(rules, rows, sizes, low)
    if len(best) > max_rules:
        warn(
            f"no alpha down to {low:g} meets max_rules={max_rules}; "
            f"returning the {len(best)} rules at alpha={low:g}",
            PylfitWarning,
            stacklevel=2,
        )
        return best
    high = alpha
    for _ in range(_SEARCH_STEPS):
        middle = math.sqrt(low * high)
        candidate = _generalize_all(rules, rows, sizes, middle)
        if len(candidate) <= max_rules:
            low, best = middle, candidate
        else:
            high = middle
    return best


def _generalize_all(
    rules: list[Rule], rows: _Rows, sizes: dict[str, int], alpha: float
) -> list[Rule]:
    """Every rule generalized at ``alpha``, merged, made minimal, sorted."""
    found = _minimal(
        {_generalize(rule, rows, alpha) for rule in rules}, rows, sizes, alpha
    )
    heads = {rule.head for rule in found if rule.body}
    return sorted(
        (rule for rule in found if rule.body or rule.head not in heads), key=str
    )


def _extensions(sizes: dict[str, int], broad: Rule, extra: int) -> int:
    """How many rules extend ``broad`` by ``extra`` conditions.

    The elementary symmetric sum of degree ``extra`` of the domain sizes
    of the features ``broad`` has no condition on: a condition on each
    of ``extra`` of them, any value.
    """
    sums = [1] + [0] * extra
    for variable, size in sizes.items():
        if variable not in broad.body:
            for k in range(extra, 0, -1):
                sums[k] += sums[k - 1] * size
    return sums[extra]


def _nested_pairs(rules: set[Rule]) -> list[tuple[Rule, Rule]]:
    """Every ``(broad, narrow)`` of ``rules`` where ``broad`` is more general.

    Rules with no condition are left out. A rule's broader partners are
    found by looking up each of its ``2**k - 2`` proper sub-bodies, or
    by comparing it with every rule of its head when there are fewer: on
    thousands of rules of short bodies, comparing every two would take
    seconds.
    """
    by_head: dict[tuple[str, str], dict[frozenset[tuple[str, str]], Rule]] = {}
    for rule in rules:
        if rule.body:
            by_head.setdefault(rule.head, {})[frozenset(rule.body.items())] = rule
    pairs: list[tuple[Rule, Rule]] = []
    for group in by_head.values():
        for body, narrow in group.items():
            if 2 ** len(body) - 2 < len(group):
                items = sorted(body)
                for size in range(1, len(body)):
                    for sub in itertools.combinations(items, size):
                        broad = group.get(frozenset(sub))
                        if broad is not None:
                            pairs.append((broad, narrow))
            else:
                pairs.extend(
                    (broad, narrow)
                    for broad in group.values()
                    if len(broad.body) < len(body)
                    and broad.is_more_general_than(narrow)
                )
    return pairs


def _minimal(
    rules: set[Rule], rows: _Rows, sizes: dict[str, int], alpha: float
) -> set[Rule]:
    """``rules`` with no rule more general than another of its head.

    Each such pair, a broad rule and a narrow one, keeps one: the narrow
    rule when, among the broad rule's rows, those it matches hold the
    head value more often than the rest, by ``fisher_greater`` at
    ``alpha`` divided by ``_extensions``; the broad rule otherwise. The
    pairs are taken in order of their text, skipping those that lost a
    rule already; a rule with no condition is left to the caller.
    """
    pairs = sorted(_nested_pairs(rules), key=lambda pair: (str(pair[0]), str(pair[1])))
    dropped: set[Rule] = set()
    for broad, narrow in pairs:
        if broad in dropped or narrow in dropped:
            continue
        positive, negative = rows.head(*broad.head)
        inside = rows.body(frozenset(narrow.body.items()))
        rest = rows.body(frozenset(broad.body.items())) & ~inside
        p = rows.p_value(
            (inside & positive).bit_count(),
            (inside & negative).bit_count(),
            (rest & positive).bit_count(),
            (rest & negative).bit_count(),
        )
        extra = len(narrow.body) - len(broad.body)
        needed = p * _extensions(sizes, broad, extra) <= alpha
        dropped.add(broad if needed else narrow)
    return rules - dropped
