"""Tests for ``generalize`` and the Fisher test it runs.

``fisher_greater`` is checked against p-values hard-coded from
``scipy.stats.fisher_exact(table, alternative="greater")`` and against
exact sums of ``math.comb`` terms. The generalization itself runs on a
small dataset where ``y`` is ``a`` and ``z`` is ``b``, so which
conditions are needed is known; which rule of a nested pair is kept,
on tables built for each p-value.
"""

import itertools
import math
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from pylfit import Dataset, PylfitWarning, Rule, generalize
from pylfit._generalize import _nested_pairs, fisher_greater
from pylfit.algorithms import GULA
from pylfit.models import WeightedRuleSetModel

#: ``(table, p)``, with ``p`` from ``scipy.stats.fisher_exact`` 1.17.1.
SCIPY = [
    ((8, 2, 1, 5), 0.024475524475524483),
    ((0, 3, 3, 0), 1.0),
    ((3, 0, 0, 3), 0.05000000000000001),
    ((10, 0, 0, 10), 5.412544112234515e-06),
    ((1, 9, 11, 3), 0.9999663480953022),
    ((0, 0, 5, 5), 1.0),
    ((40, 10, 30, 160), 1.7795420266930318e-17),
    ((120, 5, 3, 110), 2.754091588476163e-57),
]


def _exact(a: int, b: int, c: int, d: int) -> Fraction:
    """The one-sided p-value as an exact sum of ``math.comb`` terms."""
    row, column, total = a + b, a + c, a + b + c + d
    return sum(
        (
            Fraction(math.comb(column, x) * math.comb(total - column, row - x))
            for x in range(a, min(row, column) + 1)
        ),
        Fraction(0),
    ) / math.comb(total, row)


@pytest.mark.parametrize(("table", "expected"), SCIPY)
def test_fisher_greater_matches_scipy(table, expected):
    assert fisher_greater(*table) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    "table", [(5, 1, 2, 7), (0, 4, 6, 2), (12, 3, 4, 15), (1, 0, 0, 0), (0, 0, 0, 0)]
)
def test_fisher_greater_matches_exact_comb_sums(table):
    assert fisher_greater(*table) == pytest.approx(float(_exact(*table)), rel=1e-12)


@pytest.mark.parametrize(
    "table", [(0, 3, 3, 0), (0, 3, 3, 6), (0, 3, 3, 21), (4, 7, 10, 0), (0, 0, 5, 5)]
)
def test_fisher_greater_is_exactly_one_on_the_whole_tail(table):
    # `a` is the smallest count the margins allow, so every table is
    # summed. Without the shortcut, (0, 3, 3, 6) gives
    # 0.9999999999999968 on glibc, and (0, 3, 3, 0) 0.9999999999999998
    # on macOS.
    assert _exact(*table) == 1
    assert fisher_greater(*table) == 1.0


@pytest.fixture
def dataset() -> Dataset:
    """40 rows: ``a``, ``b``, ``c`` in every combination, 5 times.

    ``y`` is ``a`` and ``z`` is ``b``.
    """
    rows = [
        {"a": a, "b": b, "c": c, "y": a, "z": b}
        for a, b, c in itertools.product("01", repeat=3)
    ] * 5
    return Dataset(pd.DataFrame(rows), features=["a", "b", "c"], targets=["y", "z"])


def _rule(text: str) -> Rule:
    return Rule.from_string(text)


def test_an_unneeded_condition_is_dropped(dataset):
    rules = [_rule("y(1) :- a(1), b(1), c(0).")]
    assert generalize(rules, dataset) == [_rule("y(1) :- a(1).")]


def test_every_head_is_generalized(dataset):
    rules = [_rule("z(1) :- a(0), b(1)."), _rule("y(0) :- a(0), c(1).")]
    expected = [_rule("y(0) :- a(0)."), _rule("z(1) :- b(1).")]
    assert generalize(rules, dataset) == expected


def test_a_rule_with_no_needed_condition_is_dropped_beside_another(dataset):
    rules = [_rule("y(1) :- b(1), c(1)."), _rule("y(1) :- a(1), b(0).")]
    assert generalize(rules, dataset) == [_rule("y(1) :- a(1).")]


def test_a_rule_with_no_needed_condition_is_kept_alone_with_no_condition(dataset):
    assert generalize([_rule("y(1) :- b(1), c(1).")], dataset) == [_rule("y(1) :- .")]


def test_a_value_keeps_a_rule_when_its_only_rule_needs_no_condition(dataset):
    # y(0) :- b(0) needs no condition, y(1) :- a(1) does: y=0 keeps a
    # rule to vote with, so the y(1) rules do not win every vote.
    rules = [_rule("y(0) :- b(0)."), _rule("y(1) :- a(1), c(0).")]
    assert generalize(rules, dataset) == [_rule("y(0) :- ."), _rule("y(1) :- a(1).")]


def test_rules_that_become_the_same_are_merged(dataset):
    rules = [_rule("y(1) :- a(1), b(1)."), _rule("y(1) :- a(1), b(0), c(1).")]
    assert generalize(rules, dataset) == [_rule("y(1) :- a(1).")]


def test_an_unknown_value_matches_no_condition():
    rows = [{"a": a, "y": a} for a in "01"] * 10 + [{"a": None, "y": "1"}] * 30
    dataset = Dataset(pd.DataFrame(rows), features=["a"], targets=["y"])
    # The unknown rows hold y=1 but not a=1, so a=1 does not raise the
    # rate of y=1 enough to be needed: 10 of 10 against 30 of 40,
    # p 0.08.
    assert generalize([_rule("y(1) :- a(1).")], dataset) == [_rule("y(1) :- .")]
    assert generalize([_rule("y(1) :- a(1).")], dataset, alpha=0.1) == [
        _rule("y(1) :- a(1).")
    ]


def test_an_unknown_target_value_is_no_negative_example():
    # 3 of 3 against 0 of 3, p 0.05. Read as y=0, the 10 rows of unknown
    # y would make it 3 of 13 against 0 of 3, p 0.51.
    rows = [{"a": "1", "y": "1"}] * 3 + [{"a": "1", "y": None}] * 10
    rows += [{"a": "0", "y": "0"}] * 3
    dataset = Dataset(pd.DataFrame(rows), features=["a"], targets=["y"])
    rules = [_rule("y(1) :- a(1).")]
    assert generalize(rules, dataset, alpha=0.1) == rules


def test_a_value_the_dataset_lacks_matches_no_row(dataset):
    # A holdout split may lack a value the rules were learned with: the
    # condition is dropped, and the model refusing it takes the result.
    rules = generalize([_rule("y(1) :- a(1), b(2).")], dataset)
    assert rules == [_rule("y(1) :- a(1).")]
    WeightedRuleSetModel(dataset, rules)


def test_a_head_value_the_dataset_lacks_loses_its_rule(dataset):
    # No row to need a condition, and no model would take the rule.
    rules = [_rule("y(2) :- a(1)."), _rule("y(1) :- a(1), b(1).")]
    found = generalize(rules, dataset)
    assert found == [_rule("y(1) :- a(1).")]
    WeightedRuleSetModel(dataset, found)


def test_only_rules_of_a_head_value_the_dataset_lacks_leave_none(dataset):
    assert generalize([_rule("y(2) :- a(1)."), _rule("y(2) :- .")], dataset) == []


def test_the_rules_build_a_model(dataset):
    rules = generalize([_rule("y(1) :- a(1), b(1).")], dataset)
    model = WeightedRuleSetModel(dataset, rules)
    assert model.get_score(_rule("y(1) :- a(1).")) == (20, 0)


@pytest.mark.parametrize("alpha", [0, -0.1, 1.5])
def test_an_alpha_outside_0_to_1_is_refused(dataset, alpha):
    with pytest.raises(ValueError, match="alpha must be above 0"):
        generalize([], dataset, alpha=alpha)


@pytest.mark.parametrize("alpha", [True, "0.01", None])
def test_an_alpha_that_is_not_a_number_is_refused(dataset, alpha):
    with pytest.raises(TypeError, match="alpha must be a number"):
        generalize([], dataset, alpha=alpha)


def test_alpha_1_keeps_every_condition(dataset):
    rule = _rule("y(1) :- a(1), b(1), c(0).")
    assert generalize([rule], dataset, alpha=1) == [rule]


@pytest.mark.parametrize("text", ["y(1) :- d(1).", "w(1) :- a(1).", "a(1) :- b(1)."])
def test_a_rule_off_the_targets_and_features_is_refused(dataset, text):
    with pytest.raises(ValueError, match="names a variable"):
        generalize([_rule(text)], dataset)


@pytest.mark.parametrize(
    ("rules", "dataset_ok", "match"),
    [
        ("y(1) :- a(1).", True, "not a str"),
        ([1], True, "must be Rule objects"),
        ([], False, "expected a Dataset"),
    ],
)
def test_wrong_types_are_refused(dataset, rules, dataset_ok, match):
    with pytest.raises(TypeError, match=match):
        generalize(rules, dataset if dataset_ok else "data")


def _nested(both: int, *, constant: bool = False) -> Dataset:
    """80 rows: 20 per value pair of ``a``, ``b``; ``c`` if ``constant``.

    ``y=1`` on the 20 with ``a=1, b=1``, on ``both`` of the 20 with
    ``a=1, b=0``, never with ``a=0``; ``c``, when there, is always
    ``0``.

    ``y(1) :- a(1).`` and ``y(1) :- a(1), b(1).`` each keep every
    condition, and the narrow rule's ``b(1)`` is tested among the broad
    rule's 40 rows: 20 of 20 against ``both`` of 20.
    """
    rows = [{"a": "1", "b": "1", "y": "1"}] * 20
    rows += [{"a": "1", "b": "0", "y": "1" if i < both else "0"} for i in range(20)]
    rows += [{"a": "0", "b": b, "y": "0"} for b in "01" for _ in range(20)]
    features = ["a", "b"]
    if constant:
        rows = [{**row, "c": "0"} for row in rows]
        features.append("c")
    # The a=1, b=0 rows conflict on purpose; generalize reads them as
    # rows.
    return Dataset(
        pd.DataFrame(rows),
        features=features,
        targets=["y"],
        assume_nondeterministic=True,
    )


NESTED = [_rule("y(1) :- a(1)."), _rule("y(1) :- a(1), b(1).")]


def test_a_narrow_rule_needed_among_the_broad_rules_rows_replaces_it():
    # 20 of 20 against 12 of 20, p 0.0016, times the 2 values b could
    # take.
    assert generalize(NESTED, _nested(12)) == NESTED[1:]


def test_the_narrow_test_counts_every_condition_it_could_have_added():
    # p 0.0042, the same table as b(1)'s own test in the narrow rule,
    # which keeps it: times 2, the values of b, it passes; times 3, b's
    # or c's one value, it fails, though c holds no information.
    assert generalize(NESTED, _nested(13)) == NESTED[1:]
    assert generalize(NESTED, _nested(13, constant=True)) == NESTED[:1]


def test_a_narrow_rule_at_exactly_the_level_is_kept():
    # alpha is _nested(13)'s p-value times 2: the test is
    # p * 2 <= alpha.
    alpha = 2 * fisher_greater(20, 0, 13, 7)
    assert generalize(NESTED, _nested(13), alpha=alpha) == NESTED[1:]


def test_a_pair_that_lost_a_rule_already_is_skipped():
    # 10 rows per value pair of a, b; y=1 on 0, 0, 1 and 6 of them (00,
    # 01, 10, 11). Each rule keeps its conditions. y(1) :- a(1), b(1).
    # loses to y(1) :- a(1). (6 of 10 against 1 of 10, p 0.029, times
    # 2), so its pair with y(1) :- b(1)., which it would win (6 of 10
    # against 0 of 10, p 0.0054, times 2), is not taken: y(1) :- b(1).
    # stays.
    rows = []
    for (a, b), held in zip(["00", "01", "10", "11"], [0, 0, 1, 6], strict=True):
        rows += [{"a": a, "b": b, "y": "1" if i < held else "0"} for i in range(10)]
    dataset = Dataset(
        pd.DataFrame(rows),
        features=["a", "b"],
        targets=["y"],
        assume_nondeterministic=True,
    )
    rules = [_rule("y(1) :- a(1)."), _rule("y(1) :- b(1).")]
    narrow = _rule("y(1) :- a(1), b(1).")
    assert generalize([*rules, narrow], dataset, alpha=0.05) == rules


def test_no_rule_is_more_general_than_another_of_its_head():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.choice(["0", "1"], size=(200, 5)), columns=list("abcde"))
    rule = (frame["a"] == "1") & (frame["b"] == "1") | (frame["c"] == "1")
    flipped = rng.random(len(frame)) < 0.1
    frame["y"] = np.where(rule ^ flipped, "1", "0")
    dataset = Dataset(
        frame, features=list("abcde"), targets=["y"], assume_nondeterministic=True
    )
    rules = generalize(GULA.fit(dataset), dataset, alpha=0.2)
    assert len(rules) > 2
    for broad, narrow in itertools.permutations(rules, 2):
        assert not broad.is_more_general_than(narrow), (broad, narrow)


@pytest.mark.parametrize("count", [3, 40, 400])
def test_nested_pairs_are_every_pair_one_more_general(count):
    # Few rules are compared with each other, many by their sub-bodies.
    rng = np.random.default_rng(count)
    rules = {
        Rule(
            ("y", str(rng.integers(2))),
            {v: str(rng.integers(2)) for v in "abcdef" if rng.random() < 0.4},
        )
        for _ in range(count)
    }
    expected = {
        (broad, narrow)
        for broad in rules
        for narrow in rules
        if broad.body and broad != narrow and broad.is_more_general_than(narrow)
    }
    found = _nested_pairs(rules)
    assert len(found) == len(expected)
    assert set(found) == expected


def _weak_and_strong(c_on: int = 15) -> pd.DataFrame:
    """40 rows, ``y = a``; ``c=1`` on ``c_on`` of the 20 ``y=1`` rows.

    ``c=1`` on ``20 - c_on`` of the others too. ``y(1) :- a(1).`` is
    needed at p 7e-12; ``y(1) :- c(1).`` at p 0.0019 with ``c_on=15``,
    at p 3e-9 with 19.
    """
    rows = [{"a": "1", "c": "1" if i < c_on else "0", "y": "1"} for i in range(20)]
    rows += [
        {"a": "0", "c": "1" if i < 20 - c_on else "0", "y": "0"} for i in range(20)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def weak_and_strong() -> Dataset:
    """``_weak_and_strong()`` as a ``Dataset``: features ``a``, ``c``."""
    return Dataset(_weak_and_strong(), features=["a", "c"], targets=["y"])


def test_max_rules_lowers_alpha_until_the_cap_holds(weak_and_strong):
    rules = [_rule("y(1) :- a(1)."), _rule("y(1) :- c(1).")]
    assert generalize(rules, weak_and_strong) == rules
    assert generalize(rules, weak_and_strong, max_rules=1) == rules[:1]


def test_max_rules_searches_alpha_on_a_log_scale():
    # a(1) is needed at p 7e-12, c(1) at 3e-9: the search must land
    # between them. Halving 0.01 ten times only reaches 1e-5.
    dataset = Dataset(_weak_and_strong(19), features=["a", "c"], targets=["y"])
    rules = [_rule("y(1) :- a(1)."), _rule("y(1) :- c(1).")]
    assert generalize(rules, dataset) == rules
    assert generalize(rules, dataset, max_rules=1) == rules[:1]


def test_max_rules_already_met_changes_nothing(weak_and_strong):
    # alpha is c(1)'s own p-value. A search, which never tries alpha
    # itself, would drop it.
    rules = [_rule("y(1) :- a(1)."), _rule("y(1) :- c(1).")]
    alpha = fisher_greater(15, 5, 5, 15)
    assert generalize(rules, weak_and_strong, alpha=alpha) == rules
    assert generalize(rules, weak_and_strong, alpha=alpha, max_rules=2) == rules


def test_max_rules_counts_every_target_together():
    # weak_and_strong's rows, z a copy of y: two rules of y and one of z
    # fit 2 per target, not 2 in all.
    frame = _weak_and_strong().assign(z=lambda f: f["y"])
    dataset = Dataset(frame, features=["a", "c"], targets=["y", "z"])
    rules = [_rule("y(1) :- a(1)."), _rule("y(1) :- c(1)."), _rule("z(1) :- a(1).")]
    assert generalize(rules, dataset) == rules
    assert generalize(rules, dataset, max_rules=2) == [rules[0], rules[2]]


def test_max_rules_out_of_reach_warns_and_returns_the_floor_rules():
    rows = [{"a": a, "y": a} for a in "01"] * 100
    dataset = Dataset(pd.DataFrame(rows), features=["a"], targets=["y"])
    rules = [_rule("y(0) :- a(0)."), _rule("y(1) :- a(1).")]
    with pytest.warns(PylfitWarning, match="meets max_rules=1") as record:
        assert generalize(rules, dataset, max_rules=1) == rules
    assert record[0].filename == __file__


def test_an_alpha_below_1e_12_is_the_lowest_tried():
    # One rule per head, and generalize never drops a head's last rule.
    rows = [{"a": a, "y": a} for a in "01"]
    dataset = Dataset(pd.DataFrame(rows), features=["a"], targets=["y"])
    rules = [_rule("y(0) :- a(0)."), _rule("y(1) :- a(1).")]
    with pytest.warns(PylfitWarning, match="no alpha down to 1e-13 meets"):
        generalize(rules, dataset, alpha=1e-13, max_rules=1)


@pytest.mark.parametrize("max_rules", [0, -1])
def test_a_max_rules_below_1_is_refused(dataset, max_rules):
    with pytest.raises(ValueError, match="max_rules must be at least 1"):
        generalize([], dataset, max_rules=max_rules)


@pytest.mark.parametrize("max_rules", [True, 1.5, "2"])
def test_a_max_rules_that_is_not_an_int_is_refused(dataset, max_rules):
    with pytest.raises(TypeError, match="max_rules must be an int"):
        generalize([], dataset, max_rules=max_rules)


def test_numpy_numbers_are_accepted(weak_and_strong):
    rules = [_rule("y(1) :- a(1)."), _rule("y(1) :- c(1).")]
    capped = generalize(
        rules, weak_and_strong, alpha=np.float32(0.01), max_rules=np.int64(1)
    )
    assert capped == rules[:1]
