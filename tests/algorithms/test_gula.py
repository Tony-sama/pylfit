"""Tests for ``GULA``: its guarantees, its options, and ``BruteForce``.

Both algorithms are documented (``_gula.py``/``_brute_force.py`` module
docstrings) to compute *every* minimal, consistent rule for a given
dataset — a set that is fully determined by the data, so two correct
algorithms must agree on it exactly, not just produce equally-valid
alternatives. The file checks that agreement on every shared fixture,
and correctness, completeness and minimality as ``test_brute_force.py``
does. The rest tests ``GULA``'s options: ``heads``,
``allowed_conditions``, ``min_support``, ``negative_order``,
``min_possible_support``, and unknown values.
"""

import inspect

import numpy as np
import pandas as pd
import pytest

from pylfit import GULA, PRIDE, BruteForce, Dataset, Rule
from tests.algorithms._properties import (
    assert_complete,
    assert_correct,
    assert_heads_in_domain,
    assert_irreducible,
    assert_not_dominated,
)
from tests.algorithms._tables import as_question_marks, blanked_frame

# The coverage warning fires whenever an option or unknown values leave
# a positive row uncovered, which these tests do on purpose; the
# warning itself is tested in test_coverage_warning.py.
pytestmark = pytest.mark.filterwarnings("ignore:.*matched by no rule of their head")

# --- fit() return shape ---
#
# fit(dataset) -> list[Rule], sorted, no duplicates; stateless — no
# dependency on models/ at all, not even RuleSetModel. A caller wanting
# a model wraps the result explicitly, e.g.
# RuleSetModel(dataset, GULA.fit(dataset)).


def test_fit_returns_rules_sorted_by_text_without_duplicates(toy_dataset):
    rules = GULA.fit(toy_dataset)
    assert isinstance(rules, list)
    assert rules
    assert rules == sorted(set(rules))
    assert all(isinstance(rule, Rule) for rule in rules)


# --- toy_dataset: same hand-checked exact rule set as BruteForce ---
#
# See the comment above test_brute_force.py's exact-rule-set test for
# the by-hand derivation — reused verbatim here since both algorithms
# are documented to learn the same set of every minimal rule.


def test_toy_dataset_exact_rule_set(toy_dataset):
    rules = GULA.fit(toy_dataset)
    expected = {
        Rule(head=("y", "1"), body={"b": "1"}),
        Rule(head=("y", "1"), body={"a": "0", "c": "1"}),
        Rule(head=("y", "0"), body={"b": "0", "c": "0"}),
        Rule(head=("y", "0"), body={"a": "1", "b": "0"}),
    }
    assert set(rules) == expected


# --- degenerate_dataset: target never varies ---


def test_degenerate_dataset_yields_the_single_empty_body_rule(degenerate_dataset):
    rules = GULA.fit(degenerate_dataset)
    assert rules == [Rule(head=("y", "1"), body={})]


# --- nondeterministic_dataset: the "complete guarantee" case ---


def test_nondeterministic_target_value_with_no_negatives_gets_the_empty_body_rule(
    nondeterministic_dataset,
):
    # Same scenario as test_brute_force.py's equivalent test: a=0 has
    # both y=0 and y=1 rows, a=1 has only y=1, so dataset.negatives(
    # "y", "1") is empty — GULA's starting rule for y=1 never gets
    # specialized away from, since there's no negative example to
    # trigger it.
    assert nondeterministic_dataset.negatives("y", "1") == []
    rules = GULA.fit(nondeterministic_dataset)
    y1_rules = [rule for rule in rules if rule.head == ("y", "1")]
    assert y1_rules == [Rule(head=("y", "1"), body={})]


def test_nondeterministic_target_value_with_negatives_gets_a_specific_rule(
    nondeterministic_dataset,
):
    assert nondeterministic_dataset.negatives("y", "0") != []
    rules = GULA.fit(nondeterministic_dataset)
    y0_rules = [rule for rule in rules if rule.head == ("y", "0")]
    assert y0_rules == [Rule(head=("y", "0"), body={"a": "0"})]


# --- multi-target: targets learned independently ---


def test_targets_are_learned_independently(multi_target_dataset):
    rules = GULA.fit(multi_target_dataset)
    heads = {rule.head for rule in rules}
    assert heads == {("y1", "0"), ("y1", "1"), ("y2", "0"), ("y2", "1")}
    by_head = {rule.head: rule for rule in rules}
    assert by_head[("y1", "1")].body == {"a": "1"}
    assert by_head[("y1", "0")].body == {"a": "0"}
    assert by_head[("y2", "1")].body == {"a": "0"}
    assert by_head[("y2", "0")].body == {"a": "1"}


# --- heads: learning only some (target, value) pairs ---


@pytest.mark.parametrize(
    "heads",
    [
        [("y1", "1")],
        [("y1", "0"), ("y2", "1")],
        [("y1", "0"), ("y1", "1"), ("y2", "0"), ("y2", "1")],
    ],
)
def test_heads_learns_the_full_runs_rules_for_those_heads(multi_target_dataset, heads):
    full = GULA.fit(multi_target_dataset)
    rules = GULA.fit(multi_target_dataset, heads=heads)
    assert rules == [rule for rule in full if rule.head in set(heads)]


def test_heads_bare_target_name_means_every_value(multi_target_dataset):
    full = GULA.fit(multi_target_dataset)
    rules = GULA.fit(multi_target_dataset, heads=["y1"])
    assert rules == [rule for rule in full if rule.head[0] == "y1"]


def test_heads_keeps_min_support_per_head(toy_dataset):
    # A fractional min_support is relative to each head's own positives,
    # so learning one head alone must not change its threshold.
    full = GULA.fit(toy_dataset, min_support=0.5)
    rules = GULA.fit(toy_dataset, heads=[("y", "1")], min_support=0.5)
    assert rules == [rule for rule in full if rule.head == ("y", "1")]


def test_heads_empty_collection_learns_nothing(toy_dataset):
    assert GULA.fit(toy_dataset, heads=[]) == []


@pytest.mark.parametrize(
    ("heads", "message"),
    [
        (["Y"], "'Y' is not a target"),
        ([("Y", "1")], "'Y' is not a target"),
        ([("y1", "2")], "'2' is not a value of 'y1'"),
        ([("y1",)], "not a target or a \\(target, value\\) pair"),
        ([("y1", "1", "x")], "not a target or a \\(target, value\\) pair"),
    ],
)
def test_heads_naming_anything_unknown_raises(multi_target_dataset, heads, message):
    # A misspelt head would otherwise be skipped silently, and a string
    # read letter by letter.
    with pytest.raises(ValueError, match=message):
        GULA.fit(multi_target_dataset, heads=heads)


def test_heads_given_as_a_single_string_is_refused(multi_target_dataset):
    # A wrong type, so TypeError: "y1" would read as heads "y" and "1".
    with pytest.raises(TypeError, match="not a single string"):
        GULA.fit(multi_target_dataset, heads="y1")


def test_heads_reports_every_problem_at_once(multi_target_dataset):
    with pytest.raises(ValueError, match="is not") as error:
        GULA.fit(multi_target_dataset, heads=["Y", ("y1", "2"), ("y2", "1")])
    assert str(error.value).count("is not") == 2


def test_heads_allowed_conditions_for_another_head_is_checked_not_used(
    multi_target_dataset,
):
    alone = GULA.fit(multi_target_dataset, heads=[("y1", "1")])
    rules = GULA.fit(
        multi_target_dataset,
        heads=[("y1", "1")],
        allowed_conditions={("y1", "0"): []},
    )
    assert rules == alone
    with pytest.raises(ValueError, match="'A' is not a feature"):
        GULA.fit(
            multi_target_dataset,
            heads=[("y1", "1")],
            allowed_conditions={("y1", "0"): ["A"]},
        )


# --- allowed_conditions: background-knowledge restriction ---


@pytest.fixture
def and_function_dataset():
    """Full 3-feature truth table where ``y`` is ``a AND b``, ``c`` unused.

    ``a``/``b`` are ``y``'s true influences; ``c`` is irrelevant to it.
    Exhaustive (all 8 combinations), so restricting search to the true
    influences should reproduce the unrestricted rule set exactly, and
    restricting to the irrelevant feature should reproduce nothing.
    """
    df = pd.DataFrame(
        {
            "a": [0, 0, 0, 0, 1, 1, 1, 1],
            "b": [0, 0, 1, 1, 0, 0, 1, 1],
            "c": [0, 1, 0, 1, 0, 1, 0, 1],
            "y": [0, 0, 0, 0, 0, 0, 1, 1],
        }
    )
    return Dataset(df, features=["a", "b", "c"], targets=["y"])


def test_unrestricted_and_function_rule_set(and_function_dataset):
    # The reference rule set every restricted variant below is compared
    # against — neither unique minimal rule ever needs a condition on
    # `c`, so a correct restriction shouldn't change it.
    rules = GULA.fit(and_function_dataset)
    expected = {
        Rule(head=("y", "1"), body={"a": "1", "b": "1"}),
        Rule(head=("y", "0"), body={"a": "0"}),
        Rule(head=("y", "0"), body={"b": "0"}),
    }
    assert set(rules) == expected


def test_restricting_to_true_influences_reproduces_the_unrestricted_rule_set(
    and_function_dataset,
):
    # Bare feature-name shorthand for both heads — exact background
    # knowledge, so completeness isn't actually given up here.
    allowed_conditions = {("y", "0"): ["a", "b"], ("y", "1"): ["a", "b"]}
    rules = GULA.fit(and_function_dataset, allowed_conditions=allowed_conditions)
    assert rules == GULA.fit(and_function_dataset)


def test_explicit_pairs_and_bare_names_mix_freely(and_function_dataset):
    # Same true-influence restriction as above, but as one explicit
    # (feature, value) pair mixed with one bare feature name.
    allowed_conditions = {("y", "1"): [("a", "1"), "b"]}
    rules = GULA.fit(and_function_dataset, allowed_conditions=allowed_conditions)
    y1_rules = [rule for rule in rules if rule.head == ("y", "1")]
    assert y1_rules == [Rule(head=("y", "1"), body={"a": "1", "b": "1"})]


def test_allowed_conditions_is_scoped_per_head(and_function_dataset):
    # Only y=1 is restricted; y=0 (absent from the mapping) stays
    # unrestricted — confirms the dict is looked up per (target,
    # value), not applied blanket to every head.
    allowed_conditions = {("y", "1"): ["a", "b"]}
    rules = GULA.fit(and_function_dataset, allowed_conditions=allowed_conditions)
    y0_bodies = {
        frozenset(rule.body.items()) for rule in rules if rule.head == ("y", "0")
    }
    assert y0_bodies == {frozenset({"a": "0"}.items()), frozenset({"b": "0"}.items())}
    y1_rules = [rule for rule in rules if rule.head == ("y", "1")]
    assert y1_rules == [Rule(head=("y", "1"), body={"a": "1", "b": "1"})]


def test_restricting_to_an_irrelevant_feature_drops_completeness_but_stays_correct(
    and_function_dataset,
):
    # `c` has no bearing on `y` at all, so no candidate drawn from it
    # can ever separate the last inconsistent rule from a negative —
    # every specialization on `c` still matches other negatives, and
    # `c` is the only feature left to specialize on. The restricted
    # search ends up dropping y=1 entirely rather than keeping an
    # incorrect rule: no rule is ever added that matches a negative
    # (correctness), but no rule concludes y=1 for any state anymore
    # either (completeness lost) — the tradeoff _gula.py's module
    # docstring documents.
    allowed_conditions = {("y", "1"): ["c"]}
    rules = GULA.fit(and_function_dataset, allowed_conditions=allowed_conditions)
    y1_rules = [rule for rule in rules if rule.head == ("y", "1")]
    assert y1_rules == []
    assert not any(
        rule.head == ("y", "1") and rule.matches({"a": "1", "b": "1", "c": "0"})
        for rule in rules
    )


def test_allowed_conditions_none_for_a_head_is_unrestricted(and_function_dataset):
    # An explicit `None` entry for a head behaves the same as that head
    # being absent from the mapping entirely — both mean unrestricted.
    allowed_conditions = {("y", "1"): None}
    rules = GULA.fit(and_function_dataset, allowed_conditions=allowed_conditions)
    assert rules == GULA.fit(and_function_dataset)


@pytest.mark.parametrize(
    ("allowed_conditions", "message"),
    [
        ({("Y", "1"): ["a"]}, "'Y' is not a target"),
        ({("y", "2"): ["a"]}, "'2' is not a value of 'y'"),
        ({("y", "1"): ["A"]}, "'A' is not a feature"),
        ({("y", "1"): ["y"]}, "'y' is not a feature"),
        ({("y", "1"): [("a", "7")]}, "'7' is not a value of 'a'"),
        ({("y", "1"): "ab"}, "not a single string"),
        ({"y": ["a"]}, "not a \\(target, value\\) pair"),
        ({"y1": ["a"]}, "not a \\(target, value\\) pair"),
        ({("y", "1"): [("a",)]}, "not a feature or a \\(feature, value\\) pair"),
        ({("y", "1"): [("a", "1", "x")]}, "not a feature or a"),
    ],
)
def test_allowed_conditions_naming_anything_unknown_raises(
    and_function_dataset, allowed_conditions, message
):
    # Each of these used to run: a misspelt head as unrestricted, an
    # unknown value as a rule no state meets, a string char by char.
    with pytest.raises(ValueError, match=message):
        GULA.fit(and_function_dataset, allowed_conditions=allowed_conditions)


@pytest.mark.parametrize("algorithm", [GULA, PRIDE])
def test_allowed_conditions_may_be_a_one_pass_iterable(and_function_dataset, algorithm):
    # The check reads each entry once; the learner must still get it.
    as_list = {("y", "1"): ["a", "b"]}
    as_generator = {("y", "1"): (feature for feature in ["a", "b"])}
    assert algorithm.fit(
        and_function_dataset, allowed_conditions=as_generator
    ) == algorithm.fit(and_function_dataset, allowed_conditions=as_list)


def test_allowed_conditions_reports_every_problem_at_once(and_function_dataset):
    allowed_conditions = {("Y", "1"): ["a"], ("y", "1"): ["A", ("b", "9")]}
    with pytest.raises(ValueError, match="head") as error:
        GULA.fit(and_function_dataset, allowed_conditions=allowed_conditions)
    assert str(error.value).count("head") == 3


# --- min_support: minimum-support filter ---


def test_unrestricted_support_dataset_rule_set(support_dataset):
    # The reference rule set the min_support variants below are
    # compared against.
    rules = GULA.fit(support_dataset)
    expected = {
        Rule(head=("y", "0"), body={"a": "0"}),
        Rule(head=("y", "1"), body={"a": "1"}),
        Rule(head=("y", "1"), body={"a": "2"}),
    }
    assert set(rules) == expected


def test_min_support_absolute_count_drops_the_low_support_rule(support_dataset):
    # a=1's rule has support 1, below the threshold; a=2's rule has
    # support 4, above it — only the former is dropped.
    rules = GULA.fit(support_dataset, min_support=2)
    assert set(rules) == {
        Rule(head=("y", "0"), body={"a": "0"}),
        Rule(head=("y", "1"), body={"a": "2"}),
    }


def test_min_support_fraction_drops_the_low_support_rule(support_dataset):
    # y=1 has 5 total positives; ceil(0.3 * 5) == 2, the same effective
    # threshold as the absolute-count test above, exercised through the
    # fraction path instead.
    rules = GULA.fit(support_dataset, min_support=0.3)
    assert set(rules) == {
        Rule(head=("y", "0"), body={"a": "0"}),
        Rule(head=("y", "1"), body={"a": "2"}),
    }


def test_min_support_trades_away_completeness(support_dataset):
    # The dropped a=1 rule was the only one concluding y=1 for that
    # state — no rule covers it anymore, same tradeoff
    # allowed_conditions makes (_gula.py's module docstring).
    rules = GULA.fit(support_dataset, min_support=2)
    assert not any(
        rule.head == ("y", "1") and rule.matches({"a": "1"}) for rule in rules
    )


def test_min_support_none_is_unrestricted(support_dataset):
    assert GULA.fit(support_dataset, min_support=None) == GULA.fit(support_dataset)


@pytest.mark.parametrize("option", ["min_support", "min_possible_support"])
@pytest.mark.parametrize("fraction", [np.float32(0.5), np.float64(0.5)])
def test_support_thresholds_take_a_numpy_float(support_dataset, option, fraction):
    assert GULA.fit(support_dataset, **{option: fraction}) == GULA.fit(
        support_dataset, **{option: 0.5}
    )


@pytest.mark.parametrize("option", ["min_support", "min_possible_support"])
def test_support_thresholds_refuse_a_bool(support_dataset, option):
    with pytest.raises(TypeError, match="must be an int or float, got bool"):
        GULA.fit(support_dataset, **{option: True})


@pytest.mark.parametrize("option", ["min_support", "min_possible_support"])
def test_support_thresholds_take_a_numpy_integer(support_dataset, option):
    assert GULA.fit(support_dataset, **{option: np.int64(2)}) == GULA.fit(
        support_dataset, **{option: 2}
    )


def test_min_support_drops_an_untouched_general_rule_that_fails_threshold(
    nondeterministic_dataset,
):
    # y=1 has no negatives at all (see the equivalent test above), so
    # its general rule is never specialized against — confirms the
    # filter also applies to that never-touched starting rule, not
    # just candidates produced by specialization.
    assert nondeterministic_dataset.negatives("y", "1") == []
    assert len(nondeterministic_dataset.positives("y", "1")) == 2
    rules = GULA.fit(nondeterministic_dataset, min_support=3)
    y1_rules = [rule for rule in rules if rule.head == ("y", "1")]
    assert y1_rules == []


@pytest.mark.parametrize(
    ("min_support", "error"),
    [
        (-1, ValueError),
        (0.0, ValueError),
        (1.5, ValueError),
        (True, TypeError),
        ("2", TypeError),
    ],
)
def test_min_support_rejects_invalid_values(support_dataset, min_support, error):
    with pytest.raises(error, match="min_support"):
        GULA.fit(support_dataset, min_support=min_support)


# --- negative_order: which order negatives are processed in ---
#
# Never changes the learned rule set (_gula.py's module docstring) —
# only which order candidates get produced/discarded in along the way,
# so every test here is an equivalence check across the three orders,
# not a behavior difference the way allowed_conditions/min_support get.


def test_default_negative_order_is_similarity():
    assert inspect.signature(GULA.fit).parameters["negative_order"].default == (
        "similarity"
    )


@pytest.mark.parametrize("fixture_name", ["toy_dataset", "and_function_dataset"])
@pytest.mark.parametrize("negative_order", ["similarity", "lexicographic", "unordered"])
def test_negative_order_never_changes_the_rule_set(
    fixture_name, negative_order, request
):
    dataset = request.getfixturevalue(fixture_name)
    rules = GULA.fit(dataset, negative_order=negative_order)
    assert rules == GULA.fit(dataset, negative_order="lexicographic")


def test_negative_order_default_matches_every_explicit_order(support_dataset):
    default_rules = GULA.fit(support_dataset)
    for negative_order in ("similarity", "lexicographic", "unordered"):
        assert default_rules == GULA.fit(support_dataset, negative_order=negative_order)


@pytest.mark.parametrize(
    ("negative_order", "expected"),
    [
        # Summed Hamming distance to the others: 000 has 6, 111 has 9,
        # and the three at 7 tie, broken lexicographically.
        ("similarity", ["000", "001", "010", "100", "111"]),
        ("lexicographic", ["000", "001", "010", "100", "111"]),
        ("unordered", ["111", "001", "000", "100", "010"]),
    ],
)
def test_negative_order_gives_the_documented_sequence(negative_order, expected):
    features = ["a", "b", "c"]
    given = ["111", "001", "000", "100", "010"]
    negatives = [dict(zip(features, row, strict=True)) for row in given]
    ordered = GULA._order_negatives(negatives, features, negative_order, False)
    assert ["".join(state.values()) for state in ordered] == expected


def test_similarity_puts_the_most_typical_negative_first():
    # Not the lexicographic order: 111 is typical, 000 the outlier.
    features = ["a", "b", "c"]
    given = ["000", "111", "110", "101", "011"]
    negatives = [dict(zip(features, row, strict=True)) for row in given]
    ordered = GULA._order_negatives(negatives, features, "similarity", False)
    assert ["".join(state.values()) for state in ordered][0] == "111"
    assert ["".join(state.values()) for state in ordered][-1] == "000"


def test_verbose_counts_each_distinct_negative_once(capsys):
    df = pd.DataFrame({"a": ["0", "0", "0", "1"], "y": ["0", "0", "0", "1"]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    GULA.fit(dataset, heads=[("y", "1")], verbose=True)
    assert "y=1: 1 distinct negatives" in capsys.readouterr().err


def test_negative_order_rejects_invalid_values(toy_dataset):
    with pytest.raises(ValueError, match="negative_order must be one of"):
        GULA.fit(toy_dataset, negative_order="shuffled")


# --- unknown values ---
#
# An unknown (None in Dataset's states) never satisfies a condition and
# is never one. That is the same as writing it as an ordinary string,
# "?", and leaving every condition on "?" out through
# allowed_conditions (as_question_marks, in _tables.py): the reference
# these tests check GULA against.


def _possible_support(rule, positives):
    return sum(
        all(state[f] in (v, None) for f, v in rule.body.items()) for state in positives
    )


@pytest.mark.filterwarnings("ignore:Rows with the same feature state")
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("min_support", [None, 1, 2])
@pytest.mark.parametrize("deterministic", [False, True])
def test_unknowns_learn_what_question_marks_without_their_conditions_learn(
    seed, min_support, deterministic
):
    frame = blanked_frame(seed, deterministic=deterministic)
    dataset = Dataset(frame, features=["a", "b", "c"], targets=["y"])
    reference, allowed = as_question_marks(frame)
    rules = GULA.fit(dataset, min_support=min_support)
    assert rules == GULA.fit(
        reference, allowed_conditions=allowed, min_support=min_support
    )


@pytest.mark.filterwarnings("ignore:Rows with the same feature state")
@pytest.mark.parametrize("negative_order", ["similarity", "lexicographic", "unordered"])
def test_unknowns_learn_the_same_rules_in_every_negative_order(negative_order):
    dataset = Dataset(blanked_frame(0), features=["a", "b", "c"], targets=["y"])
    assert GULA.fit(dataset, negative_order=negative_order) == GULA.fit(dataset)


def test_unknowns_are_never_a_condition():
    dataset = Dataset(
        blanked_frame(1, rate=0.5), features=["a", "b", "c"], targets=["y"]
    )
    for rule in GULA.fit(dataset):
        assert None not in rule.body.values()


def test_an_unknown_in_a_negative_is_excluded_by_every_value():
    # The negative a=None, b=0 could hide any value of a, and differs
    # from each: every value of a separates it.
    df = pd.DataFrame(
        {
            "a": ["0", "1", "2", None],
            "b": ["1", "1", "1", "0"],
            "y": ["1", "1", "1", "0"],
        }
    )
    dataset = Dataset(df, features=["a", "b"], targets=["y"])
    rules = {rule.to_string() for rule in GULA.fit(dataset, heads=[("y", "1")])}
    assert rules == {"y(1) :- a(0).", "y(1) :- a(1).", "y(1) :- a(2).", "y(1) :- b(1)."}


def test_learning_under_assume_nondeterministic_is_consistent_with_its_negatives():
    dataset = Dataset(
        blanked_frame(2),
        features=["a", "b", "c"],
        targets=["y"],
        assume_nondeterministic=True,
    )
    rules = GULA.fit(dataset)
    for rule in rules:
        assert not any(rule.matches(n) for n in dataset.negatives(*rule.head))


# --- min_possible_support: support counting an unknown as a match ---


@pytest.mark.filterwarnings("ignore:Rows with the same feature state")
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("threshold", [1, 3, 0.25])
def test_min_possible_support_keeps_exactly_the_unrestricted_rules_meeting_it(
    seed, threshold
):
    dataset = Dataset(blanked_frame(seed), features=["a", "b", "c"], targets=["y"])
    full = GULA.fit(dataset)
    rules = GULA.fit(dataset, min_possible_support=threshold)
    expected = set()
    for rule in full:
        positives = dataset.positives(*rule.head)
        count = threshold if isinstance(threshold, int) else threshold * len(positives)
        if _possible_support(rule, positives) >= count:
            expected.add(rule)
    assert set(rules) == expected


@pytest.mark.filterwarnings("ignore:Rows with the same feature state")
@pytest.mark.parametrize("seed", range(4))
def test_min_support_and_min_possible_support_combine_as_both(seed):
    dataset = Dataset(blanked_frame(seed), features=["a", "b", "c"], targets=["y"])
    both = GULA.fit(dataset, min_support=1, min_possible_support=3)
    assert set(both) == set(GULA.fit(dataset, min_support=1)) & set(
        GULA.fit(dataset, min_possible_support=3)
    )


def test_min_possible_support_keeps_a_rule_only_unknowns_support():
    # The one y=1 row, a=0 with b unknown, could be a=0, b=0.
    df = pd.DataFrame(
        {
            "a": ["0", "0", None, "1"],
            "b": ["1", None, "0", "0"],
            "y": ["0", "1", "0", "0"],
        }
    )
    dataset = Dataset(df, features=["a", "b"], targets=["y"])
    heads = [("y", "1")]
    assert GULA.fit(dataset, heads=heads, min_support=1) == []
    assert GULA.fit(dataset, heads=heads, min_possible_support=1) == [
        Rule.from_string("y(1) :- a(0), b(0).")
    ]


@pytest.mark.parametrize("threshold", [1, 2, 0.5])
def test_min_possible_support_is_min_support_without_unknowns(
    support_dataset, threshold
):
    assert GULA.fit(support_dataset, min_possible_support=threshold) == GULA.fit(
        support_dataset, min_support=threshold
    )


@pytest.mark.parametrize(
    ("threshold", "error"),
    [(-1, ValueError), (0.0, ValueError), (1.5, ValueError), (True, TypeError)],
)
def test_min_possible_support_rejects_invalid_values(support_dataset, threshold, error):
    with pytest.raises(error, match="min_possible_support"):
        GULA.fit(support_dataset, min_possible_support=threshold)


# --- generic properties, checked against every fixture ---

_FIXTURE_NAMES = ["toy_dataset", "nondeterministic_dataset", "degenerate_dataset"]


@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
def test_every_rule_head_value_is_in_the_targets_domain(fixture_name, request):
    dataset = request.getfixturevalue(fixture_name)
    assert_heads_in_domain(dataset, GULA.fit(dataset))


@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
def test_correctness_no_rule_matches_a_negative(fixture_name, request):
    dataset = request.getfixturevalue(fixture_name)
    assert_correct(dataset, GULA.fit(dataset))


@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
def test_completeness_every_feature_state_is_matched_per_target(fixture_name, request):
    dataset = request.getfixturevalue(fixture_name)
    assert_complete(dataset, GULA.fit(dataset))


@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
def test_minimality_no_rule_is_dominated_by_another(fixture_name, request):
    dataset = request.getfixturevalue(fixture_name)
    assert_not_dominated(dataset, GULA.fit(dataset))


@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
def test_minimality_no_condition_can_be_dropped(fixture_name, request):
    dataset = request.getfixturevalue(fixture_name)
    assert_irreducible(dataset, GULA.fit(dataset))


# --- the BruteForce-vs-GULA cross-check, on every fixture ---
#
# Both algorithms compute every minimal, consistent rule (module
# docstrings), so the two rule sets must be equal exactly, not just
# equally-valid alternatives.


@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
def test_gula_and_brute_force_learn_the_same_rules(fixture_name, request):
    dataset = request.getfixturevalue(fixture_name)
    brute_force_rules = BruteForce.fit(dataset)
    gula_rules = GULA.fit(dataset)
    assert gula_rules == brute_force_rules


def test_gula_and_brute_force_agree_on_a_multi_target_dataset(multi_target_dataset):
    brute_force_rules = BruteForce.fit(multi_target_dataset)
    gula_rules = GULA.fit(multi_target_dataset)
    assert gula_rules == brute_force_rules
