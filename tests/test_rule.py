"""Tests for ``Rule``: construction, matching, generality, parsing, order.

These tests use plain Python objects (pandas only for the unknown-value
markers a ``DataFrame`` row holds), including a minimal duck-typed
state to confirm ``matches`` really only needs a ``.get`` method, not a
full ``Mapping`` or a ``pandas.Series``.
"""

import copy
import decimal
import pickle
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from pylfit import Rule


class DuckTypedState:
    """Minimal stand-in for a state: exposes ``.get``, nothing else."""

    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)


# --- Construction ---


def test_body_defaults_to_empty():
    rule = Rule(head=("y", "1"))
    assert rule.body == {}
    assert rule.size() == 0


def test_body_is_copied_at_construction():
    body = {"a": "0"}
    rule = Rule(head=("y", "1"), body=body)
    body["a"] = "9"  # mutating the original dict afterwards...
    assert rule.body.get("a") == "0"  # ...must not affect the rule


@pytest.mark.parametrize(
    "head",
    [("y", 1), (1, "1"), ["y", "1"], "y1", ("y", "1", "x"), ("y",)],
)
def test_head_not_a_tuple_of_two_str_raises(head):
    with pytest.raises(TypeError, match="not a \\(variable, value\\) tuple"):
        Rule(head=head)


def test_body_value_not_a_str_raises():
    # It would never match a state, whose values are str.
    with pytest.raises(TypeError, match="body value for 'a' is 1 \\(int\\)"):
        Rule(head=("y", "1"), body={"a": 1})


def test_body_variable_not_a_str_raises():
    with pytest.raises(TypeError, match="body variable 0 \\(int\\) is not a str"):
        Rule(head=("y", "1"), body={0: "1"})


def test_body_not_a_mapping_raises():
    with pytest.raises(TypeError, match="not a mapping"):
        Rule(head=("y", "1"), body=[("a", "1")])


def test_body_may_be_any_mapping():
    rule = Rule(head=("y", "1"), body={"a": "1"})
    assert Rule(head=("y", "1"), body=rule.body) == rule


# --- Observers: size / has_condition ---


def test_size_counts_conditions():
    rule = Rule(head=("y", "1"), body={"a": "0", "b": "1"})
    assert rule.size() == 2


def test_has_condition():
    rule = Rule(head=("y", "1"), body={"a": "0"})
    assert rule.has_condition("a") is True
    assert rule.has_condition("b") is False


# --- matches ---


def test_matches_true_when_all_conditions_hold():
    rule = Rule(head=("y", "1"), body={"a": "0", "c": "1"})
    assert rule.matches({"a": "0", "b": "1", "c": "1"}) is True


def test_matches_false_when_any_condition_fails():
    rule = Rule(head=("y", "1"), body={"a": "0", "c": "1"})
    assert rule.matches({"a": "0", "b": "1", "c": "0"}) is False


def test_matches_raises_when_state_lacks_a_tested_variable():
    # A missing key is far likelier a typo than an intended unknown,
    # which is written explicitly as None.
    rule = Rule(head=("y", "1"), body={"a": "0"})
    with pytest.raises(KeyError, match="'a': None for an unknown"):
        rule.matches({"b": "1"})


def test_equal_rules_raise_alike_whatever_order_their_body_was_built_in():
    # Which of two equal rules a set keeps is arbitrary, so their scans
    # must agree on which missing variable they report.
    built = [
        Rule(("y", "1"), {"a": "1", "b": "0"}),
        Rule(("y", "1"), {"b": "0", "a": "1"}),
    ]
    assert built[0] == built[1]
    for rule in built:
        assert list(rule.body) == ["a", "b"]
        with pytest.raises(KeyError, match="'b'"):
            rule.matches({"a": "1"})


def test_matches_ignores_missing_variables_the_body_does_not_test():
    rule = Rule(head=("y", "1"), body={"a": "0"})
    assert rule.matches({"a": "0"}) is True


@pytest.mark.parametrize(
    "unknown",
    [
        None,
        float("nan"),
        np.nan,
        np.float64("nan"),
        np.float32("nan"),
        pd.NA,
        pd.NaT,
        np.datetime64("NaT", "ns"),
        decimal.Decimal("NaN"),
    ],
    ids=[
        "None",
        "float-nan",
        "np-nan",
        "np-float64-nan",
        "np-float32-nan",
        "pd-NA",
        "pd-NaT",
        "np-NaT",
        "decimal-nan",
    ],
)
def test_matches_false_on_an_unknown_value(unknown):
    rule = Rule(head=("y", "1"), body={"a": "0"})
    assert rule.matches({"a": unknown}) is False


def test_matches_reads_unknowns_in_a_series_row():
    rule = Rule(head=("y", "1"), body={"a": "0", "b": "1"})
    row = pd.DataFrame({"a": ["0"], "b": [None]}).iloc[0]
    assert rule.matches(row) is False


def test_empty_body_matches_any_state():
    rule = Rule(head=("y", "1"), body={})
    assert rule.matches({}) is True
    assert rule.matches({"a": "0", "b": "1"}) is True


@pytest.mark.parametrize(
    ("value", "shown"),
    [(1, "1 \\(int\\)"), (["1", "2"], "\\['1', '2'\\] \\(list\\)")],
    ids=["int", "list"],
)
def test_matches_rejects_a_non_str_value_instead_of_not_matching(value, shown):
    # An int never equals a str body value, so without the check this
    # would read as a plain non-match and hide the mistake. A list is
    # no unknown either, though pandas.isna answers it with an array.
    rule = Rule(head=("y", "1"), body={"a": "1"})
    with pytest.raises(TypeError, match=f"'a' is {shown}"):
        rule.matches({"a": value})


def test_matches_does_not_check_values_after_the_first_failing_condition():
    # The scan stops at the first failure, as the docstring says.
    rule = Rule(head=("y", "1"), body={"a": "1", "b": "1"})
    assert rule.matches({"a": "0", "b": 5}) is False
    with pytest.raises(TypeError, match="'b' is 5"):
        rule.matches({"a": "1", "b": 5})


def test_matches_ignores_non_str_values_the_body_does_not_test():
    rule = Rule(head=("y", "1"), body={"a": "1"})
    assert rule.matches({"a": "1", "b": 2})


def test_matches_accepts_duck_typed_state():
    rule = Rule(head=("y", "1"), body={"a": "0", "c": "1"})
    matching_state = DuckTypedState({"a": "0", "b": "1", "c": "1"})
    non_matching_state = DuckTypedState({"a": "0", "b": "1", "c": "0"})
    assert rule.matches(matching_state) is True
    assert rule.matches(non_matching_state) is False


# --- is_more_general_than ---


def test_is_more_general_than_refuses_a_non_rule():
    with pytest.raises(TypeError, match="not a Rule"):
        Rule(("y", "1")).is_more_general_than(("y", "1"))


def test_empty_body_is_more_general_than_everything():
    general = Rule(head=("y", "1"), body={})
    specific = Rule(head=("y", "1"), body={"a": "0", "b": "1"})
    assert general.is_more_general_than(specific) is True


def test_specific_is_not_more_general_than_general():
    general = Rule(head=("y", "1"), body={})
    specific = Rule(head=("y", "1"), body={"a": "0", "b": "1"})
    assert specific.is_more_general_than(general) is False


@pytest.mark.parametrize("head", [("y", "0"), ("z", "1")])
def test_is_more_general_than_false_for_another_head(head):
    assert Rule(head).is_more_general_than(Rule(("y", "1"), {"a": "1"})) is False


def test_rule_is_more_general_than_itself():
    rule = Rule(head=("y", "1"), body={"a": "0", "b": "1"})
    assert rule.is_more_general_than(Rule(rule.head, rule.body)) is True


def test_is_more_general_than_requires_subset_of_conditions():
    subset = Rule(head=("y", "1"), body={"a": "0"})
    superset = Rule(head=("y", "1"), body={"a": "0", "b": "1"})
    assert subset.is_more_general_than(superset) is True
    assert superset.is_more_general_than(subset) is False


def test_is_more_general_than_false_for_same_size_different_conditions():
    rule_a = Rule(head=("y", "1"), body={"a": "0"})
    rule_b = Rule(head=("y", "1"), body={"b": "1"})
    assert rule_a.is_more_general_than(rule_b) is False
    assert rule_b.is_more_general_than(rule_a) is False


def test_is_more_general_than_false_for_same_variable_different_value():
    rule_a = Rule(head=("y", "1"), body={"a": "0"})
    rule_b = Rule(head=("y", "1"), body={"a": "1"})
    assert rule_a.is_more_general_than(rule_b) is False


# --- Derived rules: with_condition / without_condition ---


def test_with_condition_adds_new_variable():
    rule = Rule(head=("y", "1"), body={})
    assert rule.with_condition("a", "0").body.get("a") == "0"


def test_with_condition_replaces_existing_variable():
    rule = Rule(head=("y", "1"), body={"a": "0"})
    derived = rule.with_condition("a", "1")
    assert derived.body.get("a") == "1"
    assert derived.size() == 1  # replaced, not appended


def test_with_condition_refuses_a_value_that_is_not_a_str():
    with pytest.raises(TypeError, match="body value for 'a' is 1"):
        Rule(head=("y", "1")).with_condition("a", 1)


def test_without_condition_drops_existing_variable():
    rule = Rule(head=("y", "1"), body={"a": "0", "b": "1"})
    derived = rule.without_condition("a")
    assert derived.has_condition("a") is False
    assert derived.size() == 1


def test_without_condition_gives_an_equal_rule_if_absent():
    rule = Rule(head=("y", "1"), body={"a": "0"})
    assert rule.without_condition("z") == rule  # must not raise


def test_derived_rules_leave_the_original_unchanged():
    rule = Rule(head=("y", "1"), body={"a": "0"})
    rule.with_condition("b", "1")
    rule.without_condition("a")
    assert rule == Rule(head=("y", "1"), body={"a": "0"})


# --- Immutability ---


def test_head_and_body_cannot_be_reassigned():
    rule = Rule(head=("y", "1"), body={"a": "0"})
    with pytest.raises(AttributeError):
        rule.head = ("y", "0")  # type: ignore[misc]
    with pytest.raises(AttributeError):
        rule.body = {}  # type: ignore[misc]


def test_body_cannot_be_changed_in_place():
    rule = Rule(head=("y", "1"), body={"a": "0"})
    with pytest.raises(TypeError):
        rule.body["b"] = "1"  # type: ignore[index]


def test_a_rule_stays_findable_in_a_set():
    rule = Rule(head=("y", "1"), body={"a": "0"})
    rules = {rule}
    rule.with_condition("b", "1")
    assert rule in rules


def test_copy_module_and_pickle_give_an_equal_rule():
    rule = Rule(head=("y", "1"), body={"a": "0"})
    assert copy.copy(rule) == rule
    assert copy.deepcopy(rule) == rule
    restored = pickle.loads(pickle.dumps(rule))
    assert restored == rule
    assert hash(restored) == hash(rule)


class Tagged(Rule):
    """A subclass at module level, where pickle can find it."""

    __slots__ = ()


def test_a_subclass_survives_pickling_and_derivation():
    rule = Tagged(("y", "1"), {"a": "0"})
    assert type(pickle.loads(pickle.dumps(rule))) is Tagged
    assert type(rule.with_condition("b", "1")) is Tagged
    assert type(rule.without_condition("a")) is Tagged


# --- String representation ---


def test_to_string_sorts_conditions_alphabetically():
    rule = Rule(head=("y", "1"), body={"b": "1", "a": "0"})
    assert rule.to_string() == "y(1) :- a(0), b(1)."


def test_to_string_empty_body():
    rule = Rule(head=("y", "1"), body={})
    assert rule.to_string() == "y(1) :- ."


def test_str_matches_to_string():
    rule = Rule(head=("y", "1"), body={"a": "0"})
    assert str(rule) == rule.to_string()


@pytest.mark.parametrize(
    "rule",
    [
        Rule(("y", "1")),
        Rule(("y", "1"), {"b": "0", "a": "1"}),
        Rule(("y", "p:-q"), {"a": " x", "c, d": "[0, 40)"}),
    ],
    ids=["empty-body", "two-conditions", "beyond-from-string"],
)
def test_repr_rebuilds_an_equal_rule(rule):
    # Even for the rules from_string can't read back.
    assert eval(repr(rule)) == rule


def test_numpy_strings_are_stored_as_str():
    values = np.unique(np.array(["0", "1"]))
    names = np.unique(np.array(["a", "b", "y"]))
    rule = Rule((names[2], values[1]), {names[0]: values[0]})
    rule = rule.with_condition(names[1], values[0])
    assert repr(rule) == "Rule(head=('y', '1'), body={'a': '0', 'b': '0'})"
    parts = [*rule.head, *rule.body, *rule.body.values()]
    assert {type(part) for part in parts} == {str}


def test_repr_sorts_conditions_by_variable():
    rule = Rule(head=("y", "1"), body={"b": "0", "a": "1"})
    assert repr(rule) == "Rule(head=('y', '1'), body={'a': '1', 'b': '0'})"


# --- from_string (Horn-clause parsing) ---


def test_from_string_parses_head_and_body():
    rule = Rule.from_string("y(1) :- a(0), c(1).")
    assert rule == Rule(head=("y", "1"), body={"a": "0", "c": "1"})


def test_from_string_parses_empty_body():
    rule = Rule.from_string("y(1) :- .")
    assert rule == Rule(head=("y", "1"))


def test_from_string_trailing_period_is_optional():
    rule = Rule.from_string("y(1) :- a(0)")
    assert rule == Rule(head=("y", "1"), body={"a": "0"})


def test_from_string_tolerates_surrounding_whitespace():
    rule = Rule.from_string("  y(1)   :-   a(0) ,  c(1)  .  ")
    assert rule == Rule(head=("y", "1"), body={"a": "0", "c": "1"})


def test_from_string_is_case_sensitive_and_order_independent():
    # Body order in the text doesn't matter: it's a mapping either way.
    rule = Rule.from_string("y(1) :- c(1), a(0).")
    assert rule == Rule(head=("y", "1"), body={"a": "0", "c": "1"})
    assert Rule.from_string("Y(1) :- A(0).") != Rule(("y", "1"), {"a": "0"})


def test_from_string_round_trips_with_to_string():
    rule = Rule(head=("y", "1"), body={"a": "0", "c": "1"})
    assert Rule.from_string(rule.to_string()) == rule


def test_from_string_round_trips_empty_body():
    rule = Rule(head=("y", "1"))
    assert Rule.from_string(rule.to_string()) == rule


@pytest.mark.parametrize(
    "rule",
    [
        Rule(("a_t+1", "1"), {"b_t": "0", "c_t": "1"}),
        Rule(("petal-width", "wide"), {"sepal length": "long"}),
        Rule(("sick", "yes"), {"age": "(40, 100]", "bmi": "(18.5, 25.0)"}),
        Rule(("y", "x(1)"), {"a": "f(0)"}),
        Rule(("y", "1"), {"w": "[0, 40)", "z": "x)"}),
    ],
    ids=[
        "time-suffix",
        "punctuated-names",
        "interval-values",
        "nested-parens",
        "closing-parens",
    ],
)
def test_from_string_round_trips_names_and_values_beyond_identifiers(rule):
    # Dataset accepts any column name and value, and to_string writes
    # them as they are, so from_string has to read them back.
    assert Rule.from_string(rule.to_string()) == rule


def test_from_string_rejects_two_conditions_on_one_variable():
    with pytest.raises(ValueError, match="two conditions on 'a'"):
        Rule.from_string("y(1) :- a(0), a(1).")


def test_from_string_rejects_a_trailing_comma():
    with pytest.raises(ValueError, match=r"malformed atom .*: ''"):
        Rule.from_string("y(1) :- a(0),")


@pytest.mark.parametrize(
    "text", ["y(1) :- a(0) b(1).", "y(1) a(0) :- b(1).", "y(1) :- a(0)\n b(1)."]
)
def test_from_string_rejects_a_missing_comma(text):
    # Would read as one condition whose value is "0) b(1".
    with pytest.raises(ValueError, match="is a ',' missing"):
        Rule.from_string(text)


def test_from_string_rejects_two_commas_in_a_row():
    # Would read as a condition on a variable named ", b".
    with pytest.raises(ValueError, match="two commas in a row"):
        Rule.from_string("y(1) :- a(0), , b(1).")


def test_from_string_drops_spaces_around_a_value():
    # One of the round-trip limits its docstring lists.
    rule = Rule(("y", "1"), {"a": " x"})
    assert Rule.from_string(rule.to_string()).body == {"a": "x"}


def test_from_string_cannot_read_the_arrow_in_a_head():
    # One of the round-trip limits its docstring lists.
    with pytest.raises(ValueError, match=r"malformed atom .*: 'y\(p'"):
        Rule.from_string(Rule(("y", "p:-q")).to_string())


def test_from_string_refuses_a_text_that_is_not_a_str():
    with pytest.raises(TypeError, match="not a str"):
        Rule.from_string(5)


def test_from_string_rejects_empty_string():
    with pytest.raises(ValueError, match="from an empty string"):
        Rule.from_string("  ")


@pytest.mark.parametrize(
    "text",
    # CN2's "IF ... THEN" is deliberately not accepted.
    ["y(1) a(0).", "IF a = 0 AND c = 1 THEN y = 1"],
    ids=["no-arrow", "cn2-style"],
)
def test_from_string_rejects_missing_arrow(text):
    with pytest.raises(ValueError, match="missing ':-'"):
        Rule.from_string(text)


@pytest.mark.parametrize(
    ("text", "atom"),
    [
        ("y1 :- a(0).", "y1"),
        ("y(1) :- a0.", "a0"),
        ("(1) :- a(0).", "\\(1\\)"),
        ("y(1) :- (0).", "\\(0\\)"),
        ("y(1 :- a(0).", "y\\(1"),
        ("y(1) :- a(0.", "a\\(0"),
        # variable=value is CN2's spelling, not Horn's.
        ("y=1 :- a=0, c=1.", "y=1"),
    ],
    ids=[
        "head-no-parens",
        "body-no-parens",
        "head-no-variable",
        "body-no-variable",
        "head-unclosed",
        "body-unclosed",
        "equality-style",
    ],
)
def test_from_string_rejects_a_malformed_atom(text, atom):
    with pytest.raises(ValueError, match=f"malformed atom .*: '{atom}'"):
        Rule.from_string(text)


# --- Equality, hashing and order ---


def test_equality_with_a_non_rule_defers_to_the_other_operand():
    rule = Rule(("y", "1"))
    assert rule.__eq__(1) is NotImplemented
    assert rule != 1
    assert rule == mock.ANY


def test_equal_rules_have_same_head_and_body():
    rule_a = Rule(head=("y", "1"), body={"a": "0", "b": "1"})
    rule_b = Rule(head=("y", "1"), body={"b": "1", "a": "0"})  # built the other way
    assert rule_a == rule_b


def test_unequal_rules_by_head():
    rule_a = Rule(head=("y", "1"), body={"a": "0"})
    rule_b = Rule(head=("y", "0"), body={"a": "0"})
    assert rule_a != rule_b


def test_unequal_rules_by_body():
    rule_a = Rule(head=("y", "1"), body={"a": "0"})
    rule_b = Rule(head=("y", "1"), body={"a": "1"})
    assert rule_a != rule_b


def test_not_equal_to_a_non_rule_object():
    rule = Rule(head=("y", "1"), body={"a": "0"})
    assert (rule == {"head": ("y", "1"), "body": {"a": "0"}}) is False
    assert (rule == "y(1) :- a(0).") is False


def test_equal_rules_hash_the_same():
    rule_a = Rule(head=("y", "1"), body={"a": "0", "b": "1"})
    rule_b = Rule(head=("y", "1"), body={"b": "1", "a": "0"})
    assert hash(rule_a) == hash(rule_b)


def test_rules_usable_in_a_set():
    # A model holds its rules in a set: equal rules must collapse.
    rule_a = Rule(head=("y", "1"), body={"a": "0"})
    rule_b = Rule(head=("y", "1"), body={"a": "0"})
    assert len({rule_a, rule_b}) == 1


@pytest.mark.parametrize("name", ["__lt__", "__le__", "__gt__", "__ge__"])
def test_ordering_against_a_non_rule_defers_to_the_other_operand(name):
    rule = Rule(("y", "1"))
    assert getattr(rule, name)(1) is NotImplemented
    with pytest.raises(TypeError):
        sorted([rule, 1])


def test_rules_sort_by_their_text():
    rules = [
        Rule.from_string(text)
        for text in ("y(1) :- b(1).", "y(0) :- .", "y(1) :- a(0), b(1).")
    ]
    assert [rule.to_string() for rule in sorted(rules)] == sorted(
        rule.to_string() for rule in rules
    )
    first, second = sorted(rules)[:2]
    assert first < second
    assert first <= second
    assert second > first
    assert second >= first
    assert first <= first
    assert first >= first
    assert not first < first
    assert not first > first


def test_two_rules_sharing_a_text_still_sort_apart():
    # A variable named "a(0), b" makes {"a(0), b": "1"} read as
    # {"a": "0", "b": "1"}.
    tricky = Rule(("y", "1"), {"a(0), b": "1"})
    plain = Rule(("y", "1"), {"a": "0", "b": "1"})
    assert str(tricky) == str(plain)
    assert tricky != plain
    assert (tricky < plain) != (plain < tricky)


def test_a_rule_does_not_compare_with_text():
    with pytest.raises(TypeError):
        Rule(("y", "1")) < "y(1) :- ."  # noqa: B015
