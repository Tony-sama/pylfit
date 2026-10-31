"""Tests for ``BruteForce``: correctness, completeness, minimality.

The three guarantees ``_brute_force.py``'s module docstring documents,
checked on every shared fixture, plus a rule set worked out by hand
for ``toy_dataset`` and the conflicting target value with no negatives
at all, which a row-local ``Dataset.negatives`` would get wrong. Also:
targets learned independently, the candidates and their count,
``_judge``, the same rules as ``GULA`` on unknown values, ``verbose``,
and repeated negatives judged once.
"""

import pandas as pd
import pytest

from pylfit import GULA, BruteForce, Dataset, Rule
from tests.algorithms._properties import (
    assert_complete,
    assert_correct,
    assert_heads_in_domain,
    assert_irreducible,
    assert_not_dominated,
)
from tests.algorithms._tables import blanked_frame

# --- fit() return shape ---
#
# fit(dataset) -> list[Rule], sorted, no duplicates; stateless — no
# dependency on models/ at all, not even RuleSetModel. A caller wanting
# a model wraps the result explicitly, e.g.
# RuleSetModel(dataset, BruteForce.fit(dataset)).


def test_fit_returns_rules_sorted_by_text_without_duplicates(toy_dataset):
    rules = BruteForce.fit(toy_dataset)
    assert isinstance(rules, list)
    assert rules  # sanity: toy_dataset isn't degenerate
    assert rules == sorted(set(rules))
    assert all(isinstance(rule, Rule) for rule in rules)


# --- toy_dataset: hand-checked exact rule set ---
#
# y = 1 when b = 1, or when a = 0 and c = 1 (conftest.py's docstring).
# Worked out by hand against the 8-row full factorial:
#   y=1 rows: (0,0,1) (0,1,0) (0,1,1) (1,1,0) (1,1,1)
#   y=0 rows: (0,0,0) (1,0,0) (1,0,1)
# For y=1: "b=1" (matches all 4 b=1 rows, none of the 3 y=0 rows has
# b=1) and "a=0,c=1" (matches (0,0,1)/(0,1,1), no y=0 row has a=0,c=1)
# are the only consistent, non-dominated candidates.
# For y=0: "b=0,c=0" (matches (0,0,0)/(1,0,0)) and "a=1,b=0" (matches
# (1,0,0)/(1,0,1)) are the only consistent, non-dominated candidates —
# every size-1 candidate for y=0 turns out inconsistent (e.g. "b=0"
# alone also matches the y=1 row (0,0,1)).


def test_toy_dataset_exact_rule_set(toy_dataset):
    rules = BruteForce.fit(toy_dataset)
    expected = {
        Rule(head=("y", "1"), body={"b": "1"}),
        Rule(head=("y", "1"), body={"a": "0", "c": "1"}),
        Rule(head=("y", "0"), body={"b": "0", "c": "0"}),
        Rule(head=("y", "0"), body={"a": "1", "b": "0"}),
    }
    assert set(rules) == expected


# --- degenerate_dataset: target never varies ---


def test_degenerate_dataset_yields_the_single_empty_body_rule(degenerate_dataset):
    rules = BruteForce.fit(degenerate_dataset)
    assert rules == [Rule(head=("y", "1"), body={})]


# --- nondeterministic_dataset: the "complete guarantee" case ---


def test_nondeterministic_target_value_with_no_negatives_gets_the_empty_body_rule(
    nondeterministic_dataset,
):
    # a=0 has both y=0 and y=1 rows, a=1 has only y=1 — so "y" is
    # never negative for value "1" (every feature state has a y=1 row
    # somewhere), and dataset.negatives("y", "1") is empty. The
    # empty-body rule is then trivially consistent and maximally
    # general, so it must be the sole rule learned for y=1 — this is
    # exactly the scenario that would silently break the "complete"
    # guarantee if ``negatives`` weren't non-determinism-aware.
    assert nondeterministic_dataset.negatives("y", "1") == []
    rules = BruteForce.fit(nondeterministic_dataset)
    y1_rules = [rule for rule in rules if rule.head == ("y", "1")]
    assert y1_rules == [Rule(head=("y", "1"), body={})]


def test_nondeterministic_target_value_with_negatives_gets_a_specific_rule(
    nondeterministic_dataset,
):
    # value "0" is only ever observed at a=0, so a=1 is a genuine
    # negative for it — the learned rule must exclude a=1.
    assert nondeterministic_dataset.negatives("y", "0") != []
    rules = BruteForce.fit(nondeterministic_dataset)
    y0_rules = [rule for rule in rules if rule.head == ("y", "0")]
    assert y0_rules == [Rule(head=("y", "0"), body={"a": "0"})]


# --- multi-target: targets learned independently ---


def test_targets_are_learned_independently(multi_target_dataset):
    rules = BruteForce.fit(multi_target_dataset)
    heads = {rule.head for rule in rules}
    assert heads == {("y1", "0"), ("y1", "1"), ("y2", "0"), ("y2", "1")}
    by_head = {rule.head: rule for rule in rules}
    assert by_head[("y1", "1")].body == {"a": "1"}
    assert by_head[("y1", "0")].body == {"a": "0"}
    assert by_head[("y2", "1")].body == {"a": "0"}
    assert by_head[("y2", "0")].body == {"a": "1"}


# --- generic properties, checked against every fixture ---

_FIXTURE_NAMES = ["toy_dataset", "nondeterministic_dataset", "degenerate_dataset"]


@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
def test_every_rule_head_value_is_in_the_targets_domain(fixture_name, request):
    dataset = request.getfixturevalue(fixture_name)
    assert_heads_in_domain(dataset, BruteForce.fit(dataset))


@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
def test_correctness_no_rule_matches_a_negative(fixture_name, request):
    dataset = request.getfixturevalue(fixture_name)
    assert_correct(dataset, BruteForce.fit(dataset))


@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
def test_completeness_every_feature_state_is_matched_per_target(fixture_name, request):
    dataset = request.getfixturevalue(fixture_name)
    assert_complete(dataset, BruteForce.fit(dataset))


@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
def test_minimality_no_rule_is_dominated_by_another(fixture_name, request):
    dataset = request.getfixturevalue(fixture_name)
    assert_not_dominated(dataset, BruteForce.fit(dataset))


@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
def test_minimality_no_condition_can_be_dropped(fixture_name, request):
    dataset = request.getfixturevalue(fixture_name)
    assert_irreducible(dataset, BruteForce.fit(dataset))


@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
def test_the_fully_general_rule_is_always_a_candidate(fixture_name, request):
    # Every head's candidates include the empty-body rule, and a head
    # with no negatives keeps it: nothing contradicts it.
    dataset = request.getfixturevalue(fixture_name)
    rules = BruteForce.fit(dataset)
    for target in dataset.targets:
        for value in dataset.domain(target):
            general = Rule(head=(target, value), body={})
            assert general in BruteForce._candidates(dataset, target, value)
            if dataset.negatives(target, value) == []:
                assert general in rules


def test_candidate_count_matches_the_candidates_generated(toy_dataset):
    # The count is the progress total and the verbose summary, computed
    # up front because the candidates themselves are generated lazily.
    generated = list(BruteForce._candidates(toy_dataset, "y", "1"))
    assert BruteForce._candidate_count(toy_dataset) == len(generated) == 3**3
    assert len(set(generated)) == len(generated)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # toy_dataset: y is 1 when b is 1, or when a is 0 and c is 1;
        # y(1)'s negatives are 000, 100 and 101.
        ("y(1) :- .", (False, False)),  # matches every negative
        ("y(1) :- c(1).", (False, False)),  # matches 101
        ("y(1) :- b(1).", (True, True)),
        ("y(1) :- a(0), c(1).", (True, True)),
        ("y(1) :- a(0), b(1).", (True, False)),  # b(1) alone is consistent
        ("y(1) :- a(0), b(1), c(1).", (True, False)),
    ],
)
def test_judge_tells_consistent_from_minimal(toy_dataset, text, expected):
    rule = Rule.from_string(text)
    assert BruteForce._judge(rule, toy_dataset.negatives("y", "1")) == expected


def test_judge_empty_body_is_minimal_when_there_are_no_negatives():
    assert BruteForce._judge(Rule.from_string("y(1) :- ."), []) == (True, True)


# --- unknown values: the same rules as GULA ---
#
# GULA is tested against BruteForce on complete data; with unknowns,
# both read "matches" the same way (an unknown satisfies no condition),
# so they must still learn exactly the same rules.


@pytest.mark.filterwarnings("ignore:Rows with the same feature state")
@pytest.mark.filterwarnings("ignore:.*matched by no rule of their head")
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("assume_nondeterministic", [False, True])
def test_brute_force_and_gula_agree_with_unknowns(
    seed, deterministic, assume_nondeterministic
):
    # At 20% unknowns the assumption sets aside most negatives of these
    # random tables, which would make the comparison empty.
    rate = 0.05 if assume_nondeterministic else 0.2
    dataset = Dataset(
        blanked_frame(seed, rate=rate, deterministic=deterministic),
        features=["a", "b", "c"],
        targets=["y"],
        assume_nondeterministic=assume_nondeterministic,
    )
    assert BruteForce.fit(dataset) == GULA.fit(dataset)


def test_brute_force_never_tests_an_unknown():
    df = pd.DataFrame(
        {"a": [None, "1", "0"], "b": ["1", "1", "0"], "y": ["1", "0", "0"]}
    )
    dataset = Dataset(df, features=["a", "b"], targets=["y"])
    for rule in BruteForce.fit(dataset, warn_uncovered=False):
        assert None not in rule.body.values()


def test_verbose_counts_the_consistent_candidates(toy_dataset, capsys):
    BruteForce.fit(toy_dataset, verbose=True)
    err = capsys.readouterr().err
    total = BruteForce._candidate_count(toy_dataset)
    for target in toy_dataset.targets:
        for value in toy_dataset.domain(target):
            negatives = toy_dataset.negatives(target, value)
            consistent = sum(
                not any(rule.matches(negative) for negative in negatives)
                for rule in BruteForce._candidates(toy_dataset, target, value)
            )
            assert (
                f"{target}={value}: {total} candidates, {consistent} consistent" in err
            )


def test_repeated_negative_rows_are_judged_once(monkeypatch):
    df = pd.DataFrame({"a": ["0"] * 50 + ["1"], "y": ["0"] * 50 + ["1"]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    judge = BruteForce._judge
    seen = []

    def counting_judge(rule, negatives):
        seen.append(len(negatives))
        return judge(rule, negatives)

    monkeypatch.setattr(BruteForce, "_judge", staticmethod(counting_judge))
    rules = BruteForce.fit(dataset)
    assert set(seen) == {1}
    assert {rule.to_string() for rule in rules} == {"y(0) :- a(0).", "y(1) :- a(1)."}
