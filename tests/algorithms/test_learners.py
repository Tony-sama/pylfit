"""Tests every learner passes: ``BruteForce``, ``GULA`` and ``PRIDE``.

The return shape of ``fit``, a rule set worked out by hand for
``toy_dataset``, the degenerate and conflicting cases, targets learned
independently, and correctness, completeness and minimality on every
shared fixture.

``BruteForce`` and ``GULA`` are documented to compute *every* minimal,
consistent rule, so these hold for them on any dataset. ``PRIDE`` is
not: it promises consistent, irreducible rules covering every
positive row (``_pride.py``'s module docstring). It passes here
because every ``conftest.py`` dataset is exhaustive over its own
feature-domain product and has no coverage-redundant rules;
``test_pride.py`` shows where it differs.
"""

import pytest

from pylfit import GULA, PRIDE, BruteForce, Dataset, Rule
from tests.algorithms._properties import (
    assert_complete,
    assert_correct,
    assert_heads_in_domain,
    assert_irreducible,
    assert_not_dominated,
)
from tests.algorithms._tables import blanked_frame

LEARNERS = [BruteForce, GULA, PRIDE]

# --- fit() return shape ---
#
# fit(dataset) -> list[Rule], sorted, no duplicates; stateless — no
# dependency on models/ at all, not even RuleSetModel. A caller wanting
# a model wraps the result explicitly, e.g.
# RuleSetModel(dataset, GULA.fit(dataset)).


# SPEC: RULE-5
@pytest.mark.parametrize("learner", LEARNERS)
def test_fit_returns_rules_sorted_by_text_without_duplicates(learner, toy_dataset):
    rules = learner.fit(toy_dataset)
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


@pytest.mark.parametrize("learner", LEARNERS)
def test_toy_dataset_exact_rule_set(learner, toy_dataset):
    rules = learner.fit(toy_dataset)
    expected = {
        Rule(head=("y", "1"), body={"b": "1"}),
        Rule(head=("y", "1"), body={"a": "0", "c": "1"}),
        Rule(head=("y", "0"), body={"b": "0", "c": "0"}),
        Rule(head=("y", "0"), body={"a": "1", "b": "0"}),
    }
    assert set(rules) == expected


# --- degenerate_dataset: target never varies ---


@pytest.mark.parametrize("learner", LEARNERS)
def test_degenerate_dataset_yields_the_single_empty_body_rule(
    learner, degenerate_dataset
):
    rules = learner.fit(degenerate_dataset)
    assert rules == [Rule(head=("y", "1"), body={})]


# --- nondeterministic_dataset: the "complete guarantee" case ---


@pytest.mark.parametrize("learner", LEARNERS)
def test_nondeterministic_target_value_with_no_negatives_gets_the_empty_body_rule(
    learner, nondeterministic_dataset
):
    # a=0 has both y=0 and y=1 rows, a=1 has only y=1 — so "y" is
    # never negative for value "1" (every feature state has a y=1 row
    # somewhere), and dataset.negatives("y", "1") is empty. The
    # empty-body rule is then trivially consistent and maximally
    # general, so it must be the sole rule learned for y=1 — this is
    # exactly the scenario that would silently break the "complete"
    # guarantee if ``negatives`` weren't non-determinism-aware.
    assert nondeterministic_dataset.negatives("y", "1") == []
    rules = learner.fit(nondeterministic_dataset)
    y1_rules = [rule for rule in rules if rule.head == ("y", "1")]
    assert y1_rules == [Rule(head=("y", "1"), body={})]


@pytest.mark.parametrize("learner", LEARNERS)
def test_nondeterministic_target_value_with_negatives_gets_a_specific_rule(
    learner, nondeterministic_dataset
):
    # value "0" is only ever observed at a=0, so a=1 is a genuine
    # negative for it — the learned rule must exclude a=1.
    assert nondeterministic_dataset.negatives("y", "0") != []
    rules = learner.fit(nondeterministic_dataset)
    y0_rules = [rule for rule in rules if rule.head == ("y", "0")]
    assert y0_rules == [Rule(head=("y", "0"), body={"a": "0"})]


# --- multi-target: targets learned independently ---


@pytest.mark.parametrize("learner", LEARNERS)
def test_targets_are_learned_independently(learner, multi_target_dataset):
    rules = learner.fit(multi_target_dataset)
    heads = {rule.head for rule in rules}
    assert heads == {("y1", "0"), ("y1", "1"), ("y2", "0"), ("y2", "1")}
    by_head = {rule.head: rule for rule in rules}
    assert by_head[("y1", "1")].body == {"a": "1"}
    assert by_head[("y1", "0")].body == {"a": "0"}
    assert by_head[("y2", "1")].body == {"a": "0"}
    assert by_head[("y2", "0")].body == {"a": "1"}


# --- generic properties, checked against every fixture ---

_FIXTURE_NAMES = ["toy_dataset", "nondeterministic_dataset", "degenerate_dataset"]


# SPEC: LEARN-1
@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
@pytest.mark.parametrize("learner", LEARNERS)
def test_every_rule_head_value_is_in_the_targets_domain(learner, fixture_name, request):
    dataset = request.getfixturevalue(fixture_name)
    assert_heads_in_domain(dataset, learner.fit(dataset))


# SPEC: LEARN-2
@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
@pytest.mark.parametrize("learner", LEARNERS)
def test_correctness_no_rule_matches_a_negative(learner, fixture_name, request):
    # Unlike completeness, this holds for PRIDE on any dataset: its
    # minimization never keeps a rule that matches a negative.
    dataset = request.getfixturevalue(fixture_name)
    assert_correct(dataset, learner.fit(dataset))


# SPEC: GULA-3
@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
@pytest.mark.parametrize("learner", LEARNERS)
def test_completeness_every_feature_state_is_matched_per_target(
    learner, fixture_name, request
):
    dataset = request.getfixturevalue(fixture_name)
    assert_complete(dataset, learner.fit(dataset))


@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
@pytest.mark.parametrize("learner", LEARNERS)
def test_minimality_no_rule_is_dominated_by_another(learner, fixture_name, request):
    dataset = request.getfixturevalue(fixture_name)
    assert_not_dominated(dataset, learner.fit(dataset))


# SPEC: LEARN-3
@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
@pytest.mark.parametrize("learner", LEARNERS)
def test_minimality_no_condition_can_be_dropped(learner, fixture_name, request):
    dataset = request.getfixturevalue(fixture_name)
    assert_irreducible(dataset, learner.fit(dataset))


# SPEC: GULA-3
@pytest.mark.parametrize("learner", [BruteForce, GULA])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("seed", range(10))
def test_completeness_holds_whatever_the_unknowns_in_the_data(
    learner, deterministic, seed
):
    dataset = Dataset(
        blanked_frame(seed, deterministic=deterministic),
        features=["a", "b", "c"],
        targets=["y"],
        assume_nondeterministic=not deterministic,
    )
    assert_complete(dataset, learner.fit(dataset, warn_uncovered=False))
