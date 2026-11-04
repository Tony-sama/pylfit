"""Tests for ``PRIDE``: its guarantees, its parts, and ``GULA``.

The generic-property tests (correctness/completeness/minimality) are
``test_learners.py``'s, but ``PRIDE`` is **not** documented to compute
the same unique minimal, complete rule set ``BruteForce``/``GULA`` do
(``_pride.py``'s module docstring lists what it promises instead).
Two distinct, provable relationships:

- **``PRIDE``'s rules are always a subset of ``GULA``'s**, on any
  dataset, exhaustive or not — every PRIDE rule is individually
  consistent and irreducible, which is exactly ``GULA``'s membership
  test for its own output, so ``GULA`` (documented to find *every* such
  rule) necessarily finds each one too. ``GULA`` can still end up with
  *more* rules than ``PRIDE`` — whenever two or more irreducible rules
  exist with overlapping positive coverage, ``PRIDE`` stops once one of
  them covers what's needed, but ``GULA`` finds both regardless of
  coverage. This can happen even on an exhaustive dataset — see
  ``test_pride_can_be_a_strict_subset_of_gula_even_on_exhaustive_data``.
- **Completeness over the full feature-domain product** only holds for
  ``PRIDE`` when the dataset is exhaustive (every combination present
  as some row) —
  ``test_pride_can_be_incomplete_on_a_non_exhaustive_dataset``
  demonstrates the general (non-exhaustive) case, a separate gap from
  the subset point above.

Every dataset in ``conftest.py`` happens to have no coverage-redundant
rules for any head, so ``PRIDE``'s rules equal ``GULA``'s exactly on all
of them — confirmed by directly checking, not something to assume
holds by construction; that's what the subset-based tests below are
for.

Also tested: ``_minimize`` and ``_covering_rule``, row order and
``try_all_conditions``, unknown values and the options shared with
``GULA``, and the rebuild of a seed under a support threshold.
"""

import random
import warnings
from collections import defaultdict
from itertools import product

import pandas as pd
import pytest

from pylfit import GULA, PRIDE, Dataset, PylfitWarning, Rule
from pylfit.algorithms._pride import _Rows
from tests.algorithms._tables import as_question_marks, blanked_frame

# The coverage warning fires whenever an option or unknown values leave
# a positive row uncovered, which these tests do on purpose; the
# warning itself is tested in test_coverage_warning.py.
pytestmark = pytest.mark.filterwarnings("ignore:.*matched by no rule of their head")

# --- nondeterministic_dataset: a head with no negatives ---


@pytest.mark.parametrize("option", ["min_support", "min_possible_support"])
def test_a_head_with_no_negatives_gets_no_rule_below_the_threshold(
    nondeterministic_dataset, option
):
    # Its only candidate is the empty-body rule, matching its 2
    # positives: below a threshold of 3 it must go, as GULA drops it.
    assert len(nondeterministic_dataset.positives("y", "1")) == 2
    with pytest.warns(PylfitWarning, match=option):
        rules = PRIDE.fit(nondeterministic_dataset, **{option: 3})
    assert not [rule for rule in rules if rule.head == ("y", "1")]
    with pytest.warns(PylfitWarning):
        gula = GULA.fit(nondeterministic_dataset, **{option: 3})
    assert not [rule for rule in gula if rule.head == ("y", "1")]


@pytest.fixture
def partial_dataset():
    """A partial table: ``y=1`` iff ``a == b``, two states never seen.

    Two feature combinations — ``(a=0, b=2)`` and ``(a=2, b=0)`` —
    never appear as any row at all, neither a positive nor a negative
    for either value of ``y``. Every other combination of ``a``/``b``
    (both with domain ``{0, 1, 2}``) is present.

    Built specifically to demonstrate PRIDE's one documented gap
    against ``BruteForce``/``GULA`` (see ``_pride.py``'s module
    docstring): ``GULA``'s fully general starting rule keeps a
    genuinely unseen combination covered by default, since nothing
    ever forces a specialization away from it; PRIDE's rules are built
    purely from observed positive examples, so a combination sharing no
    rule's exact conditions can end up matched by nothing.
    """
    rows = [
        (0, 0, 1),
        (1, 1, 1),
        (2, 2, 1),
        (0, 1, 0),
        (1, 0, 0),
        (1, 2, 0),
        (2, 1, 0),
    ]
    df = pd.DataFrame(rows, columns=["a", "b", "y"])
    return Dataset(df, features=["a", "b"], targets=["y"])


# --- minimality beyond the shared fixtures ---
#
# test_learners.py checks correctness, completeness and minimality on
# the conftest.py datasets in its _FIXTURE_NAMES. All three are
# exhaustive over their own feature-domain product, which is why
# completeness holds for PRIDE there — see this file's module
# docstring and test_pride_can_be_incomplete_on_a_non_exhaustive_dataset
# below for the general case.

_FIXTURE_NAMES = [
    "toy_dataset",
    "nondeterministic_dataset",
    "degenerate_dataset",
    "support_dataset",
]


def test_minimality_drops_a_condition_the_greedy_choice_left_behind():
    # The shared fixtures never make PRIDE choose a condition it later
    # turns out not to need; this table does. The first condition picked
    # for y=0 is a(1), and c(0) alone is enough once it is added.
    frame = pd.DataFrame(
        {
            "a": ["0", "1", "1"],
            "b": ["1", "1", "1"],
            "c": ["1", "1", "0"],
            "d": ["0", "0", "1"],
            "y": ["1", "1", "0"],
        }
    )
    dataset = Dataset(frame, features=["a", "b", "c", "d"], targets=["y"])
    assert PRIDE.fit(dataset) == [
        Rule(("y", "0"), {"c": "0"}),
        Rule(("y", "1"), {"c": "1"}),
    ]


# --- PRIDE's rules are always a subset of GULA's ---
#
# The real, general invariant (see this file's and _pride.py's module
# docstrings): every PRIDE rule is individually consistent and
# irreducible, which is exactly what makes it a rule GULA also finds.
# Checked directly here rather than just argued in prose — on the
# conftest.py fixtures in _FIXTURE_NAMES, on multi_target_dataset,
# and on partial_dataset (defined above, where PRIDE is also
# incomplete over the domain product — a separate property, but the
# subset relationship holds regardless).


@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
def test_pride_rules_are_always_a_subset_of_gulas(fixture_name, request):
    dataset = request.getfixturevalue(fixture_name)
    gula_rules = GULA.fit(dataset)
    pride_rules = PRIDE.fit(dataset)
    assert set(pride_rules) <= set(gula_rules)


def test_pride_rules_are_a_subset_of_gulas_on_a_multi_target_dataset(
    multi_target_dataset,
):
    gula_rules = GULA.fit(multi_target_dataset)
    pride_rules = PRIDE.fit(multi_target_dataset)
    assert set(pride_rules) <= set(gula_rules)


def test_pride_rules_are_a_subset_of_gulas_on_a_partial_dataset(partial_dataset):
    gula_rules = GULA.fit(partial_dataset)
    pride_rules = PRIDE.fit(partial_dataset)
    assert set(pride_rules) <= set(gula_rules)


# --- exact equality, confirmed to hold on these specific fixtures ---
#
# Not a general property — see the subset tests above and
# test_pride_can_be_a_strict_subset_of_gula_even_on_exhaustive_data
# below, where a different (also exhaustive) dataset gets a strictly
# smaller PRIDE rule set. These fixtures just happen to have no
# coverage-redundant rule for any head; confirmed by running both
# algorithms, not assumed from exhaustiveness alone.


@pytest.mark.parametrize("fixture_name", _FIXTURE_NAMES)
def test_pride_and_gula_learn_the_same_rules_on_these_fixtures(fixture_name, request):
    dataset = request.getfixturevalue(fixture_name)
    gula_rules = GULA.fit(dataset)
    pride_rules = PRIDE.fit(dataset)
    assert pride_rules == gula_rules


def test_pride_and_gula_agree_on_a_multi_target_dataset(multi_target_dataset):
    gula_rules = GULA.fit(multi_target_dataset)
    pride_rules = PRIDE.fit(multi_target_dataset)
    assert pride_rules == gula_rules


# --- a strict subset, confirmed even on an exhaustive dataset ---


@pytest.fixture
def redundant_coverage_dataset():
    """A full factorial where ``GULA`` finds a rule ``PRIDE`` needn't.

    Exhaustive, over three binary features: ``GULA`` finds a
    coverage-redundant maximal rule that ``PRIDE`` never constructs.

    Found by search, not designed by hand — kept exactly as found
    since the specific rows matter (a hand-simplified version might
    lose the redundancy). ``y(1) :- f1(1), f2(0).`` is consistent and
    irreducible (so ``GULA`` keeps it), but every positive it matches
    is already covered by ``y(1) :- f0(1), f2(0).`` or
    ``y(1) :- f0(0), f1(1).``, so ``PRIDE`` never builds it.
    """
    rows = [
        (0, 0, 0, 0),
        (0, 0, 1, 0),
        (0, 1, 0, 1),
        (0, 1, 1, 1),
        (1, 0, 0, 1),
        (1, 0, 1, 1),
        (1, 1, 0, 1),
        (1, 1, 1, 0),
    ]
    df = pd.DataFrame(rows, columns=["f0", "f1", "f2", "y"])
    return Dataset(df, features=["f0", "f1", "f2"], targets=["y"])


def test_pride_can_be_a_strict_subset_of_gula_even_on_exhaustive_data(
    redundant_coverage_dataset,
):
    """Even on exhaustive data, PRIDE can learn fewer rules than GULA.

    Locks in the point this file's module docstring makes: PRIDE's
    subset-of-GULA guarantee can be a *strict* subset even when the
    dataset is exhaustive — exhaustiveness only fixes the separate,
    unseen-combination completeness gap (see
    ``test_pride_can_be_incomplete_on_a_non_exhaustive_dataset``), not
    this one.
    """
    gula_rules = GULA.fit(redundant_coverage_dataset)
    pride_rules = PRIDE.fit(redundant_coverage_dataset)

    assert set(pride_rules) < set(gula_rules)  # strict subset
    redundant_rule = Rule(head=("y", "1"), body={"f1": "1", "f2": "0"})
    assert redundant_rule in gula_rules
    assert redundant_rule not in pride_rules


# --- the one documented gap: completeness on non-exhaustive data ---


def test_pride_can_be_incomplete_on_a_non_exhaustive_dataset(partial_dataset):
    """States no row shows: GULA's rules match them, PRIDE's don't.

    Locks in PRIDE's documented completeness gap as expected
    behavior, not a latent bug — if this starts failing, either a
    change to ``_pride.py`` narrowed the gap (update its module
    docstring too) or something regressed.
    """
    gula_rules = GULA.fit(partial_dataset)
    pride_rules = PRIDE.fit(partial_dataset)

    unseen_states = [{"a": "0", "b": "2"}, {"a": "2", "b": "0"}]

    for state in unseen_states:
        assert any(rule.matches(state) for rule in gula_rules), (
            f"GULA unexpectedly failed to cover unseen state {state}"
        )
        assert not any(rule.matches(state) for rule in pride_rules), (
            f"PRIDE unexpectedly covered unseen state {state}; the "
            "documented completeness gap may have narrowed"
        )


# --- _minimize: the drops of rescanning every negative ---


def _minimize_by_rescanning(rule, negatives, features):
    """The plain loop ``_minimize`` shortcuts.

    Drops each condition in ``features`` order, keeping the drop if the
    rule then matches no negative.
    """
    for feature in [feature for feature in features if feature in rule.body]:
        shorter = rule.without_condition(feature)
        if not any(shorter.matches(negative) for negative in negatives):
            rule = shorter
    return rule


def _rows(negatives, positives=()):
    """A head's ``_Rows`` from states, as ``Dataset`` would mark them."""

    def masks(states):
        marked = defaultdict(int)
        for i, state in enumerate(states):
            for condition in state.items():
                marked[condition] |= 1 << i
        return marked

    unknown = defaultdict(int)
    for i, state in enumerate(positives):
        for feature, value in state.items():
            if value is None:
                unknown[feature] |= 1 << i
    return _Rows(
        negatives,
        masks(negatives),
        (1 << len(negatives)) - 1,
        masks(positives),
        unknown,
        (1 << len(positives)) - 1,
    )


def test_minimize_keeps_a_condition_a_drop_made_needed():
    # a is needed (the first negative violates it alone). The second
    # violates b and c only: b can go, then c can't.
    rule = Rule(head=("y", "1"), body={"a": "1", "b": "1", "c": "1"})
    negatives = [
        {"a": "0", "b": "1", "c": "1"},
        {"a": "1", "b": "0", "c": "0"},
        {"a": "0", "b": "0", "c": "0"},
    ]
    rule = PRIDE._minimize(rule, _rows(negatives), ["a", "b", "c"])
    assert rule.body == {"a": "1", "c": "1"}


@pytest.mark.parametrize("seed", range(40))
def test_minimize_drops_what_rescanning_every_negative_drops(seed):
    rng = random.Random(seed)
    features = [f"x{i}" for i in range(rng.randint(3, 7))]
    negatives = [
        {feature: rng.choice("012") for feature in features}
        for _ in range(rng.randint(1, 60))
    ]
    seed_state = {feature: rng.choice("012") for feature in features}
    # The fully specific rule on a state that is not a negative is
    # consistent, the precondition of _minimize.
    negatives = [negative for negative in negatives if negative != seed_state]
    shuffled = features[:]
    rng.shuffle(shuffled)
    body = {feature: seed_state[feature] for feature in shuffled}
    expected = Rule(head=("y", "1"), body=dict(body))
    expected = _minimize_by_rescanning(expected, negatives, shuffled)
    rule = Rule(head=("y", "1"), body=dict(body))
    rule = PRIDE._minimize(rule, _rows(negatives), shuffled)
    # dict equality: the order conditions end up in carries no meaning.
    assert rule.body == expected.body


# --- _covering_rule ---


def test_covering_rule_grows_on_the_first_disagreeing_feature(toy_dataset):
    # Nothing uncovered to keep matching: the first feature wins.
    seed = {"a": "0", "b": "0", "c": "1"}
    negative = {"a": "0", "b": "1", "c": "0"}
    rule = PRIDE._covering_rule("y", "1", seed, _rows([negative]), toy_dataset.features)
    assert rule == Rule(head=("y", "1"), body={"b": "0"})


def test_covering_rule_grows_on_the_feature_keeping_most_uncovered(toy_dataset):
    seed = {"a": "0", "b": "0", "c": "1"}
    negative = {"a": "0", "b": "1", "c": "0"}
    # b(0) keeps one of these, c(1) keeps two.
    positives = [
        {"a": "1", "b": "1", "c": "1"},
        {"a": "1", "b": "0", "c": "1"},
        {"a": "1", "b": "1", "c": "0"},
    ]
    rows = _rows([negative], positives)
    rule = PRIDE._covering_rule(
        "y", "1", seed, rows, toy_dataset.features, uncovered=rows.every_positive
    )
    assert rule == Rule(head=("y", "1"), body={"c": "1"})


def test_covering_rule_ties_go_to_the_first_feature(toy_dataset):
    seed = {"a": "0", "b": "0", "c": "1"}
    negative = {"a": "0", "b": "1", "c": "0"}
    rows = _rows([negative], [{"a": "1", "b": "0", "c": "1"}])
    rule = PRIDE._covering_rule(
        "y", "1", seed, rows, toy_dataset.features, uncovered=rows.every_positive
    )
    assert rule == Rule(head=("y", "1"), body={"b": "0"})


def test_covering_rule_starts_from_the_forced_condition(toy_dataset):
    seed = {"a": "0", "b": "0", "c": "1"}
    negative = {"a": "0", "b": "1", "c": "0"}
    rule = PRIDE._covering_rule(
        "y", "1", seed, _rows([negative]), toy_dataset.features, forced=("c", "1")
    )
    assert rule == Rule(head=("y", "1"), body={"c": "1"})


def test_minimize_tries_the_last_feature_last():
    # Either condition alone excludes the negative: the one tried last
    # stays.
    negatives = [{"a": "0", "b": "0"}]
    for last, kept in ((None, {"b": "1"}), ("a", {"a": "1"})):
        rule = Rule(head=("y", "1"), body={"a": "1", "b": "1"})
        rule = PRIDE._minimize(rule, _rows(negatives), ["a", "b"], last=last)
        assert rule.body == kept


def test_minimize_tries_conditions_in_feature_order():
    # Not in body order, which is alphabetical: renaming the variables
    # must not change which redundant condition goes.
    negatives = [{"a": "0", "b": "0"}]
    for features, kept in ((["a", "b"], {"b": "1"}), (["b", "a"], {"a": "1"})):
        rule = Rule(head=("y", "1"), body={"a": "1", "b": "1"})
        assert PRIDE._minimize(rule, _rows(negatives), features).body == kept


def test_renaming_the_variables_does_not_change_the_rules():
    rows = [
        ("1", "0", "1", "1"),
        ("0", "1", "1", "0"),
        ("1", "1", "0", "0"),
        ("1", "0", "0", "0"),
    ]
    learned = []
    for names in (["a", "b", "c"], ["z", "y", "x"]):
        df = pd.DataFrame(rows, columns=[*names, "t"])
        dataset = Dataset(df, features=names, targets=["t"])
        rules = PRIDE.fit(dataset, heads=[("t", "1")])
        renamed = dict(zip(names, "abc", strict=True))
        learned.append(
            {
                str(Rule(r.head, {renamed[f]: v for f, v in r.body.items()}))
                for r in rules
            }
        )
    assert learned[0] == learned[1]


def test_covering_rule_is_none_when_no_feature_separates_seed_and_negative(
    toy_dataset,
):
    seed = {"a": "0", "b": None, "c": "1"}
    negative = {"a": "0", "b": "1", "c": "1"}
    assert (
        PRIDE._covering_rule("y", "1", seed, _rows([negative]), toy_dataset.features)
        is None
    )


# --- row order, and try_all_conditions ---


def _truth_table_both_ways():
    """A full truth table, forwards and backwards (found by search)."""
    states = list(product(["0", "1"], repeat=3))
    targets = ["0", "0", "0", "1", "1", "0", "0", "0"]
    df = pd.DataFrame(
        [(*state, y) for state, y in zip(states, targets, strict=True)],
        columns=["a", "b", "c", "y"],
    )
    forward = Dataset(df, features=["a", "b", "c"], targets=["y"])
    backward = Dataset(
        df.iloc[::-1].reset_index(drop=True), features=["a", "b", "c"], targets=["y"]
    )
    return forward, backward


def test_row_order_can_change_the_rules_even_on_exhaustive_data():
    """Reversing a truth table's rows changes PRIDE's ``y(0)`` rules.

    Seeds are taken in row order; both orders still cover every
    positive.
    """
    forward, backward = _truth_table_both_ways()
    assert {
        r.to_string() for r in set(PRIDE.fit(forward)) - set(PRIDE.fit(backward))
    } == {
        "y(0) :- a(0), b(0).",
        "y(0) :- a(1), c(1).",
        "y(0) :- b(1), c(0).",
    }
    for dataset in (forward, backward):
        rules = PRIDE.fit(dataset)
        for value in ("0", "1"):
            for positive in dataset.positives("y", value):
                assert any(
                    rule.head == ("y", value) and rule.matches(positive)
                    for rule in rules
                )


def test_try_all_conditions_finds_the_rules_row_order_hid():
    """On the same table, every ``y(0)`` rule GULA finds, either way."""
    forward, backward = _truth_table_both_ways()
    gula = GULA.fit(forward)
    for dataset in (forward, backward):
        assert PRIDE.fit(dataset, try_all_conditions=True) == gula


# SPEC: PRIDE-1
@pytest.mark.parametrize("seed", range(20))
def test_try_all_conditions_rules_are_gula_rules_and_cover(seed):
    rng = random.Random(seed)
    features = ["a", "b", "c", "d"]
    rows = [
        {**{f: rng.choice("012") for f in features}, "y": rng.choice("01")}
        for _ in range(rng.randint(5, 40))
    ]
    df = pd.DataFrame(rows).drop_duplicates(subset=features)
    dataset = Dataset(df, features=features, targets=["y"])
    rules = PRIDE.fit(dataset, try_all_conditions=True)
    assert set(rules) <= set(GULA.fit(dataset))
    assert set(PRIDE.fit(dataset)) <= set(GULA.fit(dataset))
    for value in dataset.domain("y"):
        for positive in dataset.positives("y", value):
            assert any(r.head == ("y", value) and r.matches(positive) for r in rules)


# --- unknown values and the options shared with GULA ---
#
# As for GULA: None in a state reads as "?" written as an ordinary
# string with every "?" condition left out by allowed_conditions.


def _blanked(seed, deterministic=False):
    return Dataset(
        blanked_frame(seed, deterministic=deterministic),
        features=["a", "b", "c"],
        targets=["y"],
    )


@pytest.mark.filterwarnings("ignore:Rows with the same feature state")
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("min_support", [None, 1, 2])
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("try_all_conditions", [False, True])
def test_pride_unknowns_learn_what_question_marks_without_their_conditions_learn(
    seed, min_support, deterministic, try_all_conditions
):
    frame = blanked_frame(seed, deterministic=deterministic)
    dataset = Dataset(frame, features=["a", "b", "c"], targets=["y"])
    reference, allowed = as_question_marks(frame)
    options = {"min_support": min_support, "try_all_conditions": try_all_conditions}
    assert PRIDE.fit(dataset, **options) == PRIDE.fit(
        reference, allowed_conditions=allowed, **options
    )


# SPEC: PRIDE-1
@pytest.mark.filterwarnings("ignore:Rows with the same feature state")
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"min_support": 2},
        {"min_support": 0.3},
        {"min_possible_support": 3},
        {"min_support": 1, "min_possible_support": 3},
        {"try_all_conditions": True},
        {"try_all_conditions": True, "min_support": 2},
    ],
)
def test_pride_rules_with_unknowns_are_gula_rules(seed, options):
    dataset = _blanked(seed)
    shared = {k: v for k, v in options.items() if k != "try_all_conditions"}
    assert set(PRIDE.fit(dataset, **options)) <= set(GULA.fit(dataset, **shared))


# SPEC: PRIDE-2
@pytest.mark.filterwarnings("ignore:Rows with the same feature state")
@pytest.mark.parametrize("seed", range(8))
def test_pride_covers_every_positive_but_the_inseparable(seed):
    dataset = _blanked(seed)
    rules = PRIDE.fit(dataset)
    for value in dataset.domain("y"):
        negatives = dataset.negatives("y", value)
        for positive in dataset.positives("y", value):
            covered = any(
                rule.head == ("y", value) and rule.matches(positive) for rule in rules
            )
            assert covered != PRIDE._inseparable(positive, negatives, dataset.features)


# SPEC: PRIDE-3
@pytest.mark.filterwarnings("ignore:Rows with the same feature state")
@pytest.mark.parametrize("seed", range(4))
def test_pride_min_support_one_is_plain_pride(seed):
    dataset = _blanked(seed)
    assert PRIDE.fit(dataset, min_support=1) == PRIDE.fit(dataset)


def test_pride_never_grows_on_an_unknown():
    df = pd.DataFrame({"a": [None, "1"], "b": ["1", "1"], "y": ["1", "0"]})
    dataset = Dataset(df, features=["a", "b"], targets=["y"])
    # The y=1 seed knows only b, which doesn't separate it from a=1,
    # b=1: no rule, rather than a(None).
    assert PRIDE.fit(dataset, heads=[("y", "1")]) == []


def _binary_dataset(rows):
    """A table of 0/1 features ``x0``, ``x1``, ... and target ``y``."""
    features = [f"x{i}" for i in range(len(rows[0]) - 1)]
    df = pd.DataFrame(rows, columns=[*features, "y"]).astype(str)
    return Dataset(df, features=features, targets=["y"])


def _rules_of(rules, value):
    return sorted(rule.to_string() for rule in rules if rule.head == ("y", value))


def test_pride_support_aware_rule_keeps_the_threshold():
    # Two of the plain y=0 rules match one row each: with min_support=2
    # their seeds are rebuilt, choosing among the conditions that keep
    # the support by the positives still to cover.
    dataset = _binary_dataset(
        [
            (0, 0, 0, 1),
            (1, 0, 1, 0),
            (1, 1, 0, 0),
            (0, 1, 1, 0),
            (1, 0, 0, 0),
            (1, 1, 1, 1),
        ]
    )
    assert _rules_of(PRIDE.fit(dataset), "0") == [
        "y(0) :- x0(0), x1(1).",
        "y(0) :- x0(1), x1(0).",
        "y(0) :- x1(1), x2(0).",
    ]
    supported = PRIDE.fit(dataset, min_support=2)
    assert _rules_of(supported, "0") == ["y(0) :- x0(1), x1(0)."]


def test_pride_extra_rules_are_rebuilt_from_their_condition():
    # With a threshold, each extra rule is rebuilt around the condition
    # it was built for, so the extras stay distinct.
    dataset = _binary_dataset(
        [
            (0, 1, 1, 0, 0),
            (0, 1, 1, 1, 0),
            (1, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
            (1, 1, 1, 0, 1),
            (0, 0, 0, 1, 1),
        ]
    )
    rules = PRIDE.fit(dataset, min_support=2, try_all_conditions=True)
    assert _rules_of(rules, "0") == [
        "y(0) :- x0(0), x1(1).",
        "y(0) :- x0(0), x2(1).",
        "y(0) :- x0(0), x3(0).",
        "y(0) :- x1(0), x3(0).",
        "y(0) :- x2(0), x3(0).",
    ]


def test_pride_extra_rule_keeps_its_condition_when_dropping():
    # x1(1) is redundant once x3(0) is in, but as the extra rule's own
    # condition it is tried last, so the x1 rule survives.
    dataset = _binary_dataset(
        [(0, 1, 0, 0, 1), (0, 1, 1, 1, 0), (1, 0, 1, 0, 0), (0, 1, 0, 1, 0)]
    )
    rules = PRIDE.fit(dataset, try_all_conditions=True)
    assert _rules_of(rules, "1") == [
        "y(1) :- x0(0), x3(0).",
        "y(1) :- x1(1), x3(0).",
        "y(1) :- x2(0), x3(0).",
    ]
    # The default rule equals an extra here: each is kept once.
    learned, _ = PRIDE._fit_target_value(dataset, "y", "1", try_all_conditions=True)
    assert len(learned) == len(set(learned))


@pytest.mark.parametrize(
    ("threshold", "error"),
    [(-1, ValueError), (0.0, ValueError), (1.5, ValueError), (True, TypeError)],
)
def test_pride_min_possible_support_rejects_invalid_values(
    toy_dataset, threshold, error
):
    with pytest.raises(error, match="min_possible_support"):
        PRIDE.fit(toy_dataset, min_possible_support=threshold)


def test_pride_heads_learns_only_those_heads(toy_dataset):
    rules = PRIDE.fit(toy_dataset, heads=[("y", "1")])
    assert rules
    assert all(rule.head == ("y", "1") for rule in rules)
    assert rules == [r for r in PRIDE.fit(toy_dataset) if r.head == ("y", "1")]


def test_pride_allowed_conditions_bound_every_rule(toy_dataset):
    allowed = {("y", "1"): ["b", ("a", "0"), ("c", "1")], ("y", "0"): ["b", "c"]}
    rules = PRIDE.fit(toy_dataset, allowed_conditions=allowed, warn_uncovered=False)
    for rule in rules:
        pool = allowed[rule.head]
        for condition in rule.body.items():
            assert condition in pool or condition[0] in pool
    assert set(rules) <= set(GULA.fit(toy_dataset, allowed_conditions=allowed))


def test_verbose_progress_counts_a_seed_skipped_as_already_failed(capsys):
    # Two equal positives that allowed_conditions can't separate from
    # the negative: the second is skipped on the first's failure, and
    # still counted.
    df = pd.DataFrame(
        {"a": ["1", "1", "0"], "b": ["0", "0", "0"], "y": ["1", "1", "0"]}
    )
    dataset = Dataset(df, features=["a", "b"], targets=["y"])
    PRIDE.fit(
        dataset,
        heads=[("y", "1")],
        allowed_conditions={("y", "1"): ["b"]},
        verbose=True,
        warn_uncovered=False,
    )
    assert "covering positives: 2/2" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"heads": ["z"]}, "'z' is not a target"),
        ({"min_support": -1}, "min_support count"),
        ({"min_possible_support": 1.5}, "min_possible_support fraction"),
        ({"allowed_conditions": {("y", "1"): ["z"]}}, "'z' is not a feature"),
    ],
)
def test_pride_validates_its_options_as_gula_does(toy_dataset, options, message):
    with pytest.raises(ValueError, match=message):
        PRIDE.fit(toy_dataset, **options)


# --- Lines no other test pins: each case changes if its line does ---


def _fit_quietly(columns, **options):
    """PRIDE's rules as strings and its warnings, conflicts allowed."""
    df = pd.DataFrame(columns)
    dataset = Dataset(
        df,
        features=[column for column in df if column != "y"],
        targets=["y"],
        assume_nondeterministic=True,
        check_stand_ins=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rules = PRIDE.fit(dataset, **options)
    return sorted(rule.to_string() for rule in rules), [str(w.message) for w in caught]


def test_a_skipped_seed_a_later_rule_covers_is_not_reported_uncovered():
    # One y=1 seed misses min_support; a later seed's rule covers it.
    _, messages = _fit_quietly(
        {
            "f0": ["1", "0", "0", "0", "1", "1", "1", "1", "0"],
            "f1": ["0", "1", "0", "1", "1", "0", "1", "0", "1"],
            "f2": ["0", "0", "1", "1", "1", "1", "1", "1", "0"],
            "f3": ["0", "1", "0", "0", "0", "1", "1", "1", "0"],
            "y": ["0", "0", "0", "1", "0", "1", "1", "1", "1"],
        },
        min_support=2,
    )
    assert len(messages) == 1
    assert messages[0].startswith("PRIDE: 1 of 9 positive rows")


def test_a_rule_whose_support_equals_min_support_is_kept():
    rules, _ = _fit_quietly(
        {"f0": [None, "0", "0"], "f1": [None, "0", "0"], "y": ["1", "1", "0"]},
        min_support=2,
        try_all_conditions=True,
    )
    assert rules == ["y(1) :- ."]


def test_a_rule_whose_possible_support_equals_the_threshold_is_kept():
    rules, _ = _fit_quietly(
        {"f0": ["1", None, "1"], "f1": [None, "0", None], "y": ["1", "0", "0"]},
        min_possible_support=2,
    )
    assert rules == ["y(0) :- ."]


def test_the_rebuild_counts_an_unknown_toward_min_possible_support():
    # y(0) :- f3(1). matches one y=0 row exactly and one through its
    # unknown f3: two possible rows, enough for the threshold.
    rules, _ = _fit_quietly(
        {
            "f0": ["1", None, "1", "0"],
            "f1": [None, "1", None, None],
            "f2": ["1", None, "0", "1"],
            "f3": [None, "1", "1", None],
            "y": ["1", "1", "0", "0"],
        },
        min_possible_support=2,
    )
    assert rules == ["y(0) :- f3(1).", "y(1) :- ."]


def test_a_forced_condition_narrows_the_positives_its_rule_is_scored_on():
    rules, _ = _fit_quietly(
        {
            "f0": [None, "1", "1", "0", "0"],
            "f1": ["1", "0", None, "0", "0"],
            "f2": ["0", "0", "1", None, "0"],
            "y": ["1", "0", "1", "0", "1"],
        },
        try_all_conditions=True,
    )
    assert rules == [
        "y(0) :- f0(1), f2(0).",
        "y(0) :- f1(0).",
        "y(1) :- f0(0).",
        "y(1) :- f1(1).",
        "y(1) :- f2(1).",
    ]


@pytest.mark.parametrize("option", ["min_support", "min_possible_support"])
def test_a_rule_exactly_at_a_support_threshold_is_kept(option):
    # The plain search finds y(1) :- f3(0), with a support of exactly 2.
    # Searching again with the threshold reaches a dead end instead.
    df = pd.DataFrame(
        {
            "f1": list("02212"),
            "f2": list("00022"),
            "f3": list("10101"),
            "y": list("01011"),
        }
    )
    dataset = Dataset(df, features=["f1", "f2", "f3"], targets=["y"])
    rules = PRIDE.fit(dataset, heads=[("y", "1")], **{option: 2})
    assert [rule.to_string() for rule in rules] == [
        "y(1) :- f2(2).",
        "y(1) :- f3(0).",
    ]


def test_try_all_conditions_goes_past_an_unknown_feature():
    df = pd.DataFrame(
        {
            "f1": ["1", "0"],
            "f2": ["0", None],
            "f3": [None, "2"],
            "y": ["1", "0"],
        }
    )
    dataset = Dataset(df, features=["f1", "f2", "f3"], targets=["y"])
    rules = PRIDE.fit(
        dataset, heads=[("y", "0")], try_all_conditions=True, warn_uncovered=False
    )
    assert [rule.to_string() for rule in rules] == [
        "y(0) :- f1(0).",
        "y(0) :- f3(2).",
    ]


@pytest.mark.filterwarnings("ignore::pylfit.ConflictWarning")
def test_a_feature_with_no_known_value_changes_no_rule():
    for seed in range(30):
        df = blanked_frame(seed, rate=0.4).head(8)
        df = df.loc[:, df.notna().any()]
        features = list(df.columns[:-1])
        expected = PRIDE.fit(
            Dataset(df, features=features, targets=["y"]),
            try_all_conditions=True,
            warn_uncovered=False,
        )
        blank = df.assign(blank=None)
        rules = PRIDE.fit(
            Dataset(blank, features=["blank", *features], targets=["y"]),
            try_all_conditions=True,
            warn_uncovered=False,
        )
        assert rules == expected
