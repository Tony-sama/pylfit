"""Tests for ``BruteForce``: its candidates, and ``GULA``.

The three guarantees ``_brute_force.py``'s module docstring documents,
and the cases every learner shares, are in ``test_learners.py``. Here:
the fully general rule as a candidate, the candidates and their count,
``_judge``, the same rules as ``GULA`` on unknown values, ``verbose``,
and repeated negatives judged once.
"""

import pandas as pd
import pytest

from pylfit import GULA, BruteForce, Dataset, Rule
from tests.algorithms._tables import blanked_frame

_FIXTURE_NAMES = ["toy_dataset", "nondeterministic_dataset", "degenerate_dataset"]


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


# SPEC: GULA-2
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
