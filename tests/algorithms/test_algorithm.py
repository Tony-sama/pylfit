"""Tests for ``Algorithm``, the base class of the three algorithms.

What every ``fit`` shares: its input checks, ``verbose``, ``n_jobs``,
the support thresholds and rows whose target value is unknown.
"""

import inspect
import os

import pandas as pd
import pytest

from pylfit import GULA, PRIDE, BruteForce, Dataset, PylfitWarning, Rule
from pylfit.algorithms._algorithm import Algorithm


def test_algorithm_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Algorithm()  # type: ignore[abstract]


def test_a_subclass_without_fit_fails_loudly_when_called(toy_dataset):
    class NoFit(Algorithm):
        pass

    with pytest.raises(NotImplementedError):
        NoFit.fit(toy_dataset)


@pytest.mark.parametrize("algorithm", [GULA, PRIDE, BruteForce])
def test_unknown_target_values_are_learned_around(algorithm):
    # The row with y unknown is in no head's positives or negatives.
    df = pd.DataFrame({"a": ["0", "1", "1"], "y": ["0", "1", None]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    rules = {rule.to_string() for rule in algorithm.fit(dataset)}
    assert rules == {"y(0) :- a(0).", "y(1) :- a(1)."}


def test_needed_conditions_are_those_some_negative_violates_alone():
    rule = Rule(head=("y", "1"), body={"a": "1", "b": "1", "c": "1"})
    negatives = [
        {"a": "0", "b": "1", "c": "1"},  # violates a alone: a is needed
        {"a": "1", "b": "0", "c": "0"},  # violates b and c: says nothing
    ]
    assert Algorithm._needed_conditions(rule, negatives) == {"a"}


def test_needed_conditions_is_none_for_a_rule_matching_a_negative():
    rule = Rule(head=("y", "1"), body={"a": "1"})
    negatives = [{"a": "0", "b": "0"}, {"a": "1", "b": "0"}]
    assert Algorithm._needed_conditions(rule, negatives) is None


@pytest.mark.parametrize("algorithm", [BruteForce, GULA, PRIDE])
def test_every_algorithm_is_an_algorithm(algorithm):
    assert issubclass(algorithm, Algorithm)


@pytest.mark.parametrize("algorithm", [Algorithm, BruteForce, GULA, PRIDE])
def test_fit_takes_only_the_dataset_positionally(algorithm):
    # The options sit in a different order in each algorithm, so a
    # positional call would mean different things depending on which one
    # it reached.
    parameters = list(inspect.signature(algorithm.fit).parameters.values())
    assert parameters[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in parameters[1:])


@pytest.mark.parametrize("algorithm", [BruteForce, GULA, PRIDE])
def test_verbose_reports_to_stderr_only_and_changes_no_rule(
    toy_dataset, algorithm, capsys
):
    quiet = algorithm.fit(toy_dataset)
    assert capsys.readouterr().err == ""
    loud = algorithm.fit(toy_dataset, verbose=True)
    output = capsys.readouterr()
    assert loud == quiet
    assert output.out == ""
    assert f"[{algorithm.__name__}]" in output.err
    # The last item of every stage is always reported.
    assert "(100%)" in output.err


@pytest.mark.parametrize("algorithm", [BruteForce, GULA, PRIDE])
def test_verbose_ends_each_head_with_a_summary_line(toy_dataset, algorithm, capsys):
    rules = algorithm.fit(toy_dataset, verbose=True)
    err = capsys.readouterr().err
    for target in toy_dataset.targets:
        for value in toy_dataset.domain(target):
            kept = sum(rule.head == (target, value) for rule in rules)
            assert f"{target}={value}: " in err
            assert f"{kept} rules kept" in err


@pytest.mark.parametrize(
    ("n_jobs", "processes"), [(-1, 3), (-2, 2), (-3, 1), (-9, 1), (2, 2)]
)
def test_n_jobs_counts_the_cpus_this_process_may_use(monkeypatch, n_jobs, processes):
    # Python 3.11 and 3.12 lack os.process_cpu_count; Linux then reads
    # the affinity mask, and other systems os.cpu_count.
    monkeypatch.delattr(os, "process_cpu_count", raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    assert Algorithm._resolve_n_jobs(n_jobs) == processes
    monkeypatch.delattr(os, "sched_getaffinity")
    monkeypatch.setattr(os, "cpu_count", lambda: 3)
    assert Algorithm._resolve_n_jobs(n_jobs) == processes


@pytest.mark.parametrize("algorithm", [BruteForce, GULA, PRIDE])
def test_fit_refuses_a_dataframe_for_its_dataset(algorithm):
    frame = pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"]})
    with pytest.raises(TypeError, match="expected a Dataset, got DataFrame"):
        algorithm.fit(frame)


@pytest.mark.parametrize("algorithm", [GULA, PRIDE])
@pytest.mark.parametrize(
    "options", [{"allowed_conditions": {("y", "1"): ["a"]}}, {"min_support": -1}]
)
def test_fit_refuses_a_dataframe_whatever_options_come_with_it(algorithm, options):
    # The dataset is checked before the options, some of which read it.
    frame = pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"]})
    with pytest.raises(TypeError, match="expected a Dataset, got DataFrame"):
        algorithm.fit(frame, **options)


@pytest.mark.parametrize("algorithm", [GULA, PRIDE])
def test_allowed_conditions_must_be_a_mapping(toy_dataset, algorithm):
    with pytest.raises(TypeError, match="allowed_conditions must be a mapping"):
        algorithm.fit(toy_dataset, allowed_conditions=[(("y", "1"), ["a"])])


@pytest.mark.parametrize(
    ("fraction", "positives", "count"),
    [(0.28, 25, 7), (0.07, 100, 7), (0.14, 100, 14), (0.55, 100, 55), (0.5, 3, 2)],
)
def test_a_support_fraction_is_read_as_written(fraction, positives, count):
    # 0.28 * 25 is 7.000000000000001 in binary floating point.
    assert Algorithm._resolve_min_support(fraction, positives) == count


def test_a_support_fraction_resolves_exactly_for_every_count():
    for positives in range(1, 201):
        for count in range(1, positives + 1):
            fraction = count / positives
            assert Algorithm._resolve_min_support(fraction, positives) == count


@pytest.mark.parametrize(
    ("option", "entry"),
    [("heads", [("y", 1)]), ("allowed_conditions", {("y", "1"): [("a", 0)]})],
)
def test_a_value_that_is_not_a_str_is_named_as_such(toy_dataset, option, entry):
    # A wrong type, so TypeError, as for any value that is not a str.
    with pytest.raises(TypeError, match=r"is not a str; .* convert it with str"):
        GULA.fit(toy_dataset, **{option: entry})


@pytest.mark.parametrize("algorithm", [BruteForce, GULA, PRIDE])
def test_heads_given_as_a_mapping_are_refused(toy_dataset, algorithm):
    # Read as its keys, {"y": "1"} would learn every value of y.
    with pytest.raises(TypeError, match="not a mapping"):
        algorithm.fit(toy_dataset, heads={"y": "1"})


@pytest.mark.parametrize("algorithm", [GULA, PRIDE])
def test_conditions_given_as_a_mapping_are_refused(toy_dataset, algorithm):
    # Read as its keys, {"a": "0"} would allow every value of a.
    with pytest.raises(ValueError, match=r"\[\('a', '0'\)\], not a mapping"):
        algorithm.fit(toy_dataset, allowed_conditions={("y", "1"): {"a": "0"}})


@pytest.mark.parametrize("algorithm", [BruteForce, GULA, PRIDE])
def test_heads_that_are_not_a_collection_are_named(toy_dataset, algorithm):
    with pytest.raises(TypeError, match=r"heads must be a collection .* got int"):
        algorithm.fit(toy_dataset, heads=5)


@pytest.mark.parametrize("algorithm", [GULA, PRIDE])
def test_conditions_that_are_not_a_collection_are_named(toy_dataset, algorithm):
    with pytest.raises(TypeError, match=r"conditions must be a collection .* got int"):
        algorithm.fit(toy_dataset, allowed_conditions={("y", "1"): 3})


@pytest.mark.parametrize("algorithm", [GULA, PRIDE])
@pytest.mark.parametrize("option", ["min_support", "min_possible_support"])
def test_support_thresholds_take_their_boundaries(toy_dataset, algorithm, option):
    # A count of 0 keeps every rule; a fraction of 1.0 asks for every
    # positive row, which no rule of the toy dataset covers. Both are
    # valid.
    assert algorithm.fit(toy_dataset, **{option: 0}) == algorithm.fit(toy_dataset)
    with pytest.warns(PylfitWarning, match="matched by no rule"):
        assert algorithm.fit(toy_dataset, **{option: 1.0}) == []


@pytest.mark.parametrize("algorithm", [BruteForce, GULA, PRIDE])
@pytest.mark.parametrize("flag", ["verbose", "warn_uncovered"])
def test_a_flag_that_is_not_a_bool_is_refused(toy_dataset, algorithm, flag):
    with pytest.raises(TypeError, match=f"{flag} must be True or False"):
        algorithm.fit(toy_dataset, **{flag: "no"})


def test_try_all_conditions_that_is_not_a_bool_is_refused(toy_dataset):
    with pytest.raises(TypeError, match="try_all_conditions must be True or False"):
        PRIDE.fit(toy_dataset, try_all_conditions="False")
