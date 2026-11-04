"""Tests for the warning the algorithms give on uncovered rows.

A positive row is uncovered when no learned rule of its head matches
it. Without options, on complete data, it never happens. Unknown
values make it happen when no known feature separates a row from some
negative, for every algorithm; ``GULA``'s and ``PRIDE``'s
``min_support``, ``min_possible_support`` and ``allowed_conditions``
make it happen by design. The warning names the cause, lists the rows
with ``verbose``, and ``warn_uncovered=False`` silences it.
"""

import warnings
from unittest import mock

import pandas as pd
import pytest

from pylfit import GULA, PRIDE, BruteForce, Dataset, PylfitWarning, coverage
from pylfit.algorithms._algorithm import Algorithm

#: The algorithms with restricting options.
ALGORITHMS = [GULA, PRIDE]
ALL_ALGORITHMS = [GULA, PRIDE, BruteForce]


@pytest.fixture
def inseparable_dataset():
    """A ``y=1`` row no known feature separates from a negative.

    ``a=0, b=None`` agrees on ``a`` with the negative ``a=0, b=1``.
    """
    df = pd.DataFrame(
        {
            "a": ["0", "0", None, "1"],
            "b": ["1", None, "0", "0"],
            "y": ["0", "1", "0", "0"],
        }
    )
    return Dataset(df, features=["a", "b"], targets=["y"])


def _silent(call):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return call()


# SPEC: LEARN-6
@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_no_warning_without_options_on_complete_data(algorithm, toy_dataset):
    _silent(lambda: algorithm.fit(toy_dataset))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_no_warning_with_min_support_one_on_complete_data(algorithm, toy_dataset):
    # A rule covering a row has that row as support.
    _silent(lambda: algorithm.fit(toy_dataset, min_support=1))


# SPEC: GULA-4
@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_unknowns_that_no_feature_separates_warn(algorithm, inseparable_dataset):
    with pytest.warns(PylfitWarning, match=r"1 of 4 positive rows \(25.0%\)") as record:
        algorithm.fit(inseparable_dataset)
    message = str(record[0].message)
    assert "because of unknown values" in message
    assert record[0].filename == __file__


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_min_support_cost_is_reported_as_such(algorithm, support_dataset):
    with pytest.warns(PylfitWarning, match="1 left out by min_support"):
        algorithm.fit(support_dataset, min_support=2)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_allowed_conditions_cost_is_reported_as_such(algorithm, support_dataset):
    allowed = {("y", "1"): [("a", "2")]}
    with pytest.warns(PylfitWarning, match="1 left out by allowed_conditions"):
        algorithm.fit(support_dataset, allowed_conditions=allowed)


def test_gula_names_every_option_that_restricted_the_head(support_dataset):
    # GULA can't tell which option left a row out, so it names them all;
    # PRIDE knows, per row.
    allowed = {("y", "1"): [("a", "2")]}
    with pytest.warns(
        PylfitWarning, match="1 left out by allowed_conditions or min_support"
    ):
        GULA.fit(support_dataset, min_support=2, allowed_conditions=allowed)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_an_empty_allowed_list_still_warns(algorithm):
    # No condition may be used, so the head loses every rule: the
    # extreme case of allowed_conditions, not "no restriction".
    df = pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    with pytest.warns(PylfitWarning, match="1 left out by allowed_conditions"):
        algorithm.fit(dataset, allowed_conditions={("y", "1"): []})


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_warn_uncovered_false_skips_the_scan(algorithm, inseparable_dataset):
    # The scan costs time; turning the warning off is the way out of it.
    def scan(*_args, **_kwargs):
        raise AssertionError("scanned for uncovered rows")

    with mock.patch.object(Algorithm, "_uncovered", scan):
        algorithm.fit(inseparable_dataset, warn_uncovered=False)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_each_repeated_uncovered_row_counts(algorithm):
    # The scan answers once per distinct state, but counts every row.
    df = pd.DataFrame({"a": ["1", "1", "1", "0"], "y": ["1", "1", "1", "0"]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    with pytest.warns(PylfitWarning, match="3 of 3 positive rows .* 3 left out by"):
        algorithm.fit(dataset, heads=[("y", "1")], min_support=4)


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_verbose_lists_each_row(algorithm, inseparable_dataset):
    with pytest.warns(PylfitWarning) as record:
        algorithm.fit(inseparable_dataset, verbose=True)
    assert "y=1: {'a': '0', 'b': None} (unknown values)" in str(record[0].message)


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_without_verbose_it_says_how_to_list_the_rows(algorithm, inseparable_dataset):
    with pytest.warns(PylfitWarning, match=r"Pass verbose=True to list them\.$"):
        algorithm.fit(inseparable_dataset)


@pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
def test_warn_uncovered_false_silences_it(algorithm, inseparable_dataset):
    _silent(lambda: algorithm.fit(inseparable_dataset, warn_uncovered=False))


# SPEC: LEARN-8
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_the_warning_agrees_with_coverage(algorithm, support_dataset):
    with pytest.warns(PylfitWarning, match=r"1 of 10 positive rows \(10\.0%\)"):
        rules = algorithm.fit(support_dataset, min_support=2)
    assert coverage(support_dataset, rules) == {("y", "0"): 1.0, ("y", "1"): 0.8}
