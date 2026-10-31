"""Every warning pylfit emits has a class callers can filter it by."""

import warnings

import pandas as pd
import pytest

from pylfit import GULA, ConflictWarning, Dataset, PylfitWarning, Rule, generalize


def test_the_classes_are_user_warnings():
    assert issubclass(PylfitWarning, UserWarning)
    assert issubclass(ConflictWarning, PylfitWarning)


def test_conflicting_rows_warn_a_conflict_warning():
    df = pd.DataFrame({"a": ["0", "0"], "y": ["0", "1"]})
    with pytest.warns(ConflictWarning):
        Dataset(df, features=["a"], targets=["y"])


def test_filtering_conflict_warning_leaves_the_other_warnings():
    # A "?" may stand for a missing value: a plain PylfitWarning, still
    # shown.
    df = pd.DataFrame({"a": ["0", "0", "?"], "y": ["0", "1", "1"]})
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", ConflictWarning)
        Dataset(df, features=["a"], targets=["y"])
    assert [type(w.message) for w in record] == [PylfitWarning]


def test_too_many_integer_values_warns_a_pylfit_warning():
    df = pd.DataFrame({"a": list(range(25)) * 2, "y": ["0"] * 50})
    with pytest.warns(PylfitWarning, match="25 distinct values") as record:
        Dataset(df, features=["a"], targets=["y"])
    assert [type(w.message) for w in record] == [PylfitWarning]


def test_a_stand_in_warns_a_pylfit_warning():
    df = pd.DataFrame({"a": ["0", "?"], "y": ["0", "1"]})
    with pytest.warns(PylfitWarning, match="stand for a missing value") as record:
        Dataset(df, features=["a"], targets=["y"])
    assert [type(w.message) for w in record] == [PylfitWarning]


def test_uncovered_rows_warn_a_pylfit_warning():
    df = pd.DataFrame({"a": ["0", "1", "1"], "y": ["0", "1", "1"]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    with pytest.warns(PylfitWarning, match="matched by no rule") as record:
        GULA.fit(dataset, min_support=2)
    assert [type(w.message) for w in record] == [PylfitWarning]


def test_an_unreachable_max_rules_warns_a_pylfit_warning():
    # One rule per head, and generalize never drops a head's last rule.
    rows = [{"a": a, "y": a} for a in "01"]
    dataset = Dataset(pd.DataFrame(rows), features=["a"], targets=["y"])
    rules = [Rule.from_string(r) for r in ("y(0) :- a(0).", "y(1) :- a(1).")]
    with pytest.warns(PylfitWarning, match="meets max_rules") as record:
        generalize(rules, dataset, max_rules=1)
    assert [type(w.message) for w in record] == [PylfitWarning]
