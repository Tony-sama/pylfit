"""Tests for ``Dataset``: its checks, warnings, rows and states.

Most tests build a small ``DataFrame`` inline. ``conftest.py`` gives
``toy_dataset`` (deterministic, 3 binary features),
``nondeterministic_dataset`` (a state seen with both target values)
and ``degenerate_dataset`` (a single-valued target).
"""

import datetime
import decimal
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import pytest

from pylfit import ConflictWarning, Dataset, PylfitWarning, _dataset

# --- features and targets ---


def test_empty_features_raises():
    df = pd.DataFrame({"a": [0, 1], "y": [0, 1]})
    with pytest.raises(ValueError, match="features"):
        Dataset(df, features=[], targets=["y"])


def test_empty_targets_raises():
    df = pd.DataFrame({"a": [0, 1], "y": [0, 1]})
    with pytest.raises(ValueError, match="targets"):
        Dataset(df, features=["a"], targets=[])


def test_empty_features_checked_before_missing_columns():
    # The empty-features/targets check runs first, logically prior to
    # the missing-columns check — so a dataframe that would *also*
    # fail the missing-columns check must still fail with the
    # empty-features message, not a missing-columns one.
    df = pd.DataFrame({"a": [0, 1]})
    with pytest.raises(ValueError, match="features"):
        Dataset(df, features=[], targets=["nonexistent"])


def test_a_name_that_is_not_a_str_is_checked_before_emptiness():
    df = pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"]})
    with pytest.raises(TypeError, match="names must be str"):
        Dataset(df, features=[1], targets=[])


def test_missing_feature_column_raises():
    df = pd.DataFrame({"a": [0, 1], "y": [0, 1]})
    with pytest.raises(ValueError, match="columns not in the dataframe"):
        Dataset(df, features=["a", "b"], targets=["y"])


def test_missing_target_column_raises():
    df = pd.DataFrame({"a": [0, 1], "y": [0, 1]})
    with pytest.raises(ValueError, match="columns not in the dataframe"):
        Dataset(df, features=["a"], targets=["z"])


def test_duplicate_column_within_features_raises():
    df = pd.DataFrame({"a": [0, 1], "y": [0, 1]})
    with pytest.raises(ValueError, match="duplicate"):
        Dataset(df, features=["a", "a"], targets=["y"])


def test_same_column_as_feature_and_target_raises():
    df = pd.DataFrame({"a": [0, 1], "y": [0, 1]})
    with pytest.raises(ValueError, match="duplicate"):
        Dataset(df, features=["a"], targets=["a"])


def test_a_name_that_is_not_a_str_is_refused():
    df = pd.DataFrame({0: ["0", "1"], 1: ["0", "1"]})
    with pytest.raises(TypeError, match="must be str"):
        Dataset(df, features=[0], targets=[1])


def test_numpy_string_names_are_stored_as_str():
    df = pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"]})
    dataset = Dataset(df, features=np.array(["a"]), targets=np.array(["y"]))
    assert {type(name) for name in dataset.features + dataset.targets} == {str}
    assert repr(dataset) == "<Dataset: 2 rows, features ['a'], targets ['y']>"


@pytest.mark.parametrize(
    "features", [("a", "b"), pd.Index(["a", "b"]), iter(["a", "b"])]
)
def test_features_and_targets_take_any_collection(features):
    df = pd.DataFrame({"a": ["0", "1"], "b": ["0", "0"], "y": ["0", "1"]})
    dataset = Dataset(df, features=features, targets=("y",))
    assert dataset.features == ["a", "b"]
    assert dataset.targets == ["y"]


@pytest.mark.parametrize("argument", ["features", "targets"])
def test_a_single_string_is_refused(argument):
    df = pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"]})
    names = {"features": ["a"], "targets": ["y"]}
    names[argument] = names[argument][0]
    with pytest.raises(TypeError, match=rf"^{argument} must be a collection"):
        Dataset(df, **names)


@pytest.mark.parametrize("names", [{"a", "b"}, frozenset({"a", "b"}), {"a": 0}.keys()])
def test_features_in_a_set_are_refused(names):
    # A set's order changes between runs, and the feature order is in
    # every printed and saved model.
    df = pd.DataFrame({"a": ["0", "1"], "b": ["0", "1"], "y": ["0", "1"]})
    with pytest.raises(TypeError, match="features must have an order"):
        Dataset(df, features=names, targets=["y"])
    with pytest.raises(TypeError, match="targets must have an order"):
        Dataset(df, features=["a"], targets={"y"})


# --- the dataframe and the options ---


@pytest.mark.parametrize(
    "call",
    [
        lambda frame: Dataset(frame, features=["a"], targets=["y"]),
        Dataset.check_values,
    ],
    ids=["Dataset", "check_values"],
)
def test_a_series_for_the_dataframe_is_refused_with_a_hint(call):
    series = pd.DataFrame({"a": ["0", "1"]})["a"]
    with pytest.raises(TypeError, match="expected a DataFrame, got Series; pass"):
        call(series)


@pytest.mark.parametrize("check_stand_ins", [True, False])
def test_a_column_twice_in_the_dataframe_is_refused(check_stand_ins):
    df = pd.DataFrame([["0", "1", "0"]], columns=["a", "a", "y"])
    with pytest.raises(ValueError, match=r"^duplicate column names .*\['a'\]"):
        Dataset(df, features=["a"], targets=["y"], check_stand_ins=check_stand_ins)


def test_a_repeated_column_left_unused_is_ignored():
    df = pd.DataFrame([["0", "1", "1", "0"]], columns=["a", "n", "n", "y"])
    assert Dataset(df, features=["a"], targets=["y"]).features == ["a"]


def test_check_values_refuses_a_repeated_column_name():
    df = pd.DataFrame([[0.5, "1"], [1.5, "0"]], columns=["a", "a"])
    with pytest.raises(ValueError, match="duplicate column names"):
        Dataset.check_values(df)


@pytest.mark.parametrize("flag", ["assume_nondeterministic", "check_stand_ins"])
@pytest.mark.parametrize("value", ["False", "no", 0, None])
def test_a_flag_that_is_not_a_bool_is_refused(flag, value):
    # "False" is truthy: it would have turned the option on.
    df = pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"]})
    with pytest.raises(TypeError, match=f"{flag} must be True or False"):
        Dataset(df, features=["a"], targets=["y"], **{flag: value})


def test_a_numpy_bool_flag_is_read_as_a_bool():
    df = pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"]})
    dataset = Dataset(
        df, features=["a"], targets=["y"], assume_nondeterministic=np.bool_(True)
    )
    assert dataset.assume_nondeterministic is True


def test_dataset_options_are_keyword_only():
    df = pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"]})
    with pytest.raises(TypeError, match="positional"):
        Dataset(df, ["a"], ["y"], 5)


def test_check_values_takes_max_categories_by_name_only():
    with pytest.raises(TypeError, match="positional"):
        Dataset.check_values(pd.DataFrame({"a": ["0"]}), 5)


# --- values read as str ---


def test_values_are_coerced_to_str():
    df = pd.DataFrame({"a": [0, 1], "y": [0, 1]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    assert dataset.domain("a") == ["0", "1"]
    assert dataset.domain("y") == ["0", "1"]
    assert dataset.positives("y", "0") == [{"a": "0"}]


@pytest.mark.parametrize(
    "value",
    [1, True, "x", 2.0, np.int64(3), np.bool_(False), np.str_("s")],
    ids=["int", "bool", "str", "whole-float", "np-int", "np-bool", "np-str"],
)
@pytest.mark.parametrize("with_whole_float", [False, True])
def test_dataset_and_the_metrics_write_a_value_alike(value, with_whole_float):
    # A whole float sends the column through the other conversion; both
    # must give the text accuracy_score compares with.
    others = [1.0] if with_whole_float else ["z"]
    df = pd.DataFrame({"a": pd.Series([value, *others], dtype=object), "y": ["0", "1"]})
    dataset = Dataset(df, features=["a"], targets=["y"], check_stand_ins=False)
    assert _dataset._as_text(value) in dataset.domain("a")


@pytest.mark.parametrize(
    "value",
    [
        b"x",
        b"\xe9",
        decimal.Decimal("1.5"),
        datetime.date(2020, 1, 1),
        datetime.datetime(2020, 1, 1, 10),
        datetime.time(10),
        datetime.timedelta(days=1),
        1j,
    ],
    ids=[
        "bytes",
        "bytes-not-utf8",
        "decimal",
        "date",
        "datetime",
        "time",
        "timedelta",
        "complex",
    ],
)
def test_an_object_column_of_values_with_no_single_text_is_refused(value):
    df = pd.DataFrame({"a": pd.Series([value, "z"], dtype=object), "y": ["0", "1"]})
    with pytest.raises(ValueError, match=r"no single text.*'a'"):
        Dataset(df, features=["a"], targets=["y"])


def test_series_dt_date_is_refused_like_a_date_column():
    days = pd.Series(pd.to_datetime(["2020-01-01 10:00", "2020-01-02 11:30"]))
    df = pd.DataFrame({"day": days.dt.date, "y": ["0", "1"]})
    with pytest.raises(ValueError, match=r"'day' \(date\)"):
        Dataset(df, features=["day"], targets=["y"])


@pytest.mark.parametrize(
    "column",
    [
        pd.to_datetime(["2020-01-01", "2020-01-02"]),
        pd.to_timedelta([1, 2], unit="D"),
        pd.Series([1 + 0j, 2 + 0j]),
        pd.period_range("2020-01", periods=2, freq="M"),
    ],
)
def test_complex_date_and_duration_columns_are_refused(column):
    df = pd.DataFrame({"a": column, "y": ["0", "1"]})
    with pytest.raises(ValueError, match="complex, date and duration columns"):
        Dataset(df, ["a"], ["y"])


def test_a_date_column_is_reported_before_a_float_column():
    df = pd.DataFrame(
        {"f": [0.5, 1.5], "t": pd.to_datetime(["2020-01-01", "2020-01-02"])}
    )
    with pytest.raises(ValueError, match="date and duration columns"):
        Dataset(df, features=["f"], targets=["t"])


def test_an_unhashable_value_does_not_stop_the_float_check():
    # unique() raises on a list; the check falls back to every value.
    df = pd.DataFrame({"a": [[1], [2]], "y": ["0", "1"]})
    assert Dataset(df, ["a"], ["y"]).domain("a") == ["[1]", "[2]"]


@pytest.mark.parametrize(
    "column",
    [
        pd.Series([1, 2, None, 1], dtype="category"),
        pd.cut(pd.Series([20.0, 40.0, 200.0, 25.0]), bins=[0, 30, 50, 120]),
    ],
    ids=["int-categories", "intervals"],
)
def test_a_missing_value_does_not_change_a_category_columns_text(column):
    # With a missing value, pandas writes these through floats, "1.0"
    # and "(0.0, 30.0]"; read that way, predict would refuse "1".
    dataset = Dataset(
        pd.DataFrame({"x": column, "y": list("abca")}), features=["x"], targets=["y"]
    )
    assert dataset.domain("x") == sorted(
        {str(value) for value in column.dropna().astype(object)}
    )


# --- float columns ---
#
# A float column is refused, but pandas stores an integer column with
# a missing value as float64 (read_csv on a 0/1 column with blanks
# gives 0.0, 1.0, NaN), so a column of whole numbers is read as the
# integers it holds.


def test_float_feature_column_raises():
    df = pd.DataFrame({"a": [0.0, 1.5], "y": [0, 1]})
    with pytest.raises(ValueError, match="float"):
        Dataset(df, features=["a"], targets=["y"])


def test_float_target_column_raises():
    df = pd.DataFrame({"a": [0, 1], "y": [0.0, 1.5]})
    with pytest.raises(ValueError, match="float"):
        Dataset(df, features=["a"], targets=["y"])


def test_whole_number_float_column_reads_as_integers():
    df = pd.DataFrame({"a": [0.0, float("nan"), 1.0], "y": [0, 1, 1]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    assert dataset.domain("a") == ["0", "1"]
    assert dataset.positives("y", "1") == [{"a": None}, {"a": "1"}]


def test_all_missing_float_column_is_all_unknown():
    df = pd.DataFrame({"a": [float("nan")] * 2, "b": ["0", "1"], "y": ["0", "1"]})
    dataset = Dataset(df, features=["a", "b"], targets=["y"])
    assert dataset.domain("a") == []


@pytest.mark.parametrize("value", [0.5, float("inf")])
def test_float_column_with_a_non_whole_value_still_raises(value):
    df = pd.DataFrame({"a": [0.0, value, None], "y": [0, 1, 1]})
    with pytest.raises(ValueError, match="float"):
        Dataset(df, features=["a"], targets=["y"])


def test_a_whole_float_too_large_for_an_integer_is_refused():
    df = pd.DataFrame({"a": [1e19, 0.0], "y": ["0", "1"]})
    with pytest.raises(ValueError, match="float columns are not supported"):
        Dataset(df, ["a"], ["y"])


@pytest.mark.parametrize(
    "column",
    [
        pd.Series([23.5, "unknown", 41.25], dtype=object),
        pd.Categorical([23.5, 41.25, 23.5]),
    ],
)
def test_a_fractional_float_is_refused_in_any_column(column):
    df = pd.DataFrame({"a": column, "y": ["0", "1", "0"]})
    with pytest.raises(ValueError, match="float columns are not supported"):
        Dataset(df, ["a"], ["y"], check_stand_ins=False)


@pytest.mark.parametrize(
    "column",
    [
        [1.0, 0.0, 1.0],
        pd.Categorical([1.0, 0.0, 1.0]),
        pd.Series([1.0, "x", 0.0], dtype=object),
    ],
)
def test_a_whole_float_reads_as_its_integer_text_in_any_column(column):
    df = pd.DataFrame({"a": column, "y": ["0", "1", "0"]})
    domain = Dataset(df, ["a"], ["y"], check_stand_ins=False).domain("a")
    assert "1" in domain
    assert "1.0" not in domain


def test_an_unused_fractional_category_is_not_a_value():
    # Filtering rows leaves unused categories behind.
    column = pd.Categorical([1.0, 2.0, 1.0], categories=[1.0, 2.0, 2.5])
    df = pd.DataFrame({"a": column, "y": ["0", "1", "0"]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    assert dataset.domain("a") == ["1", "2"]


# --- missing values read as unknown ---
#
# NaN and None are unknowns, not errors: kept missing internally, never
# a domain value, handed out as None in every feature state.


@pytest.mark.parametrize("missing", [float("nan"), None])
def test_missing_feature_value_is_unknown(missing):
    df = pd.DataFrame({"a": ["0", missing, "1"], "y": ["0", "1", "0"]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    assert dataset.domain("a") == ["0", "1"]
    assert dataset.positives("y", "1") == [{"a": None}]


def test_every_state_holds_every_feature():
    # Explicit unknowns: a state never silently lacks a feature, so a
    # rule's missing key always means a mistake (Rule.matches raises).
    df = pd.DataFrame({"a": ["0", None], "b": [None, "1"], "y": ["0", "1"]})
    dataset = Dataset(df, features=["a", "b"], targets=["y"])
    assert dataset.positives("y", "0") == [{"a": "0", "b": None}]
    assert dataset.negatives("y", "0") == [{"a": None, "b": "1"}]


def test_unknown_target_leaves_the_row_out_for_that_target_only():
    df = pd.DataFrame(
        {"a": ["0", "1", "1"], "y1": ["0", None, "1"], "y2": ["0", "1", "1"]}
    )
    dataset = Dataset(df, features=["a"], targets=["y1", "y2"])
    assert dataset.domain("y1") == ["0", "1"]
    assert dataset.positives("y1", "1") == [{"a": "1"}]
    assert dataset.negatives("y1", "0") == [{"a": "1"}]
    assert dataset.positives("y2", "1") == [{"a": "1"}, {"a": "1"}]
    assert len(dataset) == 3


def test_check_values_accepts_missing_values():
    Dataset.check_values(pd.DataFrame({"a": ["0", float("nan")], "b": [None, "1"]}))


# --- the max_categories warning ---


def test_integer_column_over_max_categories_warns():
    df = pd.DataFrame({"a": list(range(25)) * 2, "y": [0] * 50})
    with pytest.warns(PylfitWarning, match="continuous"):
        Dataset(df, features=["a"], targets=["y"], max_categories=20)


def test_integer_column_under_max_categories_does_not_warn():
    df = pd.DataFrame({"a": [0, 1, 2], "y": [0, 1, 0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Dataset(df, features=["a"], targets=["y"], max_categories=20)


def test_max_categories_warns_only_above_its_default_of_5():
    df = pd.DataFrame({"a": list(range(5)), "y": ["0", "1", "0", "1", "0"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Dataset(df, features=["a"], targets=["y"], check_stand_ins=False)
    df = pd.concat([df] * 2, ignore_index=True)
    df.loc[10] = [5, "1"]
    with pytest.warns(PylfitWarning, match="6 distinct values.*continuous"):
        Dataset(df, features=["a"], targets=["y"], check_stand_ins=False)


def test_max_categories_counts_a_whole_float_column_too():
    # How pandas stores an integer column with a missing value.
    values = [float(i) for i in range(25)] * 2 + [None]
    df = pd.DataFrame({"a": values, "y": ["0"] * 51})
    with pytest.warns(PylfitWarning, match="25 distinct values.*continuous"):
        Dataset(df, features=["a"], targets=["y"], check_stand_ins=False)


def test_max_categories_zero_warns_on_any_column():
    df = pd.DataFrame({"a": [0, 1] * 2, "y": ["0", "1"] * 2})
    with pytest.warns(PylfitWarning, match="distinct values") as record:
        Dataset(df, features=["a"], targets=["y"], max_categories=0)
    assert [str(w.message).split("'")[1] for w in record] == ["a", "y"]


def test_text_column_over_max_categories_warns_of_its_categories():
    df = pd.DataFrame({"a": [f"town{i}" for i in range(8)] * 2, "y": ["0"] * 16})
    with pytest.warns(PylfitWarning, match="8 distinct values.*category of its own"):
        Dataset(df, features=["a"], targets=["y"])


def test_category_column_over_max_categories_warns_too():
    values = pd.Categorical([f"town{i}" for i in range(8)] * 2)
    df = pd.DataFrame({"a": values, "y": ["0"] * 16})
    with pytest.warns(PylfitWarning, match="category of its own"):
        Dataset(df, features=["a"], targets=["y"])


def test_a_target_over_max_categories_warns_too():
    states = [(a, b) for a in "0123" for b in "01"] * 2
    df = pd.DataFrame(
        {
            "a": [a for a, _ in states],
            "b": [b for _, b in states],
            "y": [f"c{a}{b}" for a, b in states],
        }
    )
    with pytest.warns(PylfitWarning, match="Column 'y' has 8 distinct values"):
        Dataset(df, features=["a", "b"], targets=["y"])


@pytest.mark.parametrize(
    "values", [[f"id{i}" for i in range(20)], list(range(20))], ids=["text", "int"]
)
def test_one_value_per_row_warns_of_an_identifier(values):
    df = pd.DataFrame({"a": values, "y": ["0"] * 20})
    with pytest.warns(PylfitWarning, match="Nearly one value per row: an identifier"):
        Dataset(df, features=["a"], targets=["y"])


def test_the_identifier_share_counts_known_values_only():
    # 6 distinct values in 6 known rows of 20: one per known row.
    values = [f"id{i}" for i in range(6)] + [None] * 14
    df = pd.DataFrame({"a": values, "y": ["0"] * 20})
    with pytest.warns(PylfitWarning, match="Nearly one value per row"):
        Dataset(df, features=["a"], targets=["y"], check_stand_ins=False)


def test_max_categories_warning_points_at_the_caller():
    df = pd.DataFrame({"a": list(range(25)) * 2, "y": [0] * 50})
    with pytest.warns(PylfitWarning, match="continuous") as record:
        Dataset(df, features=["a"], targets=["y"])
    assert record[0].filename == __file__
    with pytest.warns(PylfitWarning, match="continuous") as record:
        Dataset.check_values(df)
    assert record[0].filename == __file__


@pytest.mark.parametrize(
    ("max_categories", "error", "message"),
    [
        (-1, ValueError, "must be non-negative"),
        (2.5, TypeError, "must be an int"),
        ("20", TypeError, "must be an int"),
        (True, TypeError, "must be an int"),
        (None, TypeError, "must be an int"),
    ],
)
def test_max_categories_must_be_a_non_negative_int(max_categories, error, message):
    df = pd.DataFrame({"a": [0, 1], "y": [0, 1]})
    with pytest.raises(error, match=message):
        Dataset(df, features=["a"], targets=["y"], max_categories=max_categories)
    with pytest.raises(error, match=message):
        Dataset.check_values(df, max_categories=max_categories)


def test_max_categories_takes_a_numpy_integer():
    df = pd.DataFrame({"a": [0, 1], "y": [0, 1]})
    Dataset(df, features=["a"], targets=["y"], max_categories=np.int64(5))


# --- the stand-in warning ---


def test_stand_ins_warn_listing_each_finding_and_the_option():
    df = pd.DataFrame(
        {
            "smoker": ["yes", "?", "no", "no", "yes"],
            "pressure": [120, 135, 0, 110, 160],
            "y": ["1", "0", "0", "0", "1"],
        }
    )
    message = (
        "2 values may stand for a missing value: smoker '?' x1 (stand_in), "
        "pressure 0 x1 (far_zero). Replace them with None if unknown; "
        "find_missing_stand_ins(df[features + targets]) gives the full report; pass "
        "check_stand_ins=False to silence this."
    )
    with pytest.warns(PylfitWarning) as record:
        Dataset(df, features=["smoker", "pressure"], targets=["y"])
    assert [str(warning.message) for warning in record] == [message]


def test_a_single_stand_in_is_one_value():
    df = pd.DataFrame({"a": ["0", "?"], "y": ["0", "1"]})
    with pytest.warns(PylfitWarning, match=r"^1 value may stand .*a '\?' x1"):
        Dataset(df, features=["a"], targets=["y"])


def test_stand_in_warning_lists_five_and_counts_the_rest():
    df = pd.DataFrame(
        {column: ["0", "?"] for column in ["a", "b", "c", "d", "e", "f", "g"]}
        | {"y": ["0", "1"]}
    )
    with pytest.warns(PylfitWarning) as record:
        Dataset(df, features=list("abcdefg"), targets=["y"])
    message = str(record[0].message)
    assert message.startswith("7 values may stand for a missing value:")
    assert "e '?' x1 (stand_in), and 2 more." in message
    assert "f '?'" not in message


def test_stand_in_warning_counts_the_rest_from_the_sixth():
    df = pd.DataFrame({column: ["0", "?"] for column in "abcdef"} | {"y": ["0", "1"]})
    with pytest.warns(PylfitWarning) as record:
        Dataset(df, features=list("abcdef"), targets=["y"])
    assert "e '?' x1 (stand_in), and 1 more." in str(record[0].message)


def test_stand_ins_are_checked_in_targets_too():
    df = pd.DataFrame({"a": ["0", "1"], "y": ["0", "unknown"]})
    with pytest.warns(PylfitWarning, match="y 'unknown' x1"):
        Dataset(df, features=["a"], targets=["y"])


def test_stand_ins_in_other_columns_are_ignored():
    df = pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"], "note": ["?", "?"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Dataset(df, features=["a"], targets=["y"])


def test_a_far_zero_is_seen_before_values_become_strings():
    df = pd.DataFrame({"p": [0, 120, 135, 110, 160], "y": ["0", "1", "1", "1", "1"]})
    with pytest.warns(PylfitWarning, match=r"p 0 x1 \(far_zero\)"):
        Dataset(df, features=["p"], targets=["y"])


def test_stand_in_warning_points_at_the_caller():
    df = pd.DataFrame({"a": ["0", "?"], "y": ["0", "1"]})
    with pytest.warns(PylfitWarning) as record:
        Dataset(df, features=["a"], targets=["y"])
    assert record[0].filename == __file__


def test_check_stand_ins_false_silences_the_warning():
    df = pd.DataFrame({"a": ["0", "?"], "y": ["0", "1"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Dataset(df, features=["a"], targets=["y"], check_stand_ins=False)


def test_unknowns_are_not_stand_ins():
    df = pd.DataFrame({"a": ["0", None], "y": ["0", "1"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Dataset(df, features=["a"], targets=["y"])


# --- conflicting_targets and the conflict warning ---


def test_conflicting_targets_empty_for_toy_dataset(toy_dataset):
    assert toy_dataset.conflicting_targets == []


def test_conflicting_targets_names_a_conflicting_target(nondeterministic_dataset):
    assert nondeterministic_dataset.conflicting_targets == ["y"]


def test_conflicting_targets_empty_for_single_valued_target(degenerate_dataset):
    # Every row agrees on y=1 — trivially free of conflicts.
    assert degenerate_dataset.conflicting_targets == []


def test_conflicting_targets_is_per_target_not_whole_dataset():
    # y1 always agrees per feature state; y2 doesn't — must be tracked
    # independently, not collapsed into one dataset-wide bool.
    df = pd.DataFrame({"a": [0, 0, 1], "y1": [0, 0, 1], "y2": [0, 1, 1]})
    dataset = Dataset(
        df, features=["a"], targets=["y1", "y2"], assume_nondeterministic=True
    )
    assert dataset.conflicting_targets == ["y2"]


def test_conflicting_targets_ignores_rows_with_an_unknown():
    # Two a=None rows could be two different states: not a conflict.
    df = pd.DataFrame({"a": [None, None, "1"], "y": ["0", "1", "1"]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    assert dataset.conflicting_targets == []


def test_conflicting_targets_ignores_an_unknown_target():
    df = pd.DataFrame({"a": ["0", "0"], "y": ["0", None]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    assert dataset.conflicting_targets == []


def test_conflicts_warn_naming_the_targets_and_the_option():
    df = pd.DataFrame({"a": [0, 0, 1], "y1": [0, 0, 1], "y2": [0, 1, 1]})
    with pytest.warns(ConflictWarning, match=r"\['y2'\].*assume_nondeterministic"):
        Dataset(df, features=["a"], targets=["y1", "y2"])


def test_the_conflict_warning_says_what_noise_costs():
    df = pd.DataFrame({"a": ["0", "0", "1"], "y": ["0", "1", "1"]})
    with pytest.warns(
        ConflictWarning, match="stays possible for each of its values"
    ) as record:
        Dataset(df, features=["a"], targets=["y"])
    assert "a mislabelled row seen once is learned" in str(record[0].message)


def test_conflict_warning_points_at_the_caller():
    df = pd.DataFrame({"a": [0, 0], "y": [0, 1]})
    with pytest.warns(ConflictWarning) as record:
        Dataset(df, features=["a"], targets=["y"])
    assert record[0].filename == __file__


def test_assume_nondeterministic_silences_the_conflict_warning():
    df = pd.DataFrame({"a": [0, 0], "y": [0, 1]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Dataset(df, features=["a"], targets=["y"], assume_nondeterministic=True)


def test_no_conflict_no_warning():
    df = pd.DataFrame({"a": [0, 1], "y": [0, 1]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Dataset(df, features=["a"], targets=["y"])


# --- rows: len, iteration, index labels, to_dataframe ---


def test_len_matches_row_count(toy_dataset):
    assert len(toy_dataset) == 8


def test_iter_yields_every_row_in_order(toy_dataset):
    rows = list(toy_dataset)
    assert len(rows) == len(toy_dataset)
    assert dict(rows[0]) == {"a": "0", "b": "0", "c": "0", "y": "0"}
    assert [row.name for row in rows] == list(range(len(toy_dataset)))


def test_iter_keeps_duplicate_rows_and_every_target_together():
    df = pd.DataFrame(
        {"a": ["0", "0", "0"], "y1": ["1", "1", "0"], "y2": ["0", "0", None]}
    )
    dataset = Dataset(
        df, features=["a"], targets=["y1", "y2"], assume_nondeterministic=True
    )
    assert [dict(row) for row in dataset] == [
        {"a": "0", "y1": "1", "y2": "0"},
        {"a": "0", "y1": "1", "y2": "0"},
        {"a": "0", "y1": "0", "y2": None},
    ]


def test_iter_rows_are_sliceable_into_features_and_targets(toy_dataset):
    row = next(iter(toy_dataset))
    features, targets = row[toy_dataset.features], row[toy_dataset.targets]
    assert set(features.index) == {"a", "b", "c"}
    assert set(targets.index) == {"y"}


def test_iter_yields_unknowns_as_none():
    df = pd.DataFrame({"a": ["0", None], "b": [None, "1"], "y": ["1", None]})
    dataset = Dataset(df, features=["a", "b"], targets=["y"])
    rows = list(dataset)
    assert [row.to_dict() for row in rows] == [
        {"a": "0", "b": None, "y": "1"},
        {"a": None, "b": "1", "y": None},
    ]
    assert all(row.dtype == object for row in rows)


def test_iterating_an_empty_dataset_gives_no_row():
    assert list(Dataset(pd.DataFrame(columns=["a", "y"]), ["a"], ["y"])) == []


def test_iterating_leaves_the_stored_unknowns_unknown():
    dataset = Dataset(pd.DataFrame({"a": ["0", None], "y": ["1", "0"]}), ["a"], ["y"])
    before = dataset._data.isna().to_numpy().tolist()
    list(dataset)
    assert dataset._data.isna().to_numpy().tolist() == before


def test_duplicate_index_labels_never_leak():
    # A duplicate/weird caller index must never leak through.
    df = pd.DataFrame(
        {"a": [0, 1, 2], "y": [0, 1, 0]},
        index=[5, 5, 9],  # duplicate label, non-default
    )
    dataset = Dataset(df, features=["a"], targets=["y"])
    assert dataset.positives("y", "0") == [{"a": "0"}, {"a": "2"}]
    assert dataset.positives("y", "1") == [{"a": "1"}]
    assert [row.name for row in dataset] == [0, 1, 2]


def test_to_dataframe_gives_the_features_and_targets_as_read():
    df = pd.DataFrame(
        {
            "y": [1, 0, 1],
            "id": ["p1", "p2", "p3"],
            "a": [1.0, None, 0.0],
            "b": [True, False, True],
        }
    )
    dataset = Dataset(df, features=["a", "b"], targets=["y"])
    frame = dataset.to_dataframe()
    assert list(frame.columns) == ["a", "b", "y"]
    assert frame["a"].tolist()[0::2] == ["1", "0"]
    assert pd.isna(frame["a"].iloc[1])
    assert frame["b"].tolist() == ["True", "False", "True"]
    assert frame["y"].tolist() == ["1", "0", "1"]


def test_to_dataframe_keeps_the_callers_index_labels():
    # The one place the caller's index comes back (see
    # test_duplicate_index_labels_never_leak), so rows join back.
    df = pd.DataFrame(
        {"id": ["p1", "p2", "p3"], "a": ["0", "1", "2"], "y": ["0", "1", "0"]},
        index=[5, 5, 9],
    )
    frame = Dataset(df, features=["a"], targets=["y"]).to_dataframe()
    assert frame.index.tolist() == [5, 5, 9]
    assert frame.assign(id=df["id"].to_numpy())["id"].tolist() == ["p1", "p2", "p3"]


def test_to_dataframe_is_a_copy(toy_dataset):
    frame = toy_dataset.to_dataframe()
    frame.loc[:, "a"] = "9"
    frame.index = range(100, 100 + len(frame))
    assert toy_dataset.domain("a") == ["0", "1"]
    assert toy_dataset.to_dataframe()["a"].tolist() == [r["a"] for r in toy_dataset]
    assert toy_dataset.to_dataframe().index.tolist() == list(range(8))


def test_to_dataframe_survives_pickling():
    df = pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"]}, index=["r1", "r2"])
    dataset = Dataset(df, features=["a"], targets=["y"])
    restored = pickle.loads(pickle.dumps(dataset))
    pd.testing.assert_frame_equal(restored.to_dataframe(), dataset.to_dataframe())


def test_to_dataframe_of_an_empty_dataset_has_the_columns_and_no_row():
    df = pd.DataFrame({"a": pd.Series([], dtype=str), "y": pd.Series([], dtype=str)})
    frame = Dataset(df, features=["a"], targets=["y"]).to_dataframe()
    assert list(frame.columns) == ["a", "y"]
    assert frame.empty


# --- domain and domains ---


def test_domain_returns_distinct_values_sorted(toy_dataset):
    assert toy_dataset.domain("a") == ["0", "1"]
    assert toy_dataset.domain("y") == ["0", "1"]


def test_domain_is_sorted_whatever_the_row_order():
    df = pd.DataFrame({"a": ["2", "10", "0", "2"], "y": ["0", "1", "0", "0"]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    assert dataset.domain("a") == ["0", "10", "2"]


def test_domain_raises_for_unknown_variable(toy_dataset):
    with pytest.raises(ValueError, match="is not a"):
        toy_dataset.domain("nonexistent")


def test_domains_returns_a_mapping_of_variable_to_domain(toy_dataset):
    domains = toy_dataset.domains(["a", "y"])
    assert domains == {"a": ["0", "1"], "y": ["0", "1"]}


def test_domains_defaults_to_every_feature_and_target(toy_dataset):
    domains = toy_dataset.domains()
    assert set(domains.keys()) == {"a", "b", "c", "y"}
    for variable, values in domains.items():
        assert values == toy_dataset.domain(variable)


def test_domains_can_be_restricted_to_a_subset_of_variables(toy_dataset):
    domains = toy_dataset.domains(["b"])
    assert set(domains.keys()) == {"b"}


def test_domains_raises_for_unknown_variable(toy_dataset):
    with pytest.raises(ValueError, match="is not a"):
        toy_dataset.domains(["a", "nonexistent"])


def test_domains_refuses_a_single_string():
    dataset = Dataset(pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"]}), ["a"], ["y"])
    with pytest.raises(TypeError, match="not a single string"):
        dataset.domains("ab")


# --- positives and negatives ---


@pytest.mark.parametrize("method", ["positives", "negatives"])
def test_states_are_plain_dicts_of_the_features(toy_dataset, method):
    # Not pandas.Series: Rule.matches only needs `.get`, and a dict is
    # cheaper to build and cache.
    for state in getattr(toy_dataset, method)("y", "1"):
        assert type(state) is dict
        assert set(state) == {"a", "b", "c"}


def test_positives_matches_rows_with_that_value(toy_dataset):
    positives = toy_dataset.positives("y", "1")
    expected = [
        {"a": "0", "b": "0", "c": "1"},
        {"a": "0", "b": "1", "c": "0"},
        {"a": "0", "b": "1", "c": "1"},
        {"a": "1", "b": "1", "c": "0"},
        {"a": "1", "b": "1", "c": "1"},
    ]
    assert sorted(positives, key=str) == sorted(expected, key=str)


def test_negatives_excludes_rows_with_that_value(toy_dataset):
    negatives = toy_dataset.negatives("y", "1")
    expected = [
        {"a": "0", "b": "0", "c": "0"},
        {"a": "1", "b": "0", "c": "0"},
        {"a": "1", "b": "0", "c": "1"},
    ]
    assert sorted(negatives, key=str) == sorted(expected, key=str)


def test_positives_and_negatives_partition_a_deterministic_target(toy_dataset):
    positives = toy_dataset.positives("y", "1")
    negatives = toy_dataset.negatives("y", "1")
    assert len(positives) + len(negatives) == len(toy_dataset)


def test_multi_target_negatives_positives_are_per_target():
    df = pd.DataFrame({"a": [0, 1], "y1": [0, 1], "y2": [1, 0]})
    dataset = Dataset(df, features=["a"], targets=["y1", "y2"])
    assert dataset.positives("y1", "1") == [{"a": "1"}]
    assert dataset.positives("y2", "1") == [{"a": "0"}]


def test_negatives_raises_for_unknown_target(toy_dataset):
    with pytest.raises(ValueError, match="is not a"):
        toy_dataset.negatives("nonexistent", "1")


def test_positives_raises_for_unknown_target(toy_dataset):
    with pytest.raises(ValueError, match="is not a"):
        toy_dataset.positives("nonexistent", "1")


@pytest.mark.parametrize("method", ["positives", "negatives"])
def test_a_value_that_is_not_a_str_is_refused(toy_dataset, method):
    with pytest.raises(TypeError, match=r"^value for 'y' is 1 \(int\), not a str"):
        getattr(toy_dataset, method)("y", 1)


def test_a_value_outside_the_domain_is_accepted(toy_dataset):
    # No row holds it: no positive, and every row is a negative.
    assert toy_dataset.positives("y", "absent") == []
    assert len(toy_dataset.negatives("y", "absent")) == len(toy_dataset)


def test_positives_repeated_calls_return_independent_lists(toy_dataset):
    # positives() and negatives() are cached; each call still hands
    # back a list the caller may change.
    first = toy_dataset.positives("y", "1")
    first.append({"corrupted": "yes"})
    second = toy_dataset.positives("y", "1")
    assert {"corrupted": "yes"} not in second
    assert len(second) == len(first) - 1


@pytest.mark.parametrize("method", ["positives", "negatives"])
def test_changing_a_returned_state_changes_nothing_in_the_dataset(method):
    dataset = Dataset(pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"]}), ["a"], ["y"])
    first = getattr(dataset, method)("y", "1")
    first[0]["a"] = "changed"
    assert getattr(dataset, method)("y", "1") != first


def test_domain_and_the_states_hand_out_copies(toy_dataset):
    toy_dataset.domain("a").append("leak")
    toy_dataset._positive_states("y", "1").clear()
    toy_dataset._negative_states("y", "1").clear()
    assert toy_dataset.domain("a") == ["0", "1"]
    assert toy_dataset._positive_states("y", "1")
    assert toy_dataset._negative_states("y", "1")


# --- positives and negatives: a non-deterministic target ---


def test_positives_counts_every_row_with_the_value(nondeterministic_dataset):
    # Rows: (a=0,y=0), (a=0,y=1), (a=1,y=1).
    positives = nondeterministic_dataset.positives("y", "1")
    assert sorted(positives, key=str) == sorted([{"a": "0"}, {"a": "1"}], key=str)


def test_negatives_excludes_a_state_that_has_the_value_anywhere(
    nondeterministic_dataset,
):
    # a=0 has both y=0 and y=1 rows — since y=1 IS observed for a=0
    # somewhere, neither a=0 row counts as a negative for value "1",
    # even though one of them individually has y=0. A naive per-row
    # `!= value` check would wrongly include the (a=0, y=0) row here.
    negatives = nondeterministic_dataset.negatives("y", "1")
    assert negatives == []


def test_negatives_includes_a_state_that_never_has_the_value(nondeterministic_dataset):
    # a=1 is only ever seen with y=1, so it is a negative of y=0.
    negatives = nondeterministic_dataset.negatives("y", "0")
    assert negatives == [{"a": "1"}]


@pytest.mark.parametrize("value", ["0", "1"])
def test_a_positive_state_is_never_a_negative_of_its_value(
    nondeterministic_dataset, value
):
    # Unlike the deterministic case, the two need not cover every row:
    # a state seen with both values is a negative of neither.
    positives = nondeterministic_dataset.positives("y", value)
    negatives = nondeterministic_dataset.negatives("y", value)
    assert not [state for state in positives if state in negatives]


# --- positives and negatives: a single-valued target ---


def test_negatives_empty_when_target_never_takes_other_value(degenerate_dataset):
    # Every row has y=1, so nothing is ever negative for "1".
    assert degenerate_dataset.negatives("y", "1") == []


def test_positives_all_rows_when_target_is_constant(degenerate_dataset):
    positives = degenerate_dataset.positives("y", "1")
    assert len(positives) == len(degenerate_dataset)


def test_negatives_all_rows_for_a_value_never_observed(degenerate_dataset):
    # "0" never appears at all — every row's feature state qualifies
    # as a negative for it.
    negatives = degenerate_dataset.negatives("y", "0")
    assert len(negatives) == len(degenerate_dataset)


# --- negatives with unknowns ---


def _nondeterministic(df, features, targets):
    return Dataset(df, features=features, targets=targets, assume_nondeterministic=True)


def test_unknown_negative_is_kept_by_default():
    # b=0 with a unknown could be the state a=0, b=0 seen with y=1. On a
    # deterministic system its y=0 says the hidden state is another one:
    # still a true negative.
    df = pd.DataFrame({"a": ["0", None], "b": ["0", "0"], "y": ["1", "0"]})
    dataset = Dataset(df, features=["a", "b"], targets=["y"])
    assert dataset.negatives("y", "1") == [{"a": None, "b": "0"}]


def test_identical_partial_states_are_one_state_by_default():
    # Same features unknown, same known values: read as the same state,
    # as a fully known state is, so the y=1 row sets the y=0 one aside.
    df = pd.DataFrame({"a": [None, None, "1"], "y": ["1", "0", "0"]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    assert dataset.negatives("y", "1") == [{"a": "1"}]


def test_assume_nondeterministic_drops_a_negative_that_could_be_a_positive():
    df = pd.DataFrame({"a": ["0", None], "b": ["0", "0"], "y": ["1", "0"]})
    dataset = _nondeterministic(df, ["a", "b"], ["y"])
    assert dataset.negatives("y", "1") == []


def test_assume_nondeterministic_drops_when_the_unknown_is_in_the_positive():
    df = pd.DataFrame({"a": [None, "0"], "b": ["0", "0"], "y": ["1", "0"]})
    dataset = _nondeterministic(df, ["a", "b"], ["y"])
    assert dataset.negatives("y", "1") == []


def test_assume_nondeterministic_keeps_a_negative_a_known_value_separates():
    # b differs, and b is known in both rows: never the same state.
    df = pd.DataFrame({"a": ["0", None], "b": ["0", "1"], "y": ["1", "0"]})
    dataset = _nondeterministic(df, ["a", "b"], ["y"])
    assert dataset.negatives("y", "1") == [{"a": None, "b": "1"}]


def test_assume_nondeterministic_compares_unknowns_on_both_sides():
    # a unknown in one, b unknown in the other: nothing known in both
    # differs, so they could be the same state.
    df = pd.DataFrame({"a": [None, "1"], "b": ["0", None], "y": ["1", "0"]})
    dataset = _nondeterministic(df, ["a", "b"], ["y"])
    assert dataset.negatives("y", "1") == []


@pytest.mark.parametrize(
    "df",
    [
        # With a conflict (a=0, b=0 seen with y=0 and y=1)...
        pd.DataFrame(
            {"a": ["0", "0", "1", "1"], "b": ["0", "0", "1", "0"], "y": [0, 1, 1, 0]}
        ),
        # ...and without one.
        pd.DataFrame(
            {"a": ["0", "0", "1", "1"], "b": ["0", "1", "0", "1"], "y": [0, 1, 1, 1]}
        ),
    ],
)
def test_assume_nondeterministic_changes_nothing_without_unknowns(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConflictWarning)
        plain = Dataset(df, features=["a", "b"], targets=["y"])
    assumed = _nondeterministic(df, ["a", "b"], ["y"])
    for value in ("0", "1"):
        assert assumed.negatives("y", value) == plain.negatives("y", value)


def test_assume_nondeterministic_ignores_unknown_targets():
    # The y=None row is no positive of anything, so it drops nothing.
    df = pd.DataFrame({"a": ["0", None], "y": [None, "0"]})
    dataset = _nondeterministic(df, ["a"], ["y"])
    assert dataset.negatives("y", "1") == [{"a": None}]


def test_assume_nondeterministic_on_many_rows_matches_a_pairwise_check():
    # The chunked, vectorized check against a plain pairwise one.
    rng = random.Random(0)
    rows = [
        {
            "a": rng.choice(["0", "1", "2", None]),
            "b": rng.choice(["0", "1", None]),
            "c": rng.choice(["0", "1"]),
            "y": rng.choice(["0", "1", "2"]),
        }
        for _ in range(300)
    ]
    df = pd.DataFrame(rows)
    dataset = _nondeterministic(df, ["a", "b", "c"], ["y"])

    def could_equal(r, p):
        return all(
            r[f] is None or p[f] is None or r[f] == p[f] for f in ("a", "b", "c")
        )

    for value in ("0", "1", "2"):
        positives = [r for r in rows if r["y"] == value]
        expected = [
            {f: r[f] for f in ("a", "b", "c")}
            for r in rows
            if r["y"] != value and not any(could_equal(r, p) for p in positives)
        ]
        assert dataset.negatives("y", value) == expected


@pytest.mark.parametrize("chunk", [1, 2])
def test_negatives_are_the_same_whatever_the_chunk_size(monkeypatch, chunk):
    # Small chunks run the multi-chunk path, which small data never
    # reaches.
    df = pd.DataFrame(
        {
            "a": ["0", None, "1", "1", None, "0"],
            "b": [None, "1", "0", None, "0", "1"],
            "y": ["1", "0", "0", "1", "0", "0"],
        }
    )
    dataset = Dataset(df, ["a", "b"], ["y"], assume_nondeterministic=True)
    expected = dataset.negatives("y", "1")
    monkeypatch.setattr(_dataset, "_PAIR_CHUNK", chunk)
    fresh = Dataset(df, ["a", "b"], ["y"], assume_nondeterministic=True)
    assert fresh.negatives("y", "1") == expected


# --- attributes, repr, pickling ---


@pytest.mark.parametrize(
    "attribute",
    ["features", "targets", "conflicting_targets", "assume_nondeterministic"],
)
def test_attributes_are_read_only(toy_dataset, attribute):
    with pytest.raises(AttributeError, match="has no setter"):
        setattr(toy_dataset, attribute, getattr(toy_dataset, attribute))


def test_changing_a_returned_attribute_leaves_the_dataset_alone(toy_dataset):
    toy_dataset.features.append("y")
    toy_dataset.targets.clear()
    toy_dataset.conflicting_targets.append("y")
    assert toy_dataset.features == ["a", "b", "c"]
    assert toy_dataset.targets == ["y"]
    assert toy_dataset.conflicting_targets == []


def test_repr_counts_rows_and_names_the_variables(toy_dataset):
    assert repr(toy_dataset) == (
        f"<Dataset: {len(toy_dataset)} rows, features ['a', 'b', 'c'], targets ['y']>"
    )
    one = Dataset(pd.DataFrame({"a": ["0"], "y": ["1"]}), features=["a"], targets=["y"])
    assert repr(one) == "<Dataset: 1 row, features ['a'], targets ['y']>"


def test_a_pickled_dataset_leaves_its_row_caches_behind(toy_dataset):
    # `n_jobs` sends the dataset to each worker; the caches are many
    # times the table's size there, cheaper to rebuild than to send.
    positives = toy_dataset.positives("y", "1")
    negatives = toy_dataset.negatives("y", "1")
    restored = pickle.loads(pickle.dumps(toy_dataset))
    assert restored._positives_cache == {}
    assert restored._negatives_cache == {}
    assert toy_dataset._positives_cache
    assert restored.positives("y", "1") == positives
    assert restored.negatives("y", "1") == negatives
    assert restored.domains() == toy_dataset.domains()
