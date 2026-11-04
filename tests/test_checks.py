"""Tests for ``find_missing_stand_ins`` and ``DEFAULT_STAND_INS``."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pylfit import DEFAULT_STAND_INS, find_missing_stand_ins


def _findings(report):
    return [tuple(row) for row in report.itertuples(index=False)]


# --- stand-in strings ---


# SPEC: CHECK-1
def test_stand_ins_match_case_insensitively_with_spaces_ignored():
    df = pd.DataFrame({"a": ["x", " ? ", "Null", "NaN", "  ", "Unknown", "x"]})
    assert _findings(find_missing_stand_ins(df)) == [
        ("a", "  ", 1, "stand_in"),
        ("a", " ? ", 1, "stand_in"),
        ("a", "NaN", 1, "stand_in"),
        ("a", "Null", 1, "stand_in"),
        ("a", "Unknown", 1, "stand_in"),
    ]


# SPEC: CHECK-1
def test_a_stand_in_is_counted_over_its_rows():
    df = pd.DataFrame({"a": ["?", "?", "x", "?"]})
    assert _findings(find_missing_stand_ins(df)) == [("a", "?", 3, "stand_in")]


# SPEC: CHECK-2
def test_real_missing_values_are_not_reported():
    df = pd.DataFrame({"a": ["x", None, float("nan"), "y"]})
    assert find_missing_stand_ins(df).empty


# SPEC: CHECK-1
def test_the_stand_in_list_is_configurable():
    df = pd.DataFrame({"a": ["?", "n.d.", "x"]})
    assert _findings(find_missing_stand_ins(df, stand_ins=["N.D."])) == [
        ("a", "n.d.", 1, "stand_in")
    ]


def test_a_supplied_stand_in_ignores_its_own_spaces():
    df = pd.DataFrame({"a": ["n.d.", "x"]})
    report = find_missing_stand_ins(df, stand_ins=["n.d. "])
    assert _findings(report) == [("a", "n.d.", 1, "stand_in")]


def test_default_list_holds_deepchecks_markers_and_uci_question_mark():
    for marker in ("none", "null", "nan", "na", "", "?"):
        assert marker in DEFAULT_STAND_INS


# --- zeros far from the rest of a column ---


def test_a_far_zero_in_an_integer_column_is_reported():
    df = pd.DataFrame({"pressure": [120, 135, 0, 110, 160, 140, 0]})
    assert _findings(find_missing_stand_ins(df)) == [("pressure", 0, 2, "far_zero")]


# SPEC: CHECK-3
def test_a_far_zero_read_as_text_is_reported_in_each_spelling():
    df = pd.DataFrame({"pressure": ["120", "135", "0", "110", "0.0", "160"]})
    assert _findings(find_missing_stand_ins(df)) == [
        ("pressure", "0", 1, "far_zero"),
        ("pressure", "0.0", 1, "far_zero"),
    ]


@pytest.mark.parametrize(
    ("values", "far"),
    [
        ([40, 50, 60, 70, 80, 90, 100, 0], True),  # zero 1.8 IQR below Q1
        ([40, 40, 40, 55, 70, 70, 70, 0], False),  # inside 1.5 IQR, not 1
        ([0, 10, 20, 20, 40], True),  # linear quartiles, 17.5 and 25
        ([100, 120, 140, 0], True),  # exactly three distinct non-zeros
        ([1] * 40 + [2, 3, 4, 5, 6, 7] + [0] * 3, True),  # quartiles over rows
        ([30, 30, 30, 40, 50, 50, 50, 0], False),  # on the fence: not far
    ],
)
def test_far_zero_follows_tukeys_fences(values, far):
    report = find_missing_stand_ins(pd.DataFrame({"a": values}))
    assert (not report.empty) == far


def test_a_zero_among_small_counts_is_not_reported():
    # 0 among 1..5 is inside Tukey's fences: an ordinary count.
    df = pd.DataFrame({"visits": [0, 1, 2, 3, 4, 5, 2, 1]})
    assert find_missing_stand_ins(df).empty


def test_a_zero_between_negative_and_positive_values_is_not_reported():
    df = pd.DataFrame({"temperature": [-10, -5, 0, 5, 10, 3]})
    assert find_missing_stand_ins(df).empty


@pytest.mark.parametrize(
    "values",
    [
        [0, 1, 1, 0, 1],
        ["0", "1", "1", "0"],
        [True, False, True],
        [0, 50, 60, 0],  # two distinct non-zero values: too few
    ],
)
def test_boolean_like_or_small_columns_are_never_checked_for_zeros(values):
    assert find_missing_stand_ins(pd.DataFrame({"a": values})).empty


def test_a_column_mostly_zero_is_not_reported():
    # Zero is the usual value of an amount most rows don't have.
    df = pd.DataFrame({"loss": [0] * 6 + [1400, 1800, 2100, 2500, 1900]})
    assert find_missing_stand_ins(df).empty
    df = pd.DataFrame({"loss": [0] * 5 + [1400, 1800, 2100, 2500, 1900]})
    assert _findings(find_missing_stand_ins(df)) == [("loss", 0, 5, "far_zero")]


def test_a_column_with_text_is_not_numeric():
    df = pd.DataFrame({"a": ["120", "135", "0", "high", "160"]})
    assert find_missing_stand_ins(df).empty


# SPEC: CHECK-3
@pytest.mark.parametrize(
    "values",
    [
        pd.Series([True, False, 120, 135, 110, 160], dtype=object),
        pd.Series([False, 0, 120, 135, 110, 160], dtype=object),  # False == 0
        pd.Categorical([True, False, 120, 135, 110, 160]),
        [120, 135, 0, 110, 160, float("inf")],
        ["120", "135", "0", "110", "160", "-inf"],
    ],
)
def test_a_column_with_a_bool_or_an_infinity_is_not_numeric(values):
    assert find_missing_stand_ins(pd.DataFrame({"a": values})).empty


# SPEC: CHECK-3
def test_a_numpy_false_is_not_a_zero():
    df = pd.DataFrame(
        {"a": pd.Series([np.False_, 120, 135, 110, 160, 140], dtype=object)}
    )
    assert find_missing_stand_ins(df).empty


def test_a_complex_column_is_not_numeric_here():
    z = np.array([100 + 1j, 120 + 2j, 110 + 1j, 130 + 3j, 140 + 1j, 7j])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert find_missing_stand_ins(pd.DataFrame({"z": z})).empty


# SPEC: CHECK-3
@pytest.mark.parametrize(
    "values",
    [
        pd.to_datetime([0, 120, 135, 110, 160, 140], unit="s"),
        pd.to_timedelta([0, 120, 135, 110, 160, 140], unit="s"),
        pd.Series(pd.to_timedelta([0, 120, 135, 110, 160, 140])).astype(object),
    ],
)
def test_a_date_or_a_duration_is_not_a_number_here(values):
    assert find_missing_stand_ins(pd.DataFrame({"a": values})).empty


def test_stand_ins_are_left_out_before_the_zero_check():
    df = pd.DataFrame({"pressure": ["120", "?", "135", "0", "110", "160"]})
    assert _findings(find_missing_stand_ins(df)) == [
        ("pressure", "?", 1, "stand_in"),
        ("pressure", "0", 1, "far_zero"),
    ]


# --- categorical columns ---


def test_a_categorical_column_reports_only_the_stand_ins_present():
    df = pd.DataFrame(
        {"a": pd.Categorical(["x", "?", "y", "?"], categories=["x", "y", "?", "null"])}
    )
    assert _findings(find_missing_stand_ins(df)) == [("a", "?", 2, "stand_in")]


def test_a_categorical_column_reports_only_its_far_zeros():
    df = pd.DataFrame({"pressure": pd.Categorical([120, 135, 0, 110, 160, 140])})
    assert _findings(find_missing_stand_ins(df)) == [("pressure", 0, 1, "far_zero")]


# --- arguments ---


# SPEC: CHECK-1
def test_columns_limits_the_check():
    df = pd.DataFrame({"a": ["?", "x"], "b": ["?", "y"]})
    assert _findings(find_missing_stand_ins(df, columns=["b"])) == [
        ("b", "?", 1, "stand_in")
    ]


def test_an_unknown_column_raises():
    with pytest.raises(ValueError, match="'z'"):
        find_missing_stand_ins(pd.DataFrame({"a": ["x"]}), columns=["z"])


def test_a_column_named_twice_is_checked_once():
    df = pd.DataFrame({"a": ["?", "x"]})
    assert _findings(find_missing_stand_ins(df, columns=["a", "a"])) == [
        ("a", "?", 1, "stand_in")
    ]


def test_a_duplicated_dataframe_column_raises():
    df = pd.DataFrame([["?", "x", "y"]], columns=["a", "a", "b"])
    with pytest.raises(ValueError, match=r"duplicate column names .*\['a'\]"):
        find_missing_stand_ins(df)
    # Only the columns checked must be unique.
    assert find_missing_stand_ins(df, columns=["b"]).empty


def test_columns_in_a_set_are_refused():
    df = pd.DataFrame({"a": ["?"], "b": ["?"]})
    with pytest.raises(TypeError, match="columns must have an order"):
        find_missing_stand_ins(df, columns={"a", "b"})


# SPEC: DATA-10
@pytest.mark.parametrize("parameter", ["stand_ins", "columns"])
def test_a_single_string_is_refused(parameter):
    # A str iterates as characters: "null" would check "n", "u", "l".
    df = pd.DataFrame({"a": ["NULL", "x"]})
    with pytest.raises(TypeError, match=r"\['a'\], not a single string"):
        find_missing_stand_ins(df, **{parameter: "a"})


def test_stand_ins_that_are_not_strings_are_refused():
    df = pd.DataFrame({"a": ["?", "1"]})
    with pytest.raises(TypeError, match="stand_ins must hold strings"):
        find_missing_stand_ins(df, stand_ins=["?", 0])


def test_a_series_is_refused_with_a_hint():
    with pytest.raises(TypeError, match="expected a DataFrame, got Series"):
        find_missing_stand_ins(pd.DataFrame({"a": ["?"]})["a"])


def test_options_are_taken_by_name_only():
    with pytest.raises(TypeError, match="positional argument"):
        find_missing_stand_ins(pd.DataFrame({"a": ["?"]}), ["?"])


# --- the report ---


# SPEC: CHECK-2
def test_an_empty_report_keeps_its_columns():
    report = find_missing_stand_ins(pd.DataFrame({"a": ["x", "y"]}))
    assert list(report.columns) == ["column", "value", "count", "issue"]
    assert report.empty


@pytest.mark.parametrize("labels", [["a", "b"], [0, 1]])
def test_the_report_has_the_same_dtypes_empty_or_not(labels):
    found = find_missing_stand_ins(pd.DataFrame([["?", "x"]], columns=labels))
    empty = find_missing_stand_ins(pd.DataFrame([["y", "x"]], columns=labels))
    assert not found.empty
    assert empty.empty
    assert found.dtypes.to_dict() == empty.dtypes.to_dict()
    assert found["count"].dtype == "int64"
    assert found["column"].dtype == pd.Index(labels).dtype


@pytest.mark.parametrize("dtype", ["Int64", "UInt8", "Float64"])
def test_a_nullable_column_reports_a_python_number(dtype):
    df = pd.DataFrame({"p": pd.array([120, 135, 0, 110, 160, 140], dtype=dtype)})
    (value,) = find_missing_stand_ins(df)["value"]
    assert value == 0
    assert not isinstance(value, np.generic)


# SPEC: CHECK-1
def test_the_default_list_is_the_public_one():
    values = list(DEFAULT_STAND_INS)
    report = find_missing_stand_ins(pd.DataFrame({"a": values}))
    assert isinstance(DEFAULT_STAND_INS, tuple)
    assert set(report["value"]) == set(values)


# SPEC: CHECK-2
def test_findings_are_grouped_by_column_in_the_order_checked():
    df = pd.DataFrame({"a": ["?", "x", "?"], "b": ["null", "?", "x"]})
    report = find_missing_stand_ins(df, columns=["b", "a"])
    assert list(report.columns) == ["column", "value", "count", "issue"]
    assert list(report["column"]) == ["b", "b", "a"]


# SPEC: CHECK-3
def test_the_dataframe_is_not_changed():
    df = pd.DataFrame(
        {
            "a": ["?", " NULL ", "x", None],
            "b": [0, 120, 135, 110],
            "c": pd.Categorical(["?", "x", "x", "x"]),
        }
    )
    before = df.copy(deep=True)
    find_missing_stand_ins(df)
    pd.testing.assert_frame_equal(df, before)
