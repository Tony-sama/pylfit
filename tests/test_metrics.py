"""Tests for the metrics: accuracy, ``coverage`` and ``rule_recovery``.

The accuracy tests are plain unit tests over hand-built
``y_true``/``y_pred`` lists, most with no ``Dataset`` or model: a
caller is expected to have already split a target-bearing
``DataFrame`` and called the model's ``predict`` before handing the
aligned results here.
``coverage`` measures rules on a ``Dataset``; ``rule_recovery`` compares
two rule sets, alone or at given states.
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from pylfit import (
    GULA,
    Dataset,
    Rule,
    RuleSetModel,
    accuracy_by_target,
    accuracy_score,
    coverage,
    rule_recovery,
)

# --- accuracy_score ---


def test_accuracy_score_all_correct():
    y_true = [{"y": "1"}, {"y": "0"}]
    y_pred = [{"y": "1"}, {"y": "0"}]
    assert accuracy_score(y_true, y_pred, "y") == 1.0


def test_accuracy_score_all_incorrect():
    y_true = [{"y": "1"}, {"y": "0"}]
    y_pred = [{"y": "0"}, {"y": "1"}]
    assert accuracy_score(y_true, y_pred, "y") == 0.0


def test_accuracy_score_partial():
    y_true = [{"y": "1"}, {"y": "0"}, {"y": "1"}, {"y": "0"}]
    y_pred = [{"y": "1"}, {"y": "0"}, {"y": "0"}, {"y": "1"}]
    assert accuracy_score(y_true, y_pred, "y") == 0.5


def test_accuracy_score_missing_target_key_in_prediction_raises():
    # RuleSetModel.predict always holds every target: a row without it
    # is a mistake in the call, not a wrong prediction.
    y_true = [{"y": "1"}, {"y": "0"}]
    y_pred = [{"y": "1"}, {}]
    with pytest.raises(KeyError, match="y_pred row 1 has no 'y'"):
        accuracy_score(y_true, y_pred, "y")


def test_accuracy_score_missing_target_key_in_actual_values_raises():
    with pytest.raises(KeyError, match="y_true row 0 has no 'Y'"):
        accuracy_score([{"y": "1"}], [{"Y": "1"}], "Y")


def test_accuracy_score_only_looks_at_the_requested_target():
    # Multi-target rows: a wrong y2 prediction must not affect y1's
    # score, and vice versa.
    y_true = [{"y1": "1", "y2": "0"}]
    y_pred = [{"y1": "1", "y2": "1"}]
    assert accuracy_score(y_true, y_pred, "y1") == 1.0
    assert accuracy_score(y_true, y_pred, "y2") == 0.0


# SPEC: METRIC-1
def test_accuracy_score_compares_actual_values_as_strings():
    # `RuleSetModel.predict` returns strings; `y_true` built with
    # `to_dict("records")` from an integer or bool column does not.
    y_true = [{"y": 1}, {"y": 0}, {"y": True}]
    y_pred = [{"y": "1"}, {"y": "0"}, {"y": "True"}]
    assert accuracy_score(y_true, y_pred, "y") == 1.0


@pytest.mark.parametrize(
    "y_pred",
    [
        [{"y": 1}, {"y": 0}, {"y": True}],
        [{"y": 1.0}, {"y": 0.0}, {"y": True}],
        pd.DataFrame({"y": [1, 0, True]}, dtype=object),
    ],
    ids=["int", "whole-float", "dataframe"],
)
def test_accuracy_score_compares_predicted_values_as_strings(y_pred):
    # Another classifier's predictions score as pylfit's own would.
    y_true = [{"y": "1"}, {"y": "0"}, {"y": "True"}]
    assert accuracy_score(y_true, y_pred, "y") == 1.0


# SPEC: METRIC-1
@pytest.mark.parametrize("unknown", [None, float("nan"), pd.NA, pd.NaT])
def test_accuracy_score_counts_an_unknown_prediction_as_incorrect(unknown):
    # A prediction no rule made is a real outcome, not left out of the
    # denominator; it misses even an actual value written as its text.
    y_true = [{"y": "1"}, {"y": str(unknown)}]
    score = accuracy_score(y_true, [{"y": "1"}, {"y": unknown}], "y")
    assert type(score) is float
    assert score == 0.5


@pytest.mark.parametrize(
    "y_true", [pd.Series(["1", "0"]), ["yes", "no"]], ids=["series", "labels"]
)
def test_accuracy_score_refuses_rows_that_are_not_mappings(y_true):
    # The sklearn shape, one label per row, is the likely mistake.
    with pytest.raises(TypeError, match=r"y_true row 0 is .* not a mapping"):
        accuracy_score(y_true, [{"y": "1"}, {"y": "0"}], "y")


def test_accuracy_score_reads_a_whole_float_as_dataset_does():
    # An integer column with a missing value is stored as floats.
    test = pd.DataFrame({"y": [0, 1, None]})
    y_true = test.iloc[:2].to_dict("records")
    assert y_true == [{"y": 0.0}, {"y": 1.0}]
    assert accuracy_score(y_true, [{"y": "0"}, {"y": "1"}], "y") == 1.0


def test_accuracy_score_keeps_a_fraction_as_written():
    assert accuracy_score([{"y": 0.5}], [{"y": "0.5"}], "y") == 1.0


def test_accuracy_score_reads_a_numpy_whole_float_as_its_integer_text():
    assert accuracy_score([{"y": np.float32(1.0)}], [{"y": "1"}], "y") == 1.0


# SPEC: METRIC-1
@pytest.mark.parametrize(
    "unknown", [None, float("nan"), pd.NA, pd.NaT, np.float32("nan")]
)
def test_accuracy_score_leaves_out_a_row_with_an_unknown_actual_value(unknown):
    y_true = [{"y": "1"}, {"y": unknown}, {"y": "0"}]
    y_pred = [{"y": "1"}, {"y": "1"}, {"y": "1"}]
    assert accuracy_score(y_true, y_pred, "y") == 0.5


# SPEC: METRIC-1
def test_accuracy_score_with_no_known_actual_value_raises():
    with pytest.raises(ValueError, match="no row of y_true has a known value for 'y'"):
        accuracy_score([{"y": None}], [{"y": "1"}], "y")


def test_accuracy_score_takes_a_dataframe_of_actual_values():
    test = pd.DataFrame({"a": ["0", "1", "1"], "y": [0, 1, None]})
    y_pred = [{"y": "0"}, {"y": "0"}, {"y": "1"}]
    assert accuracy_score(test[["y"]], y_pred, "y") == 0.5
    assert accuracy_score(test, y_pred, "y") == accuracy_score(
        test.to_dict("records"), y_pred, "y"
    )


@pytest.mark.parametrize(
    "column",
    [
        ["0", "1", "0", "1"],
        [0, 1, 0, 1],
        [False, True, False, True],
        [0.0, 1.0, 0.0, None],
        pd.array([0, 1, 0, None], dtype="Int64"),
        pd.Categorical([0.0, 1.0, 0.0, 1.0]),
        pd.Categorical(["0", "1", "0", "1"]),
        pd.Series([0.0, 1.0, 0.0, 1.0], dtype=object),
        pd.cut([1, 5, 1, 5], bins=[0, 2, 6]),
    ],
)
def test_accuracy_score_reads_a_dataframe_as_dataset_stores_it(column):
    # The model is right on every known row, so whatever the column's
    # type, the score is 1.0 only if both read its values the same way.
    df = pd.DataFrame({"a": ["0", "1", "0", "1"], "y": column})
    dataset = Dataset(df, features=["a"], targets=["y"], check_stand_ins=False)
    model = RuleSetModel(dataset, GULA.fit(dataset))
    y_pred = model.predict(df[["a"]], random_state=0)
    assert accuracy_score(df[["y"]], y_pred, "y") == 1.0
    assert accuracy_by_target(df[["y"]], y_pred, ["y"]) == {"y": 1.0}


def test_accuracy_score_takes_a_dataframe_of_predictions():
    y_true = pd.DataFrame({"y": ["1", "0"]})
    y_pred = pd.DataFrame({"a": ["0", "0"], "y": ["1", "1"]})
    assert accuracy_score(y_true, y_pred, "y") == 0.5
    assert accuracy_by_target(y_true, y_pred, ["y"]) == {"y": 0.5}


@pytest.mark.parametrize("box", [np.array, pd.Series])
def test_accuracy_score_takes_any_sized_sequence_of_rows(box):
    y_true = box([{"y": "1"}, {"y": "0"}])
    assert accuracy_score(y_true, [{"y": "1"}, {"y": "1"}], "y") == 0.5


def test_accuracy_score_takes_rows_that_are_series():
    # As df.iterrows() or df.loc[i] give them.
    y_true = [pd.Series({"y": "1"}), pd.Series({"y": "0"})]
    y_pred = [pd.Series({"y": "1"}), pd.Series({"y": "1"})]
    assert accuracy_score(y_true, y_pred, "y") == 0.5


def test_accuracy_score_dataframe_without_the_target_raises():
    # A name the table lacks is a ValueError; a row lacking a key, a
    # KeyError.
    frame = pd.DataFrame({"a": ["0"]})
    with pytest.raises(ValueError, match="y_true has no column 'y'"):
        accuracy_score(frame, [{"y": "1"}], "y")
    with pytest.raises(ValueError, match="y_true has no column 'y'"):
        accuracy_by_target(frame, [{"y": "1"}], ["y"])
    with pytest.raises(ValueError, match="y_pred has no column 'y'"):
        accuracy_score([{"y": "1"}], frame, "y")


@pytest.mark.parametrize(
    "rows",
    [None, 3, "y", b"y", iter([{"y": "1"}])],
    ids=["none", "int", "str", "bytes", "iterator"],
)
def test_accuracy_score_names_what_it_takes_for_rows(rows):
    with pytest.raises(TypeError, match="y_true must be a DataFrame or a list"):
        accuracy_score(rows, [{"y": "1"}], "y")
    with pytest.raises(TypeError, match="y_pred must be a DataFrame or a list"):
        accuracy_score([{"y": "1"}], rows, "y")


# SPEC: METRIC-1
def test_accuracy_score_mismatched_lengths_raises():
    y_true = [{"y": "1"}, {"y": "0"}]
    y_pred = [{"y": "1"}]
    with pytest.raises(ValueError, match="same length"):
        accuracy_score(y_true, y_pred, "y")


def test_accuracy_score_empty_input_raises():
    with pytest.raises(ValueError, match="non-empty"):
        accuracy_score([], [], "y")


# --- accuracy_by_target ---


def test_accuracy_by_target_returns_one_accuracy_per_target():
    y_true = [{"y1": "1", "y2": "0"}, {"y1": "0", "y2": "1"}]
    y_pred = [{"y1": "1", "y2": "1"}, {"y1": "0", "y2": "1"}]
    assert accuracy_by_target(y_true, y_pred, ["y1", "y2"]) == {"y1": 1.0, "y2": 0.5}


# SPEC: METRIC-2
def test_accuracy_by_target_matches_calling_accuracy_score_directly():
    y_true = [{"y1": "1", "y2": "0"}, {"y1": "1", "y2": "1"}]
    y_pred = [{"y1": "0", "y2": "0"}, {"y1": "1", "y2": "0"}]
    report = accuracy_by_target(y_true, y_pred, ["y1", "y2"])
    assert report["y1"] == accuracy_score(y_true, y_pred, "y1")
    assert report["y2"] == accuracy_score(y_true, y_pred, "y2")


def test_accuracy_by_target_takes_a_dataframe_of_actual_values():
    test = pd.DataFrame({"y1": ["1", "0"], "y2": [0, 1]})
    y_pred = [{"y1": "1", "y2": "1"}, {"y1": "0", "y2": "1"}]
    assert accuracy_by_target(test, y_pred, ["y1", "y2"]) == {"y1": 1.0, "y2": 0.5}


@pytest.mark.parametrize("targets", ["y", "yz"])
def test_accuracy_by_target_refuses_a_single_string(targets):
    rows = [{"y": "1", "z": "0"}]
    with pytest.raises(TypeError, match=r"^targets must be a collection"):
        accuracy_by_target(rows, rows, targets)


def test_accuracy_by_target_refuses_targets_it_would_use_up():
    # A generator would be emptied converting y_true, scoring nothing.
    y_true = pd.DataFrame({"y": ["1"]})
    with pytest.raises(TypeError, match="not a one-pass iterator"):
        accuracy_by_target(y_true, [{"y": "1"}], (t for t in ["y"]))
    assert accuracy_by_target(y_true, [{"y": "1"}], y_true.columns) == {"y": 1.0}


# SPEC: METRIC-2
def test_accuracy_by_target_takes_a_target_listed_twice(recwarn):
    frame = pd.DataFrame({"y": ["1", "0"]})
    assert accuracy_by_target(frame, [{"y": "1"}, {"y": "0"}], ["y", "y"]) == {"y": 1.0}
    assert not recwarn.list


def test_accuracy_by_target_empty_targets_returns_empty_dict():
    # No targets, no accuracy_score call: even y_true and y_pred of
    # different lengths don't raise.
    assert accuracy_by_target([{"y": "1"}], [{"y": "0"}], []) == {}


def test_accuracy_by_target_propagates_mismatched_length_error():
    y_true = [{"y": "1"}, {"y": "0"}]
    y_pred = [{"y": "1"}]
    with pytest.raises(ValueError, match="same length"):
        accuracy_by_target(y_true, y_pred, ["y"])


# --- coverage ---


# SPEC: LEARN-8
def test_coverage_is_one_for_every_head_of_an_unrestricted_rule_set(toy_dataset):
    rules = GULA.fit(toy_dataset)
    assert coverage(toy_dataset, rules) == {("y", "0"): 1.0, ("y", "1"): 1.0}


# SPEC: LEARN-8
def test_coverage_of_a_head_without_rules_is_zero(toy_dataset):
    rules = GULA.fit(toy_dataset, heads=[("y", "1")])
    assert coverage(toy_dataset, rules)[("y", "0")] == 0.0


def test_coverage_does_not_count_a_row_hiding_a_tested_value():
    df = pd.DataFrame({"a": ["1", None, "0"], "y": ["1", "1", "0"]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    rules = {Rule.from_string("y(1) :- a(1)."), Rule.from_string("y(0) :- a(0).")}
    assert coverage(dataset, rules) == {("y", "0"): 1.0, ("y", "1"): 0.5}


# SPEC: LEARN-8
def test_coverage_refuses_a_rule_on_another_datasets_variables(toy_dataset):
    # ValueError, as a model given the same rules raises.
    with pytest.raises(ValueError, match="'zz', which is not a feature"):
        coverage(toy_dataset, [Rule(("y", "1"), {"zz": "0"})])
    with pytest.raises(ValueError, match="'q', which is not a target"):
        coverage(toy_dataset, [Rule(("q", "1"))])


def test_coverage_counts_a_rule_for_an_unseen_value_for_no_head(toy_dataset):
    shares = coverage(toy_dataset, [Rule(("y", "unseen"))])
    assert set(shares.values()) == {0.0}


def test_coverage_takes_a_models_rules(toy_dataset):
    model = RuleSetModel(toy_dataset, GULA.fit(toy_dataset))
    assert coverage(toy_dataset, model.rules)[("y", "1")] == 1.0


def test_coverage_refuses_rule_strings(toy_dataset):
    with pytest.raises(TypeError, match=r"not a Rule; read it with Rule\.from_string"):
        coverage(toy_dataset, ["y(1) :- a(1)."])
    # Only a str gets the hint.
    with pytest.raises(TypeError, match=r"holds 1 \(int\), not a Rule$"):
        coverage(toy_dataset, [1])


def test_coverage_takes_the_dataset_first_as_a_model_does(toy_dataset):
    with pytest.raises(TypeError, match="expected a Dataset"):
        coverage(GULA.fit(toy_dataset), toy_dataset)


def test_coverage_refuses_one_rule_as_text(toy_dataset):
    with pytest.raises(
        TypeError, match=r"not a str; read a rule with Rule\.from_string"
    ):
        coverage(toy_dataset, "y(1) :- a(1).")


# --- rule_recovery ---

TRUE = Rule.from_string("y(1) :- a(1), b(1).")


# SPEC: METRIC-3
@pytest.mark.parametrize(
    ("learned", "category"),
    [
        ("y(1) :- a(1), b(1).", "exact"),
        ("y(1) :- a(1), b(1), c(0).", "too_specific"),
        ("y(1) :- a(1).", "too_general"),
        ("y(1) :- a(1), c(0).", "partial"),
        ("y(1) :- a(0), c(0).", None),
    ],
)
def test_rule_recovery_puts_a_pair_in_the_same_category_on_both_sides(
    learned, category
):
    scores = rule_recovery([TRUE], [Rule.from_string(learned)])
    assert scores["learned"][category or "spurious"] == 1.0
    assert scores["true"][category or "missing"] == 1.0


# SPEC: METRIC-3
def test_rule_recovery_shares_sum_to_1_up_to_rounding():
    truth = [Rule.from_string(f"y(1) :- a({value}).") for value in range(3)]
    extra = [Rule.from_string(f"y(1) :- b({value}).") for value in range(7)]
    scores = rule_recovery(truth, truth[:2] + extra)
    assert sum(scores["learned"].values()) == pytest.approx(1)
    assert sum(scores["true"].values()) == pytest.approx(1)


# SPEC: METRIC-3
def test_rule_recovery_counts_a_rule_given_twice_once():
    good, bad = TRUE, Rule.from_string("y(1) :- a(0), c(0).")
    other = Rule.from_string("y(0) :- b(0).")
    plain = rule_recovery([good, other], [good, bad])
    assert rule_recovery([good, other], [good, good, good, bad]) == plain
    assert rule_recovery([good, good, other], [good, bad]) == plain


# SPEC: METRIC-3
def test_rule_recovery_counts_a_rule_given_twice_once_at_a_state():
    state = dict(TRUE.body)
    twin = Rule((TRUE.head[0], "2"), {})
    plain = rule_recovery([TRUE], [TRUE, twin], [state])
    assert rule_recovery([TRUE], [TRUE, TRUE, TRUE, twin], [state]) == plain
    assert plain["precision"] == 0.5


# SPEC: METRIC-4
def test_rule_recovery_learned_shares_are_means_with_states():
    states = [dict(TRUE.body), dict.fromkeys(TRUE.body, "9")]
    fallback = Rule((TRUE.head[0], "0"), {})
    scores = rule_recovery([TRUE, fallback], [TRUE], states)
    assert sum(scores["learned"].values()) == 0.5


# SPEC: METRIC-3
def test_rule_recovery_precision_and_recall_count_exact_and_too_specific():
    learned = [
        Rule.from_string(text)
        for text in (
            "y(1) :- a(1), b(1), c(0).",
            "y(1) :- a(1).",
            "y(1) :- a(1), c(0).",
            "y(1) :- a(0), c(0).",
        )
    ]
    scores = rule_recovery([TRUE], learned)
    assert scores["precision"] == 0.25
    assert scores["recall"] == 1.0
    assert scores["true"]["too_specific"] == 1.0
    assert sum(scores["learned"].values()) == 1.0


# SPEC: METRIC-3
def test_rule_recovery_gives_each_rule_its_best_relation():
    other = Rule.from_string("y(1) :- c(1).")
    scores = rule_recovery([TRUE, other], [TRUE])
    assert scores["learned"]["exact"] == 1.0
    assert scores["true"] == {
        "exact": 0.5,
        "too_specific": 0.0,
        "too_general": 0.0,
        "partial": 0.5,
        "missing": 0.0,
    }


def test_rule_recovery_compares_only_rules_of_the_same_head():
    # The right conditions for the other value: no true rule of y=0.
    scores = rule_recovery([TRUE], [Rule.from_string("y(0) :- a(1), b(1).")])
    assert scores["learned"]["spurious"] == 1.0
    assert scores["true"]["missing"] == 1.0


# SPEC: METRIC-3
def test_rule_recovery_without_learned_rules_scores_zero():
    scores = rule_recovery([TRUE], [])
    assert scores["precision"] == scores["recall"] == 0.0
    assert set(scores["learned"].values()) == {0.0}
    assert scores["true"]["missing"] == 1.0


def _or_table():
    # y = (a and b) or c, every state.
    rows = [
        {"a": a, "b": b, "c": c, "y": str(int((a == b == "1") or c == "1"))}
        for a, b, c in itertools.product("01", repeat=3)
    ]
    return pd.DataFrame(rows)


def test_rule_recovery_precision_is_correctness_on_the_complete_table():
    table = _or_table()
    full = Dataset(table, features=["a", "b", "c"], targets=["y"])
    truth = GULA.fit(full)
    train = Dataset(
        table.iloc[[0, 1, 2, 4, 6]], features=["a", "b", "c"], targets=["y"]
    )
    learned = GULA.fit(train)
    rows = table.to_dict("records")
    seen = set()
    for rule in learned:
        correct = all(row["y"] == rule.head[1] for row in rows if rule.matches(row))
        assert rule_recovery(truth, [rule])["precision"] == float(correct)
        seen.add(correct)
    assert seen == {True, False}  # the training rows leave some rules wrong


def test_rule_recovery_of_the_true_rules_is_perfect_alone_and_per_state():
    table = _or_table()
    truth = GULA.fit(Dataset(table, features=["a", "b", "c"], targets=["y"]))
    assert rule_recovery(truth, truth)["precision"] == 1.0
    per_state = rule_recovery(truth, truth, table)
    assert per_state["precision"] == per_state["recall"] == 1.0


# SPEC: METRIC-4
def test_rule_recovery_per_state_scores_a_state_no_learned_rule_matches_zero():
    table = _or_table()
    truth = GULA.fit(Dataset(table, features=["a", "b", "c"], targets=["y"]))
    learned = [Rule.from_string("y(1) :- c(1).")]
    scores = rule_recovery(truth, learned, table)
    # c=1 in half the states, where that rule is exact; nothing
    # elsewhere.
    assert scores["precision"] == 0.5
    assert scores["learned"]["exact"] == 0.5
    # At a=b=c=1, y(1) :- a(1), b(1). matches too and is not recovered:
    # recall 1 in 3 states, 1/2 in one, 0 in 4.
    assert scores["recall"] == 3.5 / 8


def test_rule_recovery_reads_states_as_dataset_reads_them():
    states = pd.DataFrame({"a": [1], "b": [1]})
    as_text = rule_recovery([TRUE], [TRUE], [{"a": "1", "b": "1"}])
    assert rule_recovery([TRUE], [TRUE], states) == as_text


@pytest.mark.parametrize("unknown", [None, float("nan")])
def test_rule_recovery_refuses_a_state_with_an_unknown_tested_value(unknown):
    with pytest.raises(ValueError, match="unknown value for 'b'"):
        rule_recovery([TRUE], [TRUE], [{"a": "1", "b": unknown}])


def test_rule_recovery_reads_a_feature_only_a_learned_rule_tests():
    true = [Rule.from_string("y(1) :- a(1).")]
    learned = [Rule.from_string("y(1) :- a(1), c(1).")]
    states = pd.DataFrame({"a": ["1"], "c": ["1"]})
    assert rule_recovery(true, learned, states)["precision"] == 1.0


def test_rule_recovery_refuses_a_state_without_a_tested_feature():
    with pytest.raises(KeyError, match="states row 0 has no 'b'"):
        rule_recovery([TRUE], [TRUE], [{"a": "1"}])
    with pytest.raises(ValueError, match="states has no column 'b'"):
        rule_recovery([TRUE], [TRUE], pd.DataFrame({"a": ["1"]}))


# SPEC: METRIC-4
def test_rule_recovery_refuses_a_state_no_true_rule_matches():
    # One true rule is not every minimal rule of the system.
    with pytest.raises(ValueError, match="no true rule on 'y' matches states row 0"):
        rule_recovery([TRUE], [TRUE], [{"a": "0", "b": "1"}])


def test_rule_recovery_refuses_empty_inputs():
    with pytest.raises(ValueError, match="true_rules must be non-empty"):
        rule_recovery([], [TRUE])
    with pytest.raises(ValueError, match="states must be non-empty"):
        rule_recovery([TRUE], [TRUE], [])
    constant = Rule(("y", "1"))
    with pytest.raises(ValueError, match="states must be non-empty"):
        rule_recovery([constant], [constant], pd.DataFrame({"a": []}))


def test_rule_recovery_refuses_a_learned_rule_on_another_target():
    with pytest.raises(ValueError, match="concludes on 'z', which no true rule"):
        rule_recovery([TRUE], [Rule.from_string("z(1) :- a(1).")])


def test_rule_recovery_refuses_rules_as_text():
    with pytest.raises(TypeError, match="true_rules must be a collection of Rule"):
        rule_recovery("y(1) :- a(1).", [TRUE])
    with pytest.raises(TypeError, match="learned_rules holds 'y"):
        rule_recovery([TRUE], ["y(1) :- a(1)."])


def test_rule_recovery_refuses_a_row_that_is_not_a_mapping():
    with pytest.raises(TypeError, match="states row 0 is 'a', not a mapping"):
        rule_recovery([TRUE], [TRUE], ["a"])
    series = pd.Series({"a": "1", "b": "1"})
    assert rule_recovery([TRUE], [TRUE], [series])["precision"] == 1.0


def test_rule_recovery_counts_every_state_when_no_rule_tests_a_feature():
    constant = Rule(("y", "1"))
    states = pd.DataFrame({"a": ["0", "1"]})
    assert rule_recovery([constant], [constant], states)["precision"] == 1.0
