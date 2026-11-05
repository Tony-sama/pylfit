"""Tests for ``RuleSetModel``: an unweighted, equal-power set of rules.

Every rule carries equal power here — a match makes a value *a*
candidate, not *the* one, and ``predict_proba`` splits uniformly across
whatever ``predict_candidates`` matched, since there's no notion of
score to weight by (that's ``WeightedRuleSetModel``, see
``test_weighted_rule_set_model.py``). A target nothing matched falls
back to its whole domain rather than to no prediction, and ``predict``
draws randomly among tied values, so the tests below sweep a fixed
range of ``random_state`` seeds instead of asserting one outcome —
deterministic as a test, while still proving the draw covers every
tied value and never a non-tied one. Tests build rule sets by hand
with plain ``Rule`` objects rather than depending on
``BruteForce``/``GULA``.
"""

import json
import os
import subprocess
import sys
import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from pylfit import Dataset, Rule, RuleSetModel, WeightedRuleSetModel


@pytest.fixture
def toy_model(toy_dataset, toy_rules):
    return RuleSetModel(toy_dataset, toy_rules)


# --- Construction ---


def test_rules_is_required_no_default(toy_dataset):
    with pytest.raises(TypeError):
        RuleSetModel(toy_dataset)  # no implicit empty-model default


def test_empty_rules_must_be_explicit(toy_dataset):
    model = RuleSetModel(toy_dataset, set())
    assert len(model.rules) == 0
    assert model.rules == ()


def test_duplicate_rules_collapse(toy_dataset):
    rule = Rule(head=("y", "1"), body={"b": "1"})
    model = RuleSetModel(
        toy_dataset, [rule, Rule(rule.head, rule.body), Rule(rule.head, rule.body)]
    )
    assert len(model.rules) == 1


def test_features_and_targets_copied_from_dataset(toy_dataset, toy_rules):
    model = RuleSetModel(toy_dataset, toy_rules)
    assert model.features == toy_dataset.features
    assert model.targets == toy_dataset.targets


# --- Read-only rules, features, targets ---


def test_rules_is_a_tuple_sorted_by_text(toy_model, toy_rules):
    assert isinstance(toy_model.rules, tuple)
    assert toy_model.rules == tuple(sorted(set(toy_rules)))


def test_rules_cannot_be_assigned(toy_model):
    with pytest.raises(AttributeError):
        toy_model.rules = set()


@pytest.mark.parametrize("name", ["features", "targets"])
def test_features_and_targets_are_read_only(toy_model, name):
    before = getattr(toy_model, name)
    getattr(toy_model, name).append("extra")
    assert getattr(toy_model, name) == before
    with pytest.raises(AttributeError):
        setattr(toy_model, name, ["extra"])


# --- domain ---
#
# RuleSetModel.domain is a validating @property capturing every feature
# *and* target's allowed values from the training Dataset, freely
# overridable afterward, and part of RuleSetModel.__eq__ (see the
# Equality section below).


# SPEC: MODEL-1
def test_domain_captured_from_dataset_at_construction(toy_model):
    assert toy_model.domain == {
        "a": ["0", "1"],
        "b": ["0", "1"],
        "c": ["0", "1"],
        "y": ["0", "1"],
    }


def test_domain_setter_takes_any_iterable_and_the_getter_sorts_it(toy_model):
    toy_model.domain = {
        "a": ["2", "0", "1"],  # a list, out of order
        "b": {"0", "1"},
        "c": ("1", "0"),
        "y": {"0", "1"},
    }
    assert toy_model.domain["a"] == ["0", "1", "2"]
    assert toy_model.domain["c"] == ["0", "1"]


def test_domain_is_freely_overridable_after_construction(toy_model):
    # e.g. with the full population's domain, not just what an
    # unlucky training split happened to contain.
    broader = {
        variable: [*values, "9"] for variable, values in toy_model.domain.items()
    }
    toy_model.domain = broader
    assert "9" in toy_model.domain["a"]


def test_domain_setter_rejects_a_missing_variable(toy_model):
    incomplete = {k: v for k, v in toy_model.domain.items() if k != "c"}
    with pytest.raises(ValueError, match="missing variable"):
        toy_model.domain = incomplete


def test_domain_setter_rejects_an_extra_variable(toy_model):
    extra = dict(toy_model.domain)
    extra["z"] = {"0"}
    with pytest.raises(ValueError, match="unexpected variable"):
        toy_model.domain = extra


def test_domain_setter_rejects_a_str_instead_of_a_collection(toy_model):
    bad = dict(toy_model.domain)
    bad["a"] = "01"  # would otherwise be split into {"0", "1"}
    with pytest.raises(ValueError, match="'a' is given the str '01'"):
        toy_model.domain = bad


def test_domain_setter_rejects_non_str_values(toy_model):
    bad = dict(toy_model.domain)
    bad["a"] = [0, 1]
    with pytest.raises(TypeError, match="'a' has non-str value"):
        toy_model.domain = bad


def test_domain_setter_reports_missing_and_extra_together(toy_model):
    bad = {k: v for k, v in toy_model.domain.items() if k != "c"}
    bad["z"] = {"0"}
    with pytest.raises(ValueError, match="missing variable") as excinfo:
        toy_model.domain = bad
    assert "unexpected variable" in str(excinfo.value)


# --- Truth value ---


def test_a_model_without_rules_is_truthy(toy_dataset):
    # No __len__: a model is not a container, so `model or default`
    # keeps it.
    model = RuleSetModel(toy_dataset, [])
    assert model
    with pytest.raises(TypeError):
        len(model)


# --- predict_candidates / predict_proba: every match a candidate ---


# SPEC: MODEL-2
def test_predict_candidates_single_matching_rule(toy_model):
    # a=0, c=1 matches only "y(1) :- a(0), c(1)." (b=0 rules out
    # "b=1"; a=0 rules out both y=0 rules).
    candidates = toy_model.predict_candidates({"a": "0", "b": "0", "c": "1"})
    assert candidates["y"] == ["1"]


def test_predict_candidates_no_match_falls_back_to_the_whole_domain(toy_dataset):
    # Open-world: nothing matched means nothing was ruled out, so every
    # known value of the target stays possible — not an empty set.
    model = RuleSetModel(toy_dataset, [Rule(head=("y", "1"), body={"a": "1"})])
    candidates = model.predict_candidates({"a": "0", "b": "0", "c": "0"})
    assert candidates["y"] == ["0", "1"]


def test_predict_proba_uniform_split_across_conflicting_values(toy_dataset):
    # Two rules with no body at all — both always match, concluding
    # opposite values for the same target — exercises the uniform
    # 1/(number of candidates) split, since there's no score to weight
    # by.
    rule_1 = Rule(head=("y", "1"), body={})
    rule_0 = Rule(head=("y", "0"), body={})
    model = RuleSetModel(toy_dataset, [rule_1, rule_0])

    candidates = model.predict_candidates({"a": "0", "b": "0", "c": "0"})
    assert candidates["y"] == ["0", "1"]

    probabilities = model.predict_proba({"a": "0", "b": "0", "c": "0"})
    assert probabilities["y"] == {"1": 0.5, "0": 0.5}


def test_predict_proba_and_predict_agree_for_a_clear_case(toy_model):
    # predict_proba takes a state or a DataFrame, predict only a
    # DataFrame (list[dict] out), so each is called its own way here.
    state = {"a": "0", "b": "0", "c": "1"}
    assert toy_model.predict_proba(state)["y"] == {"0": 0.0, "1": 1.0}
    assert toy_model.predict(pd.DataFrame([state])) == [{"y": "1"}]


def test_predict_with_no_rules_at_all_is_uniform_over_the_domain():
    # An empty model rules nothing out, so it doesn't predict "nothing"
    # — it predicts "anything", uniformly.
    df = pd.DataFrame({"a": [0, 1], "y": [0, 1]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    model = RuleSetModel(dataset, set())
    assert model.predict_candidates({"a": "0"}) == {"y": ["0", "1"]}
    assert model.predict_proba({"a": "0"}) == {"y": {"0": 0.5, "1": 0.5}}

    dataframe = pd.DataFrame({"a": ["0"]})
    # Seeds swept rather than repeated unseeded calls, so the test is
    # deterministic while still proving both values get drawn.
    drawn = {model.predict(dataframe, random_state=seed)[0]["y"] for seed in range(20)}
    assert drawn == {"0", "1"}


# SPEC: MODEL-8
def test_each_tied_value_is_equally_likely():
    df = pd.DataFrame({"a": ["0"] * 3, "y": ["p", "q", "r"]})
    dataset = Dataset(df, features=["a"], targets=["y"], assume_nondeterministic=True)
    model = RuleSetModel(dataset, [Rule(("y", v), {}) for v in "pqr"])
    rows = pd.DataFrame({"a": ["0"] * 3000})
    drawn = [row["y"] for row in model.predict(rows, random_state=0)]
    # 1000 each, within about four standard deviations of 26.
    assert all(abs(drawn.count(value) - 1000) < 100 for value in "pqr")


def test_predict_breaks_ties_randomly_not_toward_the_smallest(toy_dataset):
    # Every rule carries equal weight, so any multi-candidate case is a
    # tie by construction. Resolving it toward the smallest value would
    # bias every conflict toward one class; it's drawn fairly instead.
    rule_1 = Rule(head=("y", "1"), body={})
    rule_0 = Rule(head=("y", "0"), body={})
    model = RuleSetModel(toy_dataset, [rule_1, rule_0])

    dataframe = pd.DataFrame([{"a": "0", "b": "0", "c": "0"}])
    drawn = {model.predict(dataframe, random_state=seed)[0]["y"] for seed in range(20)}
    assert drawn == {"0", "1"}


# SPEC: MODEL-9
def test_predict_random_state_makes_a_tie_reproducible(toy_dataset):
    rule_1 = Rule(head=("y", "1"), body={})
    rule_0 = Rule(head=("y", "0"), body={})
    model = RuleSetModel(toy_dataset, [rule_1, rule_0])

    dataframe = pd.DataFrame([{"a": "0", "b": "0", "c": "0"}])
    first = model.predict(dataframe, random_state=7)
    assert all(model.predict(dataframe, random_state=7) == first for _ in range(5))


# --- predict(dataframe): batch shape and input validation ---
#
# This section exercises: missing feature, extra column, target column
# present, and out-of-domain value are all hard failures with no
# opt-out; a row not covering the full training domain is fine
# (silent), since that's expected, especially for a single row.


def test_predict_returns_one_prediction_per_row_in_order(toy_model):
    dataframe = pd.DataFrame(
        [
            {"a": "0", "b": "0", "c": "1"},  # matches "y(1) :- a(0), c(1)."
            {"a": "1", "b": "0", "c": "0"},  # matches "y(0) :- a(1), b(0)."
        ]
    )
    assert toy_model.predict(dataframe) == [{"y": "1"}, {"y": "0"}]


def test_predict_coerces_non_str_feature_values(toy_model):
    dataframe = pd.DataFrame([{"a": 0, "b": 0, "c": 1}])  # ints, not str
    assert toy_model.predict(dataframe) == [{"y": "1"}]


def test_predict_row_not_covering_full_training_domain_is_fine(toy_model):
    # A single row only ever exercises one value per feature — it
    # doesn't need to "cover" the full training domain to be valid.
    dataframe = pd.DataFrame([{"a": "0", "b": "1", "c": "0"}])
    assert toy_model.predict(dataframe) == [{"y": "1"}]  # matches "y(1) :- b(1)."


def test_predict_missing_feature_column_raises(toy_model):
    dataframe = pd.DataFrame([{"a": "0", "b": "0"}])  # missing c
    with pytest.raises(ValueError, match="missing required feature"):
        toy_model.predict(dataframe)


# SPEC: MODEL-11
def test_predict_target_column_present_raises(toy_model):
    dataframe = pd.DataFrame([{"a": "0", "b": "0", "c": "1", "y": "1"}])
    with pytest.raises(ValueError, match="target column"):
        toy_model.predict(dataframe)


def test_predict_unexpected_column_raises(toy_model):
    dataframe = pd.DataFrame([{"a": "0", "b": "0", "c": "1", "z": "9"}])
    with pytest.raises(ValueError, match="unexpected column"):
        toy_model.predict(dataframe)


def test_predict_reports_missing_target_and_unexpected_columns_together(toy_model):
    # missing b, c; leaked target y; unexpected z — all three column
    # problems are meant to surface in one combined message.
    dataframe = pd.DataFrame([{"a": "0", "y": "1", "z": "9"}])
    with pytest.raises(ValueError, match="missing required feature") as excinfo:
        toy_model.predict(dataframe)
    message = str(excinfo.value)
    assert "target column" in message
    assert "unexpected column" in message


def test_an_unknown_does_not_stop_the_check_of_the_other_values(toy_model):
    with pytest.raises(ValueError, match="out-of-domain"):
        toy_model.predict_candidates({"a": None, "b": "7", "c": "1"})


def test_predict_out_of_domain_value_raises(toy_model):
    dataframe = pd.DataFrame([{"a": "9", "b": "0", "c": "1"}])
    with pytest.raises(ValueError, match="out-of-domain"):
        toy_model.predict(dataframe)


def test_predict_reads_nan_as_unknown_not_as_out_of_domain(toy_model):
    # Not the string "nan", which check_domains would reject.
    dataframe = pd.DataFrame([{"a": float("nan"), "b": "1", "c": "1"}])
    assert toy_model.predict(dataframe) == [{"y": "1"}]


# --- Unknown values: fallback and strict ---


@pytest.fixture
def unknowns_model():
    dataset = Dataset(
        pd.DataFrame({"a": ["0", "1"], "b": ["0", "1"], "y": ["0", "1"]}),
        features=["a", "b"],
        targets=["y"],
    )
    rules = [Rule.from_string("y(1) :- a(1)."), Rule.from_string("y(0) :- a(0), b(0).")]
    return RuleSetModel(dataset, rules)


# SPEC: MODEL-3
def test_fallback_candidates_are_the_values_of_rules_that_could_match(
    unknowns_model,
):
    # a unknown, b=1: y(1) :- a(1). could match; y(0) needs b=0.
    state = {"a": None, "b": "1"}
    assert unknowns_model.predict_candidates(state) == {"y": ["1"]}


def test_strict_candidates_fall_back_to_the_whole_domain(unknowns_model):
    state = {"a": None, "b": "1"}
    assert unknowns_model.predict_candidates(state, unknowns="strict") == {
        "y": ["0", "1"]
    }


def test_fallback_keeps_a_strict_match(unknowns_model):
    # b unknown: y(1) :- a(1). matches for sure; y(0) could, but a
    # strict match exists, so the fallback never runs.
    assert unknowns_model.predict_candidates({"a": "1", "b": None}) == {"y": ["1"]}


def test_a_non_str_value_beside_an_unknown_is_refused(unknowns_model):
    # check_state refuses it before either pass; match_weight's own
    # refusal is tested directly in test_helpers.py.
    with pytest.raises(TypeError, match="state value for 'b' is 0"):
        unknowns_model.predict_candidates({"a": None, "b": 0})


def test_fallback_with_no_possible_rule_gives_the_whole_domain():
    dataset = Dataset(
        pd.DataFrame({"a": ["0", "1"], "b": ["0", "1"], "y": ["0", "1"]}),
        features=["a", "b"],
        targets=["y"],
    )
    model = RuleSetModel(dataset, [Rule.from_string("y(0) :- a(0), b(0).")])
    assert model.predict_candidates({"a": None, "b": "1"}) == {"y": ["0", "1"]}


@pytest.mark.parametrize("unknowns", ["fallback", "strict"])
def test_unknowns_modes_agree_without_unknowns(toy_model, unknowns):
    for a, b, c in [("0", "0", "0"), ("1", "0", "1"), ("0", "1", "1")]:
        state = {"a": a, "b": b, "c": c}
        assert toy_model.predict_candidates(
            state, unknowns=unknowns
        ) == toy_model.predict_candidates(state)


def test_unknowns_mode_is_validated(unknowns_model):
    with pytest.raises(ValueError, match="unknowns must be one of"):
        unknowns_model.predict_candidates({"a": "1", "b": "1"}, unknowns="partial")


def test_predict_forwards_unknowns(unknowns_model):
    rows = pd.DataFrame({"a": [None], "b": ["1"]})
    assert unknowns_model.predict(rows) == [{"y": "1"}]
    assert unknowns_model.predict_proba(rows, unknowns="strict") == [
        {"y": {"0": 0.5, "1": 0.5}}
    ]


def test_fallback_still_raises_on_a_missing_key(unknowns_model):
    with pytest.raises(KeyError, match="'a': None for an unknown"):
        unknowns_model.predict_candidates({"b": "1"})


# --- Single-state input: checked as a DataFrame row is ---


# SPEC: MODEL-11
@pytest.mark.parametrize("method", ["predict_candidates", "predict_proba"])
def test_single_state_out_of_domain_value_raises(toy_model, method):
    with pytest.raises(
        ValueError, match=r"out-of-domain value\(s\) for feature 'a': 7"
    ):
        getattr(toy_model, method)({"a": "7", "b": "1", "c": "0"})


@pytest.mark.parametrize("method", ["predict_candidates", "predict_proba"])
def test_single_state_missing_feature_raises_even_if_no_rule_tests_it(
    toy_dataset, method
):
    model = RuleSetModel(toy_dataset, [Rule(head=("y", "1"), body={"b": "1"})])
    with pytest.raises(KeyError, match="'a': None, 'c': None"):
        getattr(model, method)({"b": "1"})


@pytest.mark.parametrize("method", ["predict_candidates", "predict_proba"])
def test_single_state_non_str_value_raises(toy_model, method):
    with pytest.raises(TypeError, match="not a str"):
        getattr(toy_model, method)({"a": 0, "b": "1", "c": "0"})


def test_single_state_ignores_extra_keys(toy_model):
    state = {"a": "0", "b": "1", "c": "0"}
    assert toy_model.predict_candidates(
        {**state, "y": "1", "note": 3}
    ) == toy_model.predict_candidates(state)


def test_single_state_accepts_a_dataframe_row(toy_model):
    row = pd.DataFrame({"a": ["0"], "b": ["1"], "c": ["0"]}).iloc[0]
    assert toy_model.predict_candidates(row) == {"y": ["1"]}


def test_predict_reads_whole_number_floats_as_integers():
    # As Dataset does: 1.0 is "1", so a column pandas made float still
    # predicts like the integers it holds.
    dataset = Dataset(
        pd.DataFrame({"a": [0, 1], "y": ["0", "1"]}), features=["a"], targets=["y"]
    )
    model = RuleSetModel(dataset, {Rule.from_string("y(1) :- a(1).")})
    assert model.predict(pd.DataFrame({"a": [1.0]})) == [{"y": "1"}]


def test_predict_column_problems_are_checked_before_domain_problems(toy_model):
    # Missing c *and* an out-of-domain "a" at once — the column-set
    # failure must be the one reported, since it's checked first.
    dataframe = pd.DataFrame([{"a": "9", "b": "0"}])
    with pytest.raises(ValueError, match="missing required feature"):
        toy_model.predict(dataframe)


def test_predict_float_feature_column_raises(toy_model):
    # The same float refusal Dataset.__init__ applies at training time.
    dataframe = pd.DataFrame([{"a": 0.5, "b": "0", "c": "1"}])
    with pytest.raises(ValueError, match="float"):
        toy_model.predict(dataframe)


def test_predict_does_not_repeat_the_max_categories_warning():
    # The 25-value column passed at max_categories=30; predict, whose
    # domain check refuses any unseen value, doesn't warn on it again.
    df = pd.DataFrame({"a": list(range(25)), "y": ["1"] * 25})
    dataset = Dataset(df, features=["a"], targets=["y"], max_categories=30)
    model = RuleSetModel(dataset, [Rule(head=("y", "1"), body={})])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.predict(pd.DataFrame({"a": list(range(25))}))


# --- Persistence: save / from_file ---


# SPEC: FILE-1
def test_save_from_file_round_trip(toy_model, tmp_path):
    path = tmp_path / "model.json"
    toy_model.save(str(path))
    loaded = RuleSetModel.from_file(str(path))
    assert loaded == toy_model


def test_save_and_from_file_accept_a_path_object(toy_model, tmp_path):
    path = tmp_path / "model.json"
    toy_model.save(path)
    assert RuleSetModel.from_file(path) == toy_model


# SPEC: FILE-3
def test_saved_file_is_readable_json_with_display_strings(toy_model, tmp_path):
    path = tmp_path / "model.json"
    toy_model.save(str(path))
    payload = json.loads(path.read_text())

    assert payload["schema_version"] == 2
    assert payload["model_class"] == "RuleSetModel"
    assert isinstance(payload["pylfit_version"], str)
    assert payload["features"] == toy_model.features
    assert payload["targets"] == toy_model.targets
    assert "value_frequencies" not in payload  # WeightedRuleSetModel-only

    rule_entry = next(
        entry
        for entry in payload["rules"]
        if entry["head"] == ["y", "1"] and entry["body"] == {"b": "1"}
    )
    assert rule_entry["display"] == "y(1) :- b(1)."
    assert "score" not in rule_entry  # WeightedRuleSetModel-only


# SPEC: FILE-2
def test_from_file_rejects_unknown_schema_version(toy_model, tmp_path):
    path = tmp_path / "model.json"
    toy_model.save(str(path))
    payload = json.loads(path.read_text())
    payload["schema_version"] = 999
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="schema_version"):
        RuleSetModel.from_file(str(path))


def test_from_file_rejects_a_weighted_rule_set_model_file(
    toy_dataset, toy_rules, tmp_path
):
    path = tmp_path / "model.json"
    WeightedRuleSetModel(toy_dataset, toy_rules).save(str(path))

    with pytest.raises(ValueError, match="WeightedRuleSetModel"):
        RuleSetModel.from_file(str(path))


def test_from_file_rejects_domain_missing_a_variable(toy_model, tmp_path):
    path = tmp_path / "model.json"
    toy_model.save(str(path))
    payload = json.loads(path.read_text())
    del payload["domain"]["c"]
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="missing variable"):
        RuleSetModel.from_file(str(path))


@pytest.mark.parametrize("key", ["features", "targets"])
@pytest.mark.parametrize("names", ["abc", ["a", 1], {"a": "b"}])
def test_from_file_rejects_names_that_are_not_a_list_of_str(
    toy_model, tmp_path, key, names
):
    # A bare str would otherwise be read as one name per character.
    path = tmp_path / "model.json"
    toy_model.save(path)
    payload = json.loads(path.read_text())
    payload[key] = names
    path.write_text(json.dumps(payload))

    with pytest.raises(TypeError, match=f"'{key}' must be a list of str"):
        RuleSetModel.from_file(path)


def test_save_overwrites_existing_file(toy_dataset, toy_model, tmp_path):
    path = tmp_path / "model.json"
    toy_model.save(str(path))

    smaller = RuleSetModel(toy_dataset, [Rule(head=("y", "1"), body={"b": "1"})])
    smaller.save(str(path))

    loaded = RuleSetModel.from_file(str(path))
    assert loaded == smaller
    assert loaded != toy_model


# --- Equality ---


def test_equal_models_from_the_same_dataset_and_rules(toy_dataset, toy_rules):
    model_a = RuleSetModel(toy_dataset, toy_rules)
    model_b = RuleSetModel(
        toy_dataset, [Rule(rule.head, rule.body) for rule in toy_rules]
    )
    assert model_a == model_b


# SPEC: MODEL-12
def test_models_with_different_rules_are_unequal(toy_dataset, toy_rules):
    model_a = RuleSetModel(toy_dataset, toy_rules)
    model_b = RuleSetModel(toy_dataset, toy_rules[:-1])
    assert model_a != model_b


def test_models_with_different_domain_are_unequal(toy_dataset, toy_rules):
    # Two models fit on the same data but with differently-overridden
    # domains (e.g. one broadened to the full population's) must
    # compare unequal — domain affects check_domains, so it's part of
    # a model's predictive behavior.
    model_a = RuleSetModel(toy_dataset, toy_rules)
    model_b = RuleSetModel(toy_dataset, toy_rules)
    model_b.domain = {
        variable: [*values, "9"] for variable, values in model_b.domain.items()
    }
    assert model_a != model_b


def test_models_with_same_domain_values_via_different_iterable_types_are_equal(
    toy_dataset, toy_rules
):
    # domain always normalizes to a set at assignment, so handing in a
    # list of the same values shouldn't create a spurious inequality.
    model_a = RuleSetModel(toy_dataset, toy_rules)
    model_b = RuleSetModel(toy_dataset, toy_rules)
    model_b.domain = {
        variable: list(values) for variable, values in model_b.domain.items()
    }
    assert model_a == model_b


def test_not_equal_to_a_non_model_object(toy_model):
    assert (toy_model == "not a model") is False
    assert (toy_model != "not a model") is True


def test_never_equal_to_a_weighted_rule_set_model_with_identical_rules(
    toy_dataset, toy_rules
):
    # type(other) is type(self), not isinstance — a
    # WeightedRuleSetModel carries scores RuleSetModel has no concept
    # of, so the two are never equal even with identical rules.
    plain = RuleSetModel(toy_dataset, toy_rules)
    weighted = WeightedRuleSetModel(toy_dataset, toy_rules)
    assert plain != weighted
    assert weighted != plain


# --- RuleSetModel stays unhashable ---


def test_model_is_unhashable(toy_model):
    with pytest.raises(TypeError, match="unhashable type"):
        hash(toy_model)


# --- String representation ---


def test_str_sorts_rules(toy_model):
    text = str(toy_model)
    lines = text.splitlines()
    rule_lines = [line for line in lines if line.startswith("  ")]
    assert rule_lines == sorted(rule_lines)


def test_str_has_no_score_suffix(toy_model):
    text = str(toy_model)
    rule_lines = [line for line in text.splitlines() if line.startswith("  ")]
    for line in rule_lines:
        assert "(+" not in line


def test_repr_is_a_one_line_summary(toy_model):
    assert repr(toy_model) == (
        f"<RuleSetModel: {len(toy_model.rules)} rules, "
        f"features {toy_model.features}, targets {toy_model.targets}>"
    )


# --- Rules outside the model are rejected ---


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("Y(1) :- a(0).", "'Y' is not a target"),
        ("y(2) :- a(0).", "'2' is not a value of 'y'"),
        ("y(1) :- A(0).", "'A' is not a feature"),
        ("y(1) :- a(7).", "'7' is not a value of 'a'"),
    ],
)
@pytest.mark.parametrize("model_class", [RuleSetModel, WeightedRuleSetModel])
def test_a_rule_outside_the_model_is_rejected(toy_dataset, model_class, text, message):
    # A misspelt rule would otherwise sit in the model and never match.
    rule = Rule.from_string(text)
    with pytest.raises(ValueError, match=message):
        model_class(toy_dataset, [rule])


def test_narrowing_the_domain_below_a_rule_is_rejected(toy_dataset):
    model = RuleSetModel(toy_dataset, [Rule.from_string("y(1) :- a(0).")])
    narrowed = {variable: set(values) for variable, values in model.domain.items()}
    narrowed["a"] = {"1"}
    with pytest.raises(ValueError, match="'0' is not a value of 'a'"):
        model.domain = narrowed
    assert model.domain["a"] == ["0", "1"]


def test_loading_a_file_with_a_rule_outside_the_model_raises(toy_dataset, tmp_path):
    path = tmp_path / "model.json"
    RuleSetModel(toy_dataset, [Rule.from_string("y(1) :- a(0).")]).save(str(path))
    payload = json.loads(path.read_text())
    payload["rules"][0]["body"] = {"A": "0"}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="'A' is not a feature"):
        RuleSetModel.from_file(str(path))


def test_a_subclass_model_equals_one_of_its_own_class(toy_dataset):
    class Custom(RuleSetModel):
        pass

    model = Custom(toy_dataset, [Rule.from_string("y(1) :- a(0).")])
    assert model == Custom(toy_dataset, model.rules)
    assert model != RuleSetModel(toy_dataset, model.rules)


# --- Regression tests: one per fix ---


def test_changing_the_returned_domain_changes_nothing_in_the_model(toy_model):
    domain = toy_model.domain
    domain["a"].append("9")
    domain["y"] = []
    assert toy_model.domain["a"] == ["0", "1"]
    assert toy_model.domain["y"] == ["0", "1"]


def test_predict_takes_a_numpy_integer_seed(toy_model):
    frame = pd.DataFrame([{"a": "0", "b": "0", "c": "0"}])
    assert toy_model.predict(frame, random_state=np.int64(3)) == toy_model.predict(
        frame, random_state=3
    )


@pytest.mark.parametrize("seed", [1.5, "3", True])
def test_predict_refuses_a_seed_that_is_not_an_integer(toy_model, seed):
    frame = pd.DataFrame([{"a": "0", "b": "0", "c": "0"}])
    with pytest.raises(TypeError, match="random_state must be an int"):
        toy_model.predict(frame, random_state=seed)


@pytest.mark.parametrize("model_class", [RuleSetModel, WeightedRuleSetModel])
def test_a_bad_option_is_refused_even_on_an_empty_frame(
    toy_dataset, toy_rules, model_class
):
    model = model_class(toy_dataset, toy_rules)
    empty = pd.DataFrame(columns=["a", "b", "c"])
    methods = [model.predict, model.predict_proba]
    if model_class is WeightedRuleSetModel:
        methods.append(model.predict_likelihood)
    for method in methods:
        with pytest.raises(ValueError, match="unknowns must be one of"):
            method(empty, unknowns="loose")


def test_a_rule_that_is_not_a_rule_is_refused(toy_dataset):
    with pytest.raises(TypeError, match=r"Rule\.from_string"):
        RuleSetModel(toy_dataset, ["y(1) :- b(1)."])


def test_predict_refuses_a_repeated_column(toy_model):
    frame = pd.DataFrame([["0", "0", "0", "1"]], columns=["a", "b", "c", "c"])
    with pytest.raises(ValueError, match="more than once: c"):
        toy_model.predict(frame)


def test_predict_reports_columns_whose_names_mix_types(toy_model):
    frame = pd.DataFrame([["0", "0", "0", "1", "2"]], columns=["a", "b", "c", 0, "z"])
    with pytest.raises(ValueError, match="unexpected column"):
        toy_model.predict(frame)


def _saved_payload(model, tmp_path):
    path = tmp_path / "model.json"
    model.save(path)
    return json.loads(path.read_text(encoding="utf-8")), path


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        (lambda p: p["domain"].update(a="01"), ValueError, "single"),
        (lambda p: p.update(targets=["a"]), ValueError, "more than once"),
        (lambda p: p.update(rules={}), TypeError, "'rules' must be a list"),
        (lambda p: p.update(model_class="Model"), ValueError, "not 'RuleSetModel'$"),
        (lambda p: p["domain"]["a"].append(1), TypeError, "str"),
        (lambda p: p.pop("features"), KeyError, "features"),
        (lambda p: p.pop("targets"), KeyError, "targets"),
        (lambda p: p.pop("domain"), KeyError, "domain"),
        (lambda p: p.pop("rules"), KeyError, "rules"),
        (lambda p: p["rules"][0].pop("head"), KeyError, "head"),
        (lambda p: p["rules"][0].pop("body"), KeyError, "body"),
        (lambda p: p.update(domain=[["0", "1"]]), TypeError, "must be a mapping"),
        (lambda p: p["rules"][0].update(head="y1"), TypeError, "head must be a"),
        # Only the object save writes: dict() would read these as
        # bodies.
        (lambda p: p["rules"][0].update(body=[]), TypeError, "got \\[\\]"),
        (
            lambda p: p["rules"][0].update(body=[["a", "1"]]),
            TypeError,
            "got \\[\\['a', '1'\\]\\]",
        ),
        (
            lambda p: p["rules"][0].update(body="x"),
            TypeError,
            "save writes it, got 'x'",
        ),
    ],
)
def test_from_file_refuses_a_malformed_file(
    toy_model, tmp_path, change, error, message
):
    payload, path = _saved_payload(toy_model, tmp_path)
    change(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(error, match=message):
        RuleSetModel.from_file(path)


# SPEC: FILE-4
def test_equal_models_save_the_same_file(toy_dataset, tmp_path):
    one = RuleSetModel(toy_dataset, [Rule(("y", "1"), {"a": "0", "c": "1"})])
    other = RuleSetModel(toy_dataset, [Rule(("y", "1"), {"c": "1", "a": "0"})])
    one.save(tmp_path / "one.json")
    other.save(tmp_path / "other.json")
    assert (tmp_path / "one.json").read_bytes() == (
        tmp_path / "other.json"
    ).read_bytes()


def test_equality_with_a_non_model_defers_to_the_other_operand(toy_model):
    assert toy_model.__eq__(1) is NotImplemented
    assert toy_model == mock.ANY


# SPEC: MODEL-8
def test_a_target_with_an_empty_domain_is_predicted_as_none(toy_dataset):
    model = RuleSetModel(toy_dataset, [])
    model.domain = {**model.domain, "y": set()}
    assert model.predict(pd.DataFrame([{"a": "0", "b": "0", "c": "0"}])) == [
        {"y": None}
    ]


def test_prediction_options_are_keyword_only(toy_dataset, toy_rules):
    # predict(X, "strict") would otherwise read "strict" as random_state
    # and predict with the fallback without a word.
    model = WeightedRuleSetModel(toy_dataset, toy_rules)
    frame = pd.DataFrame([{"a": "0", "b": "0", "c": "0"}])
    state = {"a": "0", "b": "0", "c": "0"}
    for call in (
        lambda: model.predict(frame, "strict"),
        lambda: model.predict_proba(state, "strict"),
        lambda: model.predict_likelihood(state, "strict"),
        lambda: model.predict_candidates(state, "strict"),
        lambda: WeightedRuleSetModel(toy_dataset, toy_rules, True),
    ):
        with pytest.raises(TypeError, match="positional"):
            call()


def test_predict_refuses_a_single_state(toy_model):
    with pytest.raises(TypeError, match="predict takes a DataFrame"):
        toy_model.predict({"a": "0", "b": "0", "c": "0"})


@pytest.mark.parametrize("model_class", [RuleSetModel, WeightedRuleSetModel])
def test_a_model_refuses_a_dataframe_for_its_dataset(model_class, toy_rules):
    frame = pd.DataFrame({"a": ["0"], "b": ["0"], "c": ["0"], "y": ["0"]})
    with pytest.raises(TypeError, match="wrap it first"):
        model_class(frame, toy_rules)


def test_a_lone_best_value_takes_no_draw():
    # Only a tie draws: rows with one best value before a tied row leave
    # the tied row's draw where it was.
    dataset = Dataset(
        pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"]}), features=["a"], targets=["y"]
    )
    model = RuleSetModel(
        dataset, [Rule.from_string("y(0) :- ."), Rule.from_string("y(1) :- a(1).")]
    )
    tied = pd.DataFrame({"a": ["1"]})
    for seed in range(12):
        alone = model.predict(tied, random_state=seed)[-1]
        for decided in range(1, 4):
            frame = pd.concat(
                [pd.DataFrame({"a": ["0"] * decided}), tied], ignore_index=True
            )
            assert model.predict(frame, random_state=seed)[-1] == alone


def test_a_negative_seed_is_refused(toy_model):
    # random.Random seeds from abs(seed), so -1 would repeat 1.
    frame = pd.DataFrame([{"a": "0", "b": "0", "c": "0"}])
    with pytest.raises(ValueError, match="0 or more"):
        toy_model.predict(frame, random_state=-1)


def test_rules_with_the_same_text_save_in_a_fixed_order():
    # "a(0), b" as a variable name makes the second rule print as the
    # first.
    frame = pd.DataFrame(
        {"a": ["0", "1"], "b": ["1", "0"], "a(0), b": ["1", "0"], "y": ["1", "0"]}
    )
    dataset = Dataset(frame, features=["a", "b", "a(0), b"], targets=["y"])
    first = Rule(("y", "1"), {"a": "0", "b": "1"})
    second = Rule(("y", "1"), {"a(0), b": "1"})
    assert str(first) == str(second)
    # The body breaks the tie: "a" sorts before "a(0), b".
    assert RuleSetModel(dataset, [second, first]).rules == (first, second)


SAVE_SCRIPT = """
import sys
import pandas as pd
from pylfit import Dataset, Rule, RuleSetModel
values = [str(i) for i in range(12)]
frame = pd.DataFrame(
    {"a": values, "b": values[::-1], "a(0), b": values, "y": values}
)
dataset = Dataset(frame, features=["a", "b", "a(0), b"], targets=["y"])
rules = [Rule(("y", "1"), {"a": "0", "b": "1"}), Rule(("y", "1"), {"a(0), b": "1"})]
RuleSetModel(dataset, rules).save(sys.argv[1])
"""


# SPEC: FILE-1
def test_a_save_that_fails_leaves_the_file_as_it_was(tmp_path):
    df = pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"]})
    model = RuleSetModel(Dataset(df, features=["a"], targets=["y"]), [])
    path = tmp_path / "model.json"
    model.save(path)
    saved = path.read_bytes()
    model.domain = {"a": ["0", "\udcff"], "y": ["0", "1"]}
    with pytest.raises(UnicodeEncodeError):
        model.save(path)
    assert path.read_bytes() == saved


# SPEC: FILE-4
def test_save_does_not_depend_on_the_order_a_domain_was_assigned_in(tmp_path):
    df = pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    first = RuleSetModel(dataset, [])
    second = RuleSetModel(dataset, [])
    second.domain = {"y": ["0", "1"], "a": ["0", "1"]}
    first.save(tmp_path / "first.json")
    second.save(tmp_path / "second.json")
    assert first == second
    assert (tmp_path / "first.json").read_bytes() == (
        tmp_path / "second.json"
    ).read_bytes()


# SPEC: FILE-4
@pytest.mark.slow
def test_save_writes_the_same_bytes_in_every_process(tmp_path):
    # The domain's sets and the rules' set iterate in an order that
    # changes with the hash seed; the file must not.
    script = tmp_path / "save.py"
    script.write_text(SAVE_SCRIPT)
    files = set()
    for seed in range(6):
        path = tmp_path / f"model-{seed}.json"
        subprocess.run(
            [sys.executable, str(script), str(path)],
            capture_output=True,
            timeout=120,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            check=True,
        )
        files.add(path.read_bytes())
    assert len(files) == 1


def test_saved_file_lists_domain_values_sorted_with_an_indent(tmp_path):
    values = [str(i) for i in range(12)]
    dataset = Dataset(
        pd.DataFrame({"a": values, "y": values}),
        features=["a"],
        targets=["y"],
        max_categories=12,
    )
    path = tmp_path / "model.json"
    RuleSetModel(dataset, []).save(path)
    text = path.read_text(encoding="utf-8")
    assert json.loads(text)["domain"] == {"a": sorted(values), "y": sorted(values)}
    assert text.startswith('{\n  "schema_version"')
    assert text.endswith("}\n")


def test_saved_file_writes_accented_names_as_they_are(tmp_path):
    frame = pd.DataFrame({"âge": ["très", "peu"], "résultat": ["oui", "non"]})
    dataset = Dataset(frame, features=["âge"], targets=["résultat"])
    model = RuleSetModel(dataset, [Rule.from_string("résultat(oui) :- âge(très).")])
    path = tmp_path / "model.json"
    model.save(path)
    text = path.read_text(encoding="utf-8")
    assert '"display": "résultat(oui) :- âge(très)."' in text
    assert RuleSetModel.from_file(path) == model


def test_repr_says_one_rule_in_the_singular(toy_dataset, toy_rules):
    model = RuleSetModel(toy_dataset, toy_rules[:1])
    assert repr(model).startswith("<RuleSetModel: 1 rule, ")


def test_models_over_the_same_variables_in_other_roles_are_unequal():
    frame = pd.DataFrame({"a": ["0", "1"], "b": ["0", "1"], "y": ["0", "1"]})
    ab = RuleSetModel(Dataset(frame, features=["a", "b"], targets=["y"]), [])
    ba = RuleSetModel(Dataset(frame, features=["b", "a"], targets=["y"]), [])
    b_as_target = RuleSetModel(Dataset(frame, features=["a"], targets=["b", "y"]), [])
    y_first = RuleSetModel(Dataset(frame, features=["a"], targets=["y", "b"]), [])
    assert ab != ba
    assert ab != b_as_target
    assert b_as_target != y_first


def test_fallback_applies_only_to_a_target_no_rule_matches():
    frame = pd.DataFrame(
        {
            "a": ["0", "1", "0", "1"],
            "b": ["0", "0", "1", "1"],
            "t1": ["0", "1", "0", "1"],
            "t2": ["0", "0", "1", "1"],
        }
    )
    dataset = Dataset(frame, features=["a", "b"], targets=["t1", "t2"])
    rules = [
        Rule.from_string("t1(1) :- a(1)."),  # matches outright
        Rule.from_string("t1(0) :- b(1)."),  # could match, but t1 is matched
        Rule.from_string("t2(1) :- b(1)."),  # t2's only possible match
    ]
    model = RuleSetModel(dataset, rules)
    assert model.predict_candidates({"a": "1", "b": None}) == {
        "t1": ["1"],
        "t2": ["1"],
    }


@pytest.mark.parametrize(
    "state", [None, [{"a": "0", "b": "0", "c": "0"}], "a"], ids=["none", "list", "str"]
)
def test_a_single_state_that_is_not_a_mapping_is_a_type_error(toy_model, state):
    with pytest.raises(TypeError, match="a state is a mapping"):
        toy_model.predict_candidates(state)


def test_a_dataframe_given_as_one_state_is_named(toy_model):
    frame = pd.DataFrame([{"a": "0", "b": "0", "c": "0"}])
    with pytest.raises(TypeError, match="got a DataFrame"):
        toy_model.predict_candidates(frame)


def test_a_date_column_in_predict_input_says_before_predicting(toy_model):
    frame = pd.DataFrame({"a": pd.to_datetime(["2020-01-01"]), "b": ["0"], "c": ["0"]})
    with pytest.raises(ValueError, match="before predicting"):
        toy_model.predict(frame)


def test_integer_column_labels_are_named_as_not_str():
    frame = pd.DataFrame([["0", "0"], ["1", "1"]])
    named = frame.copy()
    named.columns = named.columns.astype(str)
    model = RuleSetModel(Dataset(named, features=["0"], targets=["1"]), [])
    with pytest.raises(ValueError, match=r"column label\(s\) 0 \(int\) name a feature"):
        model.predict(frame[[0]])


def test_a_domain_with_a_key_of_another_type_names_it(toy_model):
    with pytest.raises(ValueError, match="unexpected variable\\(s\\): 1"):
        toy_model.domain = {**toy_model.domain, 1: {"x"}}


def test_a_domain_that_is_not_a_mapping_is_a_type_error(toy_model):
    with pytest.raises(TypeError, match="domain must be a mapping"):
        toy_model.domain = [["0", "1"]]


def test_from_file_refuses_a_file_that_is_not_an_object(tmp_path):
    path = tmp_path / "model.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a model file"):
        RuleSetModel.from_file(path)


def test_one_rule_as_text_is_refused_as_such(toy_dataset):
    # Read as a collection, the text was refused one character in.
    with pytest.raises(
        TypeError, match=r"not a str; read a rule with Rule\.from_string"
    ):
        RuleSetModel(toy_dataset, "y(1) :- .")


@pytest.mark.parametrize(
    "random_state", [np.random.RandomState(0), np.random.default_rng(0)]
)
def test_a_numpy_random_state_is_refused_with_what_to_pass(toy_model, random_state):
    frame = pd.DataFrame({"a": ["0"], "b": ["0"], "c": ["0"]})
    with pytest.raises(TypeError, match="pass an integer seed instead"):
        toy_model.predict(frame, random_state=random_state)


def test_a_state_mapping_may_hold_other_keys_a_dataframe_may_not(toy_model):
    state = {"a": "0", "b": "0", "c": "0"}
    assert toy_model.predict_proba({**state, "y": "1", "note": "x"}) == (
        toy_model.predict_proba(state)
    )
    with pytest.raises(ValueError, match="unexpected column"):
        toy_model.predict_proba(pd.DataFrame([{**state, "note": "x"}]))


def test_an_out_of_domain_value_says_how_to_allow_it(toy_model):
    frame = pd.DataFrame({"a": ["7"], "b": ["0"], "c": ["0"]})
    with pytest.raises(ValueError, match=r"model\.domain = full_dataset\.domains\(\)"):
        toy_model.predict(frame)
    with pytest.raises(ValueError, match=r"model\.domain = full_dataset\.domains\(\)"):
        toy_model.predict_proba({"a": "7", "b": "0", "c": "0"})
