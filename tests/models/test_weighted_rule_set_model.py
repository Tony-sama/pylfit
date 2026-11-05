"""Tests for ``WeightedRuleSetModel``, a ``RuleSetModel`` with scores.

Each rule carries per-rule support counts, turning matches into
likelihoods.

Domain/size/mutator-shape tests shared, unchanged, with the base class
live in ``test_rule_set_model.py`` — this file focuses on what
``WeightedRuleSetModel`` actually adds: scoring at construction,
``predict_likelihood``/evidence-weighted ``predict_proba``, and the
score-aware persistence/equality/string-representation overrides.
``predict``'s own input validation (missing/extra columns,
out-of-domain values, etc.) is exercised once, against the base class,
in ``test_rule_set_model.py`` — ``WeightedRuleSetModel`` inherits
``predict`` unchanged, only overriding ``predict_proba``.

Two kinds of setup are used:
- ``toy_dataset`` + the four rules ``BruteForce`` is independently
  known to learn from it (see ``test_learners.py``), for tests that
  want realistic, dataset-derived scores.
- Hand-built rules with counts set directly (``_set_score``), for tests
  that need exact, contrived numbers — most notably the vote's, which
  need positive/negative counts no small dataset is sure to produce.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from pylfit import GULA, Dataset, Rule, RuleSetModel, WeightedRuleSetModel


def _set_score(
    model: WeightedRuleSetModel, rule: Rule, positives: int, negatives: int
) -> None:
    """Gives a rule of ``model`` counts no dataset need produce.

    The public API scores rules only against a dataset; tests of the
    scoring measures need exact numbers, so they write the private
    store directly.
    """
    assert rule in model.rules
    model._rule_scores[rule] = (positives, negatives)


def _hand_scored(dataset: Dataset, scores: dict) -> WeightedRuleSetModel:
    """A model of exactly ``scores``'s rules, with the given counts."""
    model = WeightedRuleSetModel(dataset, scores)
    for rule, (positives, negatives) in scores.items():
        _set_score(model, rule, positives, negatives)
    return model


def _edited(
    model: WeightedRuleSetModel, tmp_path: Path, change: Callable[[dict], object]
) -> Path:
    """Saves ``model``, applies ``change`` to the JSON, writes it back.

    Returns the file's path, for ``from_file`` to read.
    """
    path = tmp_path / "model.json"
    model.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def toy_model(toy_dataset, toy_rules):
    return WeightedRuleSetModel(toy_dataset, toy_rules)


# --- Construction ---


def test_rules_are_scored_immediately(toy_model, toy_rules):
    for rule in toy_rules:
        assert toy_model.get_score(rule) is not None


# SPEC: MODEL-5
def test_construction_scores_match_hand_derived_values(toy_model):
    scores = {rule.to_string(): toy_model.get_score(rule) for rule in toy_model.rules}
    assert scores["y(1) :- b(1)."] == (4, 0)
    assert scores["y(1) :- a(0), c(1)."] == (2, 0)
    assert scores["y(0) :- b(0), c(0)."] == (2, 0)
    assert scores["y(0) :- a(1), b(0)."] == (2, 0)


# --- predict_likelihood, predict_proba and predict on toy_dataset ---


def test_predict_likelihood_single_matching_rule(toy_model):
    # a=0, c=1 matches only "y(1) :- a(0), c(1)." (2, 0) among the four
    # rules: b=0 rules out b(1), c=1 rules out b(0), c(0), and a=0
    # rules out a(1), b(0).
    # Its vote on (2, 0) is 2 * 2 / 2 = 2.0, its positives.
    likelihoods = toy_model.predict_likelihood({"a": "0", "b": "0", "c": "1"})
    assert likelihoods["y"] == {"1": pytest.approx(2.0)}
    # Sole candidate, so it still normalizes to a certain prediction.
    assert toy_model.predict_proba({"a": "0", "b": "0", "c": "1"})["y"] == {
        "0": 0.0,
        "1": 1.0,
    }


def test_no_evidence_is_uniform_over_the_domain_but_likelihood_stays_empty():
    # predict_likelihood is the raw evidence view: no scored matching
    # rule really is no evidence, so it reports nothing. predict_proba
    # applies the same open-world fallback as RuleSetModel, so both
    # model kinds agree that "no evidence" means "anything is possible".
    df = pd.DataFrame({"a": [0, 1], "y": [0, 1]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    model = WeightedRuleSetModel(dataset, set())
    assert model.predict_likelihood({"a": "0"}) == {"y": {}}
    assert model.predict_proba({"a": "0"}) == {"y": {"0": 0.5, "1": 0.5}}

    dataframe = pd.DataFrame({"a": ["0"]})
    drawn = {model.predict(dataframe, random_state=seed)[0]["y"] for seed in range(20)}
    assert drawn == {"0", "1"}


def test_predict_returns_one_prediction_per_row_in_order(toy_model):
    dataframe = pd.DataFrame(
        [
            {"a": "0", "b": "0", "c": "1"},  # matches "y(1) :- a(0), c(1)."
            {"a": "1", "b": "0", "c": "0"},  # matches "y(0) :- a(1), b(0)."
        ]
    )
    assert toy_model.predict(dataframe) == [{"y": "1"}, {"y": "0"}]


def test_predict_missing_feature_column_raises(toy_model):
    # One check that predict's input validation, inherited from
    # RuleSetModel, still applies through this subclass; the full set
    # is in test_rule_set_model.py.
    dataframe = pd.DataFrame([{"a": "0", "b": "0"}])  # missing c
    with pytest.raises(ValueError, match="missing required feature"):
        toy_model.predict(dataframe)


# --- The vote: support times precision ---


def test_more_evidence_outranks_less_when_neither_has_negatives(toy_dataset):
    # With no negatives the vote is the raw support: twice the
    # evidence, twice the score.
    rule_1 = Rule(head=("y", "1"), body={})
    rule_0 = Rule(head=("y", "0"), body={})
    model = _hand_scored(toy_dataset, {rule_1: (10, 0), rule_0: (5, 0)})

    likelihoods = model.predict_likelihood({"a": "0", "b": "0", "c": "0"})
    assert likelihoods["y"]["1"] == pytest.approx(10.0)
    assert likelihoods["y"]["0"] == pytest.approx(5.0)


def test_negatives_lower_a_values_likelihood(toy_dataset):
    # Two values with identical positive support are separated by their
    # negatives.
    rule_1 = Rule(head=("y", "1"), body={})
    rule_0 = Rule(head=("y", "0"), body={})
    model = _hand_scored(toy_dataset, {rule_1: (10, 0), rule_0: (10, 90)})

    likelihoods = model.predict_likelihood({"a": "0", "b": "0", "c": "0"})
    assert likelihoods["y"]["1"] == pytest.approx(10.0)
    assert likelihoods["y"]["0"] == pytest.approx(10 * 10 / 100)

    # And the decision follows the evidence rather than a coin flip.
    dataframe = pd.DataFrame([{"a": "0", "b": "0", "c": "0"}])
    drawn = {model.predict(dataframe, random_state=seed)[0]["y"] for seed in range(20)}
    assert drawn == {"1"}


# SPEC: MODEL-7
def test_predict_proba_renormalizes_likelihoods_that_dont_sum_to_one(toy_dataset):
    rule_1 = Rule(head=("y", "1"), body={})
    rule_0 = Rule(head=("y", "0"), body={})
    model = _hand_scored(toy_dataset, {rule_1: (10, 2), rule_0: (5, 3)})
    # Votes 100/12 and 25/8, summing to well over 1.
    likelihoods = model.predict_likelihood({"a": "0", "b": "0", "c": "0"})
    assert likelihoods["y"]["1"] == pytest.approx(100 / 12)
    assert likelihoods["y"]["0"] == pytest.approx(25 / 8)
    assert sum(likelihoods["y"].values()) > 1

    probabilities = model.predict_proba({"a": "0", "b": "0", "c": "0"})
    total = 100 / 12 + 25 / 8
    assert probabilities["y"]["1"] == pytest.approx((100 / 12) / total)
    assert probabilities["y"]["0"] == pytest.approx((25 / 8) / total)
    assert sum(probabilities["y"].values()) == pytest.approx(1.0)


def test_negatives_weigh_against_support(toy_dataset):
    # 10 positives with 90 negatives against 5 clean ones: each rule
    # votes support times precision, 10 * 0.1 = 1 against 5, and the
    # impure value loses.
    rule_1 = Rule(head=("y", "1"), body={})
    rule_0 = Rule(head=("y", "0"), body={})
    model = _hand_scored(toy_dataset, {rule_1: (10, 90), rule_0: (5, 0)})

    state = {"a": "0", "b": "0", "c": "0"}
    votes = model.predict_likelihood(state)
    assert votes["y"]["1"] == pytest.approx(10 * 10 / 100)  # 1.0
    assert votes["y"]["0"] == pytest.approx(5 * 5 / 5)  # 5.0
    assert model.predict(pd.DataFrame([state]), random_state=0) == [{"y": "0"}]


def test_a_large_slightly_impure_value_outranks_a_tiny_clean_one(toy_dataset):
    # A pooled purity measure, the Laplace estimate, would pick the
    # small clean value: 69/77 = 0.896 < 9/10 = 0.900. The vote,
    # 68 * 68 / 75 against 8, keeps the big one.
    big = Rule(head=("y", "1"), body={})
    small = Rule(head=("y", "0"), body={})
    model = _hand_scored(toy_dataset, {big: (68, 7), small: (8, 0)})
    df = pd.DataFrame([{"a": "0", "b": "0", "c": "0"}])
    assert model.predict(df, random_state=0) == [{"y": "1"}]


def test_rules_without_positives_are_no_evidence(toy_dataset):
    rule_1 = Rule(head=("y", "1"), body={})
    rule_0 = Rule(head=("y", "0"), body={})
    model = _hand_scored(toy_dataset, {rule_1: (0, 4), rule_0: (0, 6)})

    assert model.predict_likelihood({"a": "0", "b": "0", "c": "0"}) == {"y": {}}

    probabilities = model.predict_proba({"a": "0", "b": "0", "c": "0"})
    assert probabilities["y"]["1"] == pytest.approx(0.5)
    assert probabilities["y"]["0"] == pytest.approx(0.5)


# SPEC: MODEL-6
def test_a_rule_wrong_on_every_row_gets_no_share_of_its_own():
    df = pd.DataFrame({"a": ["0", "0", "1", "1"], "y": ["0", "0", "1", "1"]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    model = WeightedRuleSetModel(dataset, [Rule.from_string("y(0) :- a(1).")])
    assert model.get_score(next(iter(model.rules))) == (0, 2)
    assert model.predict_proba({"a": "1"}) == {"y": {"0": 0.5, "1": 0.5}}


def test_a_rule_without_positives_is_not_a_fallback_candidate():
    df = pd.DataFrame({"a": ["0", "1"], "b": ["0", "1"], "y": ["0", "1"]})
    dataset = Dataset(df, features=["a", "b"], targets=["y"])
    rule = Rule.from_string("y(0) :- a(1), b(1).")
    model = WeightedRuleSetModel(dataset, [rule])
    assert model.get_score(rule) == (0, 1)
    assert model.predict_likelihood({"a": None, "b": "1"}) == {"y": {}}


def test_predict_breaks_an_equal_score_tie_randomly(toy_dataset):
    rule_1 = Rule(head=("y", "1"), body={})
    rule_0 = Rule(head=("y", "0"), body={})
    model = _hand_scored(toy_dataset, {rule_1: (10, 0), rule_0: (10, 0)})

    dataframe = pd.DataFrame([{"a": "0", "b": "0", "c": "0"}])
    drawn = {model.predict(dataframe, random_state=seed)[0]["y"] for seed in range(20)}
    assert drawn == {"0", "1"}


def test_predict_draws_only_among_the_top_scoring_values():
    # Only the top-scoring values are drawn, never one below them,
    # however many seeds are tried.
    df = pd.DataFrame({"a": ["0", "1", "0"], "y": ["0", "1", "2"]})
    dataset = Dataset(df, features=["a"], targets=["y"], assume_nondeterministic=True)
    top_a = Rule(head=("y", "0"), body={})
    top_b = Rule(head=("y", "1"), body={})
    also_ran = Rule(head=("y", "2"), body={})
    model = _hand_scored(dataset, {top_a: (10, 0), top_b: (10, 0), also_ran: (1, 0)})

    dataframe = pd.DataFrame({"a": ["1"]})
    drawn = {model.predict(dataframe, random_state=seed)[0]["y"] for seed in range(30)}
    assert drawn == {"0", "1"}


# SPEC: MODEL-6
def test_zero_zero_rules_dont_contribute(toy_dataset):
    matching_rule = Rule(head=("y", "1"), body={})
    model = _hand_scored(toy_dataset, {matching_rule: (0, 0)})
    assert model.predict_likelihood({"a": "0", "b": "0", "c": "0"}) == {"y": {}}


def test_multiple_matching_rules_for_the_same_value_sum_their_support(toy_dataset):
    # Two different rules, same head, both matching the same state —
    # their (positives, negatives) should be summed, not just one used.
    rule_a = Rule(head=("y", "1"), body={"a": "0"})
    rule_b = Rule(head=("y", "1"), body={"b": "0"})
    model = _hand_scored(toy_dataset, {rule_a: (3, 0), rule_b: (4, 0)})

    state = {"a": "0", "b": "0", "c": "0"}  # matches both rules
    # Each rule votes, then the votes are summed: 3*3/3 + 4*4/4 = 7,
    # which with no negatives is just the pooled support.
    likelihoods = model.predict_likelihood(state)
    assert likelihoods["y"]["1"] == pytest.approx(7.0)


# --- get_score ---


def test_get_score_is_none_for_an_unscored_rule(toy_dataset):
    model = WeightedRuleSetModel(toy_dataset, [])
    unscored = Rule(head=("y", "1"), body={})
    assert model.get_score(unscored) is None


# --- Unknown values: exact counts, fallback weights ---


@pytest.fixture
def unknowns_dataset():
    """``a`` is ``"0"`` in 3 of its 4 known rows, unknown in a fifth."""
    df = pd.DataFrame(
        {
            "a": ["0", "0", "0", "1", None],
            "b": ["0", "1", "0", "1", "1"],
            "y": ["0", "1", "0", "1", "1"],
        }
    )
    return Dataset(df, features=["a", "b"], targets=["y"])


def test_value_frequencies_are_shares_of_the_known_values(unknowns_dataset):
    model = WeightedRuleSetModel(unknowns_dataset, [])
    assert model.value_frequencies == {
        "a": {"0": 0.75, "1": 0.25},
        "b": {"0": 0.4, "1": 0.6},
    }


def test_value_frequencies_cannot_be_assigned(toy_model):
    with pytest.raises(AttributeError, match="has no setter"):
        toy_model.value_frequencies = None


def test_value_frequencies_is_returned_as_a_copy(toy_model):
    # Copied before the edit, inner mappings too: an aliased inner
    # dict would change `before` along with the model and still
    # compare equal.
    before = {
        feature: dict(shares) for feature, shares in toy_model.value_frequencies.items()
    }
    first = next(iter(before))
    toy_model.value_frequencies[first]["new"] = 1
    assert toy_model.value_frequencies == before


def test_counts_are_exact_a_row_hiding_a_tested_value_counts_for_neither(
    unknowns_dataset,
):
    # The a=None, y=1 row is not known to match a(1) or a(0).
    a1, b1 = Rule.from_string("y(1) :- a(1)."), Rule.from_string("y(1) :- b(1).")
    model = WeightedRuleSetModel(unknowns_dataset, [a1, b1])
    assert model.get_score(a1) == (1, 0)
    assert model.get_score(b1) == (3, 0)


def test_fallback_weighs_each_rule_by_its_probability_of_matching(unknowns_dataset):
    rules = [Rule.from_string("y(1) :- a(1)."), Rule.from_string("y(0) :- a(0), b(0).")]
    model = WeightedRuleSetModel(unknowns_dataset, rules)
    # a unknown, b=0: no strict match. y(1) :- a(1). (+1/-0) weighs
    # 0.25, y(0) :- a(0), b(0). (+2/-0) weighs 0.75; each votes its
    # weighted positives.
    likelihood = model.predict_likelihood({"a": None, "b": "0"})
    assert likelihood["y"]["1"] == pytest.approx(0.25 * 1)
    assert likelihood["y"]["0"] == pytest.approx(0.75 * 2)


def test_strict_leaves_an_unmatched_target_empty(unknowns_dataset):
    rules = [Rule.from_string("y(1) :- a(1).")]
    model = WeightedRuleSetModel(unknowns_dataset, rules)
    state = {"a": None, "b": "0"}
    assert model.predict_likelihood(state, unknowns="strict") == {"y": {}}
    assert model.predict_proba(state, unknowns="strict") == {"y": {"0": 0.5, "1": 0.5}}


def test_fallback_applies_only_to_targets_without_a_strict_match():
    df = pd.DataFrame(
        {"a": ["0", "1"], "b": ["0", "1"], "y": ["0", "1"], "z": ["0", "1"]}
    )
    dataset = Dataset(df, features=["a", "b"], targets=["y", "z"])
    rules = [
        Rule.from_string("y(1) :- b(1)."),
        Rule.from_string("y(0) :- a(0)."),
        Rule.from_string("z(0) :- a(0)."),
    ]
    model = WeightedRuleSetModel(dataset, rules)
    likelihood = model.predict_likelihood({"a": None, "b": "1"})
    # y matches strictly: y(0) :- a(0). is not pooled in. z doesn't.
    assert set(likelihood["y"]) == {"1"}
    assert likelihood["z"]["0"] == pytest.approx(0.5 * 1)


def test_predict_on_a_dataframe_with_nan_uses_the_fallback(unknowns_dataset):
    rules = [Rule.from_string("y(1) :- a(1)."), Rule.from_string("y(0) :- a(0), b(0).")]
    model = WeightedRuleSetModel(unknowns_dataset, rules)
    rows = pd.DataFrame({"a": [float("nan")], "b": ["0"]})
    assert model.predict(rows) == [{"y": "0"}]
    assert model.predict_proba(rows)[0] == model.predict_proba({"a": None, "b": "0"})


def test_value_frequencies_survive_save_and_count_in_equality(
    unknowns_dataset, tmp_path
):
    rules = [Rule.from_string("y(1) :- a(1).")]
    model = WeightedRuleSetModel(unknowns_dataset, rules)
    path = tmp_path / "model.json"
    model.save(str(path))
    assert WeightedRuleSetModel.from_file(str(path)) == model
    other = WeightedRuleSetModel(unknowns_dataset, rules)
    other._value_frequencies["a"]["0"] = 0.5
    assert model.value_frequencies["a"]["0"] == 0.75
    assert other != model


def test_a_file_without_value_frequencies_weighs_values_equally(
    unknowns_dataset, tmp_path
):
    rules = [Rule.from_string("y(1) :- a(1).")]
    model = WeightedRuleSetModel(unknowns_dataset, rules)
    path = _edited(model, tmp_path, lambda p: p.pop("value_frequencies"))
    loaded = WeightedRuleSetModel.from_file(path)
    assert loaded.value_frequencies == {}
    # a's domain is {0, 1}: a(1) weighs 1/2, and its (1, 0) votes 0.5.
    likelihood = loaded.predict_likelihood({"a": None, "b": "0"})
    assert likelihood["y"]["1"] == pytest.approx(0.5)


# --- Persistence: save / from_file ---


def test_save_from_file_round_trip(toy_model, tmp_path):
    path = tmp_path / "model.json"
    toy_model.save(str(path))
    loaded = WeightedRuleSetModel.from_file(str(path))
    assert loaded == toy_model


def test_save_round_trip_holds_after_rescoring(toy_model, tmp_path):
    rule = next(iter(toy_model.rules))
    _set_score(toy_model, rule, 40, 2)
    toy_model.save(tmp_path / "model.json")
    assert WeightedRuleSetModel.from_file(tmp_path / "model.json") == toy_model


def test_from_file_refuses_a_rule_without_a_score(toy_model, tmp_path):
    path = _edited(toy_model, tmp_path, lambda p: p["rules"][0].update(score=None))
    with pytest.raises(ValueError, match="has no score"):
        WeightedRuleSetModel.from_file(path)


@pytest.mark.parametrize(
    ("counts", "error", "message"),
    [
        ((-1, 0), ValueError, "positives must be >= 0"),
        ((1, -2), ValueError, "negatives must be >= 0"),
        ((2.5, 0), TypeError, "positives must be an int"),
        (("3", 0), TypeError, "positives must be an int"),
        ((True, 0), TypeError, "positives must be an int"),
    ],
)
def test_from_file_refuses_an_invalid_saved_score(
    toy_model, tmp_path, counts, error, message
):
    path = _edited(
        toy_model, tmp_path, lambda p: p["rules"][0].update(score=list(counts))
    )
    with pytest.raises(error, match=message):
        WeightedRuleSetModel.from_file(path)


def test_saved_file_is_readable_json_with_display_strings(toy_model, tmp_path):
    path = tmp_path / "model.json"
    toy_model.save(str(path))
    payload = json.loads(path.read_text())

    assert payload["schema_version"] == 2
    assert payload["model_class"] == "WeightedRuleSetModel"
    assert isinstance(payload["pylfit_version"], str)
    assert payload["features"] == toy_model.features
    assert payload["targets"] == toy_model.targets
    assert payload["value_frequencies"] == toy_model.value_frequencies

    rule_entry = next(
        entry
        for entry in payload["rules"]
        if entry["head"] == ["y", "1"] and entry["body"] == {"b": "1"}
    )
    assert rule_entry["display"] == "y(1) :- b(1)."
    assert rule_entry["score"] == [4, 0]


def test_from_file_rejects_unknown_schema_version(toy_model, tmp_path):
    path = _edited(toy_model, tmp_path, lambda p: p.update(schema_version=999))
    with pytest.raises(ValueError, match="schema_version"):
        WeightedRuleSetModel.from_file(path)


def test_from_file_rejects_a_plain_rule_set_model_file(
    toy_dataset, toy_rules, tmp_path
):
    path = tmp_path / "model.json"
    RuleSetModel(toy_dataset, toy_rules).save(str(path))

    with pytest.raises(ValueError, match="RuleSetModel"):
        WeightedRuleSetModel.from_file(str(path))


def test_from_file_rejects_domain_missing_a_variable(toy_model, tmp_path):
    path = _edited(toy_model, tmp_path, lambda p: p["domain"].pop("c"))
    with pytest.raises(ValueError, match="missing variable"):
        WeightedRuleSetModel.from_file(path)


def test_save_overwrites_existing_file(toy_dataset, toy_model, tmp_path):
    path = tmp_path / "model.json"
    toy_model.save(str(path))

    smaller = WeightedRuleSetModel(
        toy_dataset, [Rule(head=("y", "1"), body={"b": "1"})]
    )
    smaller.save(str(path))

    loaded = WeightedRuleSetModel.from_file(str(path))
    assert loaded == smaller
    assert loaded != toy_model


# --- Equality ---


def test_equal_models_from_the_same_dataset_and_rules(toy_dataset, toy_rules):
    model_a = WeightedRuleSetModel(toy_dataset, toy_rules)
    model_b = WeightedRuleSetModel(
        toy_dataset, [Rule(rule.head, rule.body) for rule in toy_rules]
    )
    assert model_a == model_b


def test_models_with_different_rules_are_unequal(toy_dataset, toy_rules):
    model_a = WeightedRuleSetModel(toy_dataset, toy_rules)
    model_b = WeightedRuleSetModel(toy_dataset, toy_rules[:-1])
    assert model_a != model_b


def test_models_with_same_rules_but_different_scores_are_unequal(
    toy_dataset, toy_rules
):
    # Equality covers the scores: rules alone would let two models
    # that predict differently compare equal.
    model_a = WeightedRuleSetModel(toy_dataset, toy_rules)
    model_b = WeightedRuleSetModel(toy_dataset, toy_rules)
    one_rule = next(iter(model_b.rules))
    _set_score(model_b, one_rule, 999, 999)
    assert model_a != model_b


def test_models_with_different_domain_are_unequal(toy_dataset, toy_rules):
    model_a = WeightedRuleSetModel(toy_dataset, toy_rules)
    model_b = WeightedRuleSetModel(toy_dataset, toy_rules)
    model_b.domain = {
        variable: [*values, "9"] for variable, values in model_b.domain.items()
    }
    assert model_a != model_b


def test_not_equal_to_a_non_model_object(toy_model):
    assert (toy_model == "not a model") is False
    assert (toy_model != "not a model") is True


def test_never_equal_to_a_plain_rule_set_model_with_identical_rules(
    toy_dataset, toy_rules
):
    # A plain model carries no scores, so neither side equals the other.
    weighted = WeightedRuleSetModel(toy_dataset, toy_rules)
    plain = RuleSetModel(toy_dataset, toy_rules)
    assert weighted != plain
    assert plain != weighted


# --- WeightedRuleSetModel stays unhashable ---


def test_model_is_unhashable(toy_model):
    with pytest.raises(TypeError, match="unhashable type"):
        hash(toy_model)


# --- String representation ---


def test_str_sorts_rules_and_includes_scores(toy_model):
    text = str(toy_model)
    lines = text.splitlines()
    rule_lines = [line for line in lines if line.startswith("  ")]
    assert rule_lines == sorted(rule_lines)
    for line in rule_lines:
        assert re.search(r"  \(\+\d+/-\d+\)$", line)


def test_repr_is_a_one_line_summary(toy_model):
    assert repr(toy_model) == (
        f"<WeightedRuleSetModel: {len(toy_model.rules)} rules, "
        f"features {toy_model.features}, targets {toy_model.targets}>"
    )


# --- DataFrame input to predict_proba / predict_likelihood ---


# SPEC: MODEL-10
def test_predict_proba_on_a_dataframe_is_one_result_per_row(toy_dataset):
    model = WeightedRuleSetModel(toy_dataset, GULA.fit(toy_dataset))
    frame = pd.DataFrame({"a": ["0", "1"], "b": ["1", "0"], "c": ["0", "0"]})
    expected = [model.predict_proba(row) for row in frame.to_dict("records")]
    assert model.predict_proba(frame) == expected


def test_predict_likelihood_on_a_dataframe_converts_values_to_str(toy_dataset):
    model = WeightedRuleSetModel(toy_dataset, GULA.fit(toy_dataset))
    as_ints = pd.DataFrame({"a": [0, 1], "b": [1, 0], "c": [0, 0]})
    as_strs = as_ints.astype(str)
    assert model.predict_likelihood(as_ints) == model.predict_likelihood(as_strs)


def test_predict_proba_on_a_dataframe_rejects_what_predict_rejects(toy_dataset):
    model = WeightedRuleSetModel(toy_dataset, GULA.fit(toy_dataset))
    with pytest.raises(ValueError, match="target column"):
        model.predict_proba(
            pd.DataFrame({"a": ["0"], "b": ["0"], "c": ["0"], "y": ["0"]})
        )


@pytest.mark.parametrize("method", ["predict_likelihood", "predict_proba"])
def test_single_state_is_checked_as_a_dataframe_row_is(toy_dataset, method):
    model = WeightedRuleSetModel(toy_dataset, GULA.fit(toy_dataset))
    predict = getattr(model, method)
    with pytest.raises(ValueError, match=r"out-of-domain value\(s\) for feature 'a'"):
        predict({"a": "7", "b": "1", "c": "0"})
    with pytest.raises(KeyError, match="'c': None"):
        predict({"a": "0", "b": "1"})
    with pytest.raises(TypeError, match="not a str"):
        predict({"a": 0, "b": "1", "c": "0"})


# --- Output order ---


def _eight_way_tie():
    values = [str(value) for value in range(8)]
    df = pd.DataFrame({"a": ["0"] * 8, "y": values})
    dataset = Dataset(
        df,
        features=["a"],
        targets=["y"],
        assume_nondeterministic=True,
        max_categories=8,
    )
    return dataset, GULA.fit(dataset), values


# SPEC: MODEL-4
@pytest.mark.parametrize("model_class", [RuleSetModel, WeightedRuleSetModel])
def test_predict_proba_lists_values_in_sorted_order(model_class):
    # Built from sets, whose order varies with PYTHONHASHSEED: eight
    # tied values would come out sorted by chance once in 40320 runs.
    dataset, rules, values = _eight_way_tie()
    model = model_class(dataset, rules)
    assert list(model.predict_proba({"a": "0"})["y"]) == values


def test_predict_likelihood_lists_values_in_sorted_order():
    dataset, rules, values = _eight_way_tie()
    model = WeightedRuleSetModel(dataset, rules)
    assert list(model.predict_likelihood({"a": "0"})["y"]) == values


# --- Regression tests: one per fix ---

VOTE_TIE_SCRIPT = """
import warnings
import pandas as pd
from pylfit import Dataset, Rule, WeightedRuleSetModel
features = ["f1", "f2", "f3", "f4"]
rows = []
def add(feature, n, y):
    rows.extend([{**{f: "1" if f == feature else "0" for f in features}, "y": y}] * n)
for feature, p in (("f1", 1), ("f2", 2), ("f3", 3)):
    add(feature, p, "A")
    add(feature, 9 * p, "C")
add("f4", 6, "B")
add("f4", 54, "C")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    dataset = Dataset(pd.DataFrame(rows), features=features, targets=["y"])
texts = ("y(A) :- f1(1).", "y(A) :- f2(1).", "y(A) :- f3(1).", "y(B) :- f4(1).")
model = WeightedRuleSetModel(dataset, [Rule.from_string(text) for text in texts])
state = {feature: "1" for feature in features}
likelihood = model.predict_likelihood(state)["y"]
print(repr(likelihood["A"]), repr(likelihood["B"]))
"""


@pytest.mark.slow
def test_vote_scores_are_the_same_in_every_process(outputs_per_hash_seed):
    # A's votes 0.1 + 0.2 + 0.3 tie B's 0.6 only when summed in one
    # order; the order of a set of rules changes with the hash seed.
    outputs = outputs_per_hash_seed(VOTE_TIE_SCRIPT)
    assert len(outputs) == 1
    a, b = outputs.pop().split()
    assert a == b


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        (lambda p: p["rules"][0].update(score=[1]), TypeError, "positives, negatives"),
        (lambda p: p["rules"][0].update(score=7), TypeError, "positives, negatives"),
        (
            lambda p: p["rules"].append({**p["rules"][0], "score": [99, 0]}),
            ValueError,
            "listed twice",
        ),
        (lambda p: p.update(value_frequencies=[]), TypeError, "value_frequencies"),
        (
            lambda p: p.update(value_frequencies={"zz": {"0": 0.5}}),
            ValueError,
            "not a feature",
        ),
    ],
    ids=[
        "short-score",
        "score-not-a-list",
        "rule-twice",
        "frequencies-not-a-mapping",
        "unknown-feature",
    ],
)
def test_from_file_refuses_bad_scores_and_frequencies(
    toy_model, tmp_path, change, error, message
):
    path = _edited(toy_model, tmp_path, change)
    with pytest.raises(error, match=message):
        WeightedRuleSetModel.from_file(path)


def test_a_fallback_rule_adds_less_the_less_likely_it_matches():
    # y(1) :- a(1). matches {"a": None} with a(1)'s share of the rows;
    # as that share falls toward 0, so must y=1's likelihood.
    rule = Rule.from_string("y(1) :- a(1).")
    likelihoods = []
    for ones in (50, 10, 1):
        df = pd.DataFrame(
            {"a": ["1"] * ones + ["0"] * (1000 - ones), "y": ["1"] * 1000}
        )
        model = WeightedRuleSetModel(Dataset(df, ["a"], ["y"]), [rule])
        likelihoods.append(model.predict_likelihood({"a": None})["y"]["1"])
    assert likelihoods == sorted(likelihoods, reverse=True)
    # At a share of 1 in 1000: 1 positive, weighed 0.001.
    assert likelihoods[-1] == pytest.approx(0.001)


def test_a_model_with_a_narrowed_feature_domain_round_trips(tmp_path):
    dataset = Dataset(
        pd.DataFrame({"a": ["0", "1", "1"], "y": ["0", "1", "1"]}),
        features=["a"],
        targets=["y"],
    )
    model = WeightedRuleSetModel(dataset, [Rule.from_string("y(1) :- a(1).")])
    model.domain = {"a": ["1"], "y": ["0", "1"]}
    assert model.value_frequencies == {"a": {"1": pytest.approx(2 / 3)}}
    path = tmp_path / "model.json"
    model.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["value_frequencies"] == {
        "a": {"1": pytest.approx(2 / 3)}
    }
    assert WeightedRuleSetModel.from_file(path) == model


def test_from_file_ignores_a_share_outside_the_domain(toy_model, tmp_path):
    path = _edited(
        toy_model, tmp_path, lambda p: p["value_frequencies"]["a"].update({"9": 0.5})
    )
    assert WeightedRuleSetModel.from_file(path) == toy_model


def test_votes_tied_in_exact_arithmetic_tie_in_predict():
    # Ten rules vote 0.1 each for A, one rule votes 1.0 for B; a plain
    # sum of the ten is 0.9999999999999999 on Python 3.11.
    features = [f"f{j}" for j in range(10)] + ["g"]
    rows = []
    for i in range(10):
        state = {f"f{j}": "1" if j == i else "0" for j in range(10)} | {"g": "0"}
        rows += [{**state, "y": "A"}] + [{**state, "y": "B"}] * 9
    rows.append({**{f"f{j}": "0" for j in range(10)}, "g": "1", "y": "B"})
    dataset = Dataset(
        pd.DataFrame(rows),
        features=features,
        targets=["y"],
        assume_nondeterministic=True,
    )
    rules = [Rule.from_string(f"y(A) :- f{i}(1).") for i in range(10)]
    model = WeightedRuleSetModel(dataset, [*rules, Rule.from_string("y(B) :- g(1).")])
    state = dict.fromkeys(features, "1")
    likelihood = model.predict_likelihood(state)["y"]
    assert likelihood["A"] == likelihood["B"] == 1.0
    predicted = model.predict(pd.DataFrame([state] * 40), random_state=0)
    assert {row["y"] for row in predicted} == {"A", "B"}


def _ab_dataset(rows=None):
    """``a``, ``b`` and ``y`` over ``rows``, by default ``a or b``."""
    rows = rows or [("0", "0", "0"), ("0", "1", "1"), ("1", "0", "1"), ("1", "1", "1")]
    df = pd.DataFrame(rows, columns=["a", "b", "y"])
    return Dataset(df, features=["a", "b"], targets=["y"])


def test_a_rule_almost_sure_not_to_match_still_counts():
    # Weights of 1e-18 and 1e-20: each rule still votes its weighted
    # positive, and the ranking between them is kept.
    features = [f"f{i}" for i in range(10)]
    rows = [dict.fromkeys(features, "1") | {"y": "1"}]
    rows += [dict.fromkeys(features, "0") | {"y": "0"}]
    rows += [dict.fromkeys(features, "2") | {"y": "2"}] * 98
    dataset = Dataset(pd.DataFrame(rows), features=features, targets=["y"])
    model = WeightedRuleSetModel(
        dataset,
        [
            Rule(("y", "1"), dict.fromkeys(features[:9], "1")),
            Rule(("y", "0"), dict.fromkeys(features, "0")),
        ],
    )
    likelihood = model.predict_likelihood(dict.fromkeys(features))["y"]
    assert 0 < likelihood["0"] < likelihood["1"]


@pytest.mark.parametrize("rule", ["y(1) :- b(1).", 42])
def test_get_score_refuses_what_is_not_a_rule(toy_model, rule):
    with pytest.raises(TypeError, match="rule must be a Rule"):
        toy_model.get_score(rule)


def test_a_fallback_rule_votes_its_counts_weighed_once():
    # The rule (2, 0) matches with probability 1/2: its counts weigh
    # (1, 0) and it votes 1.0, with no second scaling.
    model = WeightedRuleSetModel(_ab_dataset(), [Rule.from_string("y(1) :- a(1).")])
    state = {"a": None, "b": "0"}
    assert model.predict_likelihood(state) == {"y": {"1": 1.0}}


def test_a_rule_no_row_backs_is_left_out_of_the_fallback():
    dataset = _ab_dataset([("0", "0", "0"), ("0", "1", "1"), ("1", "0", "1")])
    model = WeightedRuleSetModel(dataset, [Rule.from_string("y(0) :- a(1), b(1).")])
    assert model.get_score(next(iter(model.rules))) == (0, 0)
    assert model.predict_likelihood({"a": None, "b": "1"}) == {"y": {}}


def test_a_rule_the_known_values_rule_out_is_left_out_of_the_fallback():
    # b=0 is known and the rule needs b=1: its match weight is 0.
    model = WeightedRuleSetModel(_ab_dataset(), [Rule.from_string("y(1) :- b(1).")])
    assert model.predict_likelihood({"a": None, "b": "0"}) == {"y": {}}


FALLBACK_TIE_SCRIPT = """
import pandas as pd
from pylfit import Dataset, Rule, WeightedRuleSetModel
columns = ("f1", "f2", "f3", "g1", "g2", "g3")
frame = {column: ["0"] * 40 for column in columns}
for i, n in enumerate((4, 8, 14), start=1):
    for j in [*range(0, n, 2), *range(1, n, 2)]:
        frame[f"f{i}"][j] = frame[f"g{i}"][j] = "1"
df = pd.DataFrame(frame).assign(y=["X", "Y"] * 20)
dataset = Dataset(
    df, features=list(columns), targets=["y"], assume_nondeterministic=True
)
rules = [
    Rule.from_string(f"y({value}) :- {column}{i}(1).")
    for value, column in (("X", "f"), ("Y", "g"))
    for i in (1, 2, 3)
]
model = WeightedRuleSetModel(dataset, rules)
likelihood = model.predict_likelihood(dict.fromkeys(columns))["y"]
print(repr(likelihood["X"]), repr(likelihood["Y"]))
"""


@pytest.mark.slow
def test_fallback_scores_are_the_same_in_every_process(outputs_per_hash_seed):
    # X's and Y's rules have the same weights, which arrive in set
    # order; combined in another order they could differ in the last
    # bit.
    outputs = outputs_per_hash_seed(FALLBACK_TIE_SCRIPT)
    assert len(outputs) == 1
    x, y = outputs.pop().split()
    assert x == y


def test_a_target_with_an_empty_domain_gets_no_probabilities():
    df = pd.DataFrame({"a": ["0", "1"], "y": [None, None]})
    model = WeightedRuleSetModel(Dataset(df, features=["a"], targets=["y"]), [])
    assert model.predict_likelihood({"a": "0"}) == {"y": {}}
    assert model.predict_proba({"a": "0"}) == {"y": {}}


def test_vote_shares_are_uniform_over_the_domain_when_no_rule_has_positives():
    df = pd.DataFrame({"a": ["0", "1", "1", "0"], "y": ["A", "B", "B", "C"]})
    dataset = Dataset(df, features=["a"], targets=["y"], assume_nondeterministic=True)
    # Both rules are wrong on every row they match: (0, 2) each.
    rules = [Rule.from_string("y(A) :- a(1)."), Rule.from_string("y(C) :- a(1).")]
    model = WeightedRuleSetModel(dataset, rules)
    proba = model.predict_proba({"a": "1"})
    assert proba["y"] == pytest.approx({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})


@pytest.mark.parametrize("share", [-0.5, True, 7])
def test_from_file_refuses_a_share_that_is_not_a_number_in_0_1(
    toy_model, tmp_path, share
):
    path = _edited(
        toy_model, tmp_path, lambda p: p["value_frequencies"]["a"].update({"0": share})
    )
    with pytest.raises(ValueError, match="not a number in"):
        WeightedRuleSetModel.from_file(path)


def test_verbose_reports_to_stderr_only(toy_dataset, toy_rules, capsys):
    WeightedRuleSetModel(toy_dataset, toy_rules, verbose=True)
    output = capsys.readouterr()
    assert output.out == ""
    assert "[WeightedRuleSetModel]" in output.err
    assert "rules scored in" in output.err


def test_verbose_false_prints_nothing(toy_dataset, toy_rules, capsys):
    WeightedRuleSetModel(toy_dataset, toy_rules)
    WeightedRuleSetModel(toy_dataset, toy_rules, verbose=False)
    assert capsys.readouterr() == ("", "")


def test_verbose_that_is_not_a_bool_is_refused(toy_dataset):
    with pytest.raises(TypeError, match="verbose must be True or False"):
        WeightedRuleSetModel(toy_dataset, [], verbose="no")


# SPEC: FILE-5
@pytest.mark.parametrize(
    ("model_class", "name"),
    [
        (RuleSetModel, "rule_set_model_schema_2.json"),
        (WeightedRuleSetModel, "weighted_rule_set_model_schema_2.json"),
    ],
)
def test_a_file_saved_by_an_earlier_version_still_loads(model_class, name):
    # Stored files, never regenerated: a new schema has to keep reading
    # them, so add a file for it rather than replacing these.
    path = Path(__file__).resolve().parent.parent / "data" / "models" / name
    model = model_class.from_file(path)
    assert model.predict_proba({"a": "1", "b": None})["y"]["1"] == 1.0


def _two_values_model():
    df = pd.DataFrame({"a": ["0", "1"], "y": ["1", "0"]})
    return Dataset(df, features=["a"], targets=["y"])


def test_a_rule_with_no_positives_does_not_stop_the_vote():
    # y(0) sorts first and has no positive row: the next rule still votes.
    rules = [Rule.from_string("y(0) :- a(0)."), Rule.from_string("y(1) :- a(0).")]
    model = WeightedRuleSetModel(_two_values_model(), rules)
    assert model.predict_proba({"a": "0"}) == {"y": {"0": 0.0, "1": 1.0}}


def test_a_target_with_an_empty_domain_does_not_stop_the_others():
    df = pd.DataFrame({"a": ["0", "1"], "y": [None, None], "z": ["0", "1"]})
    dataset = Dataset(df, features=["a"], targets=["y", "z"])
    model = WeightedRuleSetModel(dataset, [Rule.from_string("z(1) :- a(1).")])
    assert model.predict_proba({"a": "1"}) == {"y": {}, "z": {"0": 0.0, "1": 1.0}}


@pytest.mark.parametrize(
    ("votes", "shares"),
    [
        ({"0": 0.0, "1": 0.0}, {"0": 0.5, "1": 0.5, "2": 0.0}),
        ({"0": 0.75, "1": 0.25}, {"0": 0.75, "1": 0.25, "2": 0.0}),
    ],
    ids=["too small for a float", "summing to 1"],
)
def test_predict_proba_shares_the_votes(monkeypatch, votes, shares):
    df = pd.DataFrame({"a": ["0", "1", "2"], "y": ["0", "1", "2"]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    model = WeightedRuleSetModel(dataset, [])
    monkeypatch.setattr(model, "_likelihood", lambda _state, _unknowns: {"y": votes})
    assert model.predict_proba({"a": "0"}) == {"y": shares}


@pytest.mark.parametrize("share", [0, 1, 0.5])
def test_from_file_reads_a_share_up_to_its_bounds(tmp_path, share):
    path = tmp_path / "model.json"
    WeightedRuleSetModel(_two_values_model(), []).save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["value_frequencies"]["a"]["0"] = share
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert WeightedRuleSetModel.from_file(path).value_frequencies["a"]["0"] == share


def test_from_file_refuses_a_share_above_1(tmp_path):
    path = tmp_path / "model.json"
    WeightedRuleSetModel(_two_values_model(), []).save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["value_frequencies"]["a"]["0"] = 1.5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="value_frequencies gives a=0"):
        WeightedRuleSetModel.from_file(path)
