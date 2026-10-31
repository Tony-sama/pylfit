"""Predict heart disease from clinical data
========================================

From a table of clinical measurements to rules that explain each
prediction.

Run it:

    python3 examples/tutorial_heart_disease.py

``data/heart-disease.csv`` is the Cleveland part of the UCI Heart
Disease dataset: 303 patients, 13 measurements each, and ``num``, the
diagnosis of coronary artery disease from 0 (none) to 4. As in the
literature on this data, the target here is presence (1 to 4) against
absence (0).
"""

from __future__ import annotations

# sphinx_gallery_start_ignore
# The docs build runs this file from its own directory, without `__file__`.
if "__file__" not in globals():
    __file__ = "tutorial_heart_disease.py"
# sphinx_gallery_end_ignore

import time
from pathlib import Path

import pandas as pd

from pylfit import (
    GULA,
    PRIDE,
    Dataset,
    Rule,
    WeightedRuleSetModel,
    accuracy_score,
    generalize,
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "heart-disease.csv"

TARGET = "disease"

#: Share of the patients held out for scoring.
TEST_SHARE = 0.2

#: Fixed so the numbers in the text stay reproducible.
RANDOM_SEED = 0

#: Rules shown per class for each explained patient.
TOP_RULES = 3

#: Below every held-out score of `GULA` over seeds 0 to 9 (0.75 to
#: 0.85).
MIN_ACCURACY = 0.70

# %%
# 1. Make every value a category
# ------------------------------
#
# Read the file, and turn ``num`` into the target: ``"1"`` when it is 1
# to 4, ``"0"`` when it is 0. Six patients have a value nobody recorded,
# written ``?`` in the file. They are read as unknown
# (``na_values="?"``), not as a category called ``?``, and stay in the
# data: pylfit learns and predicts with unknown values.

raw = pd.read_csv(DATA_PATH, na_values="?")
raw[TARGET] = (raw.pop("num") > 0).astype(int).astype(str)
raw.head()

# %%
# The six patients with an unknown value, ``NaN`` here:

raw[raw.isna().any(axis=1)]

# %%
# pylfit reads every value as a category, and five of the columns are
# measurements. ``prepare`` bins them with ``pandas.cut``, using fixed
# bins chosen before looking at the results. Two are clinical cut-offs:
# resting blood pressure at 120 and 140 mmHg (the JNC 7 stages) and
# cholesterol at 200 and 240 mg/dl (the NCEP guideline). Age, maximum
# heart rate and ST depression have no such standard and get round
# bins. The choice matters less than it looks: bins at the training
# rows' quartiles, tertiles or median score within a point of these.
# The coded columns get names, so the rules read ``sex(female)``, not
# ``sex(0)``.
#
# Bins computed from the data, as ``pandas.qcut`` does, must be computed
# on the training rows only, then applied to the held-out ones.
# Computing them on every row lets the test rows shape the features, and
# the score is then no longer a held-out score. Fixed bins avoid the
# question.


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn every column into categories pylfit can read.

    Fixed bins, so the same call works on any rows: training, held-out
    or a new patient. The labels are what the rules will show.

    Args:
        frame: Raw rows, as read from the file.

    Returns:
        One column per feature, values ``str`` or ``None`` where
        unknown, plus the target.
    """
    out = pd.DataFrame(index=frame.index)
    out["age"] = pd.cut(
        frame["age"],
        [0, 45, 55, 65, 120],
        right=False,
        labels=["<45", "45-54", "55-64", "65+"],
    )
    out["sex"] = frame["sex"].map({0: "female", 1: "male"})
    out["cp"] = frame["cp"].map(
        {1: "typical", 2: "atypical", 3: "non-anginal", 4: "asymptomatic"}
    )
    out["trestbps"] = pd.cut(
        frame["trestbps"],
        [0, 120, 140, 300],
        right=False,
        labels=["normal", "elevated", "high"],
    )
    out["chol"] = pd.cut(
        frame["chol"],
        [0, 200, 240, 1000],
        right=False,
        labels=["desirable", "borderline", "high"],
    )
    out["fbs"] = frame["fbs"].map({0: "no", 1: "yes"})
    out["restecg"] = frame["restecg"].map({0: "normal", 1: "st-t", 2: "lvh"})
    out["thalach"] = pd.cut(
        frame["thalach"], [0, 120, 150, 250], labels=["low", "mid", "high"]
    )
    out["exang"] = frame["exang"].map({0: "no", 1: "yes"})
    out["oldpeak"] = pd.cut(
        frame["oldpeak"], [-1, 0, 2, 10], labels=["absent", "moderate", "marked"]
    )
    out["slope"] = frame["slope"].map({1: "up", 2: "flat", 3: "down"})
    out["ca"] = frame["ca"].map({0: "0", 1: "1", 2: "2", 3: "3"})
    out["thal"] = frame["thal"].map({3: "normal", 6: "fixed", 7: "reversible"})
    out = out.astype(object).where(out.notna(), None)
    out[TARGET] = frame[TARGET]
    return out


data = prepare(raw)
data.head()

# %%
# The same five patients as above, now in categories: a resting blood
# pressure of 145 is ``high``, a cholesterol of 233 ``borderline``.
#
# 2. Score on held-out patients
# -----------------------------
#
# The patients are shuffled, a fifth of them are held out, and ``GULA``
# and ``PRIDE`` learn from the rest. The model is given the values of
# every patient (``model.domain``), so a bin only the held-out patients
# fall into is not an error.

features = [column for column in data.columns if column != TARGET]
domain = Dataset(data, features=features, targets=[TARGET]).domains()
print(
    f"{DATA_PATH.name}: {len(data)} patients, {len(features)} features, "
    f"{int(data[TARGET].astype(int).sum())} with disease"
)

shuffled = data.sample(frac=1, random_state=RANDOM_SEED, ignore_index=True)
n_test = round(len(data) * TEST_SHARE)
test = shuffled[:n_test].reset_index(drop=True)
train = shuffled[n_test:].reset_index(drop=True)
print(f"Learning from {len(train)} patients, scoring on the other {len(test)}.")


def fit_and_score(
    name: str, algorithm, generalized: bool = False
) -> tuple[WeightedRuleSetModel, list, float]:
    """Learn on the training patients and score on the held-out ones.

    Args:
        name: How to label the printed line.
        algorithm: `GULA` or `PRIDE`.
        generalized: If True, drops the conditions the training rows
            do not need, with `generalize`, before building the model.

    Returns:
        The `WeightedRuleSetModel`, its predictions on the held-out
        patients, and their accuracy.
    """
    start = time.perf_counter()
    dataset = Dataset(train, features=features, targets=[TARGET])
    rules = algorithm.fit(dataset)
    if generalized:
        rules = generalize(rules, dataset)
    model = WeightedRuleSetModel(dataset, rules)
    model.domain = domain
    seconds = time.perf_counter() - start
    predictions = model.predict(test[features], random_state=RANDOM_SEED)
    accuracy = accuracy_score(test, predictions, TARGET)
    print(
        f"  {name:<21} {len(model.rules):>5} rules in {seconds:.2f} s, "
        f"{accuracy:.3f} accuracy on the held-out patients"
    )
    return model, predictions, accuracy


gula, gula_predictions, gula_accuracy = fit_and_score("GULA", GULA)
_ = fit_and_score("PRIDE", PRIDE)

# %%
# On this script's seed ``GULA`` scores 0.80 on the 61 held-out patients
# and ``PRIDE`` 0.70; with so few test patients another seed moves the
# score by several points (``GULA`` 0.75 to 0.85 over ten seeds). Each
# fit's time is printed too: about 3.7 s for ``GULA``, which learns some
# 12000 rules, a few hundredths of a second for ``PRIDE``'s 58. For
# reference, on the same categories and the same ten seeds,
# scikit-learn's logistic regression scores about 0.84 and a random
# forest about 0.82: pylfit is not here to win on accuracy. What it
# gives for about the same accuracy is the explanation.
#
# 3. Read why
# -----------
#
# For two held-out patients, one predicted right and one wrong, print
# the rules that match, the strongest first. Each rule comes with its
# counts on the training patients: how many it matches that have its
# conclusion ("for"), and how many that have the other one ("against").
# ``WeightedRuleSetModel`` adds up the counts of every matching rule and
# predicts the class with more evidence. ``GULA`` learns every rule the
# data supports, about 12000 here, and close to a hundred match each
# patient, so ``explain`` shows how many matched, their counts added up,
# and the strongest three of each class.


def explain(model: WeightedRuleSetModel, patient: dict, label: str) -> None:
    """Print a patient, the prediction, and the strongest matching rules.

    Args:
        model: The fitted `GULA` model.
        patient: One prepared row, target included.
        label: How to introduce the patient.
    """
    features = model.features
    state = {feature: patient[feature] for feature in features}
    likelihood = model.predict_likelihood(state)[TARGET]
    predicted = max(likelihood, key=likelihood.get)
    print(f"\n{label}: actual disease={patient[TARGET]}, predicted {predicted}")
    print("  " + ", ".join(f"{f}={v}" for f, v in state.items() if v is not None))
    for value in sorted(likelihood):
        matching = [
            rule
            for rule in model.rules
            if rule.head == (TARGET, value) and rule.matches(state)
        ]
        matching.sort(
            key=lambda rule: (-model.get_score(rule)[0], len(rule.body), str(rule))
        )
        # What the model weighs: every matching rule's counts, added up.
        pooled = sum(model.get_score(rule)[0] for rule in matching)
        print(
            f"  disease={value}: {len(matching)} matching rules, {pooled} for in all; "
            f"the strongest:"
        )
        for rule in matching[:TOP_RULES]:
            positives, negatives = model.get_score(rule)
            print(f"    {rule}  ({positives} for, {negatives} against)")


print(f"GULA learned {len(gula.rules)} rules on the training patients.")
right = wrong = None
for index in range(len(test)):
    correct = gula_predictions[index][TARGET] == test.loc[index, TARGET]
    if correct and right is None:
        right = index
    if not correct and wrong is None:
        wrong = index
explain(gula, test.loc[right].to_dict(), "A patient predicted right")
explain(gula, test.loc[wrong].to_dict(), "A patient predicted wrong")

# %%
# The first patient is a clear case: 859 for no disease against 6 for
# disease. The second has no disease but is predicted to, and the rules
# say why. A woman with a normal resting ECG and an upward ST slope
# points to no disease (60 in all); high cholesterol, high blood
# pressure and chest pain on exercise point to disease (271), and
# outweigh it. A clinician can see which rules decided, and judge them.
#
# 4. Read everything
# ------------------
#
# ``PRIDE`` stops once every training patient is covered: on all 303
# patients it learns 74 rules, and two match a typical patient. Its
# whole model fits on a screen and can be checked rule by rule, for a
# few points of accuracy. Choose ``GULA`` to explain each prediction,
# ``PRIDE`` to read the whole model.

everyone = Dataset(data, features=features, targets=[TARGET])
pride = WeightedRuleSetModel(everyone, PRIDE.fit(everyone))
rules = sorted(
    pride.rules, key=lambda rule: (rule.head, -pride.get_score(rule)[0], str(rule))
)
print(f"PRIDE on all {len(data)} patients: {len(rules)} rules, the whole model:")
for rule in rules:
    positives, negatives = pride.get_score(rule)
    print(f"  {rule}  ({positives} for, {negatives} against)")

# %%
# 5. Only the conditions the data needs
# -------------------------------------
#
# ``GULA``'s rules fit every training patient, so many carry conditions
# that only dodge a few patients. ``generalize(rules, dataset)`` drops,
# one at a time, each condition the training patients do not need: a
# condition is needed when, among the patients matching the rest of the
# rule, those it holds for have the rule's conclusion significantly more
# often (a Fisher exact test at ``alpha=0.01``; lower keeps fewer).
# Rules that become the same are merged, and of a rule and a broader
# one, the narrow one stays only if the broad rule's patients need its
# extra conditions, so no rule is a narrower copy of another:
#
# .. code-block:: python
#
#     rules = generalize(GULA.fit(dataset), dataset)
#     model = WeightedRuleSetModel(dataset, rules)

general, _, _ = fit_and_score("GULA, generalized", GULA, generalized=True)
_ = fit_and_score("PRIDE, generalized", PRIDE, generalized=True)

# %%
# ``GULA``'s 12000 rules become about 27 over ten seeds (22 on this
# script's seed), and the model scores as well or better (0.82 against
# 0.81; on this script's seed 0.82 against 0.80). ``PRIDE``'s 56 or so
# become about 17 and score better, 0.82 against 0.75 over ten seeds
# (0.79 against 0.70 on this script's seed): ``PRIDE`` leaves 4 to 9 of
# the 61 held-out patients matched by no rule, drawn at random, and its
# generalized rules at most one. The strongest generalized ``GULA``
# rules, each class in turn:

print(f"GULA's rules generalized: {len(general.rules)}; the strongest:")
for value in ("0", "1"):
    rules = [rule for rule in general.rules if rule.head == (TARGET, value)]
    rules.sort(key=lambda rule: (-general.get_score(rule)[0], str(rule)))
    for rule in rules[:TOP_RULES]:
        positives, negatives = general.get_score(rule)
        print(f"  {rule}  ({positives} for, {negatives} against)")

# %%
# The counts show a rule can be needed and still weak: ``age(55-64)``
# raises the rate of disease, but 33 patients that age have none.
#
# Each rule is generalized on its own, so a rule can come out beside a
# narrower copy of itself, and only one of the two is kept. Here
# ``disease(1) :- cp(asymptomatic).`` came out beside the same rule
# with ``exang(yes)``: among patients with asymptomatic chest pain,
# exercise angina raises the rate of disease, so the narrow rule is
# kept. ``disease(1) :- ca(2).`` came out beside the same rule with
# ``slope(flat)``, which no training patient contradicts, but 18
# patients are too few once every condition that could have been added
# is counted, so the broad rule is kept:

training = Dataset(train, features=features, targets=[TARGET])
for text in (
    "disease(1) :- cp(asymptomatic).",
    "disease(1) :- cp(asymptomatic), exang(yes).",
    "disease(1) :- ca(2).",
    "disease(1) :- ca(2), slope(flat).",
):
    rule = Rule.from_string(text)
    positives, negatives = WeightedRuleSetModel(training, [rule]).get_score(rule)
    kept = "kept" if rule in general.rules else "dropped"
    print(f"  {rule}  ({positives} for, {negatives} against): {kept}")

# %%
# For a model of a given size,
# ``generalize(rules, dataset, max_rules=20)`` lowers ``alpha`` until at
# most 20 rules are left. Over ten seeds, ``alpha`` from 0.0001 to 0.2
# keeps 16 to 98 of ``GULA``'s rules and the accuracy stays at 0.80 to
# 0.82, so the size is yours to choose.
#
# The assertion at the end checks that ``GULA`` reaches
# ``MIN_ACCURACY``, set below its score on every seed from 0 to 9.

assert gula_accuracy >= MIN_ACCURACY, (
    f"GULA's accuracy {gula_accuracy:.3f} is below the {MIN_ACCURACY} "
    f"this script expects."
)
print("Passed.")
