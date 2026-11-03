"""Learning and predicting with unknown values
============================================

What pylfit's reading of unknown values buys, against the usual ways
around them.

Run it:

    python3 examples/benchmark_unknowns.py

pylfit reads ``NaN`` and ``None`` as unknown: a value that existed but
was not recorded. It is never a condition and never satisfies one, so
every algorithm learns from the known values only, and a model
predicting a row with unknowns matches strictly, then falls back to the
rules that could match. This script measures that against the methods
people use instead, on four deterministic datasets: the car evaluation
data and three Boolean networks' full transition tables (``n6s1c2``,
``n7s3``, ``randomnet_n7k3``, from the test suite).
"""

from __future__ import annotations

# sphinx_gallery_start_ignore
# The docs build runs this file from its own directory, without `__file__`.
if "__file__" not in globals():
    __file__ = "benchmark_unknowns.py"
# sphinx_gallery_end_ignore

import itertools
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from _bnet import generate_transitions

from pylfit import (
    GULA,
    Dataset,
    PylfitWarning,
    WeightedRuleSetModel,
    accuracy_by_target,
)

EXAMPLES = Path(__file__).resolve().parent
CAR_PATH = EXAMPLES / "data" / "car.csv"
#: The vendored networks live with the regression suite that also reads them.
BNET_DIR = EXAMPLES.parent / "tests" / "data" / "bnet"
DATASETS = ["car", "n6s1c2", "n7s3", "randomnet_n7k3"]

#: Share of feature values blanked.
RATES = [0.1, 0.3]
#: One split and one blanking per seed; `predict` breaks ties with it too.
SEEDS = [0, 1, 2]
TRAIN_FRACTION = 0.7

#: The value the `question_mark` arm writes for an unknown.
QUESTION_MARK = "?"
#: The most completions the `expand` arm makes of one row; past it, they
#: are drawn at random.
MAX_COMPLETIONS = 64

#: Part 1, how the training rows' unknowns are handled; complete test rows.
LEARNING_ARMS = [
    "complete",
    "pylfit",
    "complete_cases",
    "mode",
    "question_mark",
    "expand",
    "assume",
]
#: Part 2, how the test rows' unknowns are handled; pylfit's rules.
PREDICTION_ARMS = ["fallback", "strict", "mode"]

# Expected, and pylfit's own: a filled or expanded table can hold
# conflicting rows, and the `question_mark` arm's "?" is flagged as a
# likely stand-in for a missing value, which it is on purpose.
warnings.simplefilter("ignore", PylfitWarning)

# %%
# 1. Load the data
# ----------------
#
# Every value is read as text. A network's table is its full transition
# table, one row per state: the features end in ``_t``, the targets in
# ``_t+1``.


def load(name: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Loads a dataset as text columns, with its features and targets.

    Args:
        name: ``"car"`` or a network's file stem.

    Returns:
        ``(table, features, targets)``. A network's table is its full
        transition table, one row per state.
    """
    if name == "car":
        table = pd.read_csv(CAR_PATH, dtype=str)
        return table, list(table.columns[:-1]), [table.columns[-1]]
    table = generate_transitions(BNET_DIR / f"{name}.bnet").astype(str)
    features = [column for column in table.columns if column.endswith("_t")]
    targets = [column for column in table.columns if column.endswith("_t+1")]
    return table, features, targets


for name in DATASETS:
    table, features, targets = load(name)
    print(
        f"{name}: {len(table)} rows, {len(features)} features, {len(targets)} targets"
    )

# %%
# 2. Blank some values
# --------------------
#
# Each dataset is shuffled and split 70/30, then 10% or 30% of the
# feature values are blanked at random, independently of everything
# else; targets stay known. Here are the first car rows at 30%:


def blank(table: pd.DataFrame, features: list[str], rate: float, rng) -> pd.DataFrame:
    """Replaces each feature value by a missing one with probability ``rate``.

    Args:
        table: The rows to blank; not modified.
        features: The columns to blank. Targets stay known.
        rate: The probability of blanking each value.
        rng: The NumPy generator to draw with.

    Returns:
        A copy of ``table`` with the drawn feature values missing.
    """
    blanked = table.copy()
    mask = rng.random((len(table), len(features))) < rate
    blanked[features] = blanked[features].mask(mask)
    return blanked


car, car_features, _ = load("car")
blank(car.head(), car_features, 0.3, np.random.default_rng(0))

# %%
# 3. Seven ways to learn from them
# --------------------------------
#
# The training rows have unknowns; the test rows are complete, so only
# the learning differs. Each arm is one way to handle the unknowns:
#
# - ``complete``: no blanking at all, the ceiling.
# - ``pylfit``: the unknowns kept, read as above.
# - ``complete_cases``: every training row with an unknown dropped.
# - ``mode``: each unknown filled with its column's most common value.
# - ``question_mark``: each unknown written as the value ``"?"``, which
#   rules can then test.
# - ``expand``: each row replaced by every completion of its unknowns,
#   at most 64, drawn at random past that.
# - ``assume``: pylfit's reading with
#   ``Dataset(..., assume_nondeterministic=True)``, as pylfit 0.x read
#   unknowns.
#
# Two of them need a helper: the most common values, and the
# completions.


def modes(train: pd.DataFrame, features: list[str]) -> dict[str, str]:
    """Each feature's most common known training value, the smallest on a tie.

    Args:
        train: The training rows, possibly with missing values.
        features: The feature columns.

    Returns:
        The fill-in value of each feature.
    """
    return {feature: min(train[feature].mode()) for feature in features}


def expand(
    table: pd.DataFrame, features: list[str], domain: dict[str, list[str]], rng
) -> pd.DataFrame:
    """Replaces each row by every completion of its missing values.

    Past ``MAX_COMPLETIONS`` completions, that many are drawn at random.

    Args:
        table: Rows with missing feature values.
        features: The feature columns.
        domain: Each feature's values.
        rng: The NumPy generator to draw with.

    Returns:
        The completed rows; a row without missing values is kept as is.
    """
    rows = []
    for row in table.to_dict("records"):
        holes = [feature for feature in features if pd.isna(row[feature])]
        choices = [sorted(domain[feature]) for feature in holes]
        if np.prod([len(values) for values in choices]) <= MAX_COMPLETIONS:
            fills = list(itertools.product(*choices))
        else:
            fills = [
                tuple(rng.choice(values) for values in choices)
                for _ in range(MAX_COMPLETIONS)
            ]
        rows += [{**row, **dict(zip(holes, fill, strict=True))} for fill in fills]
    return pd.DataFrame(rows, columns=table.columns)


# %%
# Every arm learns with ``GULA.fit(dataset, min_support=1)`` and
# predicts with ``WeightedRuleSetModel``. The model's domain is every
# value of the whole dataset, so any test row is accepted.


def fit(train, features, targets, domain, **kw):
    """Learns with ``GULA`` and wraps the rules in a ``WeightedRuleSetModel``.

    Args:
        train: The training rows.
        features: The feature columns.
        targets: The target columns.
        domain: Every value of every variable, so any test row is accepted.
        **kw: Passed to ``Dataset``.

    Returns:
        ``(model, seconds)``: the model, and the time ``GULA.fit`` took.
    """
    dataset = Dataset(train, features=features, targets=targets, **kw)
    # The first fit on a dataset also builds each head's positive and
    # negative rows, once; built here, untimed, they don't count against
    # the arms with more rows.
    for target in dataset.targets:
        for value in dataset.domain(target):
            dataset.positives(target, value)
            dataset.negatives(target, value)
    started = time.perf_counter()
    rules = GULA.fit(dataset, min_support=1, warn_uncovered=False)
    seconds = time.perf_counter() - started
    model = WeightedRuleSetModel(dataset, rules)
    model.domain = {
        variable: {*values, *dataset.domain(variable)}
        for variable, values in domain.items()
    }
    return model, seconds


def accuracy(model, test, features, targets, seed, unknowns="fallback") -> float:
    """Predicts ``test`` and averages the accuracy over the targets.

    Args:
        model: The model to predict with.
        test: Test rows, with the true targets.
        features: The feature columns.
        targets: The target columns.
        seed: Seeds ``predict``'s tie-breaking.
        unknowns: ``predict``'s mode for rows with unknown values.

    Returns:
        The mean, over the targets, of the share predicted right.
    """
    predicted = model.predict(test[features], random_state=seed, unknowns=unknowns)
    scores = accuracy_by_target(test, predicted, targets)
    return sum(scores.values()) / len(scores)


# %%
# 4. Run every arm
# ----------------
#
# For each dataset, rate and seed: split, blank, learn with each arm,
# and score on the test rows. The same loop also blanks the test rows
# for part 2, below, and makes this script's check: writing the unknown
# as ``"?"`` learns ``pylfit``'s rules plus rules testing ``"?"``,
# nothing else, since a ``"?"`` differs from every value just as an
# unknown does. The whole run takes about half a minute.


def run(name: str) -> tuple[list[dict], list[dict]]:
    """Both parts, every rate and seed, on one dataset.

    Args:
        name: ``"car"`` or a network's file stem.

    Returns:
        The learning rows and the prediction rows, one per seed, rate
        and arm: accuracy, and for learning the rule count and fit time.
    """
    table, features, targets = load(name)
    domain = Dataset(table, features=features, targets=targets).domains()
    learning, prediction = [], []
    for rate, seed in itertools.product(RATES, SEEDS):
        rng = np.random.default_rng(seed)
        shuffled = table.sample(frac=1.0, random_state=seed)
        cut = int(len(shuffled) * TRAIN_FRACTION)
        train = shuffled.iloc[:cut].reset_index(drop=True)
        test = shuffled.iloc[cut:].reset_index(drop=True)
        blanked = blank(train, features, rate, rng)
        fills = modes(blanked, features)

        tables = {
            "complete": (train, {}),
            "pylfit": (blanked, {}),
            "complete_cases": (blanked.dropna(subset=features), {}),
            "mode": (blanked.fillna(fills), {}),
            "question_mark": (blanked.fillna(QUESTION_MARK), {}),
            "expand": (expand(blanked, features, domain, rng), {}),
            "assume": (blanked, {"assume_nondeterministic": True}),
        }
        models = {}
        for arm in LEARNING_ARMS:
            arm_train, kw = tables[arm]
            model, seconds = fit(arm_train, features, targets, domain, **kw)
            models[arm] = model
            learning.append(
                {
                    "dataset": name,
                    "rate": rate,
                    "arm": arm,
                    "accuracy": accuracy(model, test, features, targets, seed),
                    "milliseconds": seconds * 1000,
                    "rules": len(model.rules),
                }
            )

        without_question_marks = {
            rule
            for rule in models["question_mark"].rules
            if QUESTION_MARK not in rule.body.values()
        }
        assert without_question_marks == set(models["pylfit"].rules), (
            f"{name}, rate {rate}, seed {seed}: the rules learned with "
            '"?" as a value, less those testing "?", differ from pylfit\'s.'
        )

        blanked_test = blank(test, features, rate, rng)
        tests = {
            "fallback": (blanked_test, "fallback"),
            "strict": (blanked_test, "strict"),
            "mode": (blanked_test.fillna(fills), "fallback"),
        }
        for arm in PREDICTION_ARMS:
            arm_test, unknowns = tests[arm]
            prediction.append(
                {
                    "dataset": name,
                    "rate": rate,
                    "arm": arm,
                    "accuracy": accuracy(
                        models["pylfit"], arm_test, features, targets, seed, unknowns
                    ),
                }
            )
    return learning, prediction


def table_of(rows: list[dict], value: str, arms: list[str], decimals: int = 3) -> str:
    """Formats the mean over seeds as one row per dataset and rate.

    Args:
        rows: Result rows from ``run``.
        value: The column to average.
        arms: The arms, in column order.
        decimals: The decimals to show; ``0`` shows whole numbers.

    Returns:
        The table as text.
    """
    frame = pd.DataFrame(rows).pivot_table(
        index=["dataset", "rate"], columns="arm", values=value
    )[arms]
    frame = frame.reindex(DATASETS, level="dataset").round(decimals)
    return (frame.astype(int) if decimals == 0 else frame).to_string()


learning, prediction = [], []
for name in DATASETS:
    rows = run(name)
    learning += rows[0]
    prediction += rows[1]
print('Writing the unknown as "?" learned pylfit\'s rules plus rules')
print('testing "?", on every dataset, rate and seed, as it must.')

# %%
# 5. Which arm learns best
# ------------------------
#
# Accuracy on the complete test rows, averaged over each dataset's
# targets and over the three seeds, each with its own split and
# blanking:

print("Learning with unknowns: accuracy on complete test rows")
print(table_of(learning, "accuracy", LEARNING_ARMS))

# %%
# **pylfit's reading learns the best rules.** It is the best arm on
# every dataset at both rates. Averaged over the four datasets, at 30%
# unknowns it scores 0.858 against the ceiling's 0.942; filling in the
# mode gives 0.817, the 0.x reading 0.773, expanding 0.756, and dropping
# rows 0.733. On ``n7s3`` at 30%, it is 0.960 against the mode's 0.913
# and complete cases' 0.733. Dropping rows throws away the known values
# with them; filling or expanding them makes rules, and their counts,
# rest on values nobody observed; and the 0.x reading, right only for a
# non-deterministic system, leaves out negative examples these
# deterministic data need.
#
# 6. Writing the unknown as a value
# ---------------------------------
#
# ``question_mark`` ties ``pylfit`` everywhere, as it must on complete
# test rows: it learns the same rules, plus rules testing ``"?"``, which
# no test row holds. **It gets the same accuracy at a cost**, in rules:

print("Rules learned")
print(table_of(learning, "rules", LEARNING_ARMS, decimals=0))

# %%
# It learns 2.3 to 4.8 times as many rules, and takes up to 5.5 times as
# long (section 8). Writing a value of its own is still the right choice
# when the absence itself carries information; here it doesn't.
#
# 7. Predicting rows with unknowns
# --------------------------------
#
# Now the test rows have unknowns too, blanked at the same rate, and the
# model is ``pylfit``'s from section 4:
#
# - ``fallback``: ``predict(..., unknowns="fallback")``, the default.
# - ``strict``: ``unknowns="strict"``: a target no rule matches for sure
#   gets a random value.
# - ``mode``: each unknown of the test row filled with the training
#   column's most common value, then predicted.

print("Predicting with unknowns: accuracy of pylfit's rules")
print(table_of(prediction, "accuracy", PREDICTION_ARMS))

# %%
# **The fallback is the safest default.** Averaged over the datasets,
# at 30% unknowns it scores 0.790, against 0.779 for filling the test
# row and 0.774 for strict prediction; at 10% the three are within 0.7
# point. It gains most where strict prediction leaves rows without a
# rule, 6 points on car at 30%. Filling the test row does better on
# ``n7s3`` (0.844 against 0.830 at 30%), where a column's most common
# value is often right, and so does strict prediction (0.842); at 30%
# filling does worse on the other three.
#
# 8. Learning time
# ----------------
#
# ``GULA.fit``'s time in milliseconds, leaving out each dataset's
# one-off row preparation. Unlike the other tables, it varies with the
# machine:

print("GULA.fit milliseconds")
print(table_of(learning, "milliseconds", LEARNING_ARMS, decimals=1))

# %%
# **Learning slows down a little at this size.** ``GULA.fit`` on the
# ``pylfit`` arm takes up to 1.7 times as long as on the complete table:
# each unknown in a negative example rules out fewer rules, so more
# survive. On more features the slowdown is far larger, 4 to 11 times on
# 12- and 13-variable networks at 30%, and ``PRIDE``, which stays within
# seconds there, is the faster choice.
#
# These are three seeds, and unknowns drawn independently of the data:
# unknowns that depend on the outcome can change the ranking. The seeds
# make the accuracies and rule counts the same on every run.
