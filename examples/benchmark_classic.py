"""How you predict matters: car evaluation
=======================================

Learns car-evaluation rules with all three algorithms, and shows why
*how you predict* matters as much as *what you learned*.

Run it:

    python3 examples/benchmark_classic.py

``data/car.csv`` is the UCI Car Evaluation dataset: 1728 rows, six
categorical features, one four-valued target (``unacc``, ``acc``,
``good``, ``vgood``). It is split 80/20 with a fixed seed, every
algorithm learns from the same training rows, and every model is
scored on the same held-out rows.

**The headline is not the timings.** It is that ``GULA`` scores about
0.23 through ``RuleSetModel`` and about 0.92 through
``WeightedRuleSetModel`` — from *identical* rules. Nothing about the
learning changed. Only the way matching rules are turned into an
answer did.

**Why the full rule set cannot discriminate on its own.** ``GULA``
returns every minimal rule consistent with the training data. The car
table lists each of its 1728 feature combinations once, so a held-out
car is one the training rows never saw, and no training row rules out
any class for it. "Every" therefore includes, for each of the four
classes, some rule that covers this particular car — and this script
checks that: on every held-out row, all four classes match.
A uniform vote across matching rules therefore ties four ways every
single time, and the answer is settled by a coin toss between four
options. 0.23 is almost exactly the 0.25 that implies.

``WeightedRuleSetModel`` breaks that tie with the evidence each rule
carries — how many training examples support it against how many
contradict it; on car none do, since no two rows share their features
— so the same 1122 rules become a strong classifier.

``PRIDE`` behaves differently for a reason worth understanding: it is
deliberately *incomplete*, stopping once every training example is
covered by something. Its 193 rules usually match one class rather
than all four (the script counts how often), so it does respectably
even under a uniform vote. That is not ``PRIDE`` being better than
``GULA``; it is a smaller, more committed rule set being easier to
vote on. Weighted, they land in the same place.

The lesson to take away is that a learning algorithm and a prediction
semantics are two separate choices, and this dataset makes the second
one visible.
"""

from __future__ import annotations

# sphinx_gallery_start_ignore
# The docs build runs this file from its own directory, without `__file__`.
if "__file__" not in globals():
    __file__ = "benchmark_classic.py"
# sphinx_gallery_end_ignore

import time
from collections import Counter
from pathlib import Path

import pandas as pd

from pylfit import (
    GULA,
    PRIDE,
    BruteForce,
    Dataset,
    RuleSetModel,
    WeightedRuleSetModel,
    accuracy_score,
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "car.csv"

TARGET = "class"
TRAIN_FRACTION = 0.8

#: Fixed so the numbers in this module's docstring stay reproducible.
#: It seeds both the split and `predict`'s tie-breaking — the latter
#: matters more here than it usually would, since the full rule set
#: ties on every single row.
RANDOM_SEED = 0

ALGORITHMS = [("BruteForce", BruteForce), ("GULA", GULA), ("PRIDE", PRIDE)]

MODELS = [("uniform", RuleSetModel), ("weighted", WeightedRuleSetModel)]


def load_split() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Shuffle `car.csv` once and cut it into train and test frames.

    Returns:
        ``(train, test, features)``.
    """
    frame = pd.read_csv(DATA_PATH).astype(str)
    features = [column for column in frame.columns if column != TARGET]

    shuffled = frame.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    cut = int(len(shuffled) * TRAIN_FRACTION)
    train = shuffled.iloc[:cut].reset_index(drop=True)
    test = shuffled.iloc[cut:].reset_index(drop=True)
    return train, test, features


def classes_matched_per_row(rules, test: pd.DataFrame, features: list[str]) -> Counter:
    """How many distinct classes match each held-out row, tallied.

    This is what turns the module docstring's explanation into
    something checked rather than asserted. For the full rule set the
    answer is "all of them, every time", which is exactly why a uniform
    vote cannot separate the classes.

    Args:
        rules: The learned rules.
        test: Held-out rows.
        features: Feature column names.

    Returns:
        For each number of classes, how many held-out rows match that
        many.
    """
    rules = list(rules)
    counts: Counter = Counter()
    for _, row in test.iterrows():
        state = {feature: row[feature] for feature in features}
        counts[len({rule.head[1] for rule in rules if rule.matches(state)})] += 1
    return counts


def main() -> None:
    """Fit every algorithm, score it both ways, and print one table."""
    train, test, features = load_split()
    train_dataset = Dataset(train, features=features, targets=[TARGET])
    n_classes = len(train_dataset.domain(TARGET))

    print(
        f"car.csv: {len(train)} train / {len(test)} test rows, "
        f"{len(features)} features, {n_classes} classes\n"
    )

    header = f"{'algorithm':<12}{'fit (s)':>10}{'rules':>8}"
    header += "".join(f"{label:>12}" for label, _ in MODELS)
    print(header)
    print("-" * len(header))

    learned = {}
    for name, algorithm in ALGORITHMS:
        started = time.perf_counter()
        rules = algorithm.fit(train_dataset)
        elapsed = time.perf_counter() - started
        learned[name] = set(rules)

        line = f"{name:<12}{elapsed:>10.3f}{len(rules):>8}"
        for _, model_class in MODELS:
            model = model_class(train_dataset, rules)
            predictions = model.predict(test[features], random_state=RANDOM_SEED)
            score = accuracy_score(test, predictions, TARGET)
            line += f"{score:>12.4f}"
        print(line)

    # Same oracle the boolean-network benchmark uses.
    assert learned["BruteForce"] == learned["GULA"], (
        "BruteForce and GULA both return every minimal rule and must agree, but "
        f"BruteForce found {len(learned['BruteForce'])} rules and GULA "
        f"{len(learned['GULA'])}."
    )

    gula_matches = classes_matched_per_row(learned["GULA"], test, features)
    pride_matches = classes_matched_per_row(learned["PRIDE"], test, features)
    # The docstring's claim, checked: every held-out row matches a rule
    # of every class.
    assert set(gula_matches) == {n_classes}, (
        f"GULA's rules matched {sorted(gula_matches)} classes per held-out row, "
        f"not always all {n_classes}"
    )

    print()
    print(f"BruteForce and GULA agree exactly: {len(learned['GULA'])} rules.")
    print(
        f"GULA  matches all {n_classes} classes on every one of the {len(test)} "
        "held-out rows, so a uniform vote always ties."
    )
    print(
        f"PRIDE matches one class on {pride_matches[1]} of them, more on "
        f"{sum(n for k, n in pride_matches.items() if k > 1)}, none on "
        f"{pride_matches[0]}, so a uniform vote usually has an answer."
    )
    print("Weighting by rule support is what makes the full rule set usable.")


if __name__ == "__main__":
    main()
