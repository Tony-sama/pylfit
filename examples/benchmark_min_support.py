"""What min_support buys, and what it costs
========================================

``GULA`` with and without ``min_support=1``, on two data sets.

Run it:

    python3 examples/benchmark_min_support.py

``GULA.fit`` accepts an optional ``min_support``: discard a candidate
rule as soon as too few training examples back it, rather than
carrying it through further specialization. Specializing only ever
shrinks a rule's support, so nothing derived from a dropped candidate
could have cleared the threshold anyway.

**It finds no new rule and loses no well-supported one.** The result
is exactly the unrestricted rule set, minus every rule backed by fewer
than ``min_support`` of its own head's positives. That is what this
script checks: it runs both, filters the unrestricted rules by support
counted directly against the data, and requires the two to be
identical. A more general rule never has less support than a more
specific one, which is why dropping the unsupported candidates early
can't knock out a rule that should have survived.

Two datasets, deliberately opposite shapes:

- ``soybean-small.csv`` — 35 features, 37 training rows. Many
  features for so few rows, so most specializations end up backed by no
  example at all, and dropping them the moment their support hits
  zero prunes almost the whole search. About ten times faster.
- ``car.csv`` — 6 features, 1382 training rows. Many rows per
  feature, so candidates are well supported and fewer get dropped. Still
  about twice as fast: each candidate's support is computed from its
  parent's matched rows rather than by rescanning every row, so the
  filter costs almost nothing even when it prunes little.

**What it costs is completeness, not time.** Look at the rule counts.
A feature state covered only by a dropped rule — one no training
example backs — is left uncovered. ``WeightedRuleSetModel`` never
notices: it gives such a rule no weight, so its predictions are the
same either way. ``RuleSetModel`` counts every matching rule, so it
falls back to "anything is possible" there instead. And read as a
theory of the data, the rule set is no longer complete: a rule backed
by nothing was a guess, but it may have been a right one.
"""

from __future__ import annotations

# sphinx_gallery_start_ignore
# The docs build runs this file from its own directory, without `__file__`.
if "__file__" not in globals():
    __file__ = "benchmark_min_support.py"
# sphinx_gallery_end_ignore

import time
import warnings
from pathlib import Path

import pandas as pd

from pylfit import GULA, Dataset, PylfitWarning

DATA_DIR = Path(__file__).resolve().parent / "data"

TARGET = "class"
TRAIN_FRACTION = 0.8
RANDOM_SEED = 0

#: Config label -> `min_support` value. `None` is unrestricted.
CONFIGS = {"unrestricted": None, "min_support=1": 1}

#: ``(filename, shape)`` pairs, run in this order.
DATASETS = [
    ("soybean-small.csv", "many features, few rows, where it prunes most"),
    ("car.csv", "few features, many rows, where it prunes least"),
]


def training_split(filename: str) -> Dataset:
    """The training slice of `filename`, as a `Dataset`.

    Only the training rows are needed: this script measures fit cost,
    not held-out accuracy. The same fraction and seed as
    `benchmark_classic.py`, so the two are talking about the same rows.

    Args:
        filename: A CSV under ``examples/data/``.

    Returns:
        A `Dataset` over the training rows.
    """
    frame = pd.read_csv(DATA_DIR / filename).astype(str)
    shuffled = frame.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    train = shuffled.iloc[: int(len(shuffled) * TRAIN_FRACTION)].reset_index(drop=True)
    features = [column for column in frame.columns if column != TARGET]
    with warnings.catch_warnings():
        # Soybean's `date` is the month a sample was taken, seven values
        # that are all needed; `Dataset` warns about any column over five.
        warnings.filterwarnings(
            "ignore", message="Column 'date'", category=PylfitWarning
        )
        return Dataset(train, features=features, targets=[TARGET])


def well_supported(rules, dataset: Dataset, min_support: int) -> set:
    """The rules backed by at least `min_support` of their head's positives.

    Support is counted directly against the data, one rule at a time,
    independently of how `GULA` tracks it internally — so comparing
    `GULA`'s restricted output against this is a real check, not the
    same computation done twice.

    Args:
        rules: The learned rules.
        dataset: The dataset they were learned from.
        min_support: The threshold to apply.

    Returns:
        The subset of `rules` clearing the threshold.
    """
    return {
        rule
        for rule in rules
        if sum(1 for p in dataset.positives(*rule.head) if rule.matches(p))
        >= min_support
    }


def main() -> None:
    """Run every dataset under every config and print one table."""
    for filename, shape in DATASETS:
        dataset = training_split(filename)
        print(f"{filename}: {shape}")
        print(f"  {len(dataset.features)} features, {len(dataset)} training rows")

        baseline = None
        unrestricted = None
        for label, min_support in CONFIGS.items():
            started = time.perf_counter()
            rules = GULA.fit(dataset, min_support=min_support)
            elapsed = time.perf_counter() - started

            if baseline is None:
                baseline = elapsed
                verdict = "baseline"
            else:
                ratio = baseline / elapsed
                if ratio >= 1:
                    verdict = f"{ratio:.1f}x faster"
                else:
                    verdict = f"{1 / ratio:.1f}x slower"

            print(f"    {label:<16}{elapsed:8.2f}s{len(rules):8} rules   {verdict}")

            if min_support is None:
                unrestricted = rules
            else:
                expected = well_supported(unrestricted, dataset, min_support)
                assert set(rules) == expected, (
                    f"{filename}: min_support={min_support} kept "
                    f"{len(set(rules) - expected)} rule(s) the unrestricted run "
                    f"filtered by support doesn't, and lost "
                    f"{len(expected - set(rules))} it does."
                )
        print()

    print("On both datasets, min_support kept exactly the unrestricted rules")
    print("that clear its threshold: faster, and nothing else changed.")


if __name__ == "__main__":
    main()
