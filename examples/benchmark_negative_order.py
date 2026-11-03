"""The order GULA reads negatives in
=================================

What ``negative_order`` changes: ``GULA``'s fit time with each of its
three orders, on a Boolean network and two classic datasets.

Run it:

    python3 examples/benchmark_negative_order.py

``GULA`` starts each head from the most general rule and specializes
its kept rules against one negative example at a time.
``negative_order`` is the order it takes them in:

- ``"similarity"`` (the default) — the most typical negatives first:
  those differing from the other negatives in the fewest feature
  values, summed over all of them.
- ``"lexicographic"`` — sorted by feature values, so consecutive
  negatives share long prefixes.
- ``"unordered"`` — the order the rows came in.

**No order is fastest everywhere, which is why the option exists.** On
a 16-CPU Linux machine, ``"lexicographic"`` won on ``car.csv`` (1728
rows, 6 features) by about 30%; the two tied on the network's
transition table; on ``soybean-small.csv`` (47 rows, 35 features, over
13,000 rules) ``"similarity"`` took 1.5 s against 2.3 s. The default
is the order that saved the most time on the datasets that took
longest, not the one that wins most often. The gap can be wider the
other way: on the 12-variable ``n12c5.bnet``, ``"lexicographic"`` took
0.9 s against 2.6 s.

**Leaving the rows unordered is only as good as their order.** Every
table here is shuffled first, as a dataset collected in no particular
order would be, and there ``"unordered"`` is slower on every one:
about 30 times on the network, and 270 s against 1 s on the
12-variable ``tournier_apoptosis.bnet``, which is why the network here
is small.
A transition table generated state by state comes out nearly sorted,
and on it ``"unordered"`` is as fast as ``"lexicographic"`` — a
coincidence of how the file was written, not something to count on.

**The rules do not depend on the order, and the script checks it.**
Every order must return the same rule set.
"""

from __future__ import annotations

# sphinx_gallery_start_ignore
# The docs build runs this file from its own directory, without `__file__`.
if "__file__" not in globals():
    __file__ = "benchmark_negative_order.py"
# sphinx_gallery_end_ignore

import time
import warnings
from pathlib import Path

import pandas as pd
from _bnet import build_dataset_arguments

from pylfit import GULA, Dataset, PylfitWarning

EXAMPLES_DIR = Path(__file__).resolve().parent

#: The network to learn, from the corpus the test suite vendors. Kept
#: at 10 variables: ``"unordered"`` on a shuffled table grows fastest.
NETWORK = "faure_cellcycle.bnet"

#: The two classic datasets, under ``examples/data/``, and their target.
CSV_FILES = ["car.csv", "soybean-small.csv"]
TARGET = "class"

ORDERS = ["similarity", "lexicographic", "unordered"]
RANDOM_SEED = 0


def shuffled(table: pd.DataFrame) -> pd.DataFrame:
    """Returns ``table``'s rows in a fixed random order.

    Args:
        table: The rows to shuffle.

    Returns:
        The same rows, reordered with ``RANDOM_SEED``.
    """
    return table.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)


def datasets() -> dict[str, Dataset]:
    """Builds the network's transition table and the two CSVs.

    Returns:
        Display name -> `Dataset`, rows shuffled.
    """
    path = EXAMPLES_DIR.parent / "tests" / "data" / "bnet" / NETWORK
    table, features, targets = build_dataset_arguments(path)
    built = {NETWORK: Dataset(shuffled(table), features=features, targets=targets)}
    for filename in CSV_FILES:
        frame = pd.read_csv(EXAMPLES_DIR / "data" / filename).astype(str)
        features = [column for column in frame.columns if column != TARGET]
        with warnings.catch_warnings():
            # Soybean's `date` is the month a sample was taken, seven
            # values that are all needed; `Dataset` warns over five.
            warnings.filterwarnings(
                "ignore", message="Column 'date'", category=PylfitWarning
            )
            built[filename] = Dataset(
                shuffled(frame), features=features, targets=[TARGET]
            )
    return built


def main() -> None:
    """Fit each dataset with each order, time it, and check they agree."""
    for name, dataset in datasets().items():
        print(f"{name} ({len(dataset.features)} features):")

        # The first fit on a dataset also builds each head's positive
        # and negative rows; built here, untimed, so that whichever
        # order runs first doesn't pay for them.
        for target in dataset.targets:
            for value in dataset.domain(target):
                dataset.positives(target, value)
                dataset.negatives(target, value)

        baseline_rules = None
        for order in ORDERS:
            started = time.perf_counter()
            rules = GULA.fit(dataset, negative_order=order)
            elapsed = time.perf_counter() - started
            print(f"  {order:<14} {elapsed:6.2f}s   {len(rules)} rules")

            if baseline_rules is None:
                baseline_rules = rules
            # The order only changes which kept rules are specialized
            # when, never which minimal rules are left at the end.
            assert rules == baseline_rules, (
                f"negative_order={order!r} changed the answer on {name}: "
                f"{len(set(baseline_rules) - set(rules))} rule(s) lost and "
                f"{len(set(rules) - set(baseline_rules))} gained against "
                f"{ORDERS[0]!r}."
            )
        print()

    print("Every order returned the same rules on every dataset.")


if __name__ == "__main__":
    main()
