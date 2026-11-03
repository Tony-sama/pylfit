"""Compare the three algorithms on Boolean networks
================================================

Times ``BruteForce``, ``GULA`` and ``PRIDE`` on small Boolean networks.

Run it:

    python3 examples/benchmark_bnet.py

Each network is expanded into its full transition table — every one of
its ``2**n`` states paired with the state that follows — and each
algorithm is asked to recover the network's rules from that table. What
gets timed is ``fit()`` alone.

**The point is the comparison, not the absolute numbers.** All three
learn the same thing from the same data; they differ in how much work
they do to get there. ``BruteForce`` enumerates candidate rules
exhaustively, ``GULA`` specializes away from what the data forbids, and
``PRIDE`` covers each example with one rule and stops.

The interesting part is how the gap *grows*. On the 3-variable
networks ``BruteForce`` is two or three times slower than ``GULA``; at
5 variables about four times, at 7 about fifteen, and at 9, the last
network here, about seventy: four seconds against under a tenth of one.
That curve, not any single row, is the reason ``GULA`` exists.
It is also why the list below stops at 9 variables: one more would
turn a script meant to be run into one that has to be waited for.

**Past that, ``GULA`` meets the same curve against ``PRIDE``.** A
second table times only those two, on networks of 10 to 13 variables.
At 10 variables ``GULA`` takes about ten times ``PRIDE``'s time; at
13, about seventeen: two seconds against a tenth of one. At 15 variables
(``dinwoodie_life``, 32,768 rows, not run here) ``GULA`` took 13 s and
``PRIDE`` 0.9 s. A full transition table doubles its rows with each
variable, so both grow; ``GULA`` grows faster. Where ``GULA`` takes
seconds, ``PRIDE`` is the algorithm to start with.

**Further still, only ``PRIDE`` runs.** A third table times it on
networks of 23, 40 and 103 variables, too large for a full table: the
smallest would have 2**23 rows, 8.4 million, and hundreds of gigabytes
of memory. Each is sampled instead: 10,000 distinct states with their
successors for the first two, 1000 for the third. ``PRIDE`` fits each
in a second or two. ``GULA``, on the same rows of the first two, was
stopped unfinished after three hours and two hours. With no ``GULA``
rules to compare against, the script checks ``PRIDE``'s against the
rows: none matches a row with another value of its target, and every
row is matched by a rule for its own.

**Every run checks itself, which is why the slow algorithm is still
here.** ``BruteForce`` and ``GULA`` both return *every* minimal rule
consistent with the data, so on identical input they must return
identical rule sets, and the script fails loudly if they ever don't.
That makes the slow exhaustive one useful long after you would stop
calling it: it is the oracle the fast one is checked against.
``PRIDE`` may return fewer, so it is held to the weaker claim its
docstring actually makes — every rule it returns is one ``GULA`` also
found, never a rule of its own invention. The ``PRIDE rules`` column
shows where it kept fewer: on these full transition tables, only on
``n5s3``.

A failure here is therefore worth reading closely. It does not mean
"the benchmark broke"; it means two algorithms that are supposed to
agree no longer do.
"""

from __future__ import annotations

# sphinx_gallery_start_ignore
# The docs build runs this file from its own directory, without `__file__`.
if "__file__" not in globals():
    __file__ = "benchmark_bnet.py"
# sphinx_gallery_end_ignore

import time
from pathlib import Path

from _bnet import build_dataset_arguments, feature_target_columns, sample_transitions

from pylfit import GULA, PRIDE, BruteForce, Dataset, coverage

#: Networks to run, smallest first. Drawn from the corpus the test
#: suite vendors. The last one carries the whole point — it is where
#: `BruteForce` first costs seconds rather than milliseconds — and it
#: is also most of this table's time, about ten seconds. Larger
#: networks sit in the same directory; adding one is a one-line change,
#: and `BruteForce` is what will stop you.
NETWORKS = [
    "n3s1c1a",
    "n3s1c1b",
    "raf",
    "n5s3",
    "n6s1c2",
    "n7s3",
    "arellano_rootstem",
]

#: Networks past `BruteForce`'s reach, timed with `GULA` and `PRIDE`
#: only. `dinwoodie_life` (15 variables) adds about 15 s.
LARGER_NETWORKS = [
    "faure_cellcycle",
    "krumsiek_myeloid",
    "tournier_apoptosis",
    "saadatpour_guardcell",
]

#: Networks too large for a full table, with the number of distinct
#: states sampled from each, timed with `PRIDE` alone. `jaoude_thdiff`
#: takes 1000: its 10,000 rows take 14 s to fit and 6 GB of memory.
SAMPLED_NETWORKS = [
    ("mendoza_thelper", 10_000),
    ("klamt_tcr", 10_000),
    ("jaoude_thdiff", 1_000),
]

#: The vendored networks live with the regression suite that also
#: reads them. There is one copy, not one per consumer.
DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "bnet"

ALGORITHMS = [("BruteForce", BruteForce), ("GULA", GULA), ("PRIDE", PRIDE)]


def run_one(name: str, algorithms: list) -> dict:
    """Fit each algorithm on one network and check they agree.

    Args:
        name: Network filename stem, e.g. ``"raf"``.
        algorithms: ``(label, algorithm)`` pairs, ``GULA`` and ``PRIDE``
            among them.

    Returns:
        A row of results: network name, variable count, row count, one
        ``f"{algorithm}_s"`` timing per algorithm, and the rule counts
        `GULA` and `PRIDE` found.

    Raises:
        AssertionError: If `BruteForce`, when run, and `GULA` disagree,
            or if `PRIDE` returns a rule `GULA` did not find.
    """
    table, features, targets = build_dataset_arguments(DATA_DIR / f"{name}.bnet")
    dataset = Dataset(table, features=features, targets=targets)

    row = {"network": name, "vars": len(features), "rows": len(table)}
    # The first fit on a dataset also builds each head's positive and
    # negative rows, once, for every fit after it. Built here, untimed,
    # so the timings below compare the algorithms and not who happened
    # to run first.
    for target in dataset.targets:
        for value in dataset.domain(target):
            dataset.positives(target, value)
            dataset.negatives(target, value)
    learned = {}
    for label, algorithm in algorithms:
        started = time.perf_counter()
        learned[label] = set(algorithm.fit(dataset))
        row[f"{label}_s"] = time.perf_counter() - started

    # Both return every minimal rule, so this is equality, not containment.
    if "BruteForce" in learned:
        only_brute = sorted(map(str, learned["BruteForce"] - learned["GULA"]))
        only_gula = sorted(map(str, learned["GULA"] - learned["BruteForce"]))
        assert learned["BruteForce"] == learned["GULA"], (
            f"{name}: BruteForce and GULA both return every minimal rule and "
            f"must agree, but BruteForce found {len(learned['BruteForce'])} "
            f"rules and GULA {len(learned['GULA'])}. "
            f"Only in BruteForce: {only_brute}. Only in GULA: {only_gula}."
        )

    # PRIDE stops once every example is covered, so it may find fewer
    # rules, but never a different one.
    spurious = learned["PRIDE"] - learned["GULA"]
    assert not spurious, (
        f"{name}: PRIDE returned {len(spurious)} rule(s) GULA did not find. "
        f"PRIDE is allowed to be incomplete, never to invent: "
        f"{sorted(map(str, spurious))}."
    )

    row["rules"] = len(learned["GULA"])
    row["pride_rules"] = len(learned["PRIDE"])
    return row


def print_table(networks: list[str], algorithms: list) -> None:
    """Run each network with each algorithm and print one table.

    Args:
        networks: Network filename stems.
        algorithms: ``(label, algorithm)`` pairs, as ``run_one`` takes.
    """
    header = f"{'network':<22}{'vars':>5}{'rows':>7}{'rules':>7}{'PRIDE rules':>13}"
    header += "".join(f"{label + ' (s)':>15}" for label, _ in algorithms)
    print(header)
    print("-" * len(header))

    for name in networks:
        row = run_one(name, algorithms)
        line = f"{row['network']:<22}{row['vars']:>5}{row['rows']:>7}{row['rules']:>7}"
        line += f"{row['pride_rules']:>13}"
        line += "".join(f"{row[f'{label}_s']:>15.4f}" for label, _ in algorithms)
        print(line)
    print()


def run_sampled(name: str, rows: int) -> dict:
    """Fit `PRIDE` on sampled states of one network and check its rules.

    Args:
        name: Network filename stem, e.g. ``"klamt_tcr"``.
        rows: Number of distinct states to sample.

    Returns:
        A row of results: network name, variable count, row count,
        `PRIDE`'s time and its rule count.

    Raises:
        AssertionError: If a rule matches a row of another value of its
            target, or a row matches no rule of its own.
    """
    path = DATA_DIR / f"{name}.bnet"
    table = sample_transitions(path, rows)
    features, targets = feature_target_columns(path)
    dataset = Dataset(table, features=features, targets=targets)
    for target in dataset.targets:
        for value in dataset.domain(target):
            dataset.positives(target, value)
            dataset.negatives(target, value)

    started = time.perf_counter()
    rules = PRIDE.fit(dataset)
    elapsed = time.perf_counter() - started

    # Without GULA to compare against, the rules are checked against
    # the rows themselves: none matches a negative, all cover every row.
    for rule in rules:
        target, value = rule.head
        wrong = [row for row in dataset.negatives(target, value) if rule.matches(row)]
        assert not wrong, f"{name}: {rule} matches {len(wrong)} negative row(s)."
    shares = coverage(dataset, rules)
    uncovered = {head: share for head, share in shares.items() if share < 1}
    assert not uncovered, f"{name}: rows no rule matches, per head: {uncovered}."

    return {
        "network": name,
        "vars": len(features),
        "rows": len(table),
        "PRIDE_s": elapsed,
        "pride_rules": len(rules),
    }


def print_sampled_table() -> None:
    """Run `PRIDE` on each sampled network and print one table."""
    header = f"{'network':<22}{'vars':>5}{'rows':>7}{'PRIDE rules':>13}"
    header += f"{'PRIDE (s)':>15}"
    print(header)
    print("-" * len(header))
    for name, rows in SAMPLED_NETWORKS:
        row = run_sampled(name, rows)
        line = f"{row['network']:<22}{row['vars']:>5}{row['rows']:>7}"
        line += f"{row['pride_rules']:>13}{row['PRIDE_s']:>15.4f}"
        print(line)
    print()


def main() -> None:
    """Run every network: all three algorithms, then the larger ones."""
    print_table(NETWORKS, ALGORITHMS)
    print("BruteForce and GULA agreed on every network, as they must.")
    print("PRIDE found no rule the other two missed.\n")
    print_table(LARGER_NETWORKS, ALGORITHMS[1:])
    print("PRIDE found no rule GULA missed.\n")
    print_sampled_table()
    print("PRIDE's rules match no negative row and cover every row.")


if __name__ == "__main__":
    main()
