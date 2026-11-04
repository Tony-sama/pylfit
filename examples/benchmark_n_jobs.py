"""Learning heads in parallel
==========================

What ``n_jobs`` buys: ``GULA``'s fit time on one Boolean network, with
the heads learned in 1, 2, 4 and 8 processes.

Run it:

    python3 examples/benchmark_n_jobs.py

A head is one value of one target, and ``GULA`` learns each head's
rules without looking at any other head's, so ``n_jobs`` hands the
heads out to worker processes. ``n12c5.bnet`` is a 12-variable network:
its full transition table is 4096 rows and it has 24 heads.

**The speedup stops at the slowest head.** Each head is learned whole
by one process, so no number of processes brings a fit below the time
of its longest head. The script measures every head alone and prints
that floor next to the fits. On a 16-CPU Linux machine with Python
3.11 the serial fit took about 0.63 s, 2 processes halved it, 4 brought
it to 0.19 s — and 8 gained little more, because one head alone takes
0.13 s. On a network whose heads cost about the same, the curve goes on
further; set ``NETWORK`` below to ``"randomnet_n15k3.bnet"`` (15
variables, about 4 s serial and 15 s for the whole script) to see one.

**The rules do not depend on the process count, and the script checks
it.** Every run must return the same rule set as the serial one.

Starting a process costs little on Linux up to Python 3.13, which
copies the running one. On Linux from Python 3.14 each process is sent
the dataset and rebuilds its row caches, which costs a fraction of a
second; on macOS and Windows it also starts Python afresh, which costs
seconds on a large table: there, use ``n_jobs`` only for fits of
several seconds.
"""

from __future__ import annotations

# sphinx_gallery_start_ignore
# The docs build runs this file from its own directory, without `__file__`.
if "__file__" not in globals():
    __file__ = "benchmark_n_jobs.py"
# sphinx_gallery_end_ignore

import os
import time
from pathlib import Path

from _bnet import build_dataset_arguments

from pylfit import GULA, Dataset

#: The network to learn, from the corpus the test suite vendors. Kept
#: under 10 s for the whole script, since the docs build runs it.
NETWORK = "n12c5.bnet"

#: Process counts to time, the serial run first: it is the baseline the
#: others are compared and checked against.
N_JOBS = (1, 2, 4, 8)

DATA_PATH = Path(__file__).resolve().parent.parent / "tests" / "data" / "bnet" / NETWORK


def main() -> None:
    """Time each head alone, then the whole fit at each ``n_jobs``."""
    table, features, targets = build_dataset_arguments(DATA_PATH)
    dataset = Dataset(table, features=features, targets=targets)
    heads = [(target, value) for target in targets for value in dataset.domain(target)]

    print(
        f"Network: {NETWORK} ({len(features)} variables, {len(table)} rows, "
        f"{len(heads)} heads), {os.cpu_count()} CPUs\n"
    )

    # The first fit on a dataset also builds each head's positive and
    # negative rows; built here, untimed, so that no head's time below
    # includes them.
    for target, value in heads:
        dataset.positives(target, value)
        dataset.negatives(target, value)

    head_seconds = {}
    for head in heads:
        started = time.perf_counter()
        GULA.fit(dataset, heads=[head])
        head_seconds[head] = time.perf_counter() - started
    slowest = max(head_seconds, key=head_seconds.get)
    print(
        f"Slowest head: {slowest[0]}={slowest[1]}, {head_seconds[slowest]:.2f}s "
        f"alone; no n_jobs can go below it\n"
    )

    baseline_rules = None
    baseline_seconds = None
    for n_jobs in N_JOBS:
        started = time.perf_counter()
        rules = GULA.fit(dataset, n_jobs=n_jobs)
        elapsed = time.perf_counter() - started

        if baseline_rules is None:
            baseline_rules, baseline_seconds = rules, elapsed
            speedup = "baseline"
        else:
            speedup = f"{baseline_seconds / elapsed:.1f}x faster"
        print(f"n_jobs={n_jobs}: {elapsed:6.2f}s   {len(rules)} rules   {speedup}")

        # Heads are learned independently, so how they are spread over
        # processes cannot change what is found.
        assert rules == baseline_rules, (
            f"n_jobs={n_jobs} changed the answer: "
            f"{len(set(baseline_rules) - set(rules))} rule(s) lost and "
            f"{len(set(rules) - set(baseline_rules))} gained against n_jobs=1."
        )

    print()
    print(f"Every n_jobs returned the same {len(baseline_rules)} rules.")


if __name__ == "__main__":
    main()
