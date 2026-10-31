"""What background knowledge buys
==============================

``GULA`` with and without ``allowed_conditions``, on a real
gene-regulatory network.

Run it:

    python3 examples/benchmark_influence.py

``GULA.fit`` accepts an optional ``allowed_conditions`` argument saying,
per head, which feature conditions are even permitted in a rule body.
Without it the search considers every feature for every head. With it,
the search is confined to the features that can actually matter.

``faure_cellcycle.bnet`` is a 10-variable mammalian cell-cycle model, so
its transition table is 1024 rows — but each node is regulated by only
one to six of the ten variables. That gap is where the saving lives,
and it is large: the restricted runs finish about four times faster.

Three configurations:

- **raw** — no background knowledge at all.
- **regulators only** — each head restricted to the variables that
  appear in its update function. "These features can matter."
- **regulators and sign** — additionally pinning each *monotone*
  regulator to the one value its influence supports. A regulator that
  only ever appears negated can only push the head one way, and that
  is readable off the expression's syntax without evaluating anything.
  On this network it adds almost nothing to the regulators alone: once
  the search is confined to the right features, little is left to cut.

**All three must return identical rules, and the script checks it.**
That is the part worth dwelling on. Background knowledge here is
*exact* — it comes from the network file that generated the data —
so restricting the search throws away only work, never an answer.
Wrong or incomplete background knowledge would not corrupt the result
either; it would cost completeness: ``GULA`` would find fewer rules,
all of them among the unrestricted run's, and warn about the training
rows none of them covers. Speed is the only thing being traded when
the knowledge is right.

Which is the practical lesson: if you know something about your
domain — that a gene responds only to the factors known to bind its
promoter, that a sensor cannot depend on tomorrow's reading — telling
the learner is close to free, and here it is the difference between a
sixth of a second and four hundredths of one.
"""

from __future__ import annotations

# sphinx_gallery_start_ignore
# The docs build runs this file from its own directory, without `__file__`.
if "__file__" not in globals():
    __file__ = "benchmark_influence.py"
# sphinx_gallery_end_ignore

import time
from pathlib import Path

from _bnet import build_dataset_arguments, influences

from pylfit import GULA, PRIDE, Dataset

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "data"
    / "bnet"
    / "faure_cellcycle.bnet"
)

#: Config name -> how to build `allowed_conditions` from the network
#: path. `None` means unrestricted.
CONFIGS = {
    "raw (nothing told to the learner)": None,
    "regulators only": lambda path: influences(path, with_values=False),
    "regulators and sign": lambda path: influences(path, with_values=True),
}


def main() -> None:
    """Fit each configuration, time it, and check they all agree."""
    table, features, targets = build_dataset_arguments(DATA_PATH)
    dataset = Dataset(table, features=features, targets=targets)
    # The first fit on a dataset also converts its rows, once, for every
    # fit after it. Paid here, untimed, it doesn't count against "raw".
    PRIDE.fit(dataset)

    print(f"Network: {DATA_PATH.name} ({len(features)} variables, {len(table)} rows)\n")

    baseline_rules = None
    baseline_seconds = None

    for name, build_allowed in CONFIGS.items():
        allowed = build_allowed(DATA_PATH) if build_allowed else None

        started = time.perf_counter()
        rules = GULA.fit(dataset, allowed_conditions=allowed)
        elapsed = time.perf_counter() - started

        if baseline_rules is None:
            baseline_rules, baseline_seconds = rules, elapsed
            speedup = "baseline"
        else:
            speedup = f"{baseline_seconds / elapsed:.1f}x faster"

        print(f"{name}:\n  {elapsed:7.3f}s   {len(rules)} rules   {speedup}")

        # Exact background knowledge cannot cost an answer, so this is
        # equality rather than containment. A mismatch means either the
        # restriction is wrong or the knowledge was not exact.
        assert rules == baseline_rules, (
            f"{name} changed the answer: "
            f"{len(set(baseline_rules) - set(rules))} rule(s) lost and "
            f"{len(set(rules) - set(baseline_rules))} gained against the "
            f"unrestricted run. Background knowledge read from the network "
            f"that generated the data should restrict the search without "
            f"changing what it finds."
        )

    print()
    print(f"All three configurations agree exactly: {len(baseline_rules)} rules.")
    print("Only the search cost changed.")


if __name__ == "__main__":
    main()
