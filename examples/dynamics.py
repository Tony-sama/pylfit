"""Learn a dynamical system and simulate it
========================================

Learning a dynamical system: time series in, simulated trajectories out.

Run it:

    python3 examples/dynamics.py

pylfit's learners work on plain classification data, and that is all a
discrete dynamical system needs. Observe the system's state at time
``t`` and again at ``t+1``, put the first in feature columns and the
second in target columns, and each pair of consecutive observations
becomes an ordinary row: "from this state, each variable takes this
value next". ``GULA`` then learns, per variable, the rules that decide
its next value — the same rules it would learn on any other table.

Using what was learned to *simulate* the system takes one more idea, a
**semantics**: how the per-variable predictions combine into whole next
states. ``RuleSetModel.predict_candidates`` already gives, per target,
every value some matching rule allows; the two classic semantics are a
few lines each on top of it, written out below rather than hidden in
the package:

- **synchronous** — every variable updates at once, so the next states
  are every combination of the per-variable candidates.
- **asynchronous** — one variable updates at a time, so each next state
  differs from the current one in exactly one variable (or the state
  stays put when no variable can change).

The script uses ``faure_cellcycle.bnet``, a published 10-variable model
of the mammalian cell cycle, as the "real" system: it generates a few
short time series from it, learns from those alone, and grades what
the learned model gets right. The guarantees are asserted: every
observed transition is reproduced, an asynchronous step changes at most
one variable, and learning from every state recovers the system
exactly. The tallies in section 5 depend on which states the random
series happened to visit, so they are printed, not asserted.
"""

from __future__ import annotations

# sphinx_gallery_start_ignore
# The docs build runs this file from its own directory, without `__file__`.
if "__file__" not in globals():
    __file__ = "dynamics.py"
# sphinx_gallery_end_ignore

import itertools
import random
from pathlib import Path

import pandas as pd
from _bnet import generate_transitions

from pylfit import GULA, Dataset, RuleSetModel, WeightedRuleSetModel, rule_recovery

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "data"
    / "bnet"
    / "faure_cellcycle.bnet"
)

State = tuple[str, ...]


def transitions_from_series(series: pd.DataFrame) -> pd.DataFrame:
    """Turns one time series into consecutive ``(t, t+1)`` pairs.

    Row ``i`` of the result pairs observation ``i`` (columns suffixed
    ``_t``) with observation ``i + 1`` (suffixed ``_t+1``), so a series
    of ``n`` observations gives ``n - 1`` transitions. Pairs are taken
    *within* one series only: concatenate several series first and the
    last observation of one would be paired with the first of the next,
    a transition that never happened.
    """
    now = series.iloc[:-1].reset_index(drop=True).add_suffix("_t")
    after = series.iloc[1:].reset_index(drop=True).add_suffix("_t+1")
    return pd.concat([now, after], axis=1)


def candidates(
    model: RuleSetModel, variables: list[str], state: State
) -> list[list[str]]:
    """Per variable, the next values the learned rules allow from ``state``."""
    features = {f"{v}_t": value for v, value in zip(variables, state, strict=True)}
    allowed = model.predict_candidates(features)
    return [allowed[f"{v}_t+1"] for v in variables]


def synchronous(model: RuleSetModel, variables: list[str], state: State) -> set[State]:
    """Next states when every variable updates at once."""
    return set(itertools.product(*candidates(model, variables, state)))


def asynchronous(model: RuleSetModel, variables: list[str], state: State) -> set[State]:
    """Next states when exactly one variable updates.

    Each next state changes one variable to a value its rules allow and
    leaves the rest as they are. A state where no variable can change
    is a fixed point: its only successor is itself.
    """
    successors = set()
    for i, allowed in enumerate(candidates(model, variables, state)):
        for value in allowed:
            if value != state[i]:
                successors.add(state[:i] + (value,) + state[i + 1 :])
    return successors or {state}


# %%
# 1. The system, and a few time series observed from it
# -----------------------------------------------------

# The real system's full transition table — 2**10 states, each with
# its true successor — used here only to *generate* observations and,
# at the end, to grade the model. The learner never sees it whole.
truth_table = generate_transitions(DATA_PATH)
features = [c for c in truth_table.columns if c.endswith("_t")]
variables = [c[: -len("_t")] for c in features]
targets = [f"{v}_t+1" for v in variables]
true_next: dict[State, State] = {
    tuple(row[features]): tuple(row[targets]) for _, row in truth_table.iterrows()
}
print(f"{DATA_PATH.name}: {len(variables)} variables, {len(true_next)} states")

# A hundred short series of four observations each, from random
# starting states, following the system's own (synchronous) dynamics.
# In practice these would be read from measurements, one CSV per
# experiment, with one column per variable. Many short series beat a
# few long ones here: from any state, this system reaches one of its
# two attractors, a fixed point or a cycle of 7 states, within nine
# steps (three on average), after which a series only repeats itself.
rng = random.Random(0)
series_list = []
for _ in range(100):
    state = rng.choice(list(true_next))
    observations = [state]
    for _ in range(3):
        state = true_next[state]
        observations.append(state)
    series_list.append(pd.DataFrame(observations, columns=variables))
print(f"observed {len(series_list)} time series, e.g. the first one:")
print(series_list[0].to_string(index=False))

# %%
# 2. Time series -> transitions -> an ordinary Dataset
# ----------------------------------------------------

transitions = pd.concat(
    [transitions_from_series(series) for series in series_list], ignore_index=True
).drop_duplicates()
dataset = Dataset(transitions, features=features, targets=targets)
print(
    f"{len(transitions)} distinct transitions, covering "
    f"{transitions[features].drop_duplicates().shape[0]} of "
    f"{len(true_next)} states"
)
print(transitions.head(3).to_string(index=False))

# %%
# 3. Learn the dynamics with GULA
# -------------------------------

# Nothing dynamics-specific here: the same call as for any table.
# min_support=1 keeps only rules that explain at least one observed
# transition, which is what makes the learned model readable.
model = RuleSetModel(dataset, GULA.fit(dataset, min_support=1))
print(f"{len(model.rules)} rules. The shortest for each next value of CycD:")
for value in ("False", "True"):
    head = [rule for rule in model.rules if rule.head == ("CycD_t+1", value)]
    shortest = min(head, key=lambda rule: (rule.size(), str(rule)))
    print(f"  {shortest}   (one of {len(head)})")

# Every observed transition is reproduced exactly: from an observed
# state, synchronous semantics yields that one observed successor and
# nothing else. Correctness rules out every other value per variable,
# completeness keeps the observed one — so this is a check of what
# GULA guarantees, not a property of this particular network.
for _, row in transitions.iterrows():
    state, observed = tuple(row[features]), tuple(row[targets])
    assert synchronous(model, variables, state) == {observed}
print("every observed transition is reproduced exactly")

# %%
# 4. Simulate: synchronous and asynchronous semantics
# ---------------------------------------------------


def fmt(state: State) -> str:
    return "".join("1" if value == "True" else "0" for value in state)


print("variables:", " ".join(variables))
start = tuple(series_list[0].iloc[0])
print(f"start: {fmt(start)}")

# Synchronous simulation from an observed start, step by step. While
# the trajectory stays on observed states it is a single path; once it
# reaches a state the data never showed, several next values can stay
# possible, and it branches. Section 5 shows how to pick one
# best-supported next state instead.
state, seen = start, [start]
while True:
    successors = synchronous(model, variables, state)
    if len(successors) != 1:
        print(f"  branches into {len(successors)} next states, stopping")
        break
    (state,) = successors
    if state in seen:
        cycle = seen[seen.index(state) :]
        print(f"  revisits a state: a cycle of length {len(cycle)}")
        break
    seen.append(state)
    print(f"  -> {fmt(state)}")

asynchronous_next = asynchronous(model, variables, start)
print(f"asynchronous next states from the start: {len(asynchronous_next)}")
for successor in sorted(asynchronous_next):
    changed = [v for v, a, b in zip(variables, start, successor, strict=True) if a != b]
    print(f"  {fmt(successor)}  ({', '.join(changed) or 'none'} changed)")
assert all(
    sum(a != b for a, b in zip(start, s, strict=True)) <= 1 for s in asynchronous_next
), "an asynchronous step must change at most one variable"

# %%
# 5. How much of the real system did it learn?
# --------------------------------------------

# Graded against the full truth table, state by state — first as
# *possibilities*. On a state it never observed, the model may be
# exact, may hedge (several possible next states, the true one among
# them), or may be wrong (the true next state missing) — partial data
# supports no stronger promise. Even the rules some observation backs
# leave most unseen states open, so it mostly hedges; the complete rule
# set (no min_support) would hedge more, never less.
exact = hedged = wrong = 0
for state, successor in true_next.items():
    predicted = synchronous(model, variables, state)
    if predicted == {successor}:
        exact += 1
    elif successor in predicted:
        hedged += 1
    else:
        wrong += 1
print(f"exact:  {exact:4d} / {len(true_next)} states")
print(f"hedged: {hedged:4d} / {len(true_next)} (true next state among several)")
print(f"wrong:  {wrong:4d} / {len(true_next)} (true next state not predicted)")

# When one next state is needed rather than every possible one, weigh
# the matching rules by how many observations back them and take the
# best-supported value per variable — WeightedRuleSetModel does that,
# with no change to the rules themselves. Graded on unseen states only,
# since the observed ones are already exact.
observed_states = set(map(tuple, transitions[features].values))
unseen = [state for state in true_next if state not in observed_states]
guesses = WeightedRuleSetModel(dataset, model.rules).predict(
    pd.DataFrame(unseen, columns=features), random_state=0
)
variable_hits = state_hits = 0
for state, guess in zip(unseen, guesses, strict=True):
    hits = [
        guess[t] == value for t, value in zip(targets, true_next[state], strict=True)
    ]
    variable_hits += sum(hits)
    state_hits += all(hits)
variable_rate = variable_hits / (len(unseen) * len(targets))
print(f"\nbest single guess on the {len(unseen)} unseen states:")
print(f"  each variable's next value right: {variable_rate:.1%}")
print(f"  whole next state right:           {state_hits / len(unseen):.1%}")

# The rules can be graded too, not only their predictions. Learned from
# every state, GULA's rules are the system's own; rule_recovery compares
# the learned rules with them at each unseen state, between the rules
# that match it. Precision is the share of the rules voting there that
# are correct, recall the share of the true rules found. Every true rule
# is found, but most of the voting rules are wrong: "partial" ones vote
# for the right value with the wrong conditions, "spurious" ones for a
# value the system never takes there. Without min_support, the rules no
# observation backs vote as well.
full = Dataset(truth_table, features=features, targets=targets)
true_rules = GULA.fit(full)
unseen_states = pd.DataFrame(unseen, columns=features)
print("\nthe rules voting at the unseen states, against the true rules:")
for name, rules in (
    ("min_support=1", model.rules),
    ("no min_support", GULA.fit(dataset)),
):
    scores = rule_recovery(true_rules, rules, unseen_states)
    shares = scores["learned"]
    print(
        f"  {name:14}  precision {scores['precision']:.1%}, recall "
        f"{scores['recall']:.1%}; partial {shares['partial']:.1%}, "
        f"spurious {shares['spurious']:.1%}"
    )

# With every state observed, the guarantee from section 3 covers the
# whole system: learning from the full table recovers it exactly.
full_model = RuleSetModel(full, GULA.fit(full, min_support=1))
assert all(
    synchronous(full_model, variables, state) == {successor}
    for state, successor in true_next.items()
)
print("learned from every state instead, the model reproduces the system exactly")
