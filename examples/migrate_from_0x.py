"""Port pylfit 0.x code to 1.0
===========================

Porting pylfit 0.x code to 1.0, step by step.

Run it:

    python3 examples/migrate_from_0x.py

The two examples 0.x shipped with — learning a state-transition model
with GULA or PRIDE and predicting under three semantics, then the
weighted variant with its explanations — rewritten for 1.0. Each step
shows the 0.x call in a comment above its replacement. The guide this
accompanies is ``docs/how-to/migration.md``.

What changes is the shape of the API, not what is learned. 0.x took
state transitions and a model that learned in place
(``DMVLP(...).compile("gula").fit(dataset)``); 1.0 takes a plain table
and returns plain rules. A transition is just a row: the state at ``t``
in feature columns, the state at ``t+1`` in target columns.

GULA's rules, all three semantics' next states and the weighted
model's most likely values are asserted against what pylfit 0.5.1
itself printed on the same data, recorded below as ``EXPECTED_*``, and
come out identical. PRIDE's rules are checked only to be rules GULA
also found: 0.5.1's PRIDE walks Python sets, so its output changes with
the hash seed and there is no one output to compare. The counts and the
weighted scores are printed, not asserted. The weighted model is the
one place the two versions genuinely differ, and that section says
how.

0.x needed no helper functions; 1.0 needs three short ones for the
dynamics-specific parts it no longer provides
(``transitions_frame``, ``feature_states``, ``next_states``).
Copy them into your own code.
"""

from __future__ import annotations

# sphinx_gallery_start_ignore
# The docs build runs this file from its own directory, without `__file__`.
if "__file__" not in globals():
    __file__ = "migrate_from_0x.py"
# sphinx_gallery_end_ignore

import itertools
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from pylfit import GULA, PRIDE, Dataset, RuleSetModel, WeightedRuleSetModel

State = tuple[str, ...]

#: The data from 0.x's own example: three Boolean variables, every
#: state observed once, each paired with its next state.
TRANSITIONS = [
    (["0", "0", "0"], ["0", "0", "1"]),
    (["1", "0", "0"], ["0", "0", "0"]),
    (["0", "1", "0"], ["1", "0", "1"]),
    (["0", "0", "1"], ["0", "0", "1"]),
    (["1", "1", "0"], ["1", "0", "0"]),
    (["1", "0", "1"], ["0", "1", "0"]),
    (["0", "1", "1"], ["1", "0", "1"]),
    (["1", "1", "1"], ["1", "1", "0"]),
]
FEATURES = ["p_t_1", "q_t_1", "r_t_1"]
TARGETS = ["p_t", "q_t", "r_t"]

#: What pylfit 0.5.1 printed for ``DMVLP`` + ``compile("gula")``.
EXPECTED_GULA_RULES = {
    "p_t(0) :- q_t_1(0).",
    "p_t(1) :- q_t_1(1).",
    "q_t(0) :- p_t_1(0).",
    "q_t(0) :- r_t_1(0).",
    "q_t(1) :- p_t_1(1), r_t_1(1).",
    "r_t(0) :- p_t_1(1).",
    "r_t(1) :- p_t_1(0).",
}

#: What pylfit 0.5.1's ``model.predict([("1", "0", "1")], semantics=...)``
#: returned, per semantics.
EXPECTED_NEXT_STATES = {
    "synchronous": {("0", "1", "0")},
    "asynchronous": {("0", "0", "1"), ("1", "1", "1"), ("1", "0", "0")},
    "general": {
        ("1", "1", "1"),
        ("1", "1", "0"),
        ("1", "0", "1"),
        ("1", "0", "0"),
        ("0", "1", "1"),
        ("0", "1", "0"),
        ("0", "0", "1"),
        ("0", "0", "0"),
    },
}


def rule_strings(rules: Iterable) -> set[str]:
    """Rules as text, conditions sorted, so two rule sets compare as sets."""
    out = set()
    for rule in rules:
        head, _, body = rule.to_string().rstrip(".").partition(" :- ")
        conditions = sorted(body.split(", ")) if body else []
        out.add(head + (" :- " + ", ".join(conditions) if conditions else "") + ".")
    return out


# %%
# The three helpers: what 0.x did for you that 1.0 leaves to you
# --------------------------------------------------------------


def transitions_frame(
    data: Sequence[tuple[Sequence[str], Sequence[str]]],
    feature_names: list[str],
    target_names: list[str],
) -> pd.DataFrame:
    """Replaces 0.x's ``discrete_state_transitions_dataset_from_array``.

    Wrap the result in ``Dataset(...)`` as usual.

    Each ``(state, next_state)`` pair becomes one row: the state in the
    feature columns, the next state in the target columns. 0.x's
    ``..._from_csv`` counterpart is ``pd.read_csv(path, dtype=str)``,
    which already gives this table.
    """
    rows = [list(state) + list(next_state) for state, next_state in data]
    return pd.DataFrame(rows, columns=feature_names + target_names)


def feature_states(dataset: Dataset) -> list[State]:
    """Replaces ``model.feature_states()``: every combination of known values."""
    domains = [dataset.domain(f) for f in dataset.features]
    return list(itertools.product(*domains))


def next_states(model: RuleSetModel, state: State, semantics: str) -> set[State]:
    """Replaces ``model.predict([state], semantics=...)`` for one state.

    As in 0.x, target ``i`` is the next value of feature ``i``: the
    lists are paired by position. Per variable, the candidates are the
    values some matching rule concludes; the semantics decides how
    they combine:

    - ``"synchronous"``: every variable updates at once — every
      combination of the candidates.
    - ``"asynchronous"``: exactly one variable changes to a candidate;
      a state where none can change is its own successor.
    - ``"general"``: each variable either takes a candidate or keeps
      its current value, in any combination (so ``state`` itself is
      always among the results).

    One difference: a variable no rule matches gets ``"?"`` (or your
    ``default``) in 0.x, but its whole known domain in 1.0 — see
    ``RuleSetModel.predict_candidates``. With rules from ``GULA`` at its
    default ``min_support``, some rule always matches, so that case does
    not arise.
    """
    allowed = model.predict_candidates(dict(zip(model.features, state, strict=True)))
    candidates = [allowed[t] for t in model.targets]
    if semantics == "synchronous":
        return set(itertools.product(*candidates))
    if semantics == "asynchronous":
        successors = {
            state[:i] + (value,) + state[i + 1 :]
            for i, values in enumerate(candidates)
            for value in values
            if value != state[i]
        }
        return successors or {state}
    if semantics == "general":
        options = [{*values, state[i]} for i, values in enumerate(candidates)]
        return set(itertools.product(*options))
    raise ValueError(f"unknown semantics: {semantics!r}")


# %%
# 1. Load the transitions
# -----------------------

# 0.x:
#   dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(
#       data=data, feature_names=[...], target_names=[...])
#   dataset.summary()
df = transitions_frame(TRANSITIONS, FEATURES, TARGETS)
dataset = Dataset(df, features=FEATURES, targets=TARGETS)
print(df)

# %%
# 2. Learn with GULA
# ------------------

# 0.x:
#   model = pylfit.models.DMVLP(features=dataset.features, targets=dataset.targets)
#   model.compile(algorithm="gula")
#   model.fit(dataset=dataset)
#   model.summary()
model = RuleSetModel(dataset, GULA.fit(dataset))
print(model)

assert rule_strings(model.rules) == EXPECTED_GULA_RULES, "GULA differs from 0.5.1"
print(f"\nSame {len(EXPECTED_GULA_RULES)} rules as pylfit 0.5.1.")

# %%
# 3. Learn with PRIDE
# -------------------

# 0.x: model.compile(algorithm="pride") instead of "gula".
#
# PRIDE is a covering heuristic: it returns *a* set of rules that
# explains every observed transition and contradicts none, not *the*
# set. 1.0's PRIDE is a reimplementation, so on larger data it can pick
# different rules than 0.x did — equally valid, but not identical.
# What holds on any data is that each of its rules is one GULA also
# finds; on this small example it happens to return all of them.
pride_model = RuleSetModel(dataset, PRIDE.fit(dataset))
print(pride_model)
assert rule_strings(pride_model.rules) <= EXPECTED_GULA_RULES, "PRIDE invented a rule"

# %%
# 4. Predict next states under each semantics
# -------------------------------------------

# 0.x:
#   prediction = model.predict([state], semantics="synchronous")
#   print([s for s in prediction[state]])
state = ("1", "0", "1")
for semantics, expected in EXPECTED_NEXT_STATES.items():
    successors = next_states(model, state, semantics)
    print(f"{semantics:>12}: {sorted(successors)}")
    assert successors == expected, f"{semantics} differs from 0.5.1"
print("\nSame next states as pylfit 0.5.1 under all three semantics.")

# %%
# 5. Every transition the model allows, saved to CSV
# --------------------------------------------------

# 0.x:
#   prediction = model.predict(model.feature_states())
#   transitions = [(s1, s2) for s1 in prediction for s2 in prediction[s1]]
#   dataset = ...dataset_from_array(data=transitions, ...)
#   dataset.to_csv("tmp/output.csv")
transitions = [
    (s1, s2)
    for s1 in feature_states(dataset)
    for s2 in sorted(next_states(model, s1, "synchronous"))
]
with tempfile.TemporaryDirectory() as folder:
    output_path = Path(folder) / "transitions.csv"
    transitions_frame(transitions, FEATURES, TARGETS).to_csv(output_path, index=False)
    written = pd.read_csv(output_path, dtype=str)
print(f"{len(written)} transitions written to CSV and read back")

# %%
# 6. The weighted model, and how it differs
# -----------------------------------------

# 0.x's weighted example: state 000 observed twice with two different
# next states, so the system is no longer deterministic (and 101 is not
# observed at all). Saying so with assume_nondeterministic=True
# silences the warning Dataset gives on conflicting rows; without
# unknown values it changes nothing else.
weighted_df = transitions_frame(
    [(["0", "0", "0"], ["1", "0", "0"])]
    + [pair for pair in TRANSITIONS if pair[0] != ["1", "0", "1"]],
    FEATURES,
    TARGETS,
)
weighted_dataset = Dataset(
    weighted_df, features=FEATURES, targets=TARGETS, assume_nondeterministic=True
)

# 0.x:
#   model = pylfit.models.WDMVLP(features=..., targets=...)
#   model.compile(algorithm="gula")
#   model.fit(dataset=dataset)
#   prediction = model.predict([state])
#   proba, (likely_weight, likely_rule), (unlikely_weight, unlikely_rule) = \
#       prediction[state][variable][value]
weighted = WeightedRuleSetModel(weighted_dataset, GULA.fit(weighted_dataset))

# 0.x scored each value with two rule sets, rules for it ("likeliness")
# and rules against it ("unlikeliness"), and reported
# best_likely / (best_likely + best_unlikely) as a probability. 1.0 has
# no rules against a value. Each rule carries (positives, negatives)
# counts instead — the rows it explains and the rows it gets wrong —
# and `predict_likelihood` combines the matching rules' counts into one
# score per value. The rules are 0.x's likeliness rules, and `positives`
# is 0.x's likeliness weight, except that 1.0 counts rows where 0.x
# counted distinct states.
#
# So the scores are not 0.x's probabilities. What carries over is which
# value comes out on top.
likelihood = weighted.predict_likelihood(dict(zip(FEATURES, state, strict=True)))
for target in TARGETS:
    print(f"{target}:")
    for value, score in sorted(likelihood[target].items()):
        # 0.x's `likely_rule`: the matching rule for this value with the
        # most support.
        supporting = [
            rule
            for rule in weighted.rules
            if rule.head == (target, value)
            and rule.matches(dict(zip(FEATURES, state, strict=True)))
        ]
        best = max(supporting, key=lambda r: weighted.get_score(r)[0])
        positives, negatives = weighted.get_score(best)
        print(
            f"  {value}: {score:.2f}  best rule: {best.to_string()}"
            f"  (rows: {positives} for, {negatives} against)"
        )

# 0.5.1 gave value 0 the highest probability for all three variables
# (0.75 against 0.5, 0.25 and 0.25).
for target in TARGETS:
    scores = likelihood[target]
    assert max(scores, key=scores.get) == "0", f"{target}: top value moved"
print("\nSame most likely value as pylfit 0.5.1 for every variable.")

# Where they part: state 000 was observed with both next values of p_t,
# and of r_t. 0.x counts one state for each value and calls it a tie
# (probability 1.0 each). 1.0 scores the rules matching 000, and those
# rules also explain other rows, so one value comes out ahead.
at_000 = weighted.predict_likelihood(dict(zip(FEATURES, ("0", "0", "0"), strict=True)))
print("\nAt 000, seen with both next values of p_t and r_t (0.x: 1.0 each, a tie):")
for target in ("p_t", "r_t"):
    scores = ", ".join(f"{v}: {s:.2f}" for v, s in sorted(at_000[target].items()))
    print(f"  {target}: {scores}")

# %%
# 7. What has no 1.0 equivalent
# -----------------------------

print(
    "CDMVLP, PDMVLP, CLP, Synchronizer, Probalizer, ACEDIA, LUST, LFkT,\n"
    "counterfactuals and continuous datasets, among others, are not in\n"
    "1.0; the migration guide lists them all. Code that needs them stays\n"
    'on 0.x: pip install "pylfit<1" numpy pandas'
)
