# Learn a gene regulatory network

This tutorial learns how three genes regulate each other from time
series of their expression, simulates the learned network, and shows
how to tell what the data settles from what it leaves open. It is
written for a biologist new to pylfit; you need only a little Python.
Work through [Getting started](../getting-started.md) first if you can.

## The idea

Model each gene as on (`1`) or off (`0`). The state of the cell is the
list of every gene's value, and in each time step every gene's next
value depends on the current state: this is a Boolean network. pylfit
learns, for each gene, rules that say when it is on or off at the next
step, from observed transitions between states.

A state is written as its genes' values in the order A, B, C: `110`
means A and B on, C off. Measured expression levels become on and off
with a threshold per gene, chosen from the biology or from the data;
[Prepare your
data](../how-to/prepare-data.md#make-every-value-a-category) shows
how.

## The observations

Two experiments measured genes A, B and C at successive time points.
Each is one time course, started from one condition of the cells and
sampled at regular intervals. They are in one table, with a row per
experiment and time point:

```pycon
>>> import pandas as pd
>>> measurements = pd.DataFrame(
...     {
...         "experiment": ["e1"] * 5 + ["e2"] * 3,
...         "time": [0, 1, 2, 3, 4, 0, 1, 2],
...         "A": [0, 1, 1, 1, 0, 1, 0, 0],
...         "B": [0, 0, 1, 1, 1, 1, 1, 0],
...         "C": [0, 0, 0, 1, 1, 1, 1, 1],
...     }
... )

```

```{figure} ../images/gene-network-measurements.svg
---
alt: >-
  Experiment e1, five time points, and experiment e2, three. Each
  column gives A, B and C at one time point, 1 for on and 0 for off.
---

The table, one column per row: genes down, time across.
```

## From time series to transitions

A transition pairs one time point with the next. Put the current state
in columns ending `_t` and the next state in columns ending `_t+1`; the
suffixes are only names. Pair time points within one experiment only,
in time order: the end of one experiment and the start of another are
not a transition.

```pycon
>>> genes = ["A", "B", "C"]
>>> def transitions(series):
...     now = series.iloc[:-1].reset_index(drop=True).add_suffix("_t")
...     after = series.iloc[1:].reset_index(drop=True).add_suffix("_t+1")
...     return pd.concat([now, after], axis=1)
>>> def all_transitions(measurements):
...     parts = [
...         transitions(experiment.sort_values("time")[genes])
...         for _, experiment in measurements.groupby("experiment")
...     ]
...     return pd.concat(parts, ignore_index=True)
>>> rows = all_transitions(measurements)
>>> len(rows)
6

```

```{figure} ../images/gene-network-transitions.svg
---
alt: >-
  Six transitions, each pairing A, B and C at one time point with
  their values at the next. The end of e1 and the start of e2 are not
  paired.
---

Each time point paired with the next, within one experiment: four
transitions from e1, two from e2.
```

## Check what the data covers

The rules can only be as sure as the states they were learned from,
and a state teaches something only when the next one was measured:

```{figure} ../images/gene-network-observations.svg
---
alt: >-
  Experiment e1 goes 000, 100, 110, 111, 011. Experiment e2 goes
  111, 011, 001 and ends there. States 010 and 101 were never observed.
---

The two experiments, one state per time point.
```

The end of e1 is no gap: e2 passes through `011` and shows what
follows. But e2 ends at `001`, so nothing shows what follows `001`,
and `010` and `101` were never measured at all. Count the distinct
states a transition starts from, out of the `2**3 = 8` possible:

```pycon
>>> current = [f"{gene}_t" for gene in genes]
>>> len(rows[current].drop_duplicates())
5

```

At the three others, `001`, `010` and `101`, the rules predict with
no observation behind them: they may allow several next states, or
name one the network doesn't take. The simulation in [Simulate the
network](#simulate-the-network) reaches `001`.

## Learn the regulation

The current state is the input and the next state the output, so the
`_t` columns are the features and the `_t+1` columns the targets.
`GULA` learns every rule the data doesn't contradict, and
`min_support=1` keeps only rules that some observation backs, which
makes the rules readable; the simulation below shows what it gives up.

```pycon
>>> from pylfit import Dataset, GULA
>>> following = [f"{gene}_t+1" for gene in genes]
>>> dataset = Dataset(rows, features=current, targets=following)
>>> rules = GULA.fit(dataset, min_support=1)
>>> for rule in rules:
...     print(rule)
A_t+1(0) :- A_t(0), B_t(1).
A_t+1(0) :- C_t(1).
A_t+1(1) :- B_t(0).
A_t+1(1) :- C_t(0).
B_t+1(0) :- A_t(0).
B_t+1(1) :- A_t(1).
B_t+1(1) :- B_t(1), C_t(0).
C_t+1(0) :- A_t(0), C_t(0).
C_t+1(0) :- B_t(0).
C_t+1(1) :- B_t(1).
C_t+1(1) :- C_t(1).

```

Read `A_t+1(1) :- C_t(0).` as "A is on at the next step when C is off
now". To read regulation, start with the rules that have one condition:

- A pair like `B_t+1(1) :- A_t(1).` and `B_t+1(0) :- A_t(0).`, where the
  gene follows the regulator, means activation: A activates B.
- A pair where the values are reversed, `A_t+1(1) :- C_t(0).` and
  `A_t+1(0) :- C_t(1).`, means repression: C represses A.

Read rules with several conditions as a group. If a gene had the rules
`C_t+1(1) :- A_t(1), B_t(1).`, `C_t+1(0) :- A_t(0).` and
`C_t+1(0) :- B_t(0).`, together they say that C turns on only when both
A and B are on: A and B jointly activate C. A rule with several
conditions and no such partners holds in the states observed, but names
no single regulator; treat it as what the data allows so far. A rule
with one condition and no reversed partner is read the same way. For
A, `A_t+1(1) :- B_t(0).` fits the observations as well as the rule on
C, but its partner `A_t+1(0) :- B_t(1).` is missing: in state 110, B
was on and A was on at the next step. The data names C as A's repressor;
the rule on B holds in the states observed, but names no regulator.
`C_t+1(1) :- C_t(1).` is read the same way: it could be C maintaining
itself, but without its reversed partner the data can't tell yet.

```{figure} ../images/gene-network-regulation.svg
---
alt: >-
  Three genes. A activates B, B activates C and C represses A, each
  drawn solid. A dashed edge from B to A is marked "represses?", and
  a dashed loop on C "maintains itself?".
---

The regulation the rules name, from the two experiments.
```

An edge says that in these data one gene's value follows another's one
step later. It is a hypothesis to test, not a mechanism: the effect
may be indirect, through genes that weren't measured.

## Simulate the network

To simulate, ask which next values each gene can take, and update every
gene at once. The next states are every combination of those values, so
a state where a gene has several possible values has several next
states:

```pycon
>>> import itertools
>>> from pylfit import RuleSetModel
>>> def next_states(model, state):
...     now = {f"{gene}_t": value for gene, value in zip(genes, state)}
...     allowed = model.predict_candidates(now)
...     choices = [sorted(allowed[f"{gene}_t+1"]) for gene in genes]
...     return ["".join(values) for values in itertools.product(*choices)]
>>> def follow(model, state):
...     path = [state]
...     while True:
...         possible = next_states(model, state)
...         if len(possible) > 1:
...             return path, possible
...         state = possible[0]
...         if state in path:
...             return path + [state], []
...         path.append(state)
>>> model = RuleSetModel(dataset, rules)
>>> follow(model, "000")
(['000', '100', '110', '111', '011', '001'], ['000', '001', '100', '101'])

```

From `000`, the path is certain for five steps, then reaches `001`,
where experiment e2 ended. There, the rules that observations back
allow four next states. Picking one would be a guess: always check for
several possible next states before you trust a simulated trajectory.

```{figure} ../images/gene-network-path.svg
---
alt: >-
  The path 000, 100, 110, 111, 011, 001, one next state at each
  step, then four dashed arrows from 001 to 000, 001, 100 and 101.
---

The simulated path from `000`, every gene updated at once.
```

Four is still a narrowing. `min_support=1` dropped the rules no
observation backs, and at a state no experiment started from, those
rules add possibilities: with the complete rule set, every gene is open
at `001`, so all eight states fit the data. Learn without `min_support`
when you need every behavior the data leaves open, for example every
possible next state:

```pycon
>>> complete = RuleSetModel(dataset, GULA.fit(dataset))
>>> len(next_states(complete, "001"))
8

```

`WeightedRuleSetModel` can rank the possibilities by how many
observations back each rule, but here the ranking is weak:

```pycon
>>> from pylfit import WeightedRuleSetModel
>>> weighted = WeightedRuleSetModel(dataset, rules)
>>> proba = weighted.predict_proba({"A_t": "0", "B_t": "0", "C_t": "1"})
>>> round(proba["A_t+1"]["0"], 2), round(proba["A_t+1"]["1"], 2)
(0.6, 0.4)

```

### One gene at a time

Updating every gene at once is one update scheme, the synchronous one.
In the asynchronous one, a single gene changes per step, and any gene
that can change may be the one, so each gives a next state. A state
where no gene can change leads to itself. The rules stay the same:
they say what each gene can become, and the scheme says how many genes
change at once.

```pycon
>>> def next_states_async(model, state):
...     now = {f"{gene}_t": value for gene, value in zip(genes, state)}
...     allowed = model.predict_candidates(now)
...     after = set()
...     for i, gene in enumerate(genes):
...         for value in allowed[f"{gene}_t+1"]:
...             if value != state[i]:
...                 after.add(state[:i] + value + state[i + 1 :])
...     return sorted(after) or [state]
>>> next_states_async(model, "001")
['000', '101']
>>> all(
...     next_states_async(model, state) == next_states(model, state)
...     for state in ["000", "100", "110", "111", "011"]
... )
True

```

Every transition observed here changes exactly one gene, so the data
fits both schemes, and on the states an experiment started from they
agree. They part where the rules leave genes open: at `001`, A and C
can each change, so the synchronous scheme gives four next states,
every combination, and the asynchronous one two, one per gene. Which
scheme fits depends partly on how often you sample: sampled often
enough, at most one gene changes between two samples; sampled
sparsely, several changes fall into one step.

```{figure} ../images/gene-network-trajectory.svg
---
alt: >-
  The path 000, 100, 110, 111, 011, 001, one next state at each
  step. From 001, updating every gene at once gives 000, 001, 100 and
  101; one gene at a time gives 000 and 101.
---

The simulated path from `000`, under both schemes.
```

## Run the experiment the data asks for

The simulation names the missing observation: start an experiment from
`001`, with A and B off and C on, for example by silencing A and B and
inducing C. Add it, and learn again:

```pycon
>>> third = pd.DataFrame(
...     {
...         "experiment": ["e3"] * 3,
...         "time": [0, 1, 2],
...         "A": [0, 0, 1],
...         "B": [0, 0, 0],
...         "C": [1, 0, 0],
...     }
... )
>>> rows = all_transitions(pd.concat([measurements, third], ignore_index=True))
>>> dataset = Dataset(rows, features=current, targets=following)
>>> rules = GULA.fit(dataset, min_support=1)
>>> sorted(
...     rule.to_string() for rule in rules if rule.head[0] == "A_t+1" and rule.size() == 1
... )
['A_t+1(0) :- C_t(1).', 'A_t+1(1) :- C_t(0).']
>>> follow(RuleSetModel(dataset, rules), "000")
(['000', '100', '110', '111', '011', '001', '000'], [])

```

The new observation rules out B as A's regulator, the dashed edge in
the regulation figure: C represses A, A activates B, and B activates
C. From `000`, the network now cycles through six states and comes
back: an oscillation, as expected of a negative feedback loop, one with
an odd number of repressions, here one.

```{figure} ../images/gene-network-cycle.svg
---
alt: >-
  The six states 000, 100, 110, 111, 011 and 001 in a cycle, each
  with one arrow to the next. The states 010 and 101 stand apart,
  dashed.
---

The path from `000`, after the third experiment.
```

The states `010` and `101` are still unvisited, so what the network
does from there remains open; the experiments that settle it are, again,
ones starting from those states.

## Next steps

- [Learn from state transitions](../how-to/state-transitions.md) covers
  the same steps in brief, for any dynamical system.
- [Check learned rules against the true
  ones](../how-to/check-against-true-rules.md) grades the rules
  themselves when the network is known.
- [About LFIT rules](../explanation/about-rules.md) explains what the
  learned rules guarantee.
- [Learn a dynamical system and simulate
  it](../auto_examples/dynamics.rst) learns a published cell-cycle
  network of 10 variables and grades its predicted next states against
  the real one, state by state.
- [Conflicting
  rows](../explanation/about-data.md#conflicting-rows) covers
  replicates that disagree, and [Get weighted
  predictions](../how-to/weighted-predictions.md) how to rank what they
  allow.
- [Keep the problem a comfortable
  size](../how-to/prepare-data.md#keep-the-problem-a-comfortable-size)
  says how learning time grows with the number of genes, and what keeps
  it short.
