# Get reproducible results

This guide lists what can change between two runs of the same code, and
how to fix each one. A typical script shuffles, splits, learns,
predicts and scores; the sections below go through what can vary in
each, then output order and versions.

## Seed the shuffle and the split

Shuffle with a seed, so every run trains and tests on the same rows:

```pycon
>>> import pandas as pd
>>> table = pd.DataFrame({"a": ["0", "1"] * 5, "y": ["0", "1"] * 5})
>>> shuffled = table.sample(frac=1, random_state=0)
>>> train, test = shuffled.iloc[:8], shuffled.iloc[8:]

```

## Take the domain from the whole table

A split can leave a value only in the test rows, and `predict` raises
on a value outside the model's domain, so whether the script runs at
all would depend on the split. Set the model's domain from the whole
table: `model.domain = Dataset(table, features=..., targets=...).domains()`,
as [Getting started](../getting-started.md#score-the-model) does.

Set it the same way for every model you compare: `predict_proba`
lists every value of a target's domain, so the domain shapes its
output.

## Pass a seed to predict

When a state's candidate values are equally likely, `predict` picks one
at random. That happens often with `RuleSetModel`, and with
`WeightedRuleSetModel` whenever two values score the same. Pass
`random_state` to make the whole call reproducible:

```pycon
>>> import pandas as pd
>>> from pylfit import Dataset, Rule, RuleSetModel
>>> df = pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"]})
>>> dataset = Dataset(df, features=["a"], targets=["y"])
>>> undecided = RuleSetModel(
...     dataset, [Rule.from_string("y(0) :- ."), Rule.from_string("y(1) :- .")]
... )
>>> states = pd.DataFrame({"a": ["0", "1", "0", "1"]})
>>> undecided.predict(states, random_state=42) == undecided.predict(states, random_state=42)
True

```

The seed fixes the output of one call, on every run and every machine
with the same pylfit version, whatever `PYTHONHASHSEED` is. It doesn't
fix each row on its own: each tie takes the next random draw, so
reordering the rows, or predicting them in separate calls, can change
which value a tied row gets. Keep the rows, their order and the calls
the same.

## Report the ties

A seed makes a tied prediction repeatable, not meaningful. Count the
rows whose top value is tied, and report it with the score. The
DataFrame form of `predict_proba` gives every row at once:

```pycon
>>> def count_ties(model, frame):
...     ties = 0
...     for proba in model.predict_proba(frame):
...         for values in proba.values():
...             best = max(values.values())
...             if sum(v == best for v in values.values()) > 1:
...                 ties += 1
...                 break
...     return ties
>>> count_ties(undecided, states)
4

```

## Learning needs no seed

`GULA`, `PRIDE` and `BruteForce` have no random step. `GULA` and
`BruteForce` learn the same rule set whatever the row order; only
`GULA`'s running time depends on it. `PRIDE` builds one rule per
uncovered positive, taking them in row order, so reordering the rows
can change which rules it returns, even on data containing every
feature state. Keep the row order fixed, or sort the rows first, for
example `df.sort_values(list(df.columns), ignore_index=True)`.

## Keep pylfit's order, not a set's

Everything pylfit returns, prints or saves comes in the same order on
every run: `fit` and `model.rules` give the rules sorted by text, and
`Dataset.domain`, `model.domain` and `predict_candidates` give values
as sorted lists.

```pycon
>>> from pylfit import GULA
>>> [rule.to_string() for rule in GULA.fit(dataset)]
['y(0) :- a(0).', 'y(1) :- a(1).']

```

A `set` you build from them loses that: a `set` of strings iterates in
a different order in each Python process. Use one to compare rule sets,
and sort it before printing or writing it to a file. Don't compare two
lists of rules with `<=`: it compares them element by element, as words
in a dictionary, not as a subset. Write `set(a) <= set(b)`.

## Record the versions

Results can change between pylfit versions. Record the version you
used; saved model files record it for you:

```python
import sys
from importlib.metadata import version

print(version("pylfit"), version("pandas"), version("numpy"), sys.version.split()[0])
```

Pin the versions of pylfit, pandas and numpy in a reproducible project,
for example in a `requirements.txt`.
