# Learn from state transitions

A discrete dynamical system, such as a gene regulatory network, is a
set of variables whose next values depend on the current state. pylfit
needs no special API for it: put the state at time `t` in feature
columns and the state at `t+1` in target columns, and each observed
step becomes an ordinary row.

## Turn time series into rows

Take a two-gene network: `a` is on at the next step when `b` is off,
and `b` copies `a`. Here are two series observed from it. Each
consecutive pair of observations in one series is one transition. Pair
observations within a series only: the last observation of one series
and the first of the next are not a transition.

```pycon
>>> import pandas as pd
>>> def transitions(series):
...     now = series.iloc[:-1].reset_index(drop=True).add_suffix("_t")
...     after = series.iloc[1:].reset_index(drop=True).add_suffix("_t+1")
...     return pd.concat([now, after], axis=1)
>>> series = [
...     pd.DataFrame({"a": ["0", "1", "1", "0"], "b": ["0", "0", "1", "1"]}),
...     pd.DataFrame({"a": ["0", "0", "1"], "b": ["1", "0", "0"]}),
... ]
>>> rows = pd.concat([transitions(s) for s in series], ignore_index=True)
>>> len(rows)
5

```

## Learn the rules

Each target column gets the rules that decide that variable's next
value. Here they recover the network exactly:

```pycon
>>> from pylfit import Dataset, GULA
>>> dataset = Dataset(rows, features=["a_t", "b_t"], targets=["a_t+1", "b_t+1"])
>>> rules = GULA.fit(dataset, min_support=1)
>>> for rule in rules:
...     print(rule)
a_t+1(0) :- b_t(1).
a_t+1(1) :- b_t(0).
b_t+1(0) :- a_t(0).
b_t+1(1) :- a_t(1).

```

## Predict the next state

`RuleSetModel.predict_candidates` gives, for each variable, every next
value some rule allows. If every variable updates at once
(synchronous semantics), the next states are every combination of
those values:

```pycon
>>> import itertools
>>> from pylfit import RuleSetModel
>>> model = RuleSetModel(dataset, rules)
>>> allowed = model.predict_candidates({"a_t": "1", "b_t": "0"})
>>> list(itertools.product(allowed["a_t+1"], allowed["b_t+1"]))
[('1', '1')]

```

Updating one variable at a time instead (asynchronous semantics) takes a
few more lines on top of `predict_candidates`. [The dynamics
example](../auto_examples/dynamics.rst) shows both semantics on a
10-variable cell-cycle model, and grades the predicted next states
against the true system.
