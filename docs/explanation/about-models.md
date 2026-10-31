# About the prediction models

A learned rule set says which values are possible in a state, not which
one to predict. A model turns rules into predictions, and pylfit keeps
that step separate from learning: any algorithm's rules go into any
model, and choosing how to predict is a decision of its own. This page
explains the two models.

## RuleSetModel: every match counts equally

`RuleSetModel` treats a matching rule as "this value is possible", no
more. `predict_candidates` returns, for each target, every value some
matching rule concludes; how many rules agree is not counted.

A target that no rule matches gets its whole domain, not an empty set:
nothing in the data ruled any value out, so every value stays possible.

```pycon
>>> import pandas as pd
>>> from pylfit import Dataset, Rule, RuleSetModel
>>> df = pd.DataFrame({"a": ["0", "1", "2"], "y": ["0", "1", "2"]})
>>> dataset = Dataset(df, features=["a"], targets=["y"])
>>> model = RuleSetModel(dataset, [Rule.from_string("y(1) :- a(1).")])
>>> model.predict_candidates({"a": "0"})["y"]
['0', '1', '2']

```

`predict_proba` splits probability evenly over the candidates, giving
`0.0` to every other value of the domain, and `predict` picks the most
probable value.

## Ties are broken at random

When several values share the top probability, `predict` picks one
uniformly at random, not the first in alphabetical order. Picking the
first would favor one class every time, a bias that looks like a
result. With `RuleSetModel`, a state with several candidates is always
a tie, so the full rule set, which often matches several values, can
predict close to randomly: on the car evaluation data, 0.23 accuracy
against 0.92 for the same rules in `WeightedRuleSetModel`
([the benchmark_classic example](../auto_examples/benchmark_classic.rst)).

Pass `random_state` to make the draw reproducible; see
[Get reproducible results](../how-to/reproducible-results.md).

## WeightedRuleSetModel: weigh rules by support

`WeightedRuleSetModel` scores each rule against a dataset: positives,
the rows its body matches that have its head value, and negatives, the
rows its body matches that have another value. Predictions weigh
matching rules by those counts.

Each matching rule votes `p * p / (p + n)` for its value, from its
positives `p` and negatives `n`: support times precision. The votes of
a value's rules add up. A rule without negatives votes its positives,
so on deterministic data, where the rules `GULA`, `PRIDE` and
`BruteForce` learn have no negatives, a value's score is its matching
rules' pooled positives. That total counts a row once per matching
rule: rules of one value that match the same rows add those rows
again, so a redundant rule, or a feature copied into a second column,
raises its value's score.

On the table from [About the algorithms](about-algorithms.md), with
half its rows hidden, the vote predicts three of the four. Step through
it with Next:

````{container} pylfit-stepper
```{figure} ../images/vote-1.svg
---
alt: >-
  The eight rows of the table. Rows 011 and 111, where y is 1, and
  rows 100 and 101, where y is 0, are observed. Rows 000, 001, 010
  and 110 are hidden. The rules learned are y(1) :- b(1), y(1) :-
  a(0), y(0) :- b(0) and y(0) :- c(0).
---

The table from [About the algorithms](about-algorithms.md), with four
rows hidden. `GULA`, given the four observed rows, learns four rules,
two for each value of `y`.
```

```{figure} ../images/vote-2.svg
---
alt: >-
  The four rules with their weights. y(1) :- b(1) matches 011 and
  111, weight 2. y(1) :- a(0) matches 011, weight 1. y(0) :- b(0)
  matches 100 and 101, weight 2. y(0) :- c(0) matches 100, weight 1.
---

Each rule's weight is its support: the observed rows it matches, all
of its own value. `b(1)` and `b(0)` match two rows each, `a(0)` and
`c(0)` one.
```

```{figure} ../images/vote-3.svg
---
alt: >-
  Row 000 is outlined. The rules matching it are highlighted, y(0)
  :- b(0) with weight 2, y(0) :- c(0) with weight 1, y(1) :- a(0)
  with weight 1. y = 1 gets 1 and y = 0 gets 3, so the prediction is
  0, the hidden value.
---

`000` matches `a(0)` for `1` and `b(0)` and `c(0)` for `0`: 3 against
1, so the model predicts `0`. Right.
```

```{figure} ../images/vote-4.svg
---
alt: >-
  Row 001 is outlined. The rules matching it are highlighted, y(0)
  :- b(0) with weight 2, y(1) :- a(0) with weight 1. y = 1 gets 1
  and y = 0 gets 2, so the prediction is 0, while the hidden value
  is 1.
---

`001` matches `a(0)` for `1` and `b(0)` for `0`: 2 against 1, so the
model predicts `0`. Wrong: the hidden value is `1`.
```

```{figure} ../images/vote-5.svg
---
alt: >-
  Row 010 is outlined. The rules matching it are highlighted, y(0)
  :- c(0) with weight 1, y(1) :- a(0) with weight 1, y(1) :- b(1)
  with weight 2. y = 1 gets 3 and y = 0 gets 1, so the prediction is
  1, the hidden value.
---

`010` matches `b(1)` and `a(0)` for `1` and `c(0)` for `0`: 3 against
1, so the model predicts `1`. Right.
```

```{figure} ../images/vote-6.svg
---
alt: >-
  Row 110 is outlined. The rules matching it are highlighted, y(0)
  :- c(0) with weight 1, y(1) :- b(1) with weight 2. y = 1 gets 2
  and y = 0 gets 1, so the prediction is 1, the hidden value.
---

`110` matches `b(1)` for `1` and `c(0)` for `0`: 2 against 1, so the
model predicts `1`. Right.
```

```{figure} ../images/vote-7.svg
---
alt: >-
  The four hidden rows with their predictions. 000, 010 and 110 are
  right. 001 is predicted 0 while its hidden value is 1.
---

Three of the four hidden rows are predicted right. `001` is wrong:
`b(0)`, matching two observed rows, outvotes `a(0)`, matching one.
```

```{figure} ../images/vote-8.svg
---
alt: >-
  The four hidden rows, each marked as a tie between 0 and 1. The
  rules have no weights.
---

The same rules in `RuleSetModel`, which doesn't weigh them: each
hidden row matches a rule for `1` and one for `0`, so all four are
ties, and `predict` picks at random.
```
````

`predict_likelihood` returns the votes, here for `001`:

```pycon
>>> import pandas as pd
>>> from pylfit import GULA, Dataset, WeightedRuleSetModel
>>> df = pd.DataFrame(
...     {
...         "a": ["0", "1", "1", "1"],
...         "b": ["1", "0", "0", "1"],
...         "c": ["1", "0", "1", "1"],
...         "y": ["1", "0", "0", "1"],
...     }
... )
>>> dataset = Dataset(df, features=["a", "b", "c"], targets=["y"])
>>> model = WeightedRuleSetModel(dataset, GULA.fit(dataset))
>>> model.predict_likelihood({"a": "0", "b": "0", "c": "1"})
{'y': {'0': 2.0, '1': 1.0}}

```

Negatives lower a value's score in proportion to its support, so a
large value with a few negatives still outranks a small, perfectly
clean one. A purity measure such as the Laplace estimate
`(p + 1) / (p + n + k)`, the one CN2 (Clark and Niblett, 1989) ranks
its rules by (Clark and Boswell, 1991), does the opposite. Pooled over
a value's matching rules, `p` and `n` their sums and `k` the number of
values, it is blind to how much evidence there is once the rules have
negatives. Rules have negatives whenever the
data conflicts, or when they are made more general than the data
allows, and there the vote predicts better: on a split of the UCI
breast cancer data, `GULA`'s rules score 0.76 under the vote and 0.66
under the Laplace estimate. Where no rule has negatives, the two pick
the same values for a state without unknowns.

`predict_likelihood` returns these raw scores. `predict_proba`
normalizes them, gives `0.0` to every value no matching rule concludes,
and, like `RuleSetModel`, returns a uniform distribution over the
whole domain for a target that no rule with a nonzero score matches.
Its numbers rank the matched values against each other; they are not
calibrated probabilities. A value whose only matching rule is wrong
most of the time still gets `1.0` when no other value matches, which
is why the likelihood, which measures the evidence, is returned too.

## Predicting from unknown values

A state to predict can have unknown values, `None` in a `dict` or `NaN`
in a `DataFrame` (see [About data](about-data.md)). An unknown never
satisfies a condition, so a rule testing it doesn't match: that is the
strict reading, and every model starts with it.

On its own, the strict reading wastes evidence. A row with an unknown
often matches no rule for some target, and that target then gets the
uniform distribution, though some rules could well match. So by
default, `unknowns="fallback"`, a target that no rule matches strictly
is predicted from the rules that *could* match: each of their
conditions holds, or falls on an unknown.

- `WeightedRuleSetModel` weighs each such rule by the probability that
  it matches: its counts are multiplied by the share each value its
  conditions on unknowns require has among the training rows' known
  values (`model.value_frequencies`), taking features as independent.
  The weighted counts then vote as usual, so a rule that almost surely
  doesn't match adds almost nothing.
- `RuleSetModel` has no counts: the candidates are the values of the
  rules that could match.

The strict match comes first because it is the stronger evidence, and
falling back only where it gives nothing keeps every prediction it can
make. Matching partially everywhere was tried and is worse on data
whose rules are reliable, such as Boolean networks: rules that only
could match add their counts to those that do, and dilute them.
Filling each unknown in with its column's most common value is worse
too, the more so when unknowns are more frequent for some outcomes.
Pass `unknowns="strict"` to `predict`, `predict_proba`,
`predict_likelihood` or `predict_candidates` to turn the fallback
off. On a row without unknowns both give the same result.

```pycon
>>> import pandas as pd
>>> from pylfit import Dataset, GULA, WeightedRuleSetModel
>>> df = pd.DataFrame(
...     {
...         "a": ["0", "0", "1", "1"],
...         "b": ["0", "1", "0", "1"],
...         "y": ["0", "1", "1", "1"],
...     }
... )
>>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
>>> model = WeightedRuleSetModel(dataset, GULA.fit(dataset))
>>> rows = pd.DataFrame({"a": [None, None], "b": ["1", "0"]})
>>> model.predict(rows, random_state=0)
[{'y': '1'}, {'y': '1'}]
>>> model.predict_proba(rows, unknowns="strict")[1]
{'y': {'0': 0.5, '1': 0.5}}

```

With `b=1`, `y(1) :- b(1).` matches. With `b=0` no rule does, but
`y(1) :- a(1).` and `y(0) :- a(0), b(0).` each could, half the rows
having each value of `a`. The first has twice the support, so `y=1`
wins; the strict reading can only say it doesn't know.

## References

- Clark, P. & Niblett, T. (1989). The CN2 induction algorithm.
  Machine Learning, 3(4), 261-283.
  <https://doi.org/10.1007/BF00116835>
- Clark, P. & Boswell, R. (1991). Rule induction with CN2: some
  recent improvements. In Machine Learning — EWSL-91, 151-163.
  <https://doi.org/10.1007/BFb0017011>
