# Use your own algorithm

pylfit's models take any collection of `Rule`, whatever produced it.
An algorithm of your own is a function from a `Dataset` to a list of
rules: nothing to subclass, and only the public API to use.

## Write it

`Rule((target, value), {feature: v, ...})` builds a rule from its head
and its body, the conditions; `Rule((target, value))` has an empty
body, and holds in every state. This algorithm learns every consistent
rule with at most one condition:

```pycon
>>> from pylfit import Rule
>>> def short_rules(dataset):
...     rules = set()
...     for target in dataset.targets:
...         for value in dataset.domain(target):
...             negatives = dataset.negatives(target, value)
...             candidates = [Rule((target, value))] + [
...                 Rule((target, value), {feature: v})
...                 for feature in dataset.features
...                 for v in dataset.domain(feature)
...             ]
...             for rule in candidates:
...                 if not any(rule.matches(n) for n in negatives):
...                     rules.add(rule)
...     return sorted(rules)

```

Use `dataset.negatives(target, value)` for the states a rule must not
match, and `dataset.positives(target, value)` for the ones it should
cover. `negatives` is not every row where the target differs: a state
also seen with `value` is left out, so on data with conflicts some rows
are in neither list. Build negatives with it rather than from the rows
outside `positives`. Return the rules sorted, as pylfit's own
algorithms do: a set lists its rules in a different order in each
Python process, so anything printed from it would change from run to
run.

## Use its rules like any others

```pycon
>>> import pandas as pd
>>> from pylfit import Dataset, WeightedRuleSetModel
>>> df = pd.DataFrame(
...     {"a": ["0", "0", "1", "1"], "b": ["0", "1", "0", "1"], "y": ["0", "1", "1", "1"]}
... )
>>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
>>> rules = short_rules(dataset)
>>> [rule.to_string() for rule in rules]
['y(1) :- a(1).', 'y(1) :- b(1).']
>>> model = WeightedRuleSetModel(dataset, rules)
>>> model.predict(pd.DataFrame({"a": ["1"], "b": ["0"]}))
[{'y': '1'}]

```

The metrics, `save` and `from_file` work on the result the same way.

## Read whole rows

`positives` and `negatives` give the feature states for one target
value at a time. When your algorithm needs each row whole, every
feature and target together, iterate the dataset: each row comes as a
`pandas.Series`, with `None` where a value is unknown. A frequency
count, for example:

```pycon
>>> from collections import Counter
>>> Counter(row["y"] for row in dataset)
Counter({'1': 3, '0': 1})

```

To work on the whole table with pandas instead, call
`dataset.to_dataframe()`: the same values, with `NaN` where one is
unknown.

## Check it against BruteForce

`BruteForce` learns every correct, minimal rule, exhaustively. Use it
as the reference on small data, and check what your algorithm claims,
nothing more. `short_rules` claims correctness: none of its rules
matches a state where the target has another value.

```pycon
>>> all(not rule.matches(n) for rule in rules for n in dataset.negatives(*rule.head))
True

```

Comparing with `BruteForce` shows what it leaves out. Compare as sets:
`<=` between two lists compares them element by element, as words in a
dictionary, not as a subset. Don't test
`set(rules) <= set(BruteForce.fit(dataset))` unless your algorithm
claims minimality too: that test also fails on a correct rule some
shorter rule dominates.

```pycon
>>> from pylfit import BruteForce
>>> sorted(rule.to_string() for rule in set(BruteForce.fit(dataset)) - set(rules))
['y(0) :- a(0), b(0).']

```

It is not complete: `y(0)` needs two conditions, so `short_rules`
learns nothing for it. Nor are its rules minimal in general: a
one-condition rule can be dominated by the empty rule, and this
version keeps both.

To add an algorithm to pylfit itself, see [Add an algorithm to
pylfit](../contributing/adding-an-algorithm.md).
