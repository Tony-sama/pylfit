# A worked GULA trace

This page follows `GULA` step by step on a three-variable dataset, for
one head. The code below re-implements its loop with the public `Rule`
and `Dataset` API, prints every step, and checks the result against
`GULA.fit`; it runs as part of the test suite, so the trace stays
correct. [About the algorithms](about-algorithms.md) describes the
algorithm in words, and draws this same run as a tree.

## The data

`y` is `1` when `b` is `1`, or when `a` is `0` and `c` is `1`. All
eight states are present.

```pycon
>>> import pandas as pd
>>> from pylfit import GULA, Dataset, Rule
>>> df = pd.DataFrame(
...     [
...         [0, 0, 0, 0],
...         [0, 0, 1, 1],
...         [0, 1, 0, 1],
...         [0, 1, 1, 1],
...         [1, 0, 0, 0],
...         [1, 0, 1, 0],
...         [1, 1, 0, 1],
...         [1, 1, 1, 1],
...     ],
...     columns=["a", "b", "c", "y"],
... )
>>> dataset = Dataset(df, features=["a", "b", "c"], targets=["y"])

```

The negative examples of `y(1)` are the three states where `y` is `0`:

```pycon
>>> ["".join(n[f] for f in "abc") for n in dataset.negatives("y", "1")]
['000', '100', '101']

```

## The loop

Start from the rule with an empty body. For each negative example,
take out every rule that matches it, and replace each with its least
specializations: one more condition, on a feature the rule doesn't
constrain, with a value the negative doesn't have. Drop a
specialization that a kept rule already dominates or equals;
`is_more_general_than` checks both, since it includes equality.

```pycon
>>> def show(rules):
...     return ", ".join(sorted(rule.to_string() for rule in rules))
>>> def trace(dataset, target, value):
...     rules = [Rule((target, value))]
...     print("start:", show(rules))
...     for negative in dataset.negatives(target, value):
...         wrong = [rule for rule in rules if rule.matches(negative)]
...         rules = [rule for rule in rules if not rule.matches(negative)]
...         for rule in wrong:
...             for feature in dataset.features:
...                 if rule.has_condition(feature):
...                     continue
...                 for v in dataset.domain(feature):
...                     if v == negative[feature]:
...                         continue
...                     candidate = rule.with_condition(feature, v)
...                     if not any(kept.is_more_general_than(candidate) for kept in rules):
...                         rules.append(candidate)
...         state = "".join(negative[f] for f in dataset.features)
...         print(f"after {state}:", show(rules))
...     return set(rules)
>>> learned = trace(dataset, "y", "1")
start: y(1) :- .
after 000: y(1) :- a(1)., y(1) :- b(1)., y(1) :- c(1).
after 100: y(1) :- b(1)., y(1) :- c(1).
after 101: y(1) :- a(0), c(1)., y(1) :- b(1).

```

## What happened

1. The empty rule matches `000`, so it is replaced by its three least
   specializations, one per feature, each with the value `000` doesn't
   have.
2. `y(1) :- a(1).` matches `100`. Its specializations,
   `y(1) :- a(1), b(1).` and `y(1) :- a(1), c(1).`, are dominated by
   the kept `y(1) :- b(1).` and `y(1) :- c(1).`, so both are dropped.
3. `y(1) :- c(1).` matches `101`. Of its specializations, only
   `y(1) :- a(0), c(1).` survives: `y(1) :- b(1), c(1).` is dominated by
   `y(1) :- b(1).`.

The two rules left are the definition of `y(1)` the data was built
from, and exactly what `GULA.fit` learns for that head:

```pycon
>>> learned == {rule for rule in GULA.fit(dataset) if rule.head == ("y", "1")}
True

```

`GULA` itself differs only in bookkeeping: it processes each distinct
negative once, in its own order (the result doesn't change), checks a
specialization only against the kept rules holding the condition it
adds, and can track each rule's support for `min_support` and
`min_possible_support`.
