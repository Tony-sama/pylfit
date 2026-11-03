# Check learned rules against the true ones

When the system behind the data is known, as with a published gene
network or data you generated, you can grade the rules themselves, not
only their predictions: [`rule_recovery`](../api/metrics.md) says how
many of the true rules were found, and how many learned rules are
wrong.

## Get the true rules

`rule_recovery` needs every minimal rule of the system, which `GULA`
learns from its complete table. Take a three-gene network: `a` copies
`b`, `b` is on when `a` and `c` both are, and `c` is the opposite of
`a`.

```pycon
>>> import itertools
>>> import pandas as pd
>>> from pylfit import Dataset, GULA, rule_recovery
>>> def step(a, b, c):
...     return {"a_t+1": b, "b_t+1": str(int(a == c == "1")), "c_t+1": str(1 - int(a))}
>>> table = pd.DataFrame(
...     [
...         {"a_t": a, "b_t": b, "c_t": c, **step(a, b, c)}
...         for a, b, c in itertools.product("01", repeat=3)
...     ]
... )
>>> features, targets = ["a_t", "b_t", "c_t"], ["a_t+1", "b_t+1", "c_t+1"]
>>> true_rules = GULA.fit(Dataset(table, features=features, targets=targets))
>>> len(true_rules)
7

```

## Compare the rule sets

Learn from half of the eight states, as if the others had never been
observed, and compare:

```pycon
>>> seen = table.iloc[[0, 3, 5, 6]]
>>> dataset = Dataset(seen, features=features, targets=targets)
>>> learned = GULA.fit(dataset, min_support=1)
>>> scores = rule_recovery(true_rules, learned)
>>> len(learned), round(scores["precision"], 2), round(scores["recall"], 2)
(18, 0.39, 1.0)
>>> {name: round(share, 2) for name, share in scores["learned"].items()}
{'exact': 0.39, 'too_specific': 0.0, 'too_general': 0.0, 'partial': 0.61, 'spurious': 0.0}

```

Every true rule was found (recall 1), but only 7 of the 18 learned
rules are right (precision 0.39). The other 11 are `"partial"`: each
overlaps a true rule without being one, and matches some state nobody
observed where its value is wrong. The
[`rule_recovery`](../api/metrics.md) reference defines each category.

## Compare at the unseen states

With `states`, the comparison is made at each state, between the
rules that match it, so it grades the rules that would vote on those
rows:

```pycon
>>> unseen = table.drop(index=[0, 3, 5, 6])[features]
>>> scores = rule_recovery(true_rules, learned, unseen)
>>> round(scores["precision"], 2), round(scores["learned"]["spurious"], 2)
(0.51, 0.46)

```

At those states almost half of the votes come from `"spurious"` rules,
which conclude a value the system never gives there. [The dynamics
example](../auto_examples/dynamics.rst) runs the same comparison on a
10-variable cell-cycle model.
