# Get weighted predictions

`WeightedRuleSetModel` scores each rule by the rows that back it, and
uses those scores to rank the values a state's matching rules allow.
This guide shows how to read the scores and how they combine.

## Build a weighted model

Build it from an algorithm's rules and the dataset they came from:

```pycon
>>> import pandas as pd
>>> from pylfit import Dataset, GULA, WeightedRuleSetModel
>>> df = pd.DataFrame(
...     {
...         "a": ["0", "0", "1", "1", "1"],
...         "b": ["0", "1", "0", "1", "1"],
...         "y": ["0", "1", "1", "1", "1"],
...     }
... )
>>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
>>> model = WeightedRuleSetModel(dataset, GULA.fit(dataset))

```

If you already have a `RuleSetModel`, build the weighted model from its
rules, `WeightedRuleSetModel(dataset, rule_set.rules)`, and assign
`rule_set.domain` to it too if you had widened that one.

## Read a rule's score

Each rule's score is a `(positives, negatives)` pair: the rows its body
matches that have its head value, and those that have another value.

```pycon
>>> for rule in model.rules:
...     print(rule.to_string(), model.get_score(rule))
y(0) :- a(0), b(0). (1, 0)
y(1) :- a(1). (3, 0)
y(1) :- b(1). (3, 0)

```

## Get likelihoods and probabilities

`predict_likelihood` gives each candidate value a raw score;
`predict_proba` normalizes them to sum to 1. Both take one state, as
here, or a `DataFrame` of states:

```pycon
>>> state = {"a": "1", "b": "1"}
>>> model.predict_likelihood(state)
{'y': {'1': 6.0}}
>>> model.predict_proba(state)
{'y': {'0': 0.0, '1': 1.0}}

```

The two answer different questions. The probabilities rank the values
the matching rules conclude against each other, with `0.0` for any
other value: here only rules for `1` match, so `1` gets all of it. They
are not calibrated probabilities. The likelihood says how much
evidence backs the value, so read it, or the rules' scores, before
trusting a `1.0`. It counts a row once per matching rule: the two `y=1`
rules match 3 rows each, 2 of them shared, so the 6 positives pooled
here come from 4 rows.

`predict` takes a `DataFrame` and returns the most likely value of each
target, for each row.

## How the scores combine

Each matching rule votes `p * p / (p + n)` for its value, from its
positives `p` and negatives `n`: support times precision. The votes add
up. A rule without negatives votes its positives, so above both `y=1`
rules vote 3. A rule's negatives scale its vote down by its precision,
so a large, slightly noisy rule can still outrank a small clean one.
That keeps the ranking sound on non-deterministic data, where rows
with the same features disagree, and on rules with negatives of any
kind. See [About the prediction models](../explanation/about-models.md).

To see values compete, add a row where `a=1, b=1` gives `y=0`, against
the two rows that give `1`. `assume_nondeterministic=True` says the
disagreement is expected, which also silences the warning about it.
Then list the rules that match the state, with their votes:

```pycon
>>> mixed = pd.concat(
...     [df, pd.DataFrame({"a": ["1"], "b": ["1"], "y": ["0"]})], ignore_index=True
... )
>>> dataset = Dataset(
...     mixed, features=["a", "b"], targets=["y"], assume_nondeterministic=True
... )
>>> model = WeightedRuleSetModel(dataset, GULA.fit(dataset))
>>> for rule in model.rules:
...     if rule.matches(state):
...         p, n = model.get_score(rule)
...         print(rule.to_string(), (p, n), round(p * p / (p + n), 2))
y(0) :- a(1), b(1). (1, 2) 0.33
y(1) :- a(1). (3, 1) 2.25
y(1) :- b(1). (3, 1) 2.25
>>> model.predict_likelihood(state)
{'y': {'0': 0.3333333333333333, '1': 4.5}}

```

`y=0` now has a rule, but one row backs it and two contradict it, so
it votes a third. Each `y=1` rule loses a quarter of its 3 votes to
its one negative, and the two votes add up: `1` wins, 4.5 to a third.
