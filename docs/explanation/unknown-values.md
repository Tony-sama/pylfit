# Unknown values

This page explains how pylfit learns from, and predicts on, data where
some values were never recorded, and why it works that way. The other
explanation pages cover each part in place; this one puts them
together.

## What an unknown is

A missing value, `NaN` or `None`, is read as unknown: a value the
variable had, but that nobody recorded. That is pylfit's reading of an
unknown, and everything else follows from it:

- An unknown is not a value. It is never in a domain, and no rule tests
  for it.
- An unknown never satisfies a condition. A rule matches a row only
  when the row's known values satisfy every one of its conditions.
- The examples the algorithms learn from can be partly known. Every
  state pylfit hands out holds every feature, with `None` where the
  value is unknown.
- A row whose target is unknown is left out of that target's examples,
  and only that target's: counting it as an example of every value
  would add rows that never happened.

The reading excludes two others. An unknown is not a value of its own,
which would let rules test it, and not a "don't care" that matches
every condition, which would let a rule match a row on values it never
had. Both are covered below: the first is right when the absence itself
carries information, and the second is what `min_possible_support`
counts on purpose.

## From a table to predictions

A small table with two unknowns, one of them written as `"?"`:

```pycon
>>> import warnings
>>> import pandas as pd
>>> from pylfit import Dataset, GULA, WeightedRuleSetModel, find_missing_stand_ins
>>> from pylfit import PylfitWarning, coverage
>>> df = pd.DataFrame(
...     {
...         "fever": ["yes", "yes", "yes", "yes", "no", "no", "?", None],
...         "cough": ["yes", "no", "yes", "no", "yes", "no", "yes", "no"],
...         "flu": ["yes", "yes", "yes", "yes", "no", "no", "yes", "no"],
...     }
... )

```

pylfit reads only `NaN` and `None` as unknown. `find_missing_stand_ins`
lists the values that may stand for one, for you to decide; `Dataset`
would warn about the same ones:

```pycon
>>> for column, value, count, issue in find_missing_stand_ins(df).itertuples(index=False):
...     print(f"{column}: {value!r} x{count} ({issue})")
fever: '?' x1 (stand_in)
>>> df = df.replace("?", None)

```

Learning reads only the known values. The last two rows know only
their cough, which the rules never use, so no rule can match them for
sure; `GULA` warns about such rows, silenced here by filtering only
pylfit's warnings, and `coverage` measures them:

```pycon
>>> dataset = Dataset(df, features=["fever", "cough"], targets=["flu"])
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore", PylfitWarning)
...     rules = GULA.fit(dataset, min_support=1)
>>> [rule.to_string() for rule in rules]
['flu(no) :- fever(no).', 'flu(yes) :- fever(yes).']
>>> {head: round(share, 2) for head, share in sorted(coverage(dataset, rules).items())}
{('flu', 'no'): 0.67, ('flu', 'yes'): 0.8}

```

Predicting a row with an unknown matches strictly first. Where no rule
matches for sure, the fallback weighs each rule that could match by the
share of the value it needs among the training rows' known values:
`fever` is `yes` in two thirds of them.

```pycon
>>> model = WeightedRuleSetModel(dataset, rules)
>>> rows = pd.DataFrame({"fever": ["no", None], "cough": ["yes", "yes"]})
>>> model.predict(rows, random_state=0)
[{'flu': 'no'}, {'flu': 'yes'}]
>>> for row in model.predict_proba(rows):
...     print({value: round(p, 2) for value, p in sorted(row["flu"].items())})
{'no': 1.0, 'yes': 0.0}
{'no': 0.2, 'yes': 0.8}
>>> model.predict_proba(rows, unknowns="strict")[1]
{'flu': {'no': 0.5, 'yes': 0.5}}

```

## Why not fill them in

The usual ways around unknowns are to drop the rows that have one, to
fill each in with its column's most common value, or to replace a row
by every completion of its unknowns. Each makes the rules rest on
something other than what was observed: fewer rows, or values nobody
recorded. [The benchmark_unknowns
example](../auto_examples/benchmark_unknowns.rst) measures them
against pylfit's reading on the car evaluation data and three Boolean
networks, with values blanked at random. At 30% unknowns,
averaged over the four datasets, pylfit's reading scores 0.858, against
0.817 for filling in the mode, 0.756 for every completion, and 0.733
for dropping rows; the complete table scores 0.942. Writing each
unknown as a value of its own, such as `"?"`, scores the same as
pylfit's reading there, with 2 to 5 times as many rules.

## Each part, and why

**`Dataset`.** It reads the unknowns and hands out states with `None`;
see [About data](about-data.md). `Dataset.conflicting_targets` compares
only fully known rows, since an unknown can hide whether two rows share
a state. For the same reason the rows can't show whether the system
behind them is deterministic, so `assume_nondeterministic` lets you say
it. On a deterministic system, the default, a row whose target differs
from a value is a true negative example of it whatever its unknowns
hide. On a non-deterministic system, it is left out when it could be a
state seen with that value. pylfit 0.x always left it out; on the
deterministic data of the benchmark, at 30% unknowns, that costs 4 to
12 points depending on the dataset, 8.5 on average.

**Learning.** `GULA`, `PRIDE` and `BruteForce` learn from the known
values only, and their rules are those the data as recorded leaves
consistent; see [About the algorithms](about-algorithms.md). A positive
row that no known feature separates from some negative example can be
matched by no consistent rule, whichever algorithm learns, and they
warn when that happens. `min_support` counts only the rows a rule
matches for sure, which can drop a rule whose backing rows all hide the
value it tests; `min_possible_support` counts a row that could match
instead. Unknowns slow `GULA` down, more so on many features, where
`PRIDE` is the faster choice.

**Weights.** `WeightedRuleSetModel` counts, for each rule, the rows it
matches for sure. Counting each row by its probability of matching was
tested on Boolean networks and the car evaluation data, and did no
better overall. The known cost: when the rows of one target value have
more unknowns than the others', as when a value goes unrecorded
because of the outcome, that value's rules match fewer rows and its
predictions are under-weighted. `WeightedRuleSetModel.get_score`
documents it.

**Prediction.** `unknowns="fallback"`, the default, matches strictly,
then falls back to the rules that could match for a target no rule
matches for sure; `unknowns="strict"` stops at the strict match. See
[About the prediction models](about-models.md). In the benchmark, at
30% unknowns, the fallback averages 0.790 against 0.774 for strict
prediction and 0.779 for filling in the test row's unknowns.

## When the absence means something

Sometimes whether a value was recorded is itself information: a test
run only on patients who seem ill, a question skipped by those it
doesn't concern. pylfit's reading can't use that, since an unknown is
never a condition. Write such a value as a value of its own instead,
such as `"not measured"`, and the rules can test it. Do it for the
columns where the absence carries meaning, and leave the others
missing.

## What is not covered

- With `assume_nondeterministic=True` and unknown target values, the
  rules may miss some of what the system does: a row with an unknown
  target rules nothing in or out, so a rule of the true system can be
  lost to another row that could be the same state. 0.x treated that
  target as possibly every value, which keeps every true rule covered,
  but at 30% unknowns on Boolean networks it left almost no conditions
  in the rules and lost 6 to 8 points of accuracy. On a deterministic
  system, the default, this doesn't arise.
- Rules that only hold because of how the unknowns fell are not pruned:
  every rule consistent with the data is kept, as on complete data.
- The measurements behind these choices use unknowns drawn at random,
  and real data with many unknowns on two datasets only. Unknowns that
  depend on the outcome can change which method does best.
