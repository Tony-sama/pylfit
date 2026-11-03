# Prepare your data

This guide covers what pylfit expects of a table and how to choose an
algorithm and a model, then unknown values, conflicting rows, how large
a problem stays comfortable, and the errors you are most likely to
meet.

## Make every value a category

pylfit compares values by exact equality, so every value is a
category. A `Dataset` stores every value as a string;
`dataset.to_dataframe()` gives that table back, to check what the rules
will read or to analyze it with pandas.

- Text, `bool`, `category` and integer columns are accepted as they
  are.
- Float columns are rejected, unless every value is a whole number:
  pandas makes an integer column with a missing value a float column,
  and pylfit reads it back as integers.
- Any column with more than `max_categories` (default 5) distinct
  values gets a warning, since each value multiplies the rules `GULA`
  considers. An integer column with that many values is usually a
  measurement, a text column often has rare values worth grouping, and
  one with nearly one value per row is an identifier or a precise
  measurement. Raise `max_categories` for columns you know are
  categories, such as a region code.

An integer column loses its order: `3` and `4` become two unrelated
categories, and the rules can't say "4 or more". Bin a count or a scale
the same way you bin a measurement, unless its values really are
labels, such as codes for a region.

Turn a numeric column into categories with `pandas.cut`, choosing the
bins yourself:

```pycon
>>> import pandas as pd
>>> from pylfit import Dataset
>>> df = pd.DataFrame(
...     {
...         "age": [23.5, 40.1, 61.0, 35.2],
...         "smoker": ["yes", "no", "yes", "no"],
...         "sick": ["1", "0", "1", "0"],
...     }
... )
>>> Dataset(df, features=["age", "smoker"], targets=["sick"])
Traceback (most recent call last):
...
ValueError: float columns are not supported (not categorical): ['age']. Discretize them (e.g. into bins) before building the Dataset.
>>> df["age"] = pd.cut(df["age"], bins=[0, 30, 50, 120], labels=["young", "middle", "old"])
>>> dataset = Dataset(df, features=["age", "smoker"], targets=["sick"])
>>> dataset.domain("age")
['middle', 'old', 'young']
>>> dataset.to_dataframe()
      age smoker sick
0   young    yes    1
1  middle     no    0
2     old    yes    1
3  middle     no    0

```

Use the same bins on the data you later predict on: a value the model
never saw is an error, not a guess. Same bins are not always enough: a
bin no training row fell into is not in the model's domain either, so
add it with `model.domain = ...` if new data may land there. Keep the
binning in one function, and call it on both:

```pycon
>>> def prepare(frame):
...     frame = frame.copy()
...     frame["age"] = pd.cut(
...         frame["age"], bins=[0, 30, 50, 120], labels=["young", "middle", "old"]
...     )
...     return frame
>>> new = prepare(pd.DataFrame({"age": [25.0], "smoker": ["no"]}))
>>> new["age"].tolist()
['young']

```

Choose the bins from what the values mean, such as clinical
thresholds, or from the data, with `pandas.qcut` for bins of equal
size. Fewer bins give fewer, better-supported rules; more bins give
rules that each cover few rows. Name the bins with `labels`, or the
rules will show intervals such as `(30, 50]`.

```{figure} ../images/measurements-to-levels.svg
---
alt: >-
  A nitrate series over twelve months, cut at its mean into low and
  high, and a Phaeocystis series cut at a noise level and a bloom
  level into absent, low and high. Below, the resulting levels, one
  row per month.
---

Two choices of bins on an illustrative year of monitoring data.
Nitrate is cut at its mean, a choice from the data. For the plankton,
counts up to 50 are noise around zero and get a level of their own,
`absent`, and a bloom starts at 150, choices from what the values
mean: `pandas.cut` with `bins=[-1, 50, 150, 1000]`.
```

Make the bins cover every value: `pandas.cut` turns a value outside
every bin into `NaN`, which pylfit reads as unknown, not as an error.

[The heart disease example](../auto_examples/tutorial_heart_disease.rst)
bins five clinical measurements this way, and compares the result with
models that read the raw numbers.

## Choose features and targets

Features are what the rules read; targets are what they conclude. A
dataset can have several targets, and the algorithms learn rules for
each one separately.

Leave out columns that identify a row, such as an ID or a date: a rule
on an ID describes one row and never generalizes.

## Choose an algorithm and a model

| If you want | Use |
| --- | --- |
| Every rule the data doesn't contradict, on small problems | `GULA` |
| A smaller rule set that still covers every row, fast on large data | `PRIDE` |
| Few rules that predict well | `PRIDE` or `GULA`, then `generalize` |
| A reference to test a new algorithm against, on small data | `BruteForce` |
| The most likely value of each target | `WeightedRuleSetModel` |
| Every value the rules allow, all treated equally | `RuleSetModel` |

To predict a single value, use `WeightedRuleSetModel`. The full rule
set usually matches several values for each row, and `RuleSetModel`
treats them as equally likely, so its `predict` often picks at random.
On the car evaluation data, the same rules score 0.23 through
`RuleSetModel` and 0.92 through `WeightedRuleSetModel`
([the benchmark_classic example](../auto_examples/benchmark_classic.rst)).

## Read a rule

`sick(1) :- age(old), smoker(yes).` reads: `sick` can be `1` when `age`
is `old` and `smoker` is `yes`; see [About LFIT
rules](../explanation/about-rules.md).

## Leave unknown values missing

Write a value nobody recorded as `NaN` or `None`. pylfit reads it as
unknown: never part of a domain, never satisfying a rule's condition,
and a row whose target is unknown is left out for that target only.
Don't fill unknowns in with a guess, such as the column's most
frequent value: the rules and their counts would then rest on values
that were never observed. [About data](../explanation/about-data.md)
explains this reading.

A string such as `"?"` or `""` is an ordinary value, not an unknown,
and so is a zero written where nobody took the measurement.
`find_missing_stand_ins` lists the values that may be such stand-ins,
for you to check: strings such as `"?"`, `""`, `"null"` or `"unknown"`,
and zeros far from the rest of a numeric column, by Tukey's rule for
outliers.

```pycon
>>> from pylfit import find_missing_stand_ins
>>> raw = pd.DataFrame(
...     {
...         "smoker": ["yes", "?", "no", "no", "yes"],
...         "pressure": [120, 135, 0, 110, 160],
...         "sick": ["1", "0", "0", "0", "1"],
...     }
... )
>>> report = find_missing_stand_ins(raw)
>>> for column, value, count, issue in report.itertuples(index=False):
...     print(f"{column}: {value!r} x{count} ({issue})")
smoker: '?' x1 (stand_in)
pressure: 0 x1 (far_zero)

```

`Dataset` runs the same check on its features and targets and warns
when it finds something, listing up to five findings. Call
`find_missing_stand_ins` for the full report or other stand-in
strings, and pass `check_stand_ins=False` to `Dataset` when the values
it reports are real.

To look for a stand-in of your own as well, add it to the default list,
`DEFAULT_STAND_INS`:

```pycon
>>> from pylfit import DEFAULT_STAND_INS
>>> notes = pd.DataFrame({"smoker": ["yes", "unk", "no"]})
>>> find_missing_stand_ins(notes).empty
True
>>> find_missing_stand_ins(notes, stand_ins=(*DEFAULT_STAND_INS, "unk"))["value"].tolist()
['unk']

```

Replace a value that stands for a missing one with `None` before
building the `Dataset`, unless the absence itself means something:
then keep it as a value of its own, such as `"not measured"`, and the
rules can use it.

```pycon
>>> clean = raw.replace({"smoker": {"?": None}, "pressure": {0: None}})
>>> Dataset(clean, features=["smoker"], targets=["sick"]).domain("smoker")
['no', 'yes']

```

Every algorithm learns from unknown values, and the models predict
from them; see [About the algorithms](../explanation/about-algorithms.md)
and [About the prediction models](../explanation/about-models.md).

## Check for conflicting rows

Two rows conflict when they have the same features but different
values of a target. `Dataset.conflicting_targets` lists the targets
with one, and `Dataset` warns when it finds one:

```pycon
>>> conflicting = pd.DataFrame({"a": ["0", "0", "1"], "y": ["0", "1", "1"]})
>>> nondeterministic = Dataset(
...     conflicting, features=["a"], targets=["y"], assume_nondeterministic=True
... )
>>> nondeterministic.conflicting_targets
['y']

```

The warning names the usual causes: noise, a feature that matters but
is not in the data, or a system that is non-deterministic, able to
give several values from the same state. If the system is
non-deterministic, say so with `assume_nondeterministic=True`, as
above: it silences the warning, and on data with unknown values it
changes the negative examples, as
[About data](../explanation/about-data.md) explains. If the conflicts
are noise or a missing feature, keep the default and the warning; to
silence it there anyway, filter `pylfit.ConflictWarning` with Python's
`warnings` module.

Conflicting data is common, for example after binning, and is
not an error. A feature state seen with several target values is
possible for each of them, so rules can't rule any of them out there.
`WeightedRuleSetModel` still ranks them, by how many rows back each
rule and how many it gets wrong; see [About the prediction
models](../explanation/about-models.md).

## Keep the problem a comfortable size

`GULA`'s cost grows with the number of features and the size of their
domains far more than with the number of rows; the worst case is
exponential. `PRIDE`'s is polynomial in the size of the data (Ribeiro
et al., IJCLR 2021, [hal-03347026](https://hal.science/hal-03347026))
and grows much more slowly. Measured on the data the scripts in
`examples/` use, the 15-variable network timed on its own, a fit on a
`Dataset` already fitted once (the first fit also prepares its rows):

| Data | `GULA.fit` | `PRIDE.fit` |
| --- | --- | --- |
| car evaluation: 6 features, 1382 training rows | 0.3 s | 0.08 s |
| soybean: 35 features, 37 training rows | 1.3 s | under 0.01 s |
| Boolean network `faure_cellcycle`, 10 variables, 1024 rows | 0.18 s | 0.02 s |
| Boolean network `tournier_apoptosis`, 12 variables, 4096 rows | 1.0 s | 0.08 s |
| Boolean network `saadatpour_guardcell`, 13 variables, 8192 rows | 2.3 s | 0.14 s |
| Boolean network `randomnet_n15k3`, 15 variables, 32,768 rows | 13 s | 0.9 s |

On the networks each variable added doubles the rows. `PRIDE`'s time
about doubles with them, `GULA`'s grows two to three times, so the gap
widens: ten times at 10 variables, fifteen at 15 ([the benchmark_bnet
example](../auto_examples/benchmark_bnet.rst) prints the curve).

**If `GULA` takes more than a few seconds, use `PRIDE`.** Its rules
are some of `GULA`'s, enough to cover every row, so there are far
fewer of them. On their own they predict a few points worse, but after
[`generalize`](../api/generalize.md) the two are level: in [the heart
disease tutorial](../auto_examples/tutorial_heart_disease.rst),
`PRIDE`'s and `GULA`'s generalized rules both score 0.82 over ten
seeds. Keep `GULA` for small problems, and when you need every rule the
data supports.

`min_support` is the other lever. It drops rules that fewer rows back
than the threshold, and the rules it keeps are exactly the unrestricted
rules that meet it: with `min_support=1`, soybean's `GULA` fit takes
0.12 s. The cost is completeness: a feature state only a dropped rule
matched is no longer matched by any rule. `GULA` and `PRIDE` warn when
that leaves rows of your data uncovered, and
`pylfit.coverage(dataset, rules)` gives the covered share per head.
[About the algorithms](../explanation/about-algorithms.md) covers the
other options that restrict the search.

Two more help with a slow fit:

- `verbose=True` prints each head's progress and an estimate of the
  time left, so you know early whether a fit will finish.
- `n_jobs` learns several heads at once in separate processes; it pays
  off on fits of a few seconds or more. A script using it must fit
  under `if __name__ == "__main__":`, or the fit fails on macOS, on
  Windows, and on Linux from Python 3.14, and the error says so. See
  [Learning heads in parallel](../explanation/about-algorithms.md#learning-heads-in-parallel).

## Troubleshooting

| Error | Cause and fix |
| --- | --- |
| `float columns are not supported` | A numeric column, or floats among the values of a text column. Bin it with `pandas.cut`, as shown above. |
| `complex, date and duration columns are not supported` | Convert the column to `str`, or bin it, before building the `Dataset`. |
| `duplicate column names in the dataframe` | Two columns share a name, often after a `merge` or `concat`. Rename or drop the copy. |
| `Rows with the same feature state have different values` (warning) | Conflicting rows; see [Check for conflicting rows](#check-for-conflicting-rows). |
| `positive rows ... are matched by no rule of their head` (warning) | Unknown values that no known feature separates from a negative, or an option such as `min_support`; the message says which. Pass `warn_uncovered=False` once you know. |
| `values may stand for a missing value` (warning) | A `"?"`, `"null"`, a far zero...; see [Leave unknown values missing](#leave-unknown-values-missing). Replace them with `None`, or pass `check_stand_ins=False` if they are real values. |
| `Column '...' has N distinct values` (warning) | A measurement, many rare categories, or an identifier; the message says which it looks like. Bin it, group rare values, drop an identifier, or raise `max_categories` if every value is needed. |
| `missing required feature column(s)` | The data you predict on lacks a feature column. Pass every feature column. |
| `target column(s) must not be included in prediction input` | Pass only the feature columns, for example `df[model.features]`. |
| `unexpected column(s), not part of the trained features` | A column such as an `id` that `Dataset` ignored. Pass only `df[model.features]`. |
| `state has no entry for feature(s)` | A state passed as a `dict` lacks a feature. Add it, with `None` if its value is unknown. |
| `out-of-domain value(s) for feature` | A value the model never saw. Bin new data with the same bins, or widen `model.domain`. |
