# About data

This page explains how pylfit represents data, and why: categories
stored as strings, unknown values, and what conflicting rows change.
For the practical steps, see [Prepare your data](../how-to/prepare-data.md).

## Every value is a category, stored as a string

A rule's condition is one variable equal to one value, so values are
compared by exact equality. A `Dataset` checks its feature and target
columns once, when you create it, and stores every value as a string;
`Dataset.to_dataframe` returns that table. Everything after that,
rules, models, saved files, can rely on that without checking again. A
`DataFrame` you pass to `Dataset`, `predict` or `predict_proba` is
converted for you, so an integer column of `1`s reads as `"1"`. A
single state you pass as a `dict` is not converted: its values must be
strings, or `None` for an unknown value, and `Rule.matches` raises
`TypeError` on any other type it reaches rather than silently not
matching. It checks conditions in variable order and stops at the
first that fails, so a wrong value after that one goes unnoticed.

Floats are rejected rather than converted, because equality on
measurements is almost never meaningful: `23.5` and `23.6` would be two
unrelated categories. Binning is a modeling choice, so pylfit leaves
it to you. A float column whose values are all whole numbers is the
exception: pandas stores an integer column with a missing value that
way, so `1.0` reads as `"1"`.

## The domain

A variable's domain is the set of values it takes in the dataset. A
model records each variable's domain, and `predict` rejects a value
outside it. The domain also sets what "every feature state" means in
completeness: every combination of domain values, including ones no row
shows.

## Unknown values

A missing value, `NaN` or `None`, is read as unknown: a value the
variable had but that nobody recorded. It is never in a domain and
never satisfies a condition, and a row whose target is unknown is left
out of that target's examples only. [Unknown values](unknown-values.md)
explains this reading and follows it through learning and prediction;
this section covers how the data holds it.

Every state pylfit hands out holds every feature, with `None` where a
value is unknown:

```pycon
>>> import pandas as pd
>>> from pylfit import Dataset
>>> df = pd.DataFrame({"a": ["0", None, "1"], "b": ["0", "1", "1"], "y": ["0", "1", "1"]})
>>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
>>> dataset.positives("y", "1")
[{'a': None, 'b': '1'}, {'a': '1', 'b': '1'}]
>>> dataset.domain("a")
['0', '1']

```

A state you write by hand must also hold every variable a rule tests,
with `None` for an unknown value: `Rule.matches` raises `KeyError` on
a missing key. A missing key is far more often a misspelt or forgotten
variable than an intended unknown, and treating it as unknown would
hide the mistake behind a plausible answer. `Rule.matches` tests the
conditions in turn and stops at the first that fails, so it only
raises for a missing key it reaches: checking every key first would
slow down prediction, which calls it for every rule and state. A
model's single-state methods, such as `predict_candidates`, do check
first: every feature must be present and every known value a `str`
within its feature's domain. Other keys are ignored, where `predict`
refuses a column that is not a feature.

Only pandas' missing values, `NaN`, `None`, `pd.NA` and `NaT`, are
unknown. A string such as `"?"`, `""` or `"unknown"` is an ordinary
value: pylfit can't tell a stand-in from a real answer, such as a vote
of `"?"` meaning "neither". `find_missing_stand_ins` lists the values
worth a look, and zeros far from the rest of a numeric column, outside
Tukey's fences (Tukey, *Exploratory Data Analysis*, Addison-Wesley,
1977, ISBN 978-0-201-07616-5); you decide. `Dataset` runs the same
check and warns rather than refusing, since a reported value may be
real; `check_stand_ins=False` turns it off.

## Repeated rows are repeated evidence

pylfit keeps every row, including rows identical to another. A row
that appears three times, as three experiments with the same outcome
would, gives a rule that matches it a support of three: for
`min_support`, and in `WeightedRuleSetModel`'s counts.

```pycon
>>> import pandas as pd
>>> from pylfit import Dataset, GULA, WeightedRuleSetModel
>>> df = pd.DataFrame({"a": ["0", "0", "0", "1"], "y": ["1", "1", "1", "0"]})
>>> dataset = Dataset(df, features=["a"], targets=["y"])
>>> model = WeightedRuleSetModel(dataset, GULA.fit(dataset))
>>> print(model)
features: ['a']
targets: ['y']
rules:
  y(0) :- a(1).  (+1/-0)
  y(1) :- a(0).  (+3/-0)

```

A repeated row also counts once per occurrence against a rule it
contradicts, in the negatives of `WeightedRuleSetModel`'s counts.
Without `min_support`, `GULA` learns the same rules with or without
repeats.

If repeats in your data are artifacts, such as a row logged twice,
remove them with `DataFrame.drop_duplicates` before building the
`Dataset`.

## Conflicting rows

Two rows conflict when they have the same feature state but different
target values. `Dataset.conflicting_targets` lists the targets that have
one. Real data often has conflicts: noise, a feature that matters but is
not in the data, binning that merges states that differed, or a system
that is itself non-deterministic, able to give several values from the
same state. `Dataset` warns when it finds a conflict, so you learn it
before reading any rule.

A feature state seen with the value in any row is not a negative
example of that value, even in rows where the target has another
value. The state is possible for both values, and a correct rule must
not rule out a value the data shows is possible.

```pycon
>>> import warnings
>>> import pandas as pd
>>> from pylfit import ConflictWarning, Dataset, GULA
>>> df = pd.DataFrame({"a": ["0", "0", "1"], "y": ["0", "1", "1"]})
>>> with warnings.catch_warnings():
...     warnings.simplefilter("ignore", ConflictWarning)
...     dataset = Dataset(df, features=["a"], targets=["y"])
>>> dataset.conflicting_targets
['y']
>>> dataset.negatives("y", "1")
[]
>>> [rule.to_string() for rule in GULA.fit(dataset)]
['y(0) :- a(0).', 'y(1) :- .']

```

`y` was `1` in some row for every value of `a`, so no state is a
negative of `y(1)`, and the rule with an empty body holds. The warning
`Dataset` gives on the conflict is filtered here, as
`ConflictWarning`, so the example shows only the result.

Scoring counts differently. `WeightedRuleSetModel` counts every row a
rule gets wrong as a negative, so the conflicting row still lowers a
rule's score, in proportion to its support; see [About the prediction
models](about-models.md).

### Conflicts and unknowns: `assume_nondeterministic`

`conflicting_targets` only compares rows whose features and target are
all known: two rows that differ only where one is unknown may be two
different states. So with unknowns, the rows can't show whether the
system is deterministic, and you have to say it.

On a **deterministic** system, the default, a row whose target differs
from a value is a true negative of that value, whatever its unknowns
hide: had the hidden state given that value, the row would show it.
pylfit keeps every such row, which keeps the rules as specific as the
data requires.

On a **non-deterministic** system that reasoning fails: an unknown can
hide a state also seen with the value, and keeping the row would rule
out a value the system can take. Pass
`Dataset(..., assume_nondeterministic=True)`, and a row is left out of
a value's negatives whenever it agrees with some row of that value on
every feature known in both:

```pycon
>>> df = pd.DataFrame({"a": ["0", None], "b": ["0", "0"], "y": ["1", "0"]})
>>> Dataset(df, features=["a", "b"], targets=["y"]).negatives("y", "1")
[{'a': None, 'b': '0'}]
>>> Dataset(df, features=["a", "b"], targets=["y"], assume_nondeterministic=True).negatives(
...     "y", "1"
... )
[]

```

The second row could be `a=0, b=0`, the first row's state, so only the
non-deterministic reading leaves it out. Leaving it out on a
deterministic system would make rules more general than the data
allows, which is why it is not the default. Without unknowns the
option changes no example: a state seen with the value is left out
either way. It also silences the conflict warning.
