# Getting started

This tutorial takes you from a table of examples to a model that
predicts, and explains its predictions with rules. It takes about ten
minutes. You need Python 3.11 or later; see
[Installation](index.md#installation).

## The data

The table records, for fourteen days, the weather and whether a game of
tennis was played. Every value is a category, such as `sunny` or
`high`: pylfit learns from categories, not numbers.

```pycon
>>> import io
>>> import pandas as pd
>>> table = """outlook,temperature,humidity,wind,play
... sunny,hot,high,weak,no
... sunny,hot,high,strong,no
... overcast,hot,high,weak,yes
... rain,mild,high,weak,yes
... rain,cool,normal,weak,yes
... rain,cool,normal,strong,no
... overcast,cool,normal,strong,yes
... sunny,mild,high,weak,no
... sunny,cool,normal,weak,yes
... rain,mild,normal,weak,yes
... sunny,mild,normal,strong,yes
... overcast,mild,high,strong,yes
... overcast,hot,normal,weak,yes
... rain,mild,high,strong,no
... """
>>> df = pd.read_csv(io.StringIO(table))
>>> len(df)
14

```

## Wrap it in a dataset

A `Dataset` says which columns are features, the inputs, and which are
targets, the values to predict. It checks those columns once, up
front, and stores every value as a string.

```pycon
>>> from pylfit import Dataset
>>> dataset = Dataset(
...     df, features=["outlook", "temperature", "humidity", "wind"], targets=["play"]
... )

```

## Learn rules

`GULA` is pylfit's main learning algorithm. `fit` returns a plain list
of rules, sorted by text:

```pycon
>>> from pylfit import GULA
>>> rules = GULA.fit(dataset)
>>> len(rules)
22

```

Each rule reads "the target takes this value when these conditions
hold". For example, `play(yes) :- outlook(overcast).` says: when the
outlook is overcast, play is a possible value. The part before `:-` is
the rule's head, its conclusion; the part after is its body, the
conditions.

```pycon
>>> for rule in rules[:3]:
...     print(rule.to_string())
play(no) :- humidity(high), outlook(sunny).
play(no) :- humidity(high), temperature(cool).
play(no) :- humidity(normal), outlook(overcast), temperature(mild).

```

Some of those rules match no day in the table at all, for example
`play(no) :- humidity(high), temperature(cool).`: no day was cool and
humid. `GULA` learns the most general rules the data doesn't contradict,
all of them, including ones the data says nothing about. `min_support`
keeps only the rules that enough rows back:

```pycon
>>> supported = GULA.fit(dataset, min_support=1)
>>> len(supported)
14

```

A rule no row backs carries no evidence, only the absence of a
contradiction, so `min_support=1` is a good default when you read the
rules or rank predictions by support. Leave it unset when you need the
full, complete rule set, for example to know every value the data
leaves possible for a state it never shows (a state is one combination
of feature values): the dropped rules are exactly the ones that add
those possibilities.

If a fit is slow, pass `verbose=True` to see its progress and an
estimate of the time left. `min_support=1` usually makes `GULA` much
faster, since it has fewer rules to keep. `PRIDE` learns a smaller set
of rules, faster, and `n_jobs=-1` learns several targets' values at
once; [Keep the problem a comfortable
size](how-to/prepare-data.md#keep-the-problem-a-comfortable-size) has
measured sizes.

## Predict

To predict, wrap the rules in a model. `WeightedRuleSetModel` counts,
for each rule, how many rows back it, and lets the better-backed rules
win:

```pycon
>>> from pylfit import WeightedRuleSetModel
>>> model = WeightedRuleSetModel(dataset, supported)
>>> new = """outlook,temperature,humidity,wind
... sunny,cool,high,strong
... rain,hot,normal,weak
... """
>>> new_days = pd.read_csv(io.StringIO(new))
>>> model.predict(new_days)
[{'play': 'no'}, {'play': 'yes'}]

```

Neither day is in the table. To see why the model decided, list the
rules that match a day, with their support: how many of the rows a
rule matches have its value (positives), and how many have another
(negatives).

```pycon
>>> day = new_days.iloc[0]
>>> for rule in model.rules:
...     if rule.matches(day):
...         print(rule.to_string(), model.get_score(rule))
play(no) :- humidity(high), outlook(sunny). (3, 0)
play(yes) :- outlook(sunny), temperature(cool). (1, 0)

```

Two rules match, for different values. Three days back the first, one
the second, so the model predicts `no`.

## Score the model

To measure how well the rules generalize, learn from part of the data
and score on the rest. Shuffle the rows first, with a fixed seed so the
split is the same every run:

```pycon
>>> shuffled = df.sample(frac=1, random_state=0)
>>> train, test = shuffled.iloc[:10], shuffled.iloc[10:]
>>> train_set = Dataset(train, features=dataset.features, targets=dataset.targets)
>>> model = WeightedRuleSetModel(train_set, GULA.fit(train_set, min_support=1))

```

Here the ten training rows happen to hold every value, but a value can
appear only in the test rows, and `predict` rejects a value outside the
model's domain, the values each column holds in the data the model was
built from. Give the model the domain of the whole table, not only of
the training rows.

```pycon
>>> model.domain = dataset.domains()

```

Then predict the test rows, and compare with the true values using
`accuracy_score`:

```pycon
>>> from pylfit import accuracy_score
>>> predictions = model.predict(test[dataset.features], random_state=0)
>>> accuracy_score(test, predictions, "play")
0.75

```

When several values are equally likely, `predict` picks one at random,
so without `random_state` the score can change from run to run. Pass a
seed whenever you need the same output twice; see
[Get reproducible results](how-to/reproducible-results.md).

Four test days are far too few for the score to mean anything; on real
data, use more rows, or average over several splits.

## Next steps

- [Prepare your data](how-to/prepare-data.md): numeric columns,
  choosing targets, which algorithm and model to use.
- [Predict heart disease from clinical
  data](auto_examples/tutorial_heart_disease.rst): a whole problem,
  from the measurements to rules that explain each prediction.
- [Load data from a CSV file](how-to/load-a-csv.md), and [save a model
  and load it back](how-to/save-and-load.md).
- [About LFIT rules](explanation/about-rules.md): what "every rule the
  data doesn't contradict" means.
- The [API reference](api/index.md).
