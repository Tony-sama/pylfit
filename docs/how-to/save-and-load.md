# Save and load a model

A model saves to a JSON file, which records the features, targets,
domain and rules, for a `WeightedRuleSetModel` each rule's counts and
how often each feature value was seen, plus the pylfit version that
wrote it. The file is plain text: you can read it, diff it and keep it
in version control.

## Save and load

```pycon
>>> import os, tempfile
>>> import pandas as pd
>>> from pylfit import Dataset, GULA, WeightedRuleSetModel
>>> df = pd.DataFrame({"a": ["0", "0", "1", "1"], "y": ["0", "0", "1", "1"]})
>>> dataset = Dataset(df, features=["a"], targets=["y"])
>>> model = WeightedRuleSetModel(dataset, GULA.fit(dataset))
>>> with tempfile.TemporaryDirectory() as folder:
...     path = os.path.join(folder, "model.json")
...     model.save(path)
...     loaded = WeightedRuleSetModel.from_file(path)
>>> loaded == model
True

```

A loaded model predicts without the dataset it was learned from.

Load a file with the class that saved it. Loading it with another class
raises instead of silently dropping data:

```pycon
>>> from pylfit import RuleSetModel
>>> with tempfile.TemporaryDirectory() as folder:
...     path = os.path.join(folder, "model.json")
...     model.save(path)
...     RuleSetModel.from_file(path)  # doctest: +ELLIPSIS
Traceback (most recent call last):
...
ValueError: file was saved by 'WeightedRuleSetModel', not 'RuleSetModel'...

```

## Export rules as text

To share rules with another tool, or to write them by hand, use the
rule notation that `Rule.to_string` writes and `from_string` reads:

```pycon
>>> from pylfit import Rule
>>> text = [rule.to_string() for rule in model.rules]
>>> tuple(Rule.from_string(line) for line in text) == model.rules
True

```

The text reads back exactly unless a name or value hits one of the
limits `Rule.from_string` lists, such as a variable name containing
`(`, or a value with a space at either end, which is dropped. Names
like `a_t+1` and values like `(30, 50]` from `pandas.cut` are fine. The
JSON file of `save` has no such limit.
