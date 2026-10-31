# Load a CSV file

pylfit reads a `pandas.DataFrame`, so any file pandas can read works.
This guide loads a CSV and learns rules from it.

## Read every column as text

This guide reads a small file of patients:

```{literalinclude} patients.csv
:caption: patients.csv
:language: text
```

Read the file with `dtype=str`. Every value then arrives as a category,
and pandas doesn't turn a column of codes like `01`, `02` into
integers:

```pycon
>>> import pandas as pd
>>> df = pd.read_csv("patients.csv", dtype=str)
>>> df["smoker"].tolist()
['yes', 'no', 'yes', 'no', '?']

```

An empty cell arrives as `NaN`, which pylfit reads as an unknown value.
pandas also reads `NA`, `N/A`, `null`, `NaN` and a few similar strings
as `NaN`. `pylfit.find_missing_stand_ins(df)` lists the markers it
doesn't know, and zeros that look like a missing measurement; see
[Prepare your data](prepare-data.md). If one of them is a real value
in your file, pass `keep_default_na=False` and list your own markers in
`na_values`. A marker pandas doesn't know, such as the `?` above, stays
a value; list it in `na_values` if it means unknown:

```pycon
>>> df = pd.read_csv("patients.csv", dtype=str, na_values=["?"])
>>> df["smoker"].tolist()
['yes', 'no', 'yes', 'no', nan]

```

A column of measurements, where the numbers' order matters, needs
binning into categories first; see [Prepare your data](prepare-data.md).
With `dtype=str` that is up to you: a measurement read as text is a
category to pylfit, so the float check below never sees it; only the
warning on a column with many distinct values still does. To keep the
float check, read only the code columns as text, with a mapping such as
`dtype={"id": str, "smoker": str}`.

## Pick the columns

Name the features and targets. Columns you name in neither, like `id`
here, are ignored. A model's `predict` is stricter: pass it the
feature columns only, `df[model.features]`.

```pycon
>>> from pylfit import Dataset, GULA
>>> dataset = Dataset(df, features=["smoker", "cough"], targets=["diagnosis"])
>>> rules = GULA.fit(dataset, min_support=1)
>>> for rule in rules:
...     print(rule)
diagnosis(bronchitis) :- cough(yes), smoker(yes).
diagnosis(cold) :- cough(yes), smoker(no).
diagnosis(healthy) :- cough(no).

```

## Check the data first

A `Dataset` checks its feature and target columns when you create it,
and raises on a float column before any learning starts. To check a
table without building one, call `Dataset.check_values` on the columns
you plan to use. It returns nothing when they are fine, and raises the
error the `Dataset` would:

```pycon
>>> Dataset.check_values(df[["smoker", "cough", "diagnosis"]])
>>> Dataset.check_values(pd.DataFrame({"weight": [61.5, 80.2]}))
Traceback (most recent call last):
...
ValueError: float columns are not supported (not categorical): ['weight']. Discretize them (e.g. into bins) before building the Dataset.

```
