# pylfit

<!-- Badges, once the repository is public and hosted: tests (CI),
docs, PyPI version, license, and the release DOI. -->

<!-- start-intro -->

pylfit learns logic rules from data. Give it a table of examples, and it
returns rules such as `y(1) :- a(0), c(1).` ("`y` can be 1 when `a` is
0 and `c` is 1") that you can read, check and use to predict.

pylfit is for people who need a model they can inspect: every
prediction traces back to the rules that made it. Its main algorithm,
`GULA`, learns the most general rules the data doesn't contradict, all
of them, with guarantees: no rule contradicts an example, every
combination of feature values is matched by some rule, and no rule
keeps a condition it can do without. pylfit implements algorithms of
the Learning From Interpretation Transitions (LFIT) framework, whose
original application is learning Boolean networks, models where each
variable is on or off and its next value depends on the others', such
as gene regulatory networks, from observed state transitions. It works
on any table of categorical data.

> pylfit 1.0 is a ground-up rewrite, still in development and not yet
> on PyPI. Its API is incompatible with pylfit 0.x; see
> [Migrate from pylfit 0.x][migration].

[migration]: https://tony-sama.github.io/pylfit/how-to/migration.html

## Installation

pylfit needs Python 3.11 or later, pandas 2.3 or later and NumPy 2.2 or
later, which `pip` installs for you. 1.0 is not on PyPI yet:
`pip install pylfit` still installs the 0.5.1 release. Until it is,
install 1.0 from the repository's `next` branch, where it is developed:

```bash
pip install "git+https://github.com/Tony-sama/pylfit.git@next"
```

To keep using pylfit 0.x, pin it: `pip install "pylfit<1" numpy pandas`
(0.5.1 uses both without declaring them).

## Quickstart

A `Dataset` stores every value as a string, so the rules and the
model's predictions read `"0"`, not `0`.

```pycon
>>> import pandas as pd
>>> from pylfit import GULA, Dataset, WeightedRuleSetModel
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
>>> rules = GULA.fit(dataset)
>>> for rule in rules:
...     print(rule)
y(0) :- a(1), b(0).
y(0) :- b(0), c(0).
y(1) :- a(0), c(1).
y(1) :- b(1).
>>> model = WeightedRuleSetModel(dataset, rules)
>>> model.predict(pd.DataFrame([{"a": "0", "b": "1", "c": "0"}]))
[{'y': '1'}]

```

<!-- end-intro -->

## Where to start

- To use pylfit: [Getting started][getting-started], then
  [Prepare your data][prepare-data].
- To understand how it works: [About LFIT rules][about-rules] and
  [About the algorithms][about-algorithms].
- To contribute: [CONTRIBUTING.md][contributing].

The `examples/` directory has runnable scripts: `starter.py` is a whole
experiment to copy, `tutorial_heart_disease.py` goes from clinical
measurements to rules that explain each prediction, `dynamics.py`
learns a dynamical system, and the `benchmark_*.py` scripts compare the
algorithms and their options.

## Citing

See [Citing pylfit][citing] and `CITATION.cff`.

## Acknowledgements

pylfit is written and maintained by Tony Ribeiro (`AUTHORS.md`). It
implements the Learning From Interpretation Transitions (LFIT)
framework, developed with many colleagues over more than ten years;
they are credited on the [Credits][credits] page. pylfit 1.0 was
written with AI assistants; the same page says [which, and how][ai].

## License

GPL-3.0-only; see `LICENSE`. The Boolean-network files under
`tests/data/bnet/` come from PyBoolNet and keep its LGPL license, and
the UCI datasets under `examples/data/` are CC BY 4.0; see the README in
each directory.

See also `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` and `AUTHORS.md`.

<!-- Absolute links, so that they work on PyPI too. -->

[getting-started]: https://tony-sama.github.io/pylfit/getting-started.html
[prepare-data]: https://tony-sama.github.io/pylfit/how-to/prepare-data.html
[about-rules]: https://tony-sama.github.io/pylfit/explanation/about-rules.html
[about-algorithms]: https://tony-sama.github.io/pylfit/explanation/about-algorithms.html
[contributing]: https://github.com/Tony-sama/pylfit/blob/next/CONTRIBUTING.md
[citing]: https://tony-sama.github.io/pylfit/citing.html
[credits]: https://tony-sama.github.io/pylfit/credits.html
[ai]: https://tony-sama.github.io/pylfit/credits.html#how-pylfit-1-0-was-written
