# Contributing to pylfit

Bug reports and questions are welcome as
[GitHub issues](https://github.com/Tony-sama/pylfit/issues), and pull
requests too. For anything you would rather not discuss in public,
write to tonyribeiro.contact@gmail.com; for a security problem, follow
the [security
policy](https://github.com/Tony-sama/pylfit/blob/next/SECURITY.md)
instead of opening an issue.

## Before you start

A bug fix or a docs correction can go straight to a pull request. For
anything that adds to the public API (a new algorithm, model, option or
function), open an issue first and describe the use that needs it:
pylfit adds a name only when a workflow, an example or a published
method calls for it, because a name added in a 1.x release can only be
removed in 2.0. To try an idea without changing pylfit, see
[Use your own algorithm](docs/how-to/your-own-algorithm.md).

## Set up a development install

Use Python 3.11 or later, in a virtual environment:

```bash
git clone -b next https://github.com/Tony-sama/pylfit.git
cd pylfit
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,docs]"
git config core.hooksPath .githooks
```

The last line enables the commit-message hook, which rejects a subject
line over 72 characters. It is local configuration, so run it once per
clone.

Or open the clone in VS Code and choose *Reopen in Container*: the dev
container in `.devcontainer/` runs the install and the hook line for
you, on Python 3.11, with no virtual environment needed.

## Find your way around

- `src/pylfit/` is the package, and the only part that ships in the
  wheel. `_rule.py` and `_dataset.py` hold the data types,
  `algorithms/` the learners (`BruteForce`, `GULA`, `PRIDE`), `models/`
  the prediction models, and `_metrics.py` the scores. Modules are
  private (leading `_`); users import every public name from `pylfit`.
- `tests/` mirrors the package. Slow tests are marked `slow` and left
  out of a plain `pytest` run;
  `tests/algorithms/test_boolean_networks.py` checks every algorithm
  against published Boolean networks.
- `examples/` holds runnable scripts, each run by the test suite.
- `docs/` is the documentation site, built with Sphinx.

The pages listed under [Contributing](docs/contributing/index.md) on
the docs site explain the code you will be changing: its architecture,
the design choices behind it, the standards it follows, how to add an
algorithm and how the tests are organized.

## Write docs

Docstrings follow Google style and are reStructuredText; the full rules
are on the [standards page](docs/contributing/standards.md), under
"Writing docstrings and pages". Code on the docs pages is written as
`>>>` sessions, and `pytest` runs it too.

To build the site:

```bash
sphinx-build -n -W --keep-going -b html docs docs/_build/html
```

then open `docs/_build/html/index.html`. The first build runs every
script in `examples/` for the examples gallery, which adds about a
minute and a half, most of it in the benchmarks; later builds rerun
only the scripts that changed. A script that fails, fails the build.

## Before opening a pull request

Make sure these pass; CI runs the same ones:

```bash
pytest -m ""            # the whole suite, slow tests included
ruff check .
ruff format --check .
pydoclint src           # docstrings agree with the signatures
interrogate src         # every public object has a docstring
mypy                    # type hints: src, and tests/typing's user-style code
codespell               # spelling
sphinx-build -n -W --keep-going -b html docs docs/_build/html
                        # only if you touched a docstring, docs/ or examples/
```

The docs build fetches the Python and pandas link inventories over the
network. Offline, drop `-W`: the failed fetch is then only a warning.

Add a line to [the changelog](docs/changelog.md) under `[Unreleased]`,
in the group it belongs to (Added, Changed, Fixed...), for any change a
user would notice.

Branch off `next`, the branch cloned above, and open the pull request
against it. Keep one change per pull request; the template asks what
it changes and why.

## AI tools

AI tools are welcome; pylfit itself is written with them, as
[Credits](docs/credits.md#how-pylfit-10-was-written) describes. Read
and understand every line before you open the pull request: you answer
for it in review as for code you typed. Say in the description which
tool and model helped, and how.

## License and credit

pylfit is licensed GPL-3.0-only, and a contribution is accepted under
the same license. The git history credits every contribution; someone
whose code contributions are substantial is also added to
[AUTHORS.md](https://github.com/Tony-sama/pylfit/blob/next/AUTHORS.md).

Everyone taking part is expected to follow the [Code of
Conduct](https://github.com/Tony-sama/pylfit/blob/next/CODE_OF_CONDUCT.md).
