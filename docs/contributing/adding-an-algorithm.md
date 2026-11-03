# Add an algorithm to pylfit

This page is for adding an algorithm to pylfit itself. To use one of
your own without changing pylfit, a plain function returning a list of
`Rule` is enough: see [Use your own algorithm](../how-to/your-own-algorithm.md).

A pylfit algorithm is a class with a static `fit` method that takes a
`Dataset` and returns a list of `Rule`, sorted, without duplicates. It
holds no state and doesn't
depend on the models: callers wrap its rules in whichever model they
choose.

## Write it

Subclass `Algorithm` and implement its abstract static method
`fit(dataset, *, verbose=False)`. Every argument but `dataset` is
keyword-only, so a call means the same whichever algorithm it goes to;
options of your own go after `verbose`, with defaults. `Algorithm` is
private, the base of pylfit's own algorithms rather than part of the
public API, so this is the one place the docs import from a private
module.

pylfit's algorithms learn each head, a target value such as `y(1)`, on
its own. Write that as a static method, `_learn_head(dataset, target,
value, verbose)`, returning the head's rules and its uncovered rows:
the positive rows no rule matches, as `(head, state, reason)` triples,
or an empty list if the algorithm always covers them. Then `fit` hands
it to `Algorithm._fit_heads`, the loop the other algorithms share. That
loop takes the options every algorithm has, `heads`, `warn_uncovered`
and `n_jobs`, checks them, learns the heads in the same order on every
run, and returns `sorted(rules)` without duplicates. For `n_jobs` the
head function must be picklable: a static method is, and so is a
`functools.partial` of one, which is how `GULA` binds its own options.

`Algorithm` also gives two class methods for `verbose` output:
`ShortRules._log(message)` prints one line to stderr, tagged with the
class name, and `ShortRules._progress(stage, total)` returns an object
whose `update(current)` prints a progress line at most about once a
second, with an estimate of the time left. This one learns every
consistent rule with at most one condition:

```pycon
>>> from pylfit import Rule
>>> from pylfit.algorithms._algorithm import Algorithm
>>> class ShortRules(Algorithm):
...     @staticmethod
...     def fit(dataset, *, verbose=False, heads=None, warn_uncovered=True, n_jobs=1):
...         return ShortRules._fit_heads(
...             dataset, ShortRules._learn_head, heads, verbose, warn_uncovered, n_jobs
...         )
...
...     @staticmethod
...     def _learn_head(dataset, target, value, verbose):
...         negatives = dataset.negatives(target, value)
...         candidates = [Rule((target, value))] + [
...             Rule((target, value), {feature: v})
...             for feature in dataset.features
...             for v in dataset.domain(feature)
...         ]
...         rules = [r for r in candidates if not any(r.matches(n) for n in negatives)]
...         return rules, []

```

It is called like the algorithms already in pylfit:

```pycon
>>> import pandas as pd
>>> from pylfit import Dataset
>>> df = pd.DataFrame(
...     {"a": ["0", "0", "1", "1"], "b": ["0", "1", "0", "1"], "y": ["0", "1", "1", "1"]}
... )
>>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
>>> [rule.to_string() for rule in ShortRules.fit(dataset)]
['y(1) :- a(1).', 'y(1) :- b(1).']

```

`dataset.negatives` and `dataset.positives` give the states a rule must
not and should match; see [Use your own
algorithm](../how-to/your-own-algorithm.md) for how to use them and
how to check the result against `BruteForce`.

## What an algorithm promises

Which properties an algorithm promises is part of its docstring, and
its tests check exactly those. Compare with `BruteForce` in the way
the promise calls for: equality for an algorithm that returns every
minimal rule, a subset for one that returns some of them, as `PRIDE`
does. An algorithm that promises neither, such as a baseline that isn't
correct, gets tests of what it does promise, and no comparison.

## Checklist for a pull request

1. The class, in `src/pylfit/algorithms/_<name>.py`, with a Google-style
   docstring: a module docstring stating the properties it promises,
   and on `fit`, Args, Returns, and a short doctest `Examples:`
   section. The full rules are in [Standards](standards.md#writing-docstrings-and-pages).
2. Exports: import the class and add its name to `__all__` in
   `src/pylfit/algorithms/__init__.py` and `src/pylfit/__init__.py`;
   `tests/test_public_api.py` checks the two agree.
3. Tests, in `tests/algorithms/test_<name>.py`, using the shared
   fixtures (see [Testing](testing.md)). If it promises correctness,
   completeness or minimality, add it to `LEARNERS` in
   `tests/algorithms/test_learners.py` and to the checks in
   `tests/algorithms/test_boolean_networks.py`; add it to `ALGORITHMS`
   in `tests/algorithms/test_n_jobs.py` too. Each names the algorithms
   it runs; nothing is picked up automatically.
4. An API page, `docs/api/algorithms/<name>.md`, modeled on
   `gula.md`: its module docstring through `.. automodule::` with
   `:no-members:` and `:no-index:` (the module is private), then the
   class through `.. autoclass::` under `.. currentmodule:: pylfit`,
   so it is documented where users import it. Add its name to the
   toctree of `docs/api/algorithms/index.md`.
5. Pages that list the algorithms: [About the
   algorithms](../explanation/about-algorithms.md) and the choice table
   in [Prepare your data](../how-to/prepare-data.md).
6. A line in `docs/changelog.md`, under the unreleased version.
7. The checks in [Before opening a pull
   request](index.md#before-opening-a-pull-request). `ruff` reads its
   rules from `pyproject.toml`; write type hints in the Python 3.11
   form, `list[Rule]` and `str | None`, as the existing signatures do.

The existing tests show the pattern: `tests/algorithms/test_learners.py`
checks each property on every shared fixture, for every learner, and
`test_gula.py` compares `GULA` with `BruteForce` on each of them.
