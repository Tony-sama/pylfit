# Changelog

Every change users would notice, newest first, in the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format: a
version's changes grouped as Added, Changed, Deprecated, Removed, Fixed
and Security.

## Versioning

pylfit follows [semantic versioning](https://semver.org): a version is
`MAJOR.MINOR.PATCH`.

- The public API is every name importable from `pylfit` directly, with
  its documented behavior, and the saved-model file format.
- A breaking change to the public API comes only with a new major
  version.
- A name to be removed first gives a deprecation warning for at least
  one minor release.
- Versions ending in `aN`, `bN`, `rcN` or `.devN` are pre-releases:
  `pip install pylfit` skips them unless you pass `--pre`.

The installed version is `pylfit.__version__`:

```bash
python -c "import pylfit; print(pylfit.__version__)"
```

Model files record the pylfit version that wrote them and a schema
version. A release that changes the file format says so here.

## [Unreleased]

## [1.0.0a1] - 2026-11-05

First alpha of 1.0. A ground-up rewrite, incompatible with 0.x; see
[Migrate from pylfit 0.x](how-to/migration.md).

### Added

- Unknown values: `NaN` and `None` are read as values nobody recorded.
  `Dataset` keeps them, every algorithm learns from the known values,
  and the models predict rows with unknowns, falling back to the rules
  that could match (`unknowns="fallback"`, or `"strict"`). In 0.x an
  unknown was written `"?"`; in 1.0 `"?"` is an ordinary value. See
  [Unknown values](explanation/unknown-values.md).
- `Dataset.conflicting_targets` lists the targets with rows of the
  same features and different values, and `Dataset` warns about them.
  `Dataset(..., assume_nondeterministic=True)` states that the system
  can give several values from one state.
- `GULA` and `PRIDE` take `heads`, `allowed_conditions`, `min_support`
  and `min_possible_support`. All three algorithms warn when positive
  rows end up matched by no rule (`warn_uncovered`).
- `GULA` takes `negative_order`, the order it reads negative examples
  in: `"similarity"` (the default), `"lexicographic"` or
  `"unordered"`. It changes how long a fit takes, never its rules.
- All three algorithms learn heads in parallel processes with
  `n_jobs` (default `1`, `-1` for every CPU), replacing 0.x's
  `threads`, with the same rules as a sequential run; `BruteForce`
  takes `heads` too.
- `pylfit.coverage` gives the share of positive rows a rule set
  matches, per head. `pylfit.find_missing_stand_ins` lists values in a
  `DataFrame` that may stand for a missing one, and `Dataset` warns
  about them (`check_stand_ins=False` to silence it).
- `Dataset` warns about a feature or target column with more than
  `max_categories` distinct values (default 5), whatever its type, and
  says whether it looks like a measurement, many categories or an
  identifier.
- `pylfit.rule_recovery` compares learned rules with known true rules:
  each rule is exact, too specific, too general, partial, or spurious
  (learned) or missing (true), with a precision and a recall, over the
  whole sets or at given states, between the rules matching each.
- `pylfit.generalize(rules, dataset, *, alpha=0.01, max_rules=None)`
  drops each rule's conditions the data does not need, by a one-sided
  Fisher exact test, and keeps no rule more general than another of
  its head: on the UCI heart disease data, `GULA`'s ~12000 rules
  become ~27, and a `WeightedRuleSetModel` of them scores about the
  same accuracy. `max_rules` caps the rules returned, every target
  together, lowering `alpha` until the cap holds.
- `WeightedRuleSetModel` files record `value_frequencies`, the share of
  each feature value, which the fallback prediction uses. A file
  without it loads with every value equally likely.
- `Dataset.to_dataframe()` returns the table a `Dataset` learns from,
  every value as pylfit reads it, the rows labeled as in the
  `DataFrame` it was built from; write it to a file with pandas.
- `pylfit.PylfitWarning`, the class of every warning pylfit emits, a
  `UserWarning`, and `pylfit.ConflictWarning`, its subclass for
  conflicting rows in a `Dataset`: filter them by class rather than by
  message.

### Changed

- Data goes in as a `pandas.DataFrame` wrapped in a `Dataset`.
- Learning algorithms `GULA`, `PRIDE` and `BruteForce` return a plain
  list of rules, sorted by text; models `RuleSetModel` and
  `WeightedRuleSetModel` predict from them.
- Everything pylfit returns, prints or saves is the same on every run,
  whatever `PYTHONHASHSEED`: `model.rules` is a sorted `tuple`, and
  `Dataset.domain()`, `Dataset.domains()`, `model.domain` and
  `predict_candidates` give sorted lists. Compare rule lists as sets,
  `set(a) <= set(b)`: `<=` on two lists is not a subset test.
- Dynamics is learned as classification on `(t, t+1)` rows; the 0.x
  dynamics API is gone.
- Import from `pylfit`, or from the `pylfit.models` and
  `pylfit.algorithms` subpackages: the modules defining the code are
  private (`_`-prefixed), as in scikit-learn, and may move.
- Requires Python 3.11 or later, pandas 2.3 or later and NumPy 2.2 or
  later.
- `PRIDE` adds, of the conditions excluding a negative, the one keeping
  the most uncovered positives matched (0.x's coverage heuristics,
  rebuilt; now always on), and takes 0.x's `try_all_atoms` as
  `try_all_conditions` (default `False`), "condition" being the
  package's word for a `(feature, value)` pair. It takes examples in
  row order. 0.x's other heuristics are gone.

### Removed

- The 0.x features listed under
  [No equivalent in 1.0](how-to/migration.md#no-equivalent-in-10):
  the other models and learning algorithms, counterfactuals,
  continuous datasets, loading a model from a Boolean network file,
  and two of `PRIDE`'s heuristics. Code that needs them stays on 0.x.

Thanks to Juliette Audemard, who used the development versions: her
feedback shaped `Dataset`'s positive and negative examples, and her
prototype led to `GULA`'s `allowed_conditions`.

[Unreleased]: https://github.com/Tony-sama/pylfit/compare/v1.0.0a1...next
[1.0.0a1]: https://github.com/Tony-sama/pylfit/compare/v0.5.1...v1.0.0a1
