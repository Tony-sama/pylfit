# Architecture

pylfit's parts pass data in one direction; `generalize` is an optional
step on the rules:

```text
pandas.DataFrame ──► Dataset ──► algorithm.fit ──► list of Rule ──► model ──► predictions ──► metrics
                                                    list of Rule ──► generalize ──► list of Rule
```

| Part | Module | Role |
| --- | --- | --- |
| `Rule` | `_rule.py` | A head and a body; matches any state with a `.get` method. Imports no other pylfit module. |
| `Dataset` | `_dataset.py` | Checks a `DataFrame` once and stores every value as a string, unknowns as missing; answers positives and negatives for a head. |
| Checks | `_checks.py` | `find_missing_stand_ins`: reports values in a `DataFrame` that may stand for a missing value. `Dataset.__init__` calls it on its features and targets and warns about what it finds; nothing else calls it. |
| Algorithms | `algorithms/` | `BruteForce`, `GULA`, `PRIDE`: stateless, `fit(dataset) -> list[Rule]`, sorted by text. `Algorithm` holds what they share: the loop over heads (`heads`, `n_jobs`, the same order on every run), argument checks, progress logging, the uncovered-rows warning. |
| Models | `models/` | `RuleSetModel` and its subclass `WeightedRuleSetModel` predict from rules and save to JSON. `_helpers.py` holds shared validation functions. |
| `generalize` | `_generalize.py` | Rules and a `Dataset` to the rules without the conditions the data doesn't need, by a one-sided Fisher exact test at `alpha`. Imports no models or algorithms. |
| Metrics | `_metrics.py` | Plain functions over `(y_true, y_pred)`, `coverage` over rules and a `Dataset`, and `rule_recovery` over two rule sets; no dependency on models. |
| Warnings | `_exceptions.py` | `PylfitWarning`, the class of every warning pylfit gives, and its subclass `ConflictWarning`. |
| Progress | `_progress.py` | The `verbose` output: one tagged line to stderr, and a progress line at most about once a second. Used by the algorithms and `WeightedRuleSetModel`. |

Every module that defines code is private, its name starting with
`_`: the public names are imported from `pylfit` (or from
`pylfit.models` and `pylfit.algorithms`), so a module can be split or
renamed without breaking anyone.

Algorithms don't import models, and models don't import algorithms:
the list of rules between them is the only contract. The
[design notes](design-notes.md) say why.
