# Behavior guarantees

This page lists the behavior pylfit promises from 1.0 on, beside the
signatures in the [API reference](api/index.md). As the
[changelog](changelog.md) says, the public API changes only with a new
major version; this page spells out what that covers. Only what is
listed under [Not promised](#not-promised) MAY change in any release.
The key words MUST, MUST NOT, SHOULD and MAY are used as in [RFC
2119](https://www.rfc-editor.org/rfc/rfc2119).

Terms: a *head* is a `(target, value)` pair; a rule's *negatives* and
*positives* are `Dataset.negatives(*rule.head)` and
`Dataset.positives(*rule.head)`. A run is *unrestricted* when none of
`allowed_conditions`, `min_support` and `min_possible_support` is set.

## Data: `Dataset`

- **DATA-1.** `Dataset` MUST store every known value as a `str`. An int
  `1`, a whole float `1.0` and `"1"` all read as `"1"`, whatever the
  column's type, and `True` reads as `"True"`.
- **DATA-2.** `Dataset` MUST raise `ValueError` for a known float that
  is not a whole number, and for a known complex, date, duration,
  `bytes` or `Decimal` value, whatever the column's type and whatever
  equal value it sits beside.
- **DATA-3.** Whatever `pandas.isna` calls missing (`NaN`, `None`,
  `pd.NA`, `NaT`) MUST be read as unknown, and every other value as a
  value: `"?"` and `""` included.
- **DATA-4.** `domain(v)` MUST return the distinct known values of `v`,
  sorted as strings.
- **DATA-5.** `positives(t, x)` MUST return one state per row whose `t`
  is `x`, in row order, a repeated row once per occurrence. Each state
  maps every feature to its value, or to `None` if unknown.
- **DATA-6.** `negatives(t, x)` MUST return, in row order, each row
  whose `t` is known and is not `x` and whose feature state no row shows
  with `t` = `x`. Two states are equal only when the same features are
  unknown in both.
- **DATA-7.** With `assume_nondeterministic=True`, `negatives` MUST also
  leave out a row that agrees with some row of `x` on every feature
  known in both. Without unknown features, it MUST change nothing.
- **DATA-8.** `conflicting_targets` MUST list, in `targets` order, each
  target for which two fully known rows share a feature state and differ
  on it. `Dataset` MUST warn with `ConflictWarning` if and only if that
  list is non-empty and `assume_nondeterministic` is false.
- **DATA-9.** `Dataset` MUST warn with `PylfitWarning` when a column has
  more than `max_categories` values, and when `check_stand_ins` is true
  and `find_missing_stand_ins` finds something in it.
- **DATA-10.** `Dataset` MUST raise `TypeError` when `features` or
  `targets` is a `str` or a set. It MUST raise `ValueError` when either
  is empty or names a missing column, or when a column appears twice
  across the two.
- **DATA-11.** `to_dataframe()` MUST return a new frame of the feature
  columns then the target columns, as stored, `NaN` where unknown, with
  the index labels of the frame passed in.

```pycon
>>> import pandas as pd
>>> from pylfit import Dataset
>>> df = pd.DataFrame({"a": [1.0, None, 0.0], "b": ["x", "?", "x"], "y": [1, 0, 1]})
>>> dataset = Dataset(df, features=["a", "b"], targets=["y"], check_stand_ins=False)
>>> dataset.domain("a"), dataset.domain("b")
(['0', '1'], ['?', 'x'])
>>> dataset.positives("y", "1")
[{'a': '1', 'b': 'x'}, {'a': '0', 'b': 'x'}]
>>> dataset.negatives("y", "1")
[{'a': None, 'b': '?'}]

```

## Rules: `Rule`

- **RULE-1.** `rule.matches(state)` MUST be true if and only if `state`
  holds the `str` value of every condition of `rule`. An unknown value
  MUST NOT satisfy any condition.
- **RULE-2.** `matches` MUST raise `KeyError` for a tested variable
  missing from `state`, and `TypeError` for a known value that is not a
  `str`. It MAY return `False` before it reaches one of them.
- **RULE-3.** Two rules MUST be equal, and hash equal, if and only if
  their heads and bodies are equal. A rule MUST NOT change after it is
  built.
- **RULE-4.** `a.is_more_general_than(b)` MUST be true if and only if
  `a` and `b` have the same head and `a`'s conditions are a strict
  subset of `b`'s.
- **RULE-5.** Rules MUST sort, with `sorted` or `<`, by their
  `to_string()` text.
- **RULE-6.** `Rule.from_string(rule.to_string())` MUST equal `rule`,
  unless a name or value hits a limit listed in `Rule.from_string`'s
  documentation.

```pycon
>>> from pylfit import Rule
>>> rule = Rule.from_string("y(1) :- a(0), c(1).")
>>> rule.matches({"a": "0", "b": None, "c": "1"}), rule.matches({"a": "0", "c": None})
(True, False)
>>> Rule.from_string("y(1) :- c(1).").is_more_general_than(rule)
True
>>> rule == Rule(("y", "1"), {"c": "1", "a": "0"})
True

```

## Learning: `GULA`, `PRIDE`, `BruteForce`

These hold for all three algorithms and every combination of options.

- **LEARN-1.** `fit` MUST return a list of distinct rules sorted by
  text. Each head MUST be a target and a value of its domain, and each
  condition a feature and a value of its domain.
- **LEARN-2.** A rule `fit` returns MUST NOT match any of its negatives
  (correctness).
- **LEARN-3.** Dropping any one condition of a rule `fit` returns MUST
  make it match one of its negatives (minimality).
- **LEARN-4.** `fit(..., heads=H)` MUST return exactly the rules of the
  same call without `heads` whose head `H` selects. A bare target name
  selects every value of it.
- **LEARN-5.** `fit` MUST return the same rules and the same warnings
  for every `n_jobs`.
- **LEARN-6.** `fit` MUST warn once with `PylfitWarning` if and only if
  some positive row of a head it learned is matched by no rule of that
  head. `warn_uncovered=False` MUST silence only that warning.
- **LEARN-7.** `fit` MUST raise `TypeError` when `dataset` is not a
  `Dataset`. It MUST raise `ValueError` when `heads` or
  `allowed_conditions` names a target, feature or value the dataset
  lacks.

- **LEARN-8.** `coverage(dataset, rules)` MUST map each target of
  `dataset` and value of its domain, and nothing else, to the share of
  that head's positive rows that a rule of that head matches. It MUST
  raise `ValueError` for a rule that concludes on anything but a target
  of `dataset`, or tests anything but a feature of it.

### GULA and BruteForce

- **GULA-1.** An unrestricted `GULA.fit` MUST return exactly the rules,
  over the dataset's domain values, that are correct and minimal for
  their head.
- **GULA-2.** `BruteForce.fit` MUST return the same rules as an
  unrestricted `GULA.fit` with the same `heads`.
- **GULA-3.** After an unrestricted `GULA.fit`, every combination of the
  features' domain values MUST be matched by a rule of each target that
  has a non-empty domain (completeness). This holds with unknown values
  too.
- **GULA-4.** After an unrestricted `GULA.fit`, a positive row MUST be
  matched by a rule of its head unless some negative of that head equals
  it on every feature the row knows.
- **GULA-5.** `min_support=k` MUST keep exactly the unrestricted rules
  matching at least `k` positive rows, a row with an unknown in a tested
  feature not matching. A float `k` means that share of the positive
  rows, rounded up.
- **GULA-6.** `min_possible_support` MUST do the same, counting a row
  where each condition holds or falls on an unknown. With both set, a
  rule MUST meet both.
- **GULA-7.** With `allowed_conditions`, a head it maps to conditions
  MUST get exactly the correct, minimal rules that use only those
  conditions. A head absent from it, or mapped to `None`, is
  unrestricted.
- **GULA-8.** `GULA`'s rules MUST NOT depend on `negative_order` or on
  the order of the rows. Without `min_support` and
  `min_possible_support`, they MUST NOT depend on repeated rows either.

```pycon
>>> from pylfit import GULA, BruteForce
>>> df = pd.DataFrame(
...     {"a": ["0", "0", "1", "1"], "b": ["0", "1", "0", "1"], "y": ["0", "1", "1", "1"]}
... )
>>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
>>> [rule.to_string() for rule in GULA.fit(dataset)]
['y(0) :- a(0), b(0).', 'y(1) :- a(1).', 'y(1) :- b(1).']
>>> GULA.fit(dataset) == BruteForce.fit(dataset)
True

```

### PRIDE

- **PRIDE-1.** Every rule `PRIDE.fit` returns MUST be one that
  `GULA.fit` returns with the same `heads`, `allowed_conditions`,
  `min_support` and `min_possible_support`, whatever
  `try_all_conditions`.
- **PRIDE-2.** Unless a restricting option is set, every positive row
  MUST be matched by a `PRIDE` rule of its head, except one that some
  negative equals on every feature the row knows.
- **PRIDE-3.** `min_support=1` and `min_possible_support=1` MUST NOT
  change `PRIDE`'s rules.

## Rule post-processing: `generalize`

These hold for every `alpha`, with or without `max_rules`, whatever
rules are given, unless a statement says otherwise.

- **GEN-1.** `generalize` MUST return a list of distinct rules sorted by
  text. Each MUST have the head of a given rule and a body that is a
  subset of a given rule's body of that head.
- **GEN-2.** Each head of a given rule whose value `dataset` holds MUST
  keep at least one rule. A rule returned MUST NOT be more general than
  another rule returned.
- **GEN-3.** With `alpha` below 1, a returned rule that has a condition
  MUST match one of its positives, and a model of `dataset` MUST accept
  the result.
- **GEN-4.** The result MUST NOT depend on the order of `rules`, on
  repeated rules, or on the order of the rows.
- **GEN-5.** `generalize` MUST warn with `PylfitWarning` if and only if
  `max_rules` is set and it returns more than `max_rules` rules. When
  `alpha` alone gives at most `max_rules` rules, `max_rules` MUST NOT
  change the result.
- **GEN-6.** Without `max_rules`, with `alpha=1`, every rule returned
  MUST be one of the given rules.
- **GEN-7.** `generalize` MUST raise `ValueError` for a rule that
  concludes on anything but a target of `dataset`, or tests anything but
  a feature of it.

## Models: `RuleSetModel`, `WeightedRuleSetModel`

- **MODEL-1.** A model's `rules` MUST be the distinct rules it was
  given, as a tuple sorted by text. Its `features`, `targets` and
  `domain` MUST be those of the dataset it was built from.
- **MODEL-2.** With `unknowns="strict"`,
  `RuleSetModel.predict_candidates` MUST give each target the sorted
  head values of the rules matching the state, or its whole domain if
  none matches.
- **MODEL-3.** With `unknowns="fallback"`, `predict_candidates` MUST
  give a target no rule matches the values of the rules that could
  match: each condition holds or falls on an unknown. On a state with no
  unknown, every prediction method MUST give the same result as with
  `"strict"`.
- **MODEL-4.** `predict_proba` MUST map each target to every value of
  its domain, in sorted order, summing to 1 up to float rounding.
  `RuleSetModel` MUST give each candidate the same share, and others 0.
- **MODEL-5.** `WeightedRuleSetModel.get_score(rule)` MUST be the rows
  `rule` matches with its head value, and those it matches with another
  known value, repeats included; `None` for a rule not in the model.
- **MODEL-6.** On a state without unknowns, `predict_likelihood` MUST
  list exactly the values of the matching rules with positives, each
  with a positive likelihood. A rule with no positives is no evidence.
- **MODEL-7.** When a target's likelihoods have a positive sum,
  `WeightedRuleSetModel.predict_proba` MUST be them divided by that sum.
  A target with no likelihood entry MUST be uniform over its domain.
- **MODEL-8.** `predict` MUST give each row and target a value of
  highest `predict_proba`, ties within a relative `1e-9`, each tied
  value equally likely; `None` only for a target whose domain is empty.
- **MODEL-9.** `predict` with the same model, rows and `random_state`
  MUST give the same result in every process. A row with one highest
  value MUST get it whatever `random_state`.
- **MODEL-10.** The `DataFrame` and single-state forms of
  `predict_proba` and `predict_likelihood` MUST give the same result for
  each row.
- **MODEL-11.** Prediction MUST raise `ValueError` for a frame whose
  columns are not exactly the features, or for a value outside the
  model's `domain`. A single state MUST raise `KeyError` for a missing
  feature and ignore other keys.
- **MODEL-12.** Two models MUST be equal if and only if they have the
  same class, features, targets, rules and domain, plus the same scores
  and `value_frequencies` for `WeightedRuleSetModel`.

```pycon
>>> from pylfit import RuleSetModel, WeightedRuleSetModel
>>> model = WeightedRuleSetModel(dataset, GULA.fit(dataset))
>>> (
...     model.get_score(Rule.from_string("y(1) :- a(1).")),
...     sorted(model.predict_likelihood({"a": "1", "b": "1"})["y"]),
... )
((2, 0), ['1'])
>>> RuleSetModel(dataset, model.rules).predict_candidates({"a": "0", "b": "0"})
{'y': ['0']}
>>> model.predict_likelihood({"a": None, "b": "0"}, unknowns="strict")
{'y': {}}
>>> sorted(model.predict_likelihood({"a": None, "b": "0"})["y"])
['0', '1']

```

## Saved files

- **FILE-1.** `cls.from_file(path)` after `model.save(path)` MUST return
  a model of `model`'s class equal to `model`, for both model classes.
- **FILE-2.** `from_file` MUST raise `ValueError` for a file another
  model class saved, or with a `schema_version` it does not read.
- **FILE-3.** A saved file MUST be a UTF-8 JSON object holding
  `schema_version`, `model_class`, `pylfit_version`, `features`,
  `targets`, `domain`, and `rules`, each with `head`, `body` and a
  `display` text that `from_file` MUST ignore. A `WeightedRuleSetModel`
  file MUST also hold each rule's `score`, and `value_frequencies`.
- **FILE-4.** Equal models MUST be saved as byte-identical files by the
  same pylfit version.
- **FILE-5.** A 1.x release MUST read every file saved by an earlier 1.x
  release.

```pycon
>>> import os, tempfile
>>> with tempfile.TemporaryDirectory() as folder:
...     path = os.path.join(folder, "model.json")
...     model.save(path)
...     WeightedRuleSetModel.from_file(path) == model
True

```

## Metrics

- **METRIC-1.** `accuracy_score` MUST be the share of rows with a known
  actual value whose prediction equals it, both written as DATA-1 writes
  them; an unknown or `None` prediction is wrong. It MUST raise
  `ValueError` when `y_true` and `y_pred` differ in length or no actual
  value is known.
- **METRIC-2.** `accuracy_by_target` MUST map each target of `targets`,
  once and in order, to what `accuracy_score` gives for it.
- **METRIC-3.** Without `states`, `rule_recovery` MUST put each rule in
  the first of `exact`, `too_specific`, `too_general` and `partial` that
  it has with a rule of its head on the other side, else in `spurious`
  or `missing`. `precision` and `recall` MUST be the learned and true
  shares of `exact` plus `too_specific`. Each side's shares MUST sum to
  1 up to float rounding, the learned side's being all 0 when no learned
  rule is given. A rule given twice MUST count once, with `states` too.
- **METRIC-4.** With `states`, every number `rule_recovery` returns MUST
  be its mean, over each state and each target of a true rule, between
  the rules matching that state. It MUST raise `ValueError` when no true
  rule of a target matches a state.

## Checks: `find_missing_stand_ins`

- **CHECK-1.** `find_missing_stand_ins` MUST report as `"stand_in"`
  exactly the distinct `str` values of the columns checked whose text,
  stripped and lower-cased, is that of an entry of `stand_ins`, with
  their row counts. `stand_ins` MUST default to `DEFAULT_STAND_INS`, a
  tuple of `str`.
- **CHECK-2.** The report MUST have the columns `column`, `value`,
  `count` and `issue`, one row per finding, grouped by column in the
  order checked. It MUST NOT report a value `pandas.isna` calls missing.
- **CHECK-3.** `find_missing_stand_ins` MUST NOT change `dataframe`. A
  `"far_zero"` finding MUST be a number or a text reading 0, never a
  `bool`.

## Determinism and warnings

- **RUN-1.** Given the same input, row order and `random_state`,
  everything pylfit returns or writes MUST be the same in every process,
  whatever `PYTHONHASHSEED`. Only `verbose` output is exempt.
- **RUN-2.** Every warning pylfit gives MUST be a `PylfitWarning`,
  itself a `UserWarning`.

## Not promised

Do not rely on the following. Each MAY change in any release.

- **The text of errors and warnings.** Only their class and the
  condition that triggers them are promised. Match on the class, never
  on the message.
- **Run time, memory, and the speedup from `n_jobs`.**
- **Which rules `generalize` returns for a given `alpha`**, and which
  of a nested pair it keeps. The test and its levels MAY improve. A
  lower `alpha` usually gives fewer rules and conditions, but not
  always.
- **Which `alpha` the `max_rules` search settles on.** It MAY return
  fewer rules than a larger `alpha` would within the cap.
- **That `generalize`'s rules are correct, minimal or cover every
  positive row.** LEARN-2, LEARN-3 and LEARN-6 do not apply to them.
- **Which correct, minimal rules `PRIDE` returns.** Reordering rows, or
  removing repeated rows, MAY change them. So MAY `try_all_conditions`,
  which does not return a superset of the default run.
- **The order of domain values beyond plain `str` order.** Domains sort
  as strings, so `"10"` comes before `"2"`. No numeric or natural order
  is promised.
- **How `predict_likelihood` weighs a rule's counts.** Today a rule
  scored `(p, n)` votes `p * p / (p + n)`, and a value's votes add up.
  The formula MAY be improved.
- **Which markers and zeros `find_missing_stand_ins` looks for.** The
  list MAY grow.
- **The exact floats of `predict_proba` and `predict_likelihood`** below
  about `1e-9` relative. They are not calibrated probabilities.
- **Which value a seeded tie gets**, between pylfit versions, or when
  the rows or calls are split or reordered. Within one version, MODEL-9
  applies.
- **The output of `verbose`, `str()` and `repr()`** of datasets, rules
  and models, apart from `Rule.to_string`, which RULE-6 covers.
- **Results across versions before 1.0.** Rules, predictions and files
  of 0.x and of 1.0 pre-releases MAY differ from 1.0. A file is read
  only by the `schema_version` it records.
- **The fallback's numbers on unknown values.** How
  `WeightedRuleSetModel` weighs a rule that could match is a heuristic,
  which MAY be improved; MODEL-3 fixes only when it applies.
