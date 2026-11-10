# Migrate from pylfit 0.x

`pylfit` 1.0 is a rewrite. It keeps the LFIT learning algorithms but
not the 0.x API, so code written for 0.x will not run against 1.0
unchanged. You have two options: stay on 0.x, or port your code. This
page covers both. A runnable port of 0.x's own examples is in
`examples/migrate_from_0x.py`.

## Option 1: stay on 0.x

```bash
pip install "pylfit<1" numpy pandas
```

`numpy` and `pandas` have to be named: 0.5.1 uses both but does not
declare them, so `pip install "pylfit<1"` alone installs a package that
fails on `import pylfit`. With them, 0.5.1's two main examples ran
unchanged on Python 3.11 with numpy 2.4 and pandas 3.1, given the
`tmp/` folder one of them saves to. The source is the `v0.5.1` tag of
the repository.

Stay on 0.x if your code needs anything listed under
[No equivalent in 1.0](#no-equivalent-in-10), or reproduces a published
result.

## Option 2: port to 1.0

1.0 is a pre-release for now, and `--upgrade` replaces an installed 0.x:

```bash
pip install --upgrade "pylfit>=1.0.0a1"
```

### What maps to what

| 0.x | 1.0 |
| --- | --- |
| `preprocessing.discrete_state_transitions_dataset_from_array(data, feature_names, target_names)` | One `DataFrame` row per transition (state in the feature columns, next state in the target columns), then `Dataset(df, features, targets)` |
| `preprocessing.discrete_state_transitions_dataset_from_csv(path, ...)` | `pd.read_csv(path, dtype=str)`, then `Dataset(...)` |
| `"?"` for an unknown value | `None` or `NaN`, for example `df.replace("?", None)`; in 1.0 `"?"` is an ordinary value |
| `models.DMVLP(...)` + `compile(algorithm="gula")` + `fit(dataset)` | `RuleSetModel(dataset, GULA.fit(dataset))` |
| `compile(algorithm="pride")` | `PRIDE.fit(dataset)` |
| `models.WDMVLP(...)` | `WeightedRuleSetModel(dataset, GULA.fit(dataset))` — see [The weighted model](#the-weighted-model) |
| `fit(dataset, options={"targets_to_learn": {"y": ["1"]}})` | `GULA.fit(dataset, heads=[("y", "1")])`; a bare `"y"` in `heads` means all its values |
| `fit(dataset, options={"threads": 4})` | `GULA.fit(dataset, n_jobs=4)`, one process per head being learned; `-1` for every CPU |
| `fit(dataset, options={"heuristics": ["try_all_atoms"]})` with `"pride"` | `PRIDE.fit(dataset, try_all_conditions=True)` |
| `"max_coverage_dynamic"` or `"max_coverage_static"` in `heuristics` | Nothing: 1.0's `PRIDE` always picks conditions by coverage |
| `model.rules` | `model.rules` (a `tuple` of `Rule`, sorted by text; for other rules, build a new model: `RuleSetModel(dataset, rules)`) |
| `rule.add_condition(variable, value)`, `rule.remove_condition(variable)` | `new = rule.with_condition(variable, value)`, `new = rule.without_condition(variable)`: each returns a new rule and leaves `rule` as it was. `r = rule.copy(); r.add_condition(v, x)` becomes `r = rule.with_condition(v, x)` |
| `rule.copy()` | Nothing: a rule can't be changed after it's made, so use the rule itself wherever a copy was used |
| `model.summary()` | `print(model)` |
| `model.predict([state], semantics=...)` | `model.predict_candidates(...)` plus a few lines per semantics — `next_states` in the example |
| `model.feature_states()` | `itertools.product` over `dataset.domain(...)` — `feature_states` in the example |
| `dataset.to_csv(path)` | `dataset.to_dataframe().to_csv(path, index=False)` |
| `postprocessing.accuracy_score(model, dataset)`, `accuracy_score_from_predictions(predictions, dataset)` | `accuracy_score(y_true, y_pred, target)`, with `y_pred` from `model.predict(...)`. A different measure: 0.x compared each value's predicted probability with the values seen from each state, 1.0 counts the rows whose predicted value is the actual one |
| `postprocessing.explanation_score(model, expected_model, dataset)` | `rule_recovery(true_rules, model.rules, states)`. A different measure: 0.x scored the one rule behind each right prediction by its distance to the nearest true rule, 1.0 every rule matching each state |

1.0 has no state-transition classes. A transition is an ordinary row,
and the learners treat it like any other classification data. That is
why three short helpers stand in for what 0.x did internally:
`transitions_frame`, `feature_states` and `next_states`, all in
`examples/migrate_from_0x.py`. Copy them.

### Before and after

Both versions start from `data`, 0.x's example: each state of three
Boolean variables, paired with the state that follows it.

```python
data = [
    (["0", "0", "0"], ["0", "0", "1"]),
    (["1", "0", "0"], ["0", "0", "0"]),
    (["0", "1", "0"], ["1", "0", "1"]),
    (["0", "0", "1"], ["0", "0", "1"]),
    (["1", "1", "0"], ["1", "0", "0"]),
    (["1", "0", "1"], ["0", "1", "0"]),
    (["0", "1", "1"], ["1", "0", "1"]),
    (["1", "1", "1"], ["1", "1", "0"]),
]
```

```python
# 0.x
import pylfit

dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(
    data=data,
    feature_names=["p_t_1", "q_t_1", "r_t_1"],
    target_names=["p_t", "q_t", "r_t"],
)
model = pylfit.models.DMVLP(features=dataset.features, targets=dataset.targets)
model.compile(algorithm="gula")
model.fit(dataset=dataset)
prediction = model.predict([("1", "0", "1")], semantics="synchronous")
```

```python
# 1.0
import pandas as pd
from pylfit import GULA, Dataset, RuleSetModel

# next_states, State and import itertools: from examples/migrate_from_0x.py
features, targets = ["p_t_1", "q_t_1", "r_t_1"], ["p_t", "q_t", "r_t"]
df = pd.DataFrame([s + t for s, t in data], columns=features + targets)
dataset = Dataset(df, features=features, targets=targets)
model = RuleSetModel(dataset, GULA.fit(dataset))
successors = next_states(model, ("1", "0", "1"), "synchronous")
```

The names keep 0.x's convention: `p_t_1` is `p` at the previous step,
`t-1`, and `p_t` is the value that follows. The rest of these docs name
the current state `p_t` and the next one `p_t+1`. pylfit gives column
names no meaning, so either works; keep the one your data already uses.

## What stays the same

These were checked by running pylfit 0.5.1 and 1.0 side by side on
the same data: 0.x's own example, a non-deterministic variant, a
partial one, three random Boolean systems of 4 to 6 variables, and
50% samples of two of them.

- **`GULA` learns exactly the same rules.**
- **The synchronous, asynchronous and general semantics give exactly
  the same next states**, for every state of every system tried, with
  the helpers from the example.
- **Rules print in the same format**: `p_t(0) :- q_t_1(0).`

## What differs

- **A variable no rule matches.** 0.x predicted `"?"` for it, or the
  value you passed as `default`. 1.0 gives every known value of that
  variable: nothing in the data ruled any of them out. With `GULA`'s
  default `min_support`, some rule always matches, so this only shows
  with a higher `min_support` or with rules from elsewhere.
- **`PRIDE` can return different rules.** It is a covering heuristic:
  it finds *a* set of rules that explains every observation and
  contradicts none, not a unique one. 1.0's `PRIDE` is a
  reimplementation and can choose differently from 0.x, even on a few
  rows. 0.x's choice also changed from one run to the next, as it
  followed Python's string hashing; 1.0 gives the same rules on every
  run. Both versions' rules were checked against every dataset above:
  always consistent with the data and covering every observation.
- **Unknown values on deterministic data.** 0.x left out of a value's
  negative examples every row whose unknowns could hide a state seen
  with that value. That is right for a non-deterministic system, and
  1.0 does it with `Dataset(..., assume_nondeterministic=True)`. On a
  deterministic system those rows are true negatives, so 1.0 keeps
  them by default, and its rules are less general than 0.x's there.
- **Unknown target values.** 0.x counted a row whose target was unknown
  as a positive example of every value of that target. 1.0 leaves the
  row out for that target only, since those positives never happened.
  So on data with unknown targets 1.0's rules differ from 0.x's, with
  `assume_nondeterministic=True` too.
- **The weighted model** works differently, below.

### The weighted model

0.x's `WDMVLP` learned two rule sets per value: rules for it
("likeliness") and rules against it ("unlikeliness"). Its prediction
was a probability: the best "for" rule's weight divided by that weight
plus the best "against" rule's weight.

1.0's `WeightedRuleSetModel` has no rules against a value. Each rule
carries `(positives, negatives)` counts: the training rows it explains
and the rows it gets wrong (`model.get_score(rule)`).
`predict_likelihood` combines the matching rules' counts into one score
per value. In practice:

- **The rules are 0.x's likeliness rules**, and `positives` is 0.x's
  likeliness weight, except that 1.0 counts rows where 0.x counted
  distinct states. A state observed twice counts twice in 1.0.
- **Scores are not 0.x's probabilities**, so don't compare numbers
  across versions. Compare which value comes out on top.
- **On non-deterministic data**, such as 0.x's own weighted example,
  1.0 picks the same most likely value as 0.x everywhere except the one
  state observed with two different outcomes. There, 0.x calls it a tie
  and 1.0 picks a side.
- **On deterministic data**, the most likely value matched 0.x on
  every observed state. On states never observed, the two versions
  generalize differently: they agreed on 90% to 100% of predictions,
  and most disagreements are a tie one version breaks.
- **Explanations:** 0.x returned the best "for" rule with each value.
  In 1.0, take the matching rule with the most positives. The example
  shows how.

### Speed and memory

1.0's `GULA` learns the same rules as 0.5.1's, four to six times
faster. On the full transition table of a 10-variable Boolean network
(1024 rows), a fit took 0.08 s in 1.0 against 0.4 to 0.5 s in 0.5.1. A
first fit in 1.0 also builds the dataset's row caches: it peaked at
about 0.8 MB of memory, against 0.45 MB for 0.5.1, and the caches,
about 0.7 MB, stay with the `Dataset`. A later fit peaked at about
0.1 MB.

## No equivalent in 1.0

These 0.x features are not in 1.0. Code that needs them stays on 0.x
([Option 1](#option-1-stay-on-0x)):

- `CDMVLP`, `PDMVLP` and `CLP` models
- `Synchronizer`, `Probalizer`, ACEDIA, LUST and LFkT
- counterfactuals (`postprocessing.compute_counterfactuals` and
  `bruteforce_counterfactuals`)
- `postprocessing.explanation_score_from_predictions` and
  `postprocessing.hamming_distance`
- continuous datasets (`continuous_state_transitions_dataset_from_*`)
- loading a model from a Boolean network file
  (`dmvlp_from_boolean_network_file`). It also dropped brackets, reading
  `(a | b) & c` as `a | b & c`, so results on such a network won't carry
  over
- `PRIDE`'s `max_diversity` and `multi_thread_at_rule_level`
  heuristics
- `model.extend(dataset, feature_states)` and the
  `PRIDE.find_one_optimal_rule_of` it called, which add to a model a
  rule matching each given state, where the data allows one
