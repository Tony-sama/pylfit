# Why pylfit, and when not to use it

This page is for deciding whether pylfit fits your problem. It says
what pylfit does that other rule learners don't, and when another tool
is the better choice.

## What pylfit does differently

- It learns every rule the data doesn't contradict. `GULA` returns every
  rule that matches no negative example and can't be made more general:
  the full set, not a few rules picked to cover the data. For a question
  like "what could explain this outcome?", no rule that fits the data is
  left out. That suits small to mid-size problems: `GULA`'s worst case
  is exponential in the number of features and their domain sizes.
  `PRIDE` returns some of those rules, enough to cover every row. Its
  cost is polynomial in the size of the data (Ribeiro et al., IJCLR
  2021, [hal-03347026](https://hal.science/hal-03347026)) and doesn't
  grow with the domain sizes: on the soybean data, 35 features, it
  fits in under 0.01 s against 0.3 s for `GULA`. See
  [Prepare your data](../how-to/prepare-data.md#keep-the-problem-a-comfortable-size)
  for measured sizes.
- It also gives short rule sets, near the true ones.
  [`generalize`](../api/generalize.md) drops each condition the data
  doesn't need, keeps one rule of each pair where one is a narrower
  copy of the other, and `max_rules` caps the rules returned: on the
  UCI heart disease data, `GULA`'s ~12000 rules become ~27 and predict
  as well (0.82 held-out accuracy, against 0.81), and `PRIDE`'s become
  ~17 (0.82); see [the heart disease
  tutorial](../auto_examples/tutorial_heart_disease.rst). On Boolean
  networks learned from 70% of the states with 10% of the labels
  flipped, 92% of the rules left are true rules and the rest mostly a
  true rule a condition or two short, none spurious.
- Its guarantees are checked. Correctness, completeness and minimality
  are defined on [About LFIT rules](about-rules.md) and checked by the
  tests on example tables and on Boolean networks of up to 13
  variables; on those of up to 6, `GULA`'s rules are also checked equal
  to `BruteForce`'s exhaustive search.
- Learning and predicting are separate steps. The same rules can
  predict every possible value or a ranked value; see [About the
  prediction models](about-models.md).
- It learns dynamics. A Boolean network, or any discrete dynamical
  system, is learned from observed transitions with the same API; see
  [Learn a gene regulatory network](../tutorials/gene-network.md).
- It handles several targets at once, each with its own rules.
- It learns and predicts with unknown values, without filling them in;
  see [Unknown values](unknown-values.md).

## When to use something else

- You care about prediction accuracy, not explanation. Statistical
  machine learning methods, such as gradient-boosted trees or neural
  networks, are usually more accurate: rules give up some
  representational power in exchange for being readable.
- Your features are mostly numeric. pylfit compares categories by
  equality, so you choose the bins yourself. Other rule learners place
  thresholds on numeric features as they learn.
- Your labels are very noisy and only accuracy matters. pylfit holds up
  under wrong labels: each rule agrees with every training example, a
  mislabelled one included, but the vote of `WeightedRuleSetModel`
  outweighs the rules a few wrong labels create, and `generalize`
  recovers the true rules. With 20% of the training labels flipped, on
  Boolean networks of 9 and 10 variables, `GULA`'s rules score 0.96
  against 0.95 for a random forest, and after `generalize` 72% of the
  true rules are found exactly, in about 2 rules per node. RIPPER
  predicts better, 0.97, and finds 57% of the true rules; a decision
  tree pruned by cross-validation predicts better still, 0.985, from
  fewer rules that are less often the true ones.
- You want an estimator that plugs into scikit-learn pipelines and grid
  search. pylfit's API is its own.
- You want an ordered decision list, where the first rule that
  matches decides: RIPPER learns one class at a time, rarest first,
  and makes the most common class the default; the original CN2 is
  ordered too. pylfit's matching rules vote instead, as in CN2's
  unordered version.

## Status

pylfit 1.0 is a ground-up rewrite, in alpha: 1.0.0a1 is on PyPI as a
pre-release. The core is complete for classification: data, rules, the
three algorithms, `generalize`, the two models, saving and loading,
and metrics.

Until 1.0.0 is released, the API may still change. From 1.0.0 on,
changes follow the versioning policy in the [changelog](../changelog.md).
