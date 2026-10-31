# About the algorithms

pylfit has three learning algorithms. All three are stateless: `fit`
takes a `Dataset` and returns a list of rules, sorted by text, which
you then wrap in a model. This page explains how each searches, what
their options trade, and how `generalize` trims the rules afterwards.
The properties they guarantee are defined in
[About LFIT rules](about-rules.md).

## BruteForce: enumerate everything

`BruteForce` builds every possible rule for each `(target, value)`
pair and keeps those that are correct and minimal. Take a table where
`y` is `1` when `b` is `1`, or when `a` is `0` and `c` is `1`, with all
eight states, and the rules for `y(1)`. Step through the three steps
with Next:

````{container} pylfit-stepper
```{figure} ../images/bruteforce-1-candidates.svg
---
alt: >-
  The eight rows of the table, and the 27 rules for y(1) as a grid
  of boxes, each box listing a rule's conditions, from the empty rule
  to a(1), b(1), c(1).
---

A rule can leave each feature out or test it for `0` or `1`, so there
are 27 rules for `y(1)`, listed here by their conditions: one block
per condition on `b`, one row per condition on `a`, one column per
condition on `c`.
```

```{figure} ../images/bruteforce-2-consistent.svg
---
alt: >-
  The same grid. The rows where y is 0 are red in the table, and
  the sixteen rules that match one of them are red. The eleven others,
  among them b(1) and a(0), c(1), stay white.
---

A rule is correct when it matches no negative example of `y(1)`: no
row where `y` is `0`, in red in the table. Sixteen of the rules match
one, in red too, and eleven are left.
```

```{figure} ../images/bruteforce-3-minimal.svg
---
alt: >-
  The same grid, red and white as before. Two white rules are green,
  b(1) and a(0), c(1); the empty rule, a(0) and c(1), red, have a
  green border: each is a green rule with a condition dropped.
---

A correct rule is minimal when no other correct rule has a subset of
its conditions: dropping any one of them makes it match a negative
example (a green border marks what is left when a condition is
dropped). Two are left, in green, `y(1) :- b(1).` and
`y(1) :- a(0), c(1).`, the definition the table was built from.
```
````

`BruteForce` doesn't compare rules to find them: one pass over the
negatives judges each rule on its own. A negative that violates none
of a rule's conditions shows the rule is incorrect, and one that
violates exactly one condition shows that condition is needed.
Checking single drops is enough, since dropping conditions only widens
what a rule matches.

Its cost is exponential in the number of features and the size of
their domains, since it enumerates every combination of conditions. It
exists to be easy to trust: `GULA` is tested against it.

## GULA: specialize only when contradicted

`GULA` (General Usage LFIT Algorithm) reaches the same rule set as
`BruteForce` by a cheaper search. For each `(target, value)` pair, it
starts from the rule with an empty body, which matches every state.
Then, for each negative example in turn:

1. Every kept rule that matches the negative is removed.
2. Each removed rule is replaced by its least specializations against
   that negative: one more condition, on a feature the rule doesn't
   constrain yet, set to any value other than the negative's. The
   values are those the dataset holds for that feature, its domain.
   Where the negative's value is unknown, every value of the domain
   qualifies, since an unknown satisfies no condition.
3. A specialization that a kept rule already dominates, or that is
   already kept, is discarded. Only kept rules holding the condition the
   specialization adds can dominate it, so only those are checked; the
   argument below shows why.

On the table above, `GULA` builds eight rules instead of 27. Step
through them with Next:

````{container} pylfit-stepper
```{figure} ../images/gula-tree-1.svg
---
alt: >-
  The empty rule for y(1), alone.
---

`GULA` starts from the rule with an empty body, `y(1) :- .`, which
matches every state.
```

```{figure} ../images/gula-tree-2.svg
---
alt: >-
  Negative 000 is being processed. The empty rule matches 000.
---

Negative example 1, `000`: the empty rule matches it.
```

```{figure} ../images/gula-tree-3.svg
---
alt: >-
  Specializing against 000. The empty rule matches 000. a(1) is new.
  b(1) is new. c(1) is new.
---

The empty rule is replaced by its least specializations: one more
condition, with a value `000` doesn't have. `a(1)`, `b(1)` and `c(1)`
are kept.
```

```{figure} ../images/gula-tree-4.svg
---
alt: >-
  Negative 100 is being processed. The empty rule matches 000. a(1)
  matches 100. b(1) doesn't match 100. c(1) doesn't match 100.
---

Negative example 2, `100`: `a(1)` matches it; `b(1)` and `c(1)` don't,
and stay.
```

```{figure} ../images/gula-tree-5.svg
---
alt: >-
  Specializing against 100. The empty rule matches 000. a(1) matches
  100. b(1) doesn't match 100. c(1) doesn't match 100. a(1), b(1) is
  dominated by b(1). a(1), c(1) is dominated by c(1).
---

`a(1)` is replaced by its least specializations: one more condition,
with a value `100` doesn't have. `a(1), b(1)` and `a(1), c(1)` are
dropped: `b(1)` and `c(1)`, already kept, dominate them.
```

```{figure} ../images/gula-tree-6.svg
---
alt: >-
  Negative 101 is being processed. The empty rule matches 000. a(1)
  matches 100. b(1) doesn't match 101. c(1) matches 101. a(1), b(1)
  is dominated by b(1). a(1), c(1) is dominated by c(1).
---

Negative example 3, `101`: `c(1)` matches it; `b(1)` doesn't, and
stays.
```

```{figure} ../images/gula-tree-7.svg
---
alt: >-
  Specializing against 101. The empty rule matches 000. a(1) matches
  100. b(1) doesn't match 101. c(1) matches 101. a(1), b(1) is
  dominated by b(1). a(1), c(1) is dominated by c(1). a(0), c(1) is
  new. b(1), c(1) is dominated by b(1).
---

`c(1)` is replaced by its least specializations: one more condition,
with a value `101` doesn't have. `a(0), c(1)` is kept. `b(1), c(1)` is
dropped: `b(1)`, already kept, dominates it.
```

```{figure} ../images/gula-tree-8.svg
---
alt: >-
  Every negative processed. b(1) and a(0), c(1) are kept. The empty
  rule matches 000. a(1) matches 100. c(1) matches 101. a(1), b(1)
  is dominated by b(1). a(1), c(1) is dominated by c(1). b(1), c(1)
  is dominated by b(1).
---

Every negative example processed: the rules left, `b(1)` and `a(0),
c(1)`, are every minimal rule, the same two `BruteForce` keeps.
```
````

The negatives are taken in lexicographic order
(`negative_order="lexicographic"`), which here is the table's. The
default order, `"similarity"`, takes `100` first and reaches the same
two rules.

Rules are only ever made more specific in response to a real negative
example, so `GULA` explores far fewer candidates than `BruteForce`; the
worst case is still exponential.

### Why the result is every minimal rule

A sketch of the argument. `GULA` was introduced in Ribeiro et al.,
ILP 2018 (<https://doi.org/10.1007/978-3-319-99960-9_8>); the full
algorithm and its proofs are in Ribeiro et al., *Machine Learning*
111(10), 2022 (<https://doi.org/10.1007/s10994-021-06105-4>).

After the negatives `N` processed so far, the kept rules are exactly
the rules that are minimal with respect to `N`: consistent with every
negative in `N`, and no condition can be dropped without matching one.
At the start `N` is empty and the empty rule is the only such rule.
Processing a new negative `n` keeps that true:

- A kept rule that doesn't match `n` stays consistent, and stays
  minimal: dropping any of its conditions already made it match a
  negative in `N`.
- Every rule `M` that is minimal with respect to `N` and `n` is found.
  Drop conditions from `M` while it stays consistent with `N`, and you
  reach a rule `R` minimal with respect to `N`, which is kept. If `R`
  doesn't match `n`, it is consistent with `n` too, and since `M` is
  minimal, `M` is `R`. If `R` matches `n`, then `M` has a condition `c`
  that `n` violates and `R` lacks. `R` plus `c` is a least
  specialization of `R`, consistent with `N` and `n`, and has a subset
  of `M`'s conditions; since `M` is minimal, `M` is `R` plus `c`.
- A specialization `S` that isn't minimal is dominated by a minimal
  rule `M`: drop conditions from `S` while it stays consistent. `M` was
  already kept before `n`. `S` is `R` plus `c`, with `R` matching `n`,
  so `c` is the only condition of `S` that `n` violates, and `M`, which
  doesn't match `n`, contains `c`. If `M` were new at this step, it
  would be some kept rule `R'` plus `c`, with `R'` a strict subset of
  `R` and consistent with `N`, and then `R` wouldn't be minimal with
  respect to `N`. So the domination check, run against the rules
  already kept, discards every specialization that isn't minimal, and
  never a minimal one: no other consistent rule dominates it.

When every negative has been processed, the kept rules are the minimal
consistent rules: the rule set `BruteForce` builds by enumeration.
They are correct by construction. They are also complete: take any
feature state `s`. Some value of the target doesn't have `s` as a
negative (the value seen with `s`, or any value if the data never shows
`s`), so the rule with that value whose body is all of `s` is
consistent. Dropping its conditions while it stays consistent reaches a
minimal rule, which still matches `s` and is among the kept rules. The
argument never depends on the order of the negatives, which is why
`negative_order`, the order `GULA` processes them in, changes only the
running time. The default, `"similarity"`, takes the most typical
negatives first; no order is fastest on every dataset ([the
benchmark_negative_order
example](../auto_examples/benchmark_negative_order.rst)). The [worked
GULA trace](gula-trace.md) shows the loop on a small dataset.

## PRIDE: cover the positives

`PRIDE` works the other way round. It picks a positive example no rule
covers yet, and builds one rule around it: each time the rule matches a
negative example, it adds a condition fixing a feature where that
negative and the positive differ. Of those features, it takes the one
that leaves the rule matching the most positives no rule covers yet,
the first in `Dataset.features` order on a tie. Then it tries dropping
each condition once, keeping the drop when the rule still matches no
negative. One pass is enough: dropping conditions
only makes the rule more general, so a condition that was needed stays
needed. That also spares most of the work: one scan, the test
`BruteForce` uses, finds the conditions some negative violates alone.
Those are never dropped, so they are never tried, and only the
negatives meeting all of them can be matched again. It sets aside
every positive the new rule covers, and repeats until every positive
is covered.

On the table above, it finds the same two rules. Step through its run
with Next:

````{container} pylfit-stepper
```{figure} ../images/pride-1.svg
---
alt: >-
  The table with its positive rows in green and its negative rows,
  000, 100 and 101, in red. Row 001 is outlined as the seed. The
  rule for it is y(1) :- . with an empty body.
---

`PRIDE` takes the first positive example no rule covers, `001`, as its
seed, and starts from the rule with an empty body, `y(1) :- .`, which
matches every state, the negatives in red included.
```

```{figure} ../images/pride-2.svg
---
alt: >-
  Row 000 is outlined in red, matched by the empty rule. It differs
  from the seed 001 on c only, so the rule becomes y(1) :- c(1).
---

The first negative example, `000`, matches the rule. It differs from
the seed on `c` only, so `PRIDE` adds `c(1)`, the seed's value. With
more features to choose from, it would take the one keeping the most
uncovered positives.
```

```{figure} ../images/pride-3.svg
---
alt: >-
  Row 100 doesn't match y(1) :- c(1). Row 101, outlined in red,
  does, and differs from the seed on a only, so the rule becomes
  y(1) :- a(0), c(1).
---

`100` doesn't match `c(1)`. `101` does, and differs from the seed on
`a` only, so `PRIDE` adds `a(0)`. The rule now matches no negative.
```

```{figure} ../images/pride-4.svg
---
alt: >-
  Rows 001 and 011 are greyed out, covered by rule 1, y(1) :- a(0),
  c(1). 101 fails a(0) only and 000 fails c(1) only, both outlined in
  red, so both conditions stay.
---

Each condition is needed: `101` fails `a(0)` only, so without it the
rule would match `101`, and `000` fails `c(1)` only. The scan that
finds this tries no drop at all. The rule covers `001` and `011`,
which are set aside.
```

```{figure} ../images/pride-5.svg
---
alt: >-
  Rows 001 and 011 are greyed out, covered by rule 1. Row 010 is
  outlined as the new seed, and its rule is y(1) :- . with an empty
  body.
---

The first positive no rule covers yet, `010`, is the next seed.
`PRIDE` starts again from the rule with an empty body, `y(1) :- .`.
```

```{figure} ../images/pride-6.svg
---
alt: >-
  Row 000 is outlined in red, matched by the empty rule. It differs
  from the seed 010 on b only, so the rule becomes y(1) :- b(1),
  which rows 100 and 101 don't match.
---

`000` matches the empty rule and differs from the seed on `b` only, so
`PRIDE` adds `b(1)`. `100` and `101` don't match it: the rule matches
no negative.
```

```{figure} ../images/pride-7.svg
---
alt: >-
  Rows 010, 110 and 111 are greyed out, covered by rule 2, y(1) :-
  b(1). 000, outlined in red, fails b(1) only, so it stays.
---

`000` fails `b(1)` only, so `b(1)` stays without a drop being tried.
The rule covers `010`, `110` and `111`, which are set aside.
```

```{figure} ../images/pride-8.svg
---
alt: >-
  Every positive row is greyed out as covered. The rules learned,
  y(1) :- a(0), c(1). and y(1) :- b(1). are in green.
---

Every positive example is covered, so `PRIDE` stops. Its two rules,
`y(1) :- a(0), c(1).` and `y(1) :- b(1).`, are the two `BruteForce`
and `GULA` keep, built from three added conditions.
```
````

On this table, each negative the rule matches differs from the seed
on one feature only, so the coverage choice never has to decide, and
the scan finds every condition needed, so no drop is tried. On larger
data the choice matters, as measured below.

Each rule `PRIDE` returns is consistent and minimal, so it is one of the
rules `GULA` returns. `PRIDE` stops once the rows are covered, so it
returns fewer. A feature state the data never shows may be matched by no
rule.

Choosing the condition by the positives it keeps covering was
measured, when it was introduced, against always taking the first
feature that excludes the negative: on Boolean networks given half
their states it learned about 12% fewer rules at equal accuracy and fit
time, and given a tenth, about 60% fewer, three times faster. Each rule
covers more, so fewer seeds are needed. The first-feature choice is no
longer in `PRIDE`, so these figures are historical and can't be re-run.

`PRIDE` takes the positives and the negatives in the rows' order, so
reordering the rows can change the rules (each one still consistent
and minimal). Sort the `DataFrame` first if the rules must not depend
on the order rows came in. Sorting inside `PRIDE` was measured: with
the coverage choice it gave no fewer or more accurate rules, and cost
time. On a tie, the order of `Dataset.features` decides.

`try_all_conditions=True` builds more rules per seed: besides the usual
one, one for each condition the seed can take, that condition put in
first and the last one tried for dropping. `PRIDE` then usually finds
more of the rules `GULA` would, at a cost. On Boolean networks of 3
to 12 variables given half their states, it found up to about 10 points
more of the rules the complete state space gives, and none more on
most networks, for two to six times the fit time, growing with the
network's size, and more rules.
It is not a superset of the default run: the extra rules also cover
positives, so later seeds change, and a rule the default run learns
can be missing.

## generalize: drop what the data does not need

The three algorithms return rules that match no negative example. When
some labels are wrong, a rule then carries conditions that only dodge
a mislabelled row. [`generalize`](../api/generalize.md) runs after any
of them and drops those conditions. For each condition of a rule, it
takes the rows matching the rest of the rule and asks whether those
with the condition hold the head value more often than those without,
by a one-sided Fisher exact test. The condition with the highest
p-value goes, if that p-value is above `alpha` (0.01 by default), and
the rule is tested again, until every condition left is needed. The
default was the best level, or as good as any, on every data set
measured with fewer than about a thousand rows; on thousands of rows
with noisy labels, a lower `alpha` such as 0.001 keeps fewer chance
rules. If the default does not give good rules on your data, try
another `alpha`; [`generalize`](../api/generalize.md) gives the numbers.
For a model of a size you can read, `max_rules` caps the rules
returned, every target together, by lowering `alpha` until they fit.

Take a table of four features where `y` is `a`, with all sixteen
states, and flip one label: `1000` says `0`. Step through
`generalize` on the rules `GULA` learns, with Next:

````{container} pylfit-stepper
```{figure} ../images/generalize-1.svg
---
alt: >-
  The sixteen rows in two columns, a=0 on the left and a=1 on the
  right. Row 1000 is outlined in red, flipped from 1 to 0. GULA's
  rules for y(1) are y(1) :- a(1), b(1)., y(1) :- a(1), c(1). and
  y(1) :- a(1), d(1).
---

`GULA`'s rules for `y(1)` must not match `1000`, so none of them can
be `y(1) :- a(1).`: each adds a condition that `1000` fails. `PRIDE`
learns the same three.
```

```{figure} ../images/generalize-2.svg
---
alt: >-
  Only the rows with a=1 are shown, the rows with b=1 among them
  outlined. With b(1), 4 of 4 hold y=1; without, 3 of 4. Fisher's
  test gives p = 0.5.
---

Is `b(1)` needed in `y(1) :- a(1), b(1).`? Among the rows matching
the rest of the rule, `a(1)`, those with `b(1)` all hold `y=1`, and
those without hold it 3 times in 4. The test gives p = 0.5: so small
a difference often comes by chance.
```

```{figure} ../images/generalize-3.svg
---
alt: >-
  Only the rows with b=1 are shown, the rows with a=1 among them
  outlined. With a(1), 4 of 4 hold y=1; without, 0 of 4. Fisher's
  test gives p = 0.014. b(1) has the highest p and goes.
---

Is `a(1)` needed? Among the rows with `b(1)`, 4 of 4 against 0 of 4,
p = 0.014. `b(1)` has the highest p-value, above 0.01, so it goes.
`a(1)`'s p-value is above 0.01 too, but only one condition goes per
pass: four rows on each side are too few to show `a(1)` is needed,
eight in the next pass are enough.
```

```{figure} ../images/generalize-4.svg
---
alt: >-
  Every row is shown, the rows with a=1 outlined. With a(1), 7 of 8
  hold y=1; without, 0 of 8. Fisher's test gives p = 0.0007, so a(1)
  stays.
---

The rule's last condition is tested against every row: 7 of 8 rows
with `a(1)` hold `y=1`, none without. p = 0.0007, at most 0.01, so
`a(1)` is needed and `y(1) :- a(1).` is done. The flipped row is the
one row it matches with `y=0`.
```

```{figure} ../images/generalize-5.svg
---
alt: >-
  The rules y(1) :- a(1), c(1). and y(1) :- a(1), d(1). both become
  y(1) :- a(1)., merged with the first.
---

`c(1)` and `d(1)` only dodge `1000` too, with p = 0.5 each, so the two
other rules become `y(1) :- a(1).` as well. Rules that become the same
are merged.
```

```{figure} ../images/generalize-6.svg
---
alt: >-
  Rows 0000 and 1000 are outlined in red, the only rows y(0) :- b(0),
  c(0), d(0). matches. None of its conditions is needed, so it goes.
  The rules returned, y(0) :- a(0). and y(1) :- a(1)., are in green.
---

For `y(0)`, `GULA` learns `y(0) :- a(0).` and
`y(0) :- b(0), c(0), d(0).`, which only the flipped row backs: it
matches `0000` and `1000` alone. No condition of it is needed, p = 0.5
at each pass, so with none left it goes, as `y(0) :- a(0).` is left
for `y=0`. The rules returned are the true ones.
```
````

On this table no rule left is more general than another of its head.
When one is, `generalize` keeps one of the two: the narrow rule only
when, among the broad rule's rows, its extra conditions pass the same
test at a stricter level, as section 5 of [the heart disease
tutorial](../auto_examples/tutorial_heart_disease.rst) shows with one
pair settled each way. The [`generalize`](../api/generalize.md)
reference gives that level and what the test costs: a true rule that
matches few rows loses conditions, even on clean data.

```pycon
>>> import itertools
>>> import pandas as pd
>>> from pylfit import Dataset, GULA, generalize
>>> rows = [
...     {**dict(zip("abcd", state)), "y": state[0]}
...     for state in itertools.product("01", repeat=4)
... ]
>>> rows[8]["y"] = "0"  # state 1000, flipped
>>> dataset = Dataset(pd.DataFrame(rows), features=list("abcd"), targets=["y"])
>>> rules = GULA.fit(dataset)
>>> [rule.to_string() for rule in rules if rule.head == ("y", "1")]
['y(1) :- a(1), b(1).', 'y(1) :- a(1), c(1).', 'y(1) :- a(1), d(1).']
>>> [rule.to_string() for rule in generalize(rules, dataset)]
['y(0) :- a(0).', 'y(1) :- a(1).']

```

## Unknown values

`GULA` needs nothing special for unknown values (see
[About data](about-data.md)): an unknown satisfies no condition, so a
negative example with one is ruled out by a condition on any value of
that feature, and a positive example with one is matched only by rules
that don't test that feature. `GULA`'s argument above holds unchanged,
reading "matches" that way. So the rules are exactly the minimal rules
consistent with the negative examples as recorded, and none of them
tests for an unknown. On a deterministic system, each rule of the
system itself is still more specific than, or equal to, some learned
rule: an unknown never makes a rule match a row it would not match
with the value filled in, so the system's rules stay consistent with
the data, and every consistent rule is either learned or more specific
than a learned one.

The cost grows with the unknowns: a negative that hides a value rules
out fewer rules, so more rules survive, and each must be specialized
against the negatives after it, and on many features it grows fast.
On `soybean-small.csv` (35 features), with values hidden at random,
single runs of `GULA` took about 8 times longer than on the complete
table with 5% of the values unknown, about 50 times longer with 10%,
and did not finish within minutes at 20%. `min_support=1`, which drops
the rules no row backs, brought the 10% run from about 70 s to 2 s;
`PRIDE`, below, is the other remedy.

`BruteForce` reads unknowns the same way, and still learns exactly
`GULA`'s rules. `PRIDE` does too, so each rule it grows uses only
its seed's known features. It builds one rule per seed rather than
every rule, so it stays fast as unknowns grow: on many features with
many unknowns it is the practical choice, and its rules are still
among `GULA`'s.

Some positive rows can then be matched by no rule at all, whichever
algorithm learns. If a negative example agrees with a row on every
feature the row knows, any rule matching the row tests only those
features, with the row's values, and so matches that negative too. No
consistent rule can cover such a row. All three algorithms warn when
it happens, with the share of positive rows left uncovered; pass
`verbose=True` to list them, or `warn_uncovered=False` to silence the
warning once you know. Every other row stays covered, unless an option
below leaves it out: a row that some known feature separates from
every negative is matched by one of `GULA`'s rules, and `PRIDE` builds
a rule for it. When predicting, the models' fallback still gives such
a row the values of the rules that could match it; see [About the
prediction models](about-models.md).

## Learning some heads only

Each `(target, value)` pair, a head, is learned on its own, so
`GULA.fit(dataset, heads=[("y", "1")])` returns exactly the `y=1`
rules of a full run, without the cost of the others; `PRIDE.fit`
and `BruteForce.fit` take `heads` too. That suits
analyzing the rules of one value: on a full truth table of a Boolean
function, the `y=1` rules describe the function completely, since every
other state is `y=0`. A model knows that only if it has the `y=0` rules
too. Built from the `y=1` rules alone, it predicts `y=1` where a rule
matches and, where none does, draws a value at random rather than
predicting `y=0`.

## Learning heads in parallel

For the same reason, all three algorithms can learn several heads at
once: `n_jobs=4` learns them in four processes, `n_jobs=-1` in one
per CPU, as in scikit-learn. The rules, and any warning, are the same
as with the default `n_jobs=1`, which starts no process.

The speedup grows with the number of heads that take long to learn.
On a 16-CPU Linux machine, on the full transition table of a random
15-variable Boolean network (32,768 rows, 30 heads), `GULA` took
15.2 s with `n_jobs=1`, 4.2 s with 4 and 1.9 s with 16; `PRIDE` took
1.2 s, 0.4 s and 0.3 s. These are fits after a first fit with
`n_jobs=1` on the same `Dataset`, which builds its row caches and
takes about a second more on a table this size. A parallel fit builds
them only in its worker processes, where they are lost when it ends,
so it pays that second each time. On a 12-variable network `GULA`
gained little past 4 processes (0.83 s, then 0.70 s with 8), as one
head took 0.64 s alone: no `n_jobs` brings a fit below its slowest
head ([the benchmark_n_jobs
example](../auto_examples/benchmark_n_jobs.rst)).

These figures are for Linux up to Python 3.13, where a new process is
a copy of the running one (the `fork` start method) and costs almost
nothing. Elsewhere each process is sent the dataset, without its row
caches, and rebuilds the caches of the heads it learns. On the same
network and machine:

- Linux from Python 3.14 copies each process from a server process
  (`forkserver`): `GULA` took 4.4 s with 4 processes and 2.2 s with
  16, close to the figures above; `PRIDE` 1.0 s and 0.6 s.
- macOS and Windows start Python afresh in each process (`spawn`),
  which also imports pylfit there: `GULA` took 5.7 s with 4 and 5.3 s
  with 16, and `PRIDE` was slower than with `n_jobs=1`, 2.7 s and
  more. There, use `n_jobs` only for fits of several seconds. (These
  are Linux forced to `spawn`, not a Mac or a Windows machine.)

Under both, each new process also imports your script again, so a
script must call `fit` under `if __name__ == "__main__":`. Without it
the fit fails with `BrokenProcessPool`, and its message says to add
the guard.

## Restricting the search

These options keep every rule correct, and all give up completeness: a
feature state that only a discarded rule would have matched is left
unmatched. `GULA` and `PRIDE` both take them, and both warn when they
leave positive rows of the data uncovered, naming the option, so you
can judge what a higher `min_support` costs; `warn_uncovered=False`
silences it. `pylfit.coverage(dataset, rules)` measures the
same thing after the fact, per head, for any rule set.

`min_support` drops rules that fewer rows back than a threshold: rows,
not distinct states, so a state seen twice counts twice. Specializing a
rule only shrinks the set of rows it matches, so `GULA` drops a
candidate as soon as it falls below the threshold, along with everything
it would have been specialized into. The rules kept are exactly the
unrestricted rules that meet the threshold. On data with many features
and few rows, that is a large saving ([the benchmark_min_support
example](../auto_examples/benchmark_min_support.rst): about ten times
faster on soybean). A row with an unknown value for a feature the rule
tests doesn't count: nobody saw it back the rule.

That is a trap on data with unknowns: a rule whose backing rows all
have the value it tests hidden has a support of zero, though those rows
may well back it. `min_possible_support` counts a row as backing a rule
when it could: each of the rule's conditions holds in the row or falls
on an unknown. A rule goes when fewer rows than the threshold could
back it; at 1, only the rules no row could back go. It still only
shrinks as conditions are added, so `GULA` prunes on it during the
search, and the rules kept are exactly the unrestricted rules that meet
it. Without unknowns it is the same as `min_support`. The two
combine: set both, and a rule is kept when it meets both thresholds.

```pycon
>>> import pandas as pd
>>> from pylfit import Dataset, GULA
>>> df = pd.DataFrame(
...     {
...         "a": ["0", "0", None, "1"],
...         "b": ["1", None, "0", "0"],
...         "y": ["0", "1", "0", "0"],
...     }
... )
>>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
>>> GULA.fit(dataset, heads=[("y", "1")], min_support=1)
[]
>>> rules = GULA.fit(dataset, heads=[("y", "1")], min_possible_support=1)
>>> [rule.to_string() for rule in rules]
['y(1) :- a(0), b(0).']

```

The only row with `y=1` has `b` unknown, so it backs no rule for
certain, but it could be `a=0, b=0`.

`PRIDE` builds one rule per seed, so it can't keep a rule that misses
the threshold and look further. With `min_support`, it keeps a seed's
usual rule if it meets the threshold, and otherwise rebuilds it, adding
at each step, among the conditions that keep the threshold met, the one
leaving the most positives still to cover matched. A seed left with no
such condition, or whose rebuilt rule still falls short, gets no rule.
Every rule it keeps is one of `GULA`'s with the same threshold, and
`min_support=1` changes nothing: a rule always matches its own seed.
`min_possible_support` works the same way, and `allowed_conditions`,
below, limits the conditions it may add.

`allowed_conditions` is background knowledge: for each head, the
conditions its rules may use. The rules learned are still minimal:
dropping a condition from an allowed rule gives another allowed rule, so
the argument above applies unchanged within the allowed conditions. On
small data it also removes rules that only hold by chance. Here `y` is
`a` or `b`, and `c` is noise:

```pycon
>>> import pandas as pd
>>> from pylfit import Dataset, GULA
>>> df = pd.DataFrame(
...     {
...         "a": ["0", "0", "1", "1"],
...         "b": ["0", "1", "0", "1"],
...         "c": ["0", "1", "1", "0"],
...         "y": ["0", "1", "1", "1"],
...     }
... )
>>> dataset = Dataset(df, features=["a", "b", "c"], targets=["y"])
>>> len(GULA.fit(dataset))
7
>>> known = {("y", "0"): ["a", "b"], ("y", "1"): ["a", "b"]}
>>> [rule.to_string() for rule in GULA.fit(dataset, allowed_conditions=known)]
['y(0) :- a(0), b(0).', 'y(1) :- a(1).', 'y(1) :- b(1).']

```

When the knowledge is right, the rules it keeps are the ones you would
have kept anyway, and the search is much shorter: on a 10-gene
cell-cycle network, knowing each gene's regulators made `GULA` about 4
times faster in one measurement, with identical rules ([the
benchmark_influence example](../auto_examples/benchmark_influence.rst);
run it to measure on your machine).

## What this implementation leaves out

- `PRIDE` has one learning heuristic, the coverage choice, plus
  `try_all_conditions`; scoring by how a rule differs from earlier ones
  (0.x's `max_diversity`) was measured and left out, since rules from
  different seeds rarely contain one another. The full algorithm is in
  Ribeiro et al., IJCLR 2021
  ([hal-03347026](https://hal.science/hal-03347026)).
- Learning dynamics is plain classification on `(t, t+1)` rows. The
  update semantics of earlier pylfit versions (synchronous,
  asynchronous, and others) are not part of the package; see
  [Learn from state transitions](../how-to/state-transitions.md) and
  [Migrate from pylfit 0.x](../how-to/migration.md).
