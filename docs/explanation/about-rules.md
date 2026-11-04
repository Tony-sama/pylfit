# About LFIT rules

pylfit implements algorithms of Learning From Interpretation Transitions
(LFIT), a family of methods that learn logic programs from observations.
This page explains what a learned rule is, and the three properties the
learned rule sets have. The algorithms themselves are in [About the
algorithms](about-algorithms.md). LFIT was introduced in Inoue, Ribeiro
and Sakama, *Machine Learning* 94(1), 2014
(<https://doi.org/10.1007/s10994-013-5353-8>).

## A rule

A rule has a head, one `(variable, value)` pair, and a body, a set of
conditions on other variables:

```text
y(1) :- a(0), c(1).
```

Read it as: if `a` is 0 and `c` is 1, then `y` can be 1. The head
comes after "then", the body after "if", and each comma is an "and".

The rule matches a state when every condition in its body holds in
that state. It then says that its head value is possible there. The
notation is that of a Horn clause in logic programming: the head on
the left of `:-`, the body on the right.

A rule's body is a conjunction, and a rule set is a disjunction: `y`
can be `1` when the body of any rule concluding `y(1)` holds. With a
second rule, `y(1) :- b(1).`, the two read: if `a` is 0 and `c` is 1,
or if `b` is 1, then `y` can be 1.

## Generality

A rule is more general than another with the same head when its body
is a subset of the other's: it matches every state the other matches.
The rule-learning literature calls this subsumption. Unlike
subsumption, the relation is strict: no rule is more general than
itself. A rule dominates another when it is more general than it; the
dominated rule then adds nothing a rule set doesn't already have.

```pycon
>>> from pylfit import Rule
>>> Rule.from_string("y(1) :- a(0).").is_more_general_than(
...     Rule.from_string("y(1) :- a(0), c(1).")
... )
True

```

## A good rule, on data

pylfit learns from two kinds of data: a table of examples, where each
row is one case, and observations over time, where each row is a state
and the next one. A good rule is the same in both.

### On a table

Take the table from [Getting started](../getting-started.md): fourteen
days of weather, and whether a game of tennis was played.

```{figure} ../images/tennis-good-rule.svg
---
alt: >-
  Fourteen days of outlook, temperature, humidity, wind and play.
  The rule play(yes) :- humidity(normal), wind(weak). matches days 5,
  9, 10 and 13, and play is yes on each of them.
---

The rule's conditions hold on days 5, 9, 10 and 13 (solid boxes); on
each of them, play is yes (dashed boxes).
```

The rule reads: if humidity is normal and the wind is weak, then play
can be yes. A good rule passes three tests:

- It matches some observations: here, days 5, 9, 10 and 13.
- The data never contradicts it: each day it matches, play is yes.
  `play(yes) :- humidity(normal).` fails this test: on day 6 humidity
  is normal too, and play is no.
- It keeps only conditions that matter: adding `temperature(cool)`
  gives a rule that is never contradicted either, but only matches
  days 5 and 9 and adds nothing. The shorter rule dominates it.

The last two are what the properties below call correct and minimal,
and `GULA` learns every rule that passes both. The first is not
required: `GULA` also learns rules for states the data never shows,
which match no observation, and `min_support=1` keeps only the rules
that pass it.

### Over time

Take a year of monthly monitoring data, temperature and nitrate in two
levels, a plankton species in three, and the rules for the species'
level next month:

```{figure} ../images/a-good-rule.svg
---
alt: >-
  Twelve months of temperature, nitrate and species levels. The
  rule species_next(high) :- nitrate(high), species(low). matches in
  March and October, and the species is high in April and November.
---

The rule's conditions hold in March and October (solid boxes); the
month after, the species is high (dashed boxes). Nitrate wasn't
measured in August.
```

The rule reads: if the species is low and nitrate is high, then the
species can be high next month. It passes the same three tests. It
matches March and October, and each time the species is high the next
month. `species_next(high) :- nitrate(high).` is contradicted: nitrate
is also high in January, February, April and November, and the species
isn't high the month after. Adding `temperature(low)` matches the same
two months and adds nothing.

On this year, `species_next(high) :- species(low), temperature(low).`
passes the three tests too: temperature is low exactly in the months
nitrate is high, so the data can't tell the two apart, and `GULA`
learns both.

## Positive and negative examples

For a head `(target, value)`, a positive example is a feature state
where the target has that value. A negative example is a feature state
seen in the data where the target never has it; a state the data never
shows is a negative of nothing.

On a table, each row is one example. On the tennis table every
weather is seen once, so each row is a positive of its own value of
`play` and a negative of the other:

```{figure} ../images/tennis-positive-and-negative-examples.svg
---
alt: >-
  The tennis table drawn twice. For play = yes, the nine rows where
  play is yes are positives and the five others negatives; for
  play = no, the other way round.
---

The tennis table once per value of `play`. Each row's feature state,
not its `play`, is colored by the kind of example it is for that
value.
```

Real data is often not that clean. The same feature state can be seen
with two values of the target: through noise, a wrong record, or
through discretization, when a temperature of 21 °C and one of 23 °C
both become `mild` and only the warmer day had a game. Add to the
table a row with the weather of a day without a game, but with a game
played:

```{figure} ../images/tennis-conflicting-examples.svg
---
alt: >-
  The tennis table with a fifteenth row, sunny, mild, high, weak,
  yes, whose weather is also seen with play = no. For play = yes, that
  row is a positive and the no-game row with the same weather is
  neither positive nor negative; for play = no, the other way round.
---

The same table with a conflicting row. Each of the two rows with the
same weather is a positive of its own value, and neither is a negative
of anything.
```

A state seen with the value in any row is not a negative, even in rows
where the target has another value: a rule may conclude either value
there. [Conflicting rows](about-data.md#conflicting-rows) says how
pylfit finds such rows and what else they change.

Over time, each row is a transition, and each target of the next
state gets its examples the same way:

````{container} pylfit-stepper
```{figure} ../images/positive-and-negative-examples-1.svg
---
alt: >-
  Eight observed transitions of two variables a and b, each a state
  now and the state one step later. The panels for a_next = 0 and
  a_next = 1 are empty.
---

Eight observed transitions of two variables, `a` and `b`. Step
through the sorting of their states into the examples of each head,
with Next.
```

```{figure} ../images/positive-and-negative-examples-2.svg
---
alt: >-
  For a_next = 0, the four transitions whose next a is 0 have their
  state in green: 00, 01 and 11 are positives. The state 10 is in red,
  never followed by a = 0: the negative. The other rows are faded.
---

`a_next = 0`: a state followed by `a = 0` is a positive, in green; the
bold digit is the next value of `a`. `10` never is, so it is a
negative, in red. Faded rows add nothing: their state is already a
positive.
```

```{figure} ../images/positive-and-negative-examples-3.svg
---
alt: >-
  For a_next = 1, the positives are 00, 10 and 11 and the negative is
  01. The a_next = 0 panel stays filled.
---

`a_next = 1`: `00` is a positive again. Followed by both values, it
is a positive of both heads and a negative of neither.
```

```{figure} ../images/positive-and-negative-examples-4.svg
---
alt: >-
  The panels now hold the heads of b_next. For b_next = 0, sorted on
  the next value of b, the positives are 00, 10 and 11 and the
  negative is 01.
---

`b_next = 0`: the same sorting on the next value of `b`.
```

```{figure} ../images/positive-and-negative-examples-5.svg
---
alt: >-
  For b_next = 1, the positives are 00, 01 and 11 and the negative is
  10.
---

`b_next = 1`. Learning the rules of a head is then a classification
problem: which conditions match positives and no negative.
```
````

## The three properties

On an unrestricted run, `GULA` and `BruteForce` learn every rule that
is correct and minimal, and that set is also complete:

- Correct (also called consistent): no rule matches a negative example
  of its head. On non-deterministic data that is weaker than "never
  contradicted by a row", since a state seen with the head value isn't
  a negative of it.
- Complete: for each target, every feature state is matched by at
  least one rule. That includes feature states the data never shows.
- Minimal: a consistent rule from which dropping any condition makes
  it match a negative example. Equivalently, no rule with only some of
  its conditions is consistent: such a rule is more general than one of
  the rules with a single condition dropped, so it matches the negative
  that rule matches. No other consistent rule dominates a minimal
  rule, but one head can have several minimal rules that don't
  dominate each other.

Completeness is why a learned rule set can contain rules no row backs:
a state the data never shows still needs a rule, and the minimal rules
that match it are learned with the rest. `min_support` drops
those rules, giving up completeness. `PRIDE` learns rules that are
correct and minimal, but only some of them: enough to cover the rows
present, so it is complete only over those rows, and not on a row with
unknowns that no known feature separates from a negative, which no
correct rule can match.

In the tests, each property is checked directly on every fixture, and
`GULA`'s rule set is checked equal to `BruteForce`'s exhaustive one. The
proofs that `GULA`'s rule set is correct, complete and minimal are in
Ribeiro et al., *Machine Learning* 111(10), 2022
(<https://doi.org/10.1007/s10994-021-06105-4>).

## A rule says possible, not certain

Read a learned rule as "the data does not rule this value out here".
Several rules can match the same state for different values, and a
state no rule matches has no value ruled out. Turning that into one
predicted value is the model's job, not the rules':
see [About the prediction models](about-models.md).

## Relation to other work

- Inductive logic programming (ILP) learns logic programs too, usually
  first-order and from background knowledge. LFIT rules are
  propositional: a condition is one variable equal to one value.
- A decision tree's paths are rules too, but they partition the
  states: each state follows exactly one path. LFIT rule sets overlap,
  and the full one contains every minimal rule, not one rule per
  region.
- Covering algorithms such as CN2 and AQ learn one rule at a time and
  stop once the examples are covered. `PRIDE` works this way; `GULA`
  learns every minimal rule instead.
- LFIT was first built to learn dynamics: Boolean networks, where each
  gene's update rules are learned from observed state transitions
  (Inoue, Ribeiro and Sakama, 2014). For its first years, every
  practical use of LFIT learned the dynamics of such systems. Learning
  from a plain table of examples in practice, as in explainable AI for
  recruitment (Ortega et al., WACV Workshops 2021,
  <https://doi.org/10.1109/WACVW52041.2021.00013>),
  came later. pylfit treats both as the same problem: a transition is a
  row whose features are the state now and whose targets are the state
  next. See [Learn from state transitions](../how-to/state-transitions.md).

## Terms

| Term | Meaning in pylfit |
| --- | --- |
| Head | `Rule.head`, a `(variable, value)` pair |
| Body | `Rule.body`, a mapping of variable to value |
| Condition | one entry of the body |
| Matches | `Rule.matches(state)` |
| More general, subsumes | `Rule.is_more_general_than`; strict, unlike subsumption |
| Dominates | is more general than; a dominated rule is discarded |
| Consistent, correct | matches no negative example of its head |
| Complete (rule set) | every feature state is matched, for each target |
| Minimal (rule) | consistent, and dropping any condition makes it match a negative example |
| Positive examples | `Dataset.positives(target, value)` |
| Negative examples | `Dataset.negatives(target, value)` |
| Support | `WeightedRuleSetModel.get_score(rule)`, `(positives, negatives)` |
