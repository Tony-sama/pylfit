# Design notes

The design as it stands, and the reason for each choice. Read this
before proposing a change to one of them.

## Values are strings

`Dataset` stores every value as a string, once, at its boundary.
Conditions compare by equality only, and mixed types would make that
comparison wrong without an error: `True == 1` in Python. Not native
`int` or `bool` values, because of that silent mismatch.

## Learning and predicting are separate

An algorithm's `fit` is a static method returning a plain list of
rules, and has no dependency on the models. The same rules can then go into
any model, and how to predict becomes a choice of its own: on the car
evaluation data, one rule set scores 0.23 or 0.92 depending on the
model. Scoring rules is the weighted model's job, so a caller who only
needs rules never pays for it.

`Algorithm` declares `fit` as an abstract static method, so every
algorithm has the same entry point and type checkers know it: a list
such as `[GULA, PRIDE]` can be looped over and fitted. Each algorithm
may add keyword options after `verbose`.

## One model family, no base class

There is one family of models, and what its members share lives in
`RuleSetModel`. An abstract base declaring `predict`, `save` and
`from_file` would restate `RuleSetModel` for its own subclass, and
nothing in pylfit is written against it. A second family, storing its
rules another way, would be the time to add one above both: adding a
base class breaks no one, while removing a public one would.
`WeightedRuleSetModel` extends `RuleSetModel` through small hook
methods (`_rule_payload`, `_load_rule`, ...) rather than
by overriding `save` or `from_file` whole.

## One method per question

`predict_likelihood`, `predict_proba` and `predict` answer different
questions: raw evidence, a probability, a decision. They are separate
methods, as in scikit-learn. Not one `predict(mode=...)`, because a
wrong mode would fail silently. `predict` is written once, against
`_proba`, the unchecked single-state core of `predict_proba`, so a
subclass changes predictions by overriding `_proba` alone. The public
methods check their input, a `DataFrame` or a single state alike, then
call the private cores; `predict` checks the frame once and calls
`_proba` per row.

## Ties are random

`predict` breaks ties uniformly at random, with `random_state` to make
it reproducible. A fixed rule, such as the alphabetically first value,
would favor one class every time.

## Saved models are JSON

A saved model is a JSON file with a schema version, the class that
wrote it, and each rule stored structurally (head and body) beside a
readable string that loading ignores. Not `pickle`, because loading a
pickle can run arbitrary code and breaks silently across versions.
`from_file` refuses a file another class wrote, rather than misreading
it.

## The same output on every run

Everything pylfit returns, prints or writes is the same on every run,
given the same input, row order and `random_state`. Python seeds its
string hashing per process, so a `set` of strings iterates in a
different order each time; no set's order may reach an output. Return
a list or a dict, or sort, whichever costs no running time: `fit` and
`RuleSetModel.rules` give rules sorted by text, without duplicates, and
domains and candidates are sorted lists, as scikit-learn's `classes_`
is a sorted array. Inside, sets stay where they are fastest, such as
`RuleSetModel`'s rules and domains, and an error message that lists
what it found in one sorts it first. Only elapsed times, and the order
of `verbose` lines when heads are learned in parallel, may change
between runs. `tests/test_reproducibility.py` runs one workflow under
several hash seeds and compares every output.

A list makes one comparison a trap: `<=` between two lists compares
them element by element, as words in a dictionary, so a subset test
must convert first, `set(a) <= set(b)`.

## Public names earn their place

A name is public only when a tutorial, a how-to, an example or an
explanation page uses it. A name that only pylfit's own code calls is
private. Adding a public name later breaks no one, while removing one
after 1.0 takes a deprecation and a major release, so a name with no
use waits until it has one.
