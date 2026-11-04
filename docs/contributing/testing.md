# Testing

`pytest` runs the suite; `pytest -m ""` adds the slow tests, as CI
does.

## What is where

- `tests/` mirrors the package: `test_rule.py`, `test_dataset.py`,
  `models/`, `algorithms/`, `test_metrics.py`.
- `tests/conftest.py` builds small datasets whose correct rule sets and
  predictions can be worked out by hand, and tests assert exact results
  on them, not only that code runs:
  - `toy_dataset`: three binary features `a`, `b`, `c` and a target `y`,
    all 8 states, with `y` `1` when `b` is `1` or when `a` is `0` and
    `c` is `1`.
  - `toy_rules`: the four rules every algorithm learns from
    `toy_dataset`, built by hand for the model tests.
  - `nondeterministic_dataset`: one feature `a` and a target `y`, where
    `a=0` is seen with both `y=0` and `y=1`.
  - `degenerate_dataset`: two features and a target that is always `1`,
    so the rule with an empty body is already correct.
  - `multi_target_dataset`: one feature `a` and two targets, `y1`
    equal to `a` and `y2` its opposite.
  - `support_dataset`: one feature `a` whose values `1` and `2` lead
    to `y=1` on 1 and 4 rows, for the support thresholds.

  One more fixture is a tool, not a dataset:
  `outputs_per_hash_seed` runs a script in a subprocess per hash seed
  (`PYTHONHASHSEED`) and returns the set of its outputs, for a test
  that an output doesn't depend on the order of a `set`.
- `tests/algorithms/test_boolean_networks.py` runs `BruteForce`, `GULA`
  and `PRIDE`, each named in the file, on published Boolean networks
  from `tests/data/bnet/`, generating every state of each network, and
  checks correctness, completeness and minimality directly.
- `tests/test_examples.py` runs each script in `examples/` as a
  subprocess. Each script checks its own results with `assert`.
- `tests/test_reproducibility.py` runs one workflow in a subprocess per
  hash seed (`PYTHONHASHSEED`) and checks that every output is the same.
- `tests/typing/user_code.py` is never run: `mypy` checks it, so a
  public hint that stops accepting what the docs say it accepts, or
  stops naming the type a user gets back, fails the `mypy` check.
- `tests/test_repo_hygiene.py` checks conventions no other test would,
  such as that shipped code names no file that doesn't ship.
- Every `>>>` example in a docstring, on a page under `docs/`, and in
  the README runs as a doctest. Pages under `docs/how-to/` run from
  that folder, so they read its files by bare name
  (`docs/how-to/conftest.py`).
- A test that checks a statement of
  [Behavior guarantees](../guarantees.md) names it in a comment above
  itself, `# SPEC: GULA-1`. `test_spec_references.py` fails on a name
  the page doesn't define, and warns about a statement no test names.
- Some tests guard the project's structure rather than behavior:
  `test_public_api.py` (one import path per public name, every one in
  `__all__`), `test_layering.py` (models and algorithms don't import
  each other), `test_metadata.py` (what is written in two places
  agrees), `test_docs_markup.py` (MyST markup the docs build doesn't
  catch) and `test_commit_msg_hook.py` (the commit-message hook).

## Fast and slow

Tests that take seconds are marked `@pytest.mark.slow` and left out of
a plain `pytest` run. The split is by measured cost: run
`pytest --durations=0 -m ""` to see it. A new test that takes more than
about a second belongs in the slow tier.

## Coverage

`pytest --cov` adds a table of the lines and branches of `src/pylfit`
no test runs; CI prints it in each Linux test job. Below 99% the run
fails (`fail_under` in `pyproject.toml`). Three lines of
`algorithms/_algorithm.py` always show: the `n_jobs` worker functions,
which run only in another process, where coverage doesn't trace them.
Any other line it lists is a question, whether some behavior there
goes unchecked.

## Doctests

Doctest output must be the same in every Python process and with every
supported pandas version. Print `sorted(...)` rather than a `set`, a
value rather than a `DataFrame` repr, round floats, and seed every
random draw. A code block ends with a blank line before the closing
fence, or the fence is read as expected output.

Warnings are errors in the suite (`filterwarnings` in
`pyproject.toml`), so a test that expects one asserts it with
`pytest.warns`. One message is exempt: three examples warn about
uncovered rows on purpose, because the warning is what they show:
`GULA.fit`'s and `coverage`'s, and the one on unknown values in
`docs/explanation/about-algorithms.md`.
