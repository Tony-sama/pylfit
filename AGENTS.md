# AGENTS.md

`pylfit` learns logic rules from data — a symbolic-AI alternative to
black-box classifiers. This is the 1.0 rewrite, incompatible with
`pylfit` 0.x. See `README.md` for the user-facing overview.

## Setup and checks

`CONTRIBUTING.md` holds the development setup and every check a change
must pass; run all of them before calling a change done. In short:

```bash
pip install -e ".[dev,docs]"
git config core.hooksPath .githooks    # once per clone: commit-subject cap
pytest -m ""                           # every test, the slow ones included
```

Before opening a pull request, follow "Before opening a pull request"
and "AI tools" in `CONTRIBUTING.md`: one change per pull request, a
changelog line for anything a user would notice, and the tool and
model that helped named in the description.

## Code style

The rules for code, docstrings and doc pages are in
`docs/contributing/standards.md`, under "Writing docstrings and pages",
and `docs/contributing/testing.md` says how to keep doctest output the
same everywhere. Read both before writing a docstring.

Everything pylfit returns, prints or writes must be the same on every
run, given the same input, row order and `random_state`: never let a
`set`'s order reach an output. "The same output on every run" in
`docs/contributing/design-notes.md` says why, how, and what is exempt;
extend `tests/test_reproducibility.py` when you add an output.

## Releasing

Follow "Cut a release" in `docs/contributing/maintainer-notes.md`.

## Layout

`src/pylfit/` is the package and the only thing that ships. `tests/`,
`examples/` and `docs/` are here but stay out of the wheel.

`docs/contributing/architecture.md` shows how the parts fit together:
the pipeline from a DataFrame to predictions, and which module may
import which. `design-notes.md` beside it says why they are shaped
that way.

Benchmarks against other rule learners and research experiments are
out of scope for this repo.

## Conventions

Mechanism — **how** something works — belongs in the docstrings, next
to the code, never only in a file the reader may not have:

- Nothing in this repo names the maintainers' planning notes, which are
  kept outside it, or a directory removed from it: either pointer
  dangles for every reader. `AGENT_DOCS` and `DEPARTED_DIRS` in the
  guard below list them.
- Docstrings under `src/` don't name `tests/`, `examples/` or `docs/`,
  which stay out of the wheel: the pointer dangles for anyone reading
  from an install or from the generated doc site.

Write the substance inline instead. `tests/test_repo_hygiene.py` fails
on either; `.githooks/commit-msg` warns on a planning-note name in a
commit message.
