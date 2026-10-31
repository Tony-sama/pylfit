# Standards

The published standards pylfit follows, and where it departs from one
on purpose. A change should keep to them; a review should hold the
code to them. When something here and a tool's default disagree, this
page wins.

## Code

| Area | Standard | Checked by |
| --- | --- | --- |
| Style | [PEP 8](https://peps.python.org/pep-0008/), formatted as [Black](https://black.readthedocs.io/en/stable/) does, 88 characters a line | `ruff check`, `ruff format --check` |
| Docstrings | [PEP 257](https://peps.python.org/pep-0257/), in [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings); the rules for content are [below](#writing-docstrings-and-pages) | `pydoclint`, `interrogate` (every public object) |
| Type hints | [PEP 484](https://peps.python.org/pep-0484/); the package ships `py.typed` ([PEP 561](https://peps.python.org/pep-0561/)) | `mypy` |
| Spelling | | `codespell` |
| Test coverage | Line and branch coverage of `src/pylfit`, printed by CI's Linux test jobs; a run below 99% fails | `pytest --cov` |
| API | [scikit-learn's API conventions](https://scikit-learn.org/stable/developers/develop.html) where they apply: `predict`, `predict_proba`, `random_state`, `n_jobs` | review |

## Writing docstrings and pages

- Files are `snake_case.py`, classes `CapWords`.
- Docstrings are Google style, in reStructuredText, with type hints in
  the signature only. A public docstring is reference: a summary, what
  the object does (not why), Args, Returns and Raises, a short note for
  behavior a caller must know, then `See Also:`, `References:` and
  `Examples:`. The reasoning behind a design belongs on the docs
  pages, not in the docstring; implementation detail belongs in the
  private method that carries it.
- A cited work is listed under `References:` in a docstring, or in
  the text of a docs page, with its DOI as a full
  `https://doi.org/...` link (the form Crossref asks for, not
  `doi:...`). A work without a DOI gets its HAL link, or for a book
  its publisher and ISBN. In a docstring, the entry is a citation,
  `.. [Tukey1977] Tukey, J. W. (1977). ...`, and the text cites it as
  `[Tukey1977]_`, which the docs render as a link to the entry. Sphinx
  shares these labels across every page, so each work has one.
- A claim that needs a source the code can't give is marked
  `[paper placeholder: <claim>]`.
- Examples are doctests, run by `pytest`; [Testing](testing.md#doctests)
  says how to keep their output the same in every run.
- Docstring prose uses no "e.g." or "i.e.", no bold for emphasis, and
  "—" rather than "--".
- A finding goes into `.pydoclint-baseline.txt` only when the
  exception really comes from a called helper.

## Packaging and project

| Area | Standard | Checked by |
| --- | --- | --- |
| Project layout and tooling | The [Scientific Python Development Guide](https://learn.scientific-python.org/development/) | [`sp-repo-review`](https://learn.scientific-python.org/development/guides/repo-review/), by hand before a release |
| Metadata | [`pyproject.toml`](https://packaging.python.org/en/latest/specifications/pyproject-toml/) ([PEP 621](https://peps.python.org/pep-0621/)), license as an [SPDX expression](https://peps.python.org/pep-0639/) | `sp-repo-review`, by hand before a release |
| Supported versions | [SPEC 0](https://scientific-python.org/specs/spec-0000/) for pandas and NumPy; Python from 3.11, longer than SPEC 0 asks (see below) | CI, at the oldest and newest supported Python and at the oldest supported pandas and NumPy |
| Platforms | Linux, macOS and Windows | CI: the whole suite on Linux, the fast tier on macOS and Windows |
| Built package | What `python -m build` produces installs and passes the tests on its own | CI: `twine check --strict`, `check-wheel-contents`, the tests against the installed wheel |
| Versions | [Semantic versioning](https://semver.org), written as [PEP 440](https://peps.python.org/pep-0440/) requires; the [changelog](../changelog.md) says what the public API is | review |
| Changelog | [Keep a Changelog](https://keepachangelog.com) | review |
| Licensing | GPL-3.0-only; third-party data keeps its own license, stated in the README of the folder that holds it (`tests/data/bnet/`, `examples/data/`) | review |
| Community | [Contributor Covenant](https://www.contributor-covenant.org) 2.1 | — |
| CI security | Actions pinned to a commit, updated by Dependabot | [`zizmor`](https://docs.zizmor.sh), by hand when a file in `.github/` changes ([maintainer notes](maintainer-notes.md)) |

## Documentation

| Area | Standard | Checked by |
| --- | --- | --- |
| Structure | [Diátaxis](https://diataxis.fr): tutorials, how-to guides, explanation, reference | review |
| Prose | The [Google developer documentation style guide](https://developers.google.com/style) | review |
| Build | Every warning is an error, every cross-reference resolves, every `>>>` example runs | `sphinx-build -n -W`, `pytest` |
| Links | Every external link resolves | `sphinx-build -b linkcheck`, monthly |
| Hosting | GitHub Pages, built by the same command as CI | CI |

## Versions

Which version of each thing pylfit needs or is built with, why, and
what moves it. A version is held back only where a newer one would
change something unasked; everything else takes the latest release,
and CI shows when one breaks something.

| What | Version | Why | Moved by |
| --- | --- | --- | --- |
| Python | 3.11 or later; CI tests 3.11 and the newest release | See the departures below | Hand: the newest each October, when a Python is released; the oldest at 3.11's end of life |
| pandas, NumPy | pandas 2.3, NumPy 2.2 or later; CI also tests 2.3.0 and 2.2.0 together | SPEC 0: the feature releases of the last two years | Hand, when SPEC 0 drops a floor |
| ruff | One minor series | The formatter's output changes between minors | Hand, reformatting in the same commit |
| The other tools: pytest, mypy, pydoclint, interrogate, codespell, Sphinx and its extensions, build, twine | The latest | Pinning them would mean updating each by hand; a release that breaks something shows in CI, and is fixed then | Nothing: every install takes the latest |
| hatchling, the build backend | 1.27 or later | Older releases write the license in the outdated form | Hand, when pylfit needs a newer feature |
| pip | Any | Debian 12's pip 23.0.1 installs pylfit, editable included | — |
| GitHub Actions | A commit, with its version in a comment | A tag can be moved to other code, a commit can't | Dependabot, one pull request a month |
| Dev container | Python 3.11 | A contributor checks a change on the oldest Python users have | Hand, with the oldest Python |

The docs build, the lint job and the package job run on Python 3.11,
so each tool there is its newest release that still supports 3.11:
Sphinx 9.0, for example, since 9.1 needs Python 3.12. This lifts when
the oldest supported Python rises.

## Deliberate departures

- **pylfit is not a scikit-learn estimator.** An algorithm's `fit` is
  static and returns rules; a model is built from rules and a dataset.
  The [design notes](design-notes.md) say why.
- **Python from 3.11, longer than SPEC 0's three years.** Nothing in
  pylfit needs a newer Python, and a research tool gets installed on
  whatever Python a lab machine or cluster already has: Debian 12
  ships 3.11, Ubuntu 24.04 ships 3.12. The floor rises when 3.11
  reaches its end of life (October 2027), or when code needs a newer
  feature.
- **No pre-commit configuration and no task runner** (nox, tox).
  `CONTRIBUTING.md` lists every check, CI runs them all, and each is
  one command.
- **No external services beyond GitHub and PyPI**: no Read the Docs,
  no coverage service. Coverage is measured in CI and reported there.
- **Development tools are an extra (`.[dev]`), not a dependency
  group** ([PEP 735](https://peps.python.org/pep-0735/)). Installing a
  group needs pip 25.1 or later (April 2025), and a Python 3.11
  environment can come with an older one, which would fail a
  contributor's first command.
- **No `show_versions()`.** The bug report form gives a one-line
  command that prints the same versions.
- **mypy's `redundant-expr` check is off.** mypy is otherwise strict,
  but that check flags the `isinstance` tests that give a caller who
  doesn't type-check a clear `TypeError`, and those are deliberate.
