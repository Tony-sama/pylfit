# Maintainer notes

## Repository settings the workflows need

The workflows rely on these settings of `Tony-sama/pylfit` on GitHub
and of the `pylfit` project on PyPI; check them if a release or the
docs site stops publishing:

- **Pages**: under Settings, Pages, set the source to GitHub Actions.
  The `pages` job in `.github/workflows/ci.yml` then publishes the docs
  site on every push to the default branch.
- **The `pypi` environment**: under Settings, Environments, create one
  named `pypi` and add yourself as a required reviewer. The release
  workflow publishes from it, so a pushed tag waits for your approval.
- **PyPI**: the `pylfit` project's trusted publisher names this
  repository, the workflow `release.yml` and the environment `pypi`;
  update it if any of those names change.

## Cut a release

1. Set `version` in `pyproject.toml` and in `CITATION.cff`; a test
   fails if the two differ. Set `date-released` in `CITATION.cff` to
   the release date, `YYYY-MM-DD`, so citations carry a year.
2. In `docs/changelog.md`, rename `[Unreleased]` to the version and
   date, `## [1.0.0] - YYYY-MM-DD`, start a new empty `[Unreleased]`
   above it, and add a comparison link for each at the end.
3. Commit, and tag the commit `v<version>`, for example `v1.0.0`.
4. Push the tag. `.github/workflows/release.yml` runs the whole test
   suite, builds and checks the sdist and wheel, and waits.
5. Approve the deployment to the `pypi` environment on GitHub. The
   workflow then publishes to PyPI through trusted publishing; no API
   token is stored anywhere.
6. Once PyPI has the release, the workflow creates the GitHub release
   for the tag, with the sdist and wheel attached and a link to the
   changelog.

The workflow refuses a tag that doesn't match `version`, and runs only
in the `Tony-sama/pylfit` repository. An `aN`, `bN`, `rcN` or `.devN`
version is a pre-release: `pip install pylfit` skips it unless given
`--pre`, and the workflow publishes it to PyPI but creates no GitHub
release for it.

## Checks run by hand

Two checks are not in CI, because what they check rarely changes.

- **After changing a file in `.github/`** (a workflow or
  `dependabot.yml`), run [`zizmor`](https://docs.zizmor.sh), which looks
  for unpinned actions, leaked credentials, injectable expressions and
  a missing Dependabot cooldown:

  ```bash
  pip install zizmor
  zizmor .github
  ```

- **Before a release**, run
  [`sp-repo-review`](https://learn.scientific-python.org/development/guides/repo-review/),
  which checks the project layout and `pyproject.toml` against the
  Scientific Python Development Guide. The checks pylfit departs from
  on purpose are skipped (`[tool.repo-review]` in `pyproject.toml`; the
  [standards page](standards.md) says why), so any failure is new.

  ```bash
  pip install "sp-repo-review[cli]"
  sp-repo-review .
  ```

## Triage

Issues and pull requests come in on GitHub. A bug report is complete
when it has the pylfit and Python versions and code that reproduces
the problem; the bug-report form in `.github/ISSUE_TEMPLATE/` asks for
both. The forms there (a bug, or anything else) take
effect on GitHub once they are on the default branch; blank issues are
turned off, and security reports are pointed to `SECURITY.md`.
