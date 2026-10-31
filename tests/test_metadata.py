"""Project metadata that is written in more than one place."""

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _project() -> dict:
    """``pyproject.toml``'s ``[project]`` table."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return tomllib.loads(text)["project"]


def _citation() -> tuple[dict[str, str], list[dict[str, str]]]:
    """``CITATION.cff``'s top-level scalars, and its top-level authors.

    A line-based reading of the YAML the file uses: ``key: value``
    with the value quoted or not, and ``authors:`` entries of indented
    ``key: value`` lines, in any order. The ``authors`` of
    ``references:`` are left out.
    """
    text = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    scalars: dict[str, str] = {}
    authors: list[dict[str, str]] = []
    section = ""
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if not line.startswith(" "):
            key, value = line.split(":", 1)
            section, value = key, value.strip()
            if value and value != ">-":
                scalars[key] = value.strip('"')
        elif section == "authors":
            entry = line.strip()
            if entry.startswith("- "):
                authors.append({})
            key, value = entry.removeprefix("- ").split(":", 1)
            authors[-1][key] = value.strip().strip('"')
    return scalars, authors


def test_citation_version_matches_the_package():
    """``CITATION.cff`` cites the version ``pyproject.toml`` builds."""
    assert _citation()[0]["version"] == _project()["version"]


def test_citation_license_and_repository_match_the_package():
    """The citation's license and repository are the package's."""
    scalars, _ = _citation()
    project = _project()
    assert scalars["license"] == project["license"]
    assert scalars["repository-code"] == project["urls"]["Repository"]


def test_author_lists_match():
    """``pyproject.toml``, ``CITATION.cff`` and ``AUTHORS.md`` agree.

    The same people, in the same order.
    """
    names = [author["name"] for author in _project()["authors"]]
    cited = [
        f"{author['given-names']} {author['family-names']}" for author in _citation()[1]
    ]
    assert cited == names

    # Each author's name starts a paragraph of AUTHORS.md. Its other
    # paragraphs, on credits and contributors, start with no name, so
    # a name added to AUTHORS.md alone goes unseen.
    authors = (REPO_ROOT / "AUTHORS.md").read_text(encoding="utf-8")
    starts = [paragraph.split(" ", 2) for paragraph in authors.split("\n\n")]
    listed = [" ".join(words[:2]) for words in starts]
    assert [name for name in listed if name in names] == names
