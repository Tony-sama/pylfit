"""Guards on the docs' MyST markup that the Sphinx build doesn't catch.

**A figure's alt text that wraps must sit in a YAML option block.**
Written as ``:alt:`` with continuation lines, MyST keeps only the first
line as the alt text and starts the visible caption with the rest; the
build passes with no warning. The form that keeps a multi-line value
is::

    ```{figure} ../images/name.svg
    ---
    alt: >-
      First line of the description,
      and the rest of it.
    ---

    The caption.
    ```
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
# The pages, and the files they pull in with {include}, such as the
# README on the home page.
_SOURCES = sorted(path for path in DOCS.rglob("*.md") if "_build" not in path.parts)
INCLUDED = sorted(
    {
        (source.parent / name).resolve()
        for source in _SOURCES
        for name in re.findall(
            r"^```\{include\} (\S+)", source.read_text(encoding="utf-8"), re.MULTILINE
        )
    }
)
PAGES = _SOURCES + [path for path in INCLUDED if DOCS not in path.parents]

# A `:name:` option line, then a line at the same indent that is not
# blank, another option, or the closing fence: MyST reads it as the
# caption's start, whatever its own indent.
WRAPPED_OPTION = re.compile(r"^( *):\w+:.*\n\1(?! *$| *```|:\w+:)", re.MULTILINE)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_wrapped_colon_option(page):
    text = page.read_text(encoding="utf-8")
    lines = [
        text.count("\n", 0, match.start()) + 1
        for match in WRAPPED_OPTION.finditer(text)
    ]
    assert not lines, (
        f"{page.relative_to(ROOT)}: an option wraps at line(s) {lines}; "
        "MyST puts the wrapped part into the caption. Use a YAML option "
        "block (---, alt: >-, ---) instead."
    )


@pytest.mark.parametrize(
    ("figure", "wraps"),
    [
        (":alt: First\n  second.\n\nCaption.", True),
        (":alt: First\nsecond.\n\nCaption.", True),
        ("  :alt: First\n    second.\n\n  Caption.", True),  # in a list
        (":alt: First.\n:width: 50%\n\nCaption.", False),
        (":alt: First.\n```", False),
        ("---\nalt: >-\n  First\n  second.\n---\n\nCaption.", False),
    ],
)
def test_the_guard_finds_a_wrapped_option(figure, wraps):
    # Each case's caption checked in a built page under MyST 5.1.
    assert bool(WRAPPED_OPTION.search(figure + "\n")) == wraps


def test_guard_sees_the_pages():
    assert len(PAGES) > 10
    assert ROOT / "README.md" in PAGES
    assert ROOT / "CONTRIBUTING.md" in PAGES
