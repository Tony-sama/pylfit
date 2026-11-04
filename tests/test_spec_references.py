"""Ties ``docs/guarantees.md`` to the tests that check its statements.

A test names the statement it checks with a comment above it,
``# SPEC: GULA-1``. A reference to no statement fails, as the numbers
on that page are what the comments rely on. A statement no test names
only warns: some are checked by the page's own examples, or by a whole
class of tests, and a failure would push towards empty references.
"""

import re
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "guarantees.md"


def _statements() -> set[str]:
    """Gives the IDs the page defines."""
    return set(re.findall(r"^- \*\*([A-Z]+-\d+)\.\*\*", PAGE.read_text(), re.MULTILINE))


def _references() -> set[str]:
    """Gives the IDs the tests name."""
    return {
        name
        for path in (ROOT / "tests").rglob("*.py")
        for name in re.findall(
            r"^# SPEC: ([A-Z]+-\d+)$",
            path.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    }


def test_the_page_defines_statements():
    assert len(_statements()) > 50


def test_every_reference_names_a_statement():
    assert _references() <= _statements()


@pytest.mark.filterwarnings("default::UserWarning")
def test_a_statement_no_test_names_warns():
    missing = sorted(_statements() - _references())
    if missing:
        warnings.warn(f"no test names: {missing}", stacklevel=1)
