"""Runs the doc pages' sessions from this folder.

"Load a CSV file" reads ``patients.csv`` by its bare name, as a reader
would, so its session needs this folder as the working directory.
"""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _run_from_this_folder(monkeypatch):
    """Sets the working directory to this folder for each session."""
    monkeypatch.chdir(Path(__file__).parent)
