"""Shared pytest fixtures for the pylfit test suite.

Fixtures build small datasets, and the rules learned from one, that can
be checked by hand, so tests can assert against known-correct rule sets
and predictions, not just "it runs without crashing"; plus
``outputs_per_hash_seed``, a tool for the hash-seed tests.
"""

import os
import subprocess
import sys
from collections.abc import Iterable

import pandas as pd
import pytest

from pylfit import Dataset, Rule


@pytest.fixture
def toy_dataset():
    """The canonical toy dataset as a ``Dataset``.

    Deterministic: 3 binary features (``a``, ``b``, ``c``), 1 binary
    target (``y``), all 8 states. ``y`` is ``1`` when ``b`` is ``1``, or
    when ``a`` is ``0`` and ``c`` is ``1``.
    """
    df = pd.DataFrame(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
        ],
        columns=["a", "b", "c", "y"],
    )
    return Dataset(df, features=["a", "b", "c"], targets=["y"])


@pytest.fixture
def toy_rules():
    """The four rules every algorithm learns from ``toy_dataset``.

    ``BruteForce``, ``GULA`` and ``PRIDE`` alike. Derived by hand in
    ``test_brute_force.py``, and built by hand here so the model tests
    depend on no algorithm.
    """
    return [
        Rule(head=("y", "1"), body={"b": "1"}),
        Rule(head=("y", "1"), body={"a": "0", "c": "1"}),
        Rule(head=("y", "0"), body={"b": "0", "c": "0"}),
        Rule(head=("y", "0"), body={"a": "1", "b": "0"}),
    ]


@pytest.fixture
def nondeterministic_dataset():
    """A dataset where the same feature state disagrees on the target.

    ``a=0`` has rows with both ``y=0`` and ``y=1`` — exercises the
    branch of ``Dataset.negatives`` that data without conflicts skips.
    Built with ``assume_nondeterministic=True``, which silences the
    conflict warning and, without unknowns, changes nothing else.
    """
    df = pd.DataFrame({"a": [0, 0, 1], "y": [0, 1, 1]})
    return Dataset(df, features=["a"], targets=["y"], assume_nondeterministic=True)


@pytest.fixture
def degenerate_dataset():
    """A dataset whose target never varies.

    Every row agrees on ``y=1``, so the fully general (empty-body)
    rule is already consistent — no specialization is ever needed.
    """
    df = pd.DataFrame({"a": [0, 1, 0, 1], "b": [0, 0, 1, 1], "y": [1, 1, 1, 1]})
    return Dataset(df, features=["a", "b"], targets=["y"])


@pytest.fixture
def multi_target_dataset():
    """Two targets over the same feature, correlated oppositely with it.

    ``y1`` is deterministic on ``a`` (``y1 = a``); ``y2`` is the
    opposite (``y2 = 1 - a``). For checking that an algorithm learns
    each target's rules from its own negatives, not conflating the two.
    """
    df = pd.DataFrame({"a": [0, 0, 1, 1], "y1": [0, 0, 1, 1], "y2": [1, 1, 0, 0]})
    return Dataset(df, features=["a"], targets=["y1", "y2"])


@pytest.fixture
def support_dataset():
    """One feature whose values give ``y=1`` very different support.

    ``a=0`` (5 rows) concludes ``y=0``; ``a=1`` (1 row) and ``a=2`` (4
    rows) both conclude ``y=1``, giving two ``y=1`` rules, with 1 and 4
    of the 5 positives, to filter between.
    """
    df = pd.DataFrame(
        {
            "a": [0, 0, 0, 0, 0, 1, 2, 2, 2, 2],
            "y": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )
    return Dataset(df, features=["a"], targets=["y"])


@pytest.fixture
def outputs_per_hash_seed(tmp_path):
    """Runs a script once per hash seed and returns the set of outputs.

    Python seeds its string hashing per process (``PYTHONHASHSEED``),
    so a ``set`` of strings iterates in another order in each process.
    A set of one output means the script's output does not depend on
    that order. Call it with the script's text, its arguments, and
    optionally ``seeds``; each run must exit cleanly within 120 s.
    """

    def run(script: str, *args: str, seeds: Iterable[int] = range(6)) -> set[str]:
        path = tmp_path / "script.py"
        path.write_text(script, encoding="utf-8")
        return {
            subprocess.run(
                [sys.executable, str(path), *args],
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "PYTHONHASHSEED": str(seed)},
                check=True,
            ).stdout
            for seed in seeds
        }

    return run
