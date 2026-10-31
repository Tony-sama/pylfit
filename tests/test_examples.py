"""Smoke tests for the scripts in ``examples/``.

These exist because an example that no longer runs is worse than no
example: it is the first thing a newcomer tries, and it fails in their
environment rather than in CI. The suite therefore runs each one the
way a reader would — as a script, in a subprocess — rather than
importing pieces of it and asserting on internals.

**What is checked is that they run and that their self-checks hold**,
not that any particular number comes out. Timings are hardware, and
pinning them here would turn a slow machine into a red build. Every
script already asserts its own invariants (``BruteForce`` and ``GULA``
agree; ``PRIDE`` invents nothing; ``starter.py`` meets a threshold
chosen before the run), so a subprocess exiting zero means those held.

**Three scripts are in the fast tier, ``starter.py``,
``migrate_from_0x.py`` and ``benchmark_influence.py``; the other eight
are slow.** The split is by measured cost (under a second each, against
2 to 35 seconds), not by importance. Trimming a slow one to fit the
fast tier would leave the suite testing something other than the
script people run, so they run where ``-m ""`` runs everything: CI's
Linux jobs.
"""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run_example(name: str) -> subprocess.CompletedProcess:
    """Runs ``examples/<name>`` as a subprocess and returns the result.

    Args:
        name: Script filename, e.g. ``"starter.py"``.

    Returns:
        The completed process, with stdout and stderr captured.
    """
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        capture_output=True,
        text=True,
        cwd=EXAMPLES,
        check=False,
    )


def assert_ran_clean(result: subprocess.CompletedProcess, name: str) -> None:
    """Fails with the script's own output when it exits non-zero.

    Args:
        result: The completed process.
        name: Script filename, for the failure message.

    Raises:
        AssertionError: If the script exited non-zero.
    """
    assert result.returncode == 0, (
        f"examples/{name} exited {result.returncode}. Its own output:\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def test_migrate_from_0x_runs_and_matches_0x():
    """The 0.x porting example, whose asserts pin 0.5.1's own output.

    Exiting zero means GULA's rules, the three semantics' next states
    and the weighted model's top values still match what pylfit 0.5.1
    printed on the same data — the claims the migration guide makes.
    """
    result = run_example("migrate_from_0x.py")
    assert_ran_clean(result, "migrate_from_0x.py")
    assert "Same 7 rules as pylfit 0.5.1." in result.stdout


def test_starter_runs_and_meets_its_own_threshold():
    """``starter.py`` is the template people copy: it must pass as-is.

    A broken starter is worse than a broken benchmark: whatever it does
    gets copied into someone's own script and inherited. Exiting zero
    means it cleared the accuracy threshold it sets for itself *and*
    round-tripped the saved model. It is also the first thing a
    newcomer runs, which is why it is in the fast tier.
    """
    result = run_example("starter.py")
    assert_ran_clean(result, "starter.py")
    assert "Passed." in result.stdout


@pytest.mark.slow
def test_benchmark_classic_runs_and_its_checks_hold():
    """The car-evaluation example, including its completeness oracle.

    Exiting zero means ``BruteForce`` and ``GULA`` returned identical
    rule sets on real multi-valued data — the same claim the
    boolean-network suite makes, on a dataset shaped nothing like a
    transition table — and that ``GULA``'s rules matched every class on
    each held-out row.
    """
    result = run_example("benchmark_classic.py")
    assert_ran_clean(result, "benchmark_classic.py")
    assert "BruteForce and GULA agree exactly" in result.stdout


@pytest.mark.slow
def test_benchmark_bnet_runs_and_its_checks_hold():
    """The boolean-network example, across every network it lists.

    Exiting zero means ``BruteForce`` and ``GULA`` agreed on each
    network, and ``PRIDE`` returned no rule ``GULA`` did not find.
    """
    result = run_example("benchmark_bnet.py")
    assert_ran_clean(result, "benchmark_bnet.py")
    assert "BruteForce and GULA agreed on every network" in result.stdout


def test_benchmark_influence_runs_and_its_checks_hold():
    """``allowed_conditions`` restricts the search without changing it.

    Exiting zero means all three configurations returned identical
    rules, which is the claim exact background knowledge makes. It also
    runs ``_bnet.influences``, which reads each node's regulators, and
    the sign of the monotone ones, from a ``.bnet`` file.
    """
    result = run_example("benchmark_influence.py")
    assert_ran_clean(result, "benchmark_influence.py")
    assert "All three configurations agree exactly" in result.stdout


@pytest.mark.slow
def test_benchmark_min_support_runs_and_its_checks_hold():
    """``min_support`` keeps just the unrestricted rules that clear it.

    Checked on both datasets against the unrestricted run, filtered by
    support counted directly against the data — including the dataset
    where the filter prunes least, since the guarantee holds either way.
    """
    result = run_example("benchmark_min_support.py")
    assert_ran_clean(result, "benchmark_min_support.py")
    assert "min_support kept exactly the unrestricted rules" in result.stdout


@pytest.mark.slow
def test_benchmark_negative_order_runs_and_its_checks_hold():
    """Every ``negative_order`` returns the same rules on every dataset.

    Exiting zero means that held; which order was fastest is not
    checked, since it depends on the dataset and the machine.
    """
    result = run_example("benchmark_negative_order.py")
    assert_ran_clean(result, "benchmark_negative_order.py")
    assert "Every order returned the same rules" in result.stdout


@pytest.mark.slow
def test_benchmark_n_jobs_runs_and_its_checks_hold():
    """Every ``n_jobs`` returns the serial run's rules.

    Exiting zero means that held for each process count the script
    times; how much faster each one was is not checked.
    """
    result = run_example("benchmark_n_jobs.py")
    assert_ran_clean(result, "benchmark_n_jobs.py")
    assert "Every n_jobs returned the same" in result.stdout


@pytest.mark.slow
def test_benchmark_unknowns_runs_and_its_checks_hold():
    """Writing "?" for the unknown only adds rules that test "?".

    Exiting zero means that held on every dataset, rate and seed: a
    "?" differs from every value just as an unknown does, so the two
    readings can differ only by the rules that test "?".
    """
    result = run_example("benchmark_unknowns.py")
    assert_ran_clean(result, "benchmark_unknowns.py")
    assert "on every dataset, rate and seed, as it must." in result.stdout


@pytest.mark.slow
def test_dynamics_runs_and_its_checks_hold():
    """The time-series example: transitions, learning, simulation.

    Exiting zero means every observed transition was reproduced exactly
    under synchronous semantics, every asynchronous step changed at
    most one variable, and the model learned from the full transition
    table reproduced the whole system.
    """
    result = run_example("dynamics.py")
    assert_ran_clean(result, "dynamics.py")
    assert "every observed transition is reproduced exactly" in result.stdout


@pytest.mark.slow
def test_tutorial_heart_disease_runs_and_meets_its_own_threshold():
    """The real-data tutorial, on UCI Cleveland heart disease.

    Exiting zero means ``GULA`` cleared the accuracy the script sets for
    itself.
    """
    result = run_example("tutorial_heart_disease.py")
    assert_ran_clean(result, "tutorial_heart_disease.py")
    assert "Passed." in result.stdout
