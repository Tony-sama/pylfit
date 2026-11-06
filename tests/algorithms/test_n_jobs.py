"""Tests for ``n_jobs`` and the loop over heads all algorithms share."""

import multiprocessing
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pylfit import GULA, PRIDE, BruteForce, Dataset, PylfitWarning
from pylfit.algorithms._algorithm import Algorithm, _broken_pool_message

ALGORITHMS = [GULA, PRIDE, BruteForce]


@pytest.fixture
def two_target_dataset():
    """Two targets, one with three values, and an unknown feature."""
    df = pd.DataFrame(
        {
            "a": ["0", "0", "1", "1", "2", None],
            "b": ["0", "1", "0", "1", "1", "0"],
            "y": ["0", "1", "1", "2", "2", "0"],
            "z": ["1", "0", "0", "0", "1", "1"],
        }
    )
    return Dataset(df, features=["a", "b"], targets=["y", "z"])


def _fit_and_warnings(algorithm, dataset, **options):
    """The rules and the warning messages of one ``fit`` call."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rules = algorithm.fit(dataset, **options)
    # Python 3.12 and 3.13 warn at each fork of a multi-threaded process,
    # with its pid, and pyarrow starts a thread on import.
    return rules, [
        str(w.message) for w in caught if "use of fork()" not in str(w.message)
    ]


# SPEC: LEARN-5
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_n_jobs_learns_the_same_rules_and_warnings(algorithm, two_target_dataset):
    sequential = _fit_and_warnings(algorithm, two_target_dataset)
    parallel = _fit_and_warnings(algorithm, two_target_dataset, n_jobs=2)
    assert parallel == sequential
    assert sequential[1], "the unknown value should leave a row uncovered"


@pytest.mark.parametrize("algorithm", [GULA, PRIDE])
def test_n_jobs_keeps_per_head_options(algorithm, toy_dataset):
    options = {
        "heads": ["y"],
        "min_support": 2,
        "allowed_conditions": {("y", "1"): ["a", "b"]},
        "warn_uncovered": False,
    }
    expected = algorithm.fit(toy_dataset, **options)
    assert algorithm.fit(toy_dataset, n_jobs=-1, **options) == expected


def test_brute_force_heads_learns_only_those_heads(two_target_dataset):
    full = BruteForce.fit(two_target_dataset, warn_uncovered=False)
    some = BruteForce.fit(two_target_dataset, heads=["z"], warn_uncovered=False)
    assert some == [rule for rule in full if rule.head[0] == "z"]


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_n_jobs_zero_raises(algorithm, toy_dataset):
    with pytest.raises(ValueError, match="n_jobs must be nonzero"):
        algorithm.fit(toy_dataset, n_jobs=0)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_n_jobs_takes_a_numpy_integer(algorithm, toy_dataset):
    assert algorithm.fit(toy_dataset, n_jobs=np.int64(1)) == algorithm.fit(toy_dataset)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("n_jobs", [1.5, True, "2"])
def test_n_jobs_that_is_not_an_integer_raises(algorithm, toy_dataset, n_jobs):
    with pytest.raises(TypeError, match="n_jobs must be an int"):
        algorithm.fit(toy_dataset, n_jobs=n_jobs)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_n_jobs_none_learns_in_this_process(algorithm, toy_dataset):
    # As in scikit-learn, None means one job.
    assert Algorithm._resolve_n_jobs(None) == 1
    assert algorithm.fit(toy_dataset, n_jobs=None) == algorithm.fit(toy_dataset)


def test_verbose_with_n_jobs_logs_each_head_from_this_process(toy_dataset, capsys):
    GULA.fit(toy_dataset, verbose=True, n_jobs=2)
    err = capsys.readouterr().err
    heads = len(toy_dataset.domain("y"))
    assert err.count("heads, ") == heads
    assert "on 2 processes" in err


def test_more_jobs_than_heads_uses_one_process_per_head(capsys):
    df = pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    GULA.fit(dataset, verbose=True, n_jobs=64)
    assert "on 2 processes" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("n_jobs", "cpus", "expected"),
    [(3, 8, 3), (-1, 8, 8), (-2, 8, 7), (-8, 8, 1), (-20, 8, 1)],
)
def test_resolve_n_jobs_counts_back_from_the_cpus(monkeypatch, n_jobs, cpus, expected):
    monkeypatch.setattr("os.process_cpu_count", lambda: cpus, raising=False)
    assert Algorithm._resolve_n_jobs(n_jobs) == expected


@pytest.mark.parametrize(("platform", "expected"), [("win32", 61), ("linux", 100)])
def test_windows_caps_the_processes_at_61(monkeypatch, platform, expected):
    # ProcessPoolExecutor raises above 61 workers on Windows.
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr("os.process_cpu_count", lambda: 128, raising=False)
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    assert Algorithm._resolve_n_jobs(100) == expected
    assert Algorithm._resolve_n_jobs(-29) == expected


def _later_heads_first(_dataset, target, value, _verbose):
    """Learns nothing, slower for lower values; one row uncovered."""
    time.sleep(0.2 * (2 - int(value)))
    return [], [((target, value), {"a": value}, "unknown values")]


@pytest.mark.filterwarnings(r"ignore:.*use of fork\(\):DeprecationWarning")
def test_n_jobs_joins_the_results_in_head_order():
    # The heads finish in reverse order; the warning still lists them in
    # head order, as a sequential run does.
    df = pd.DataFrame({"a": ["0", "1", "2"], "y": ["0", "1", "2"]})
    dataset = Dataset(df, features=["a"], targets=["y"])
    with pytest.warns(PylfitWarning) as record:
        Algorithm._fit_heads(dataset, _later_heads_first, verbose=True, n_jobs=3)
    message = str(record.pop(PylfitWarning).message)
    listed = [line.split(":")[0] for line in message.splitlines()[1:]]
    assert listed == ["  y=0", "  y=1", "  y=2"]


def _fail_first_head(_dataset, _target, value, _verbose, *, directory):
    """Fails on the first head; leaves a file named after each other."""
    if value == "00":
        raise RuntimeError("the first head failed")
    time.sleep(0.2)
    Path(directory, value).touch()
    return [], []


def test_a_failed_head_cancels_the_heads_still_queued(tmp_path):
    # Ctrl-C takes the same path: without the cancel, the error surfaced
    # only once every queued head had been learned. With it, 5 of the
    # other 19 heads are learned; without it, all 19.
    values = [f"{i:02}" for i in range(20)]
    dataset = Dataset(
        pd.DataFrame({"a": values, "y": values}),
        features=["a"],
        targets=["y"],
        max_categories=20,
    )
    learn = partial(_fail_first_head, directory=tmp_path)
    with pytest.raises(RuntimeError, match="the first head failed"):
        Algorithm._fit_heads(dataset, learn, n_jobs=2)
    assert len(list(tmp_path.iterdir())) < 10


def _die(_dataset, _target, _value, _verbose):
    """Ends its worker abruptly, as running out of memory can."""
    os._exit(1)


def test_a_dead_worker_names_the_likely_causes(toy_dataset):
    with pytest.raises(BrokenProcessPool, match="runs out of memory") as raised:
        Algorithm._fit_heads(toy_dataset, _die, n_jobs=2)
    assert isinstance(raised.value.__cause__, BrokenProcessPool)


@pytest.mark.parametrize("error", [OSError, EOFError])
def test_a_pool_that_fails_on_submit_names_the_likely_causes(
    monkeypatch, toy_dataset, error
):
    # What a `forkserver` whose script import failed does to `submit`;
    # the real case runs in
    # test_a_script_needs_the_main_guard_under_spawn below, in a child
    # process coverage does not trace.
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(ProcessPoolExecutor, "submit", fail)
    with pytest.raises(BrokenProcessPool, match="runs out of memory") as raised:
        Algorithm._fit_heads(toy_dataset, _die, n_jobs=2)
    assert isinstance(raised.value.__cause__, error)


@pytest.mark.parametrize(("method", "named"), [("spawn", True), ("fork", False)])
def test_the_main_guard_is_named_only_where_workers_import_the_script(
    monkeypatch, method, named
):
    monkeypatch.setattr(multiprocessing, "get_start_method", lambda: method)
    assert ("__main__" in _broken_pool_message()) is named


@pytest.mark.parametrize(("method", "kept"), [("fork", True), ("spawn", False)])
def test_a_parallel_fit_keeps_the_row_caches_only_under_fork(
    monkeypatch, two_target_dataset, method, kept
):
    # Only the method this process reports is patched: the pool starts
    # its workers as it always does. The uncovered-row warning would
    # count every head's positives here too.
    monkeypatch.setattr(multiprocessing, "get_start_method", lambda: method)
    GULA.fit(two_target_dataset, n_jobs=2, warn_uncovered=False)
    heads = {
        (target, value)
        for target in two_target_dataset.targets
        for value in two_target_dataset.domain(target)
    }
    assert (set(two_target_dataset._positives_cache) == heads) is kept
    assert (set(two_target_dataset._negatives_cache) == heads) is kept
    assert (set(two_target_dataset._distinct_negatives_cache) == heads) is kept


FIT_FRAME = {
    "a": ["0", "1", "0", "1"],
    "b": ["0", "0", "1", "1"],
    "y": ["0", "1", "1", "1"],
}

FIT_SCRIPT = f"""\
import multiprocessing

import pandas as pd

from pylfit import GULA, Dataset


def main():
    df = pd.DataFrame({FIT_FRAME!r})
    dataset = Dataset(df, features=["a", "b"], targets=["y"])
    for rule in sorted(r.to_string() for r in GULA.fit(dataset, n_jobs=2)):
        print(rule)


multiprocessing.set_start_method(START_METHOD, force=True)
"""


@pytest.mark.slow
@pytest.mark.parametrize(
    "method",
    [
        "spawn",
        pytest.param(
            "forkserver",
            marks=pytest.mark.skipif(
                "forkserver" not in multiprocessing.get_all_start_methods(),
                reason="no forkserver on this platform",
            ),
        ),
    ],
)
@pytest.mark.parametrize("guarded", [True, False])
def test_a_script_needs_the_main_guard_under_spawn(tmp_path, method, guarded):
    # What the ``n_jobs`` docs promise for macOS, Windows (``spawn``)
    # and Linux from Python 3.14 (``forkserver``): a worker imports the
    # script again, so the fit works only from under the guard.
    # ``forkserver`` fails earlier, while starting a worker, and must
    # say the same.
    call = 'if __name__ == "__main__":\n    main()\n' if guarded else "main()\n"
    script = tmp_path / "fit.py"
    script.write_text(FIT_SCRIPT.replace("START_METHOD", repr(method)) + call)
    run = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120
    )
    if guarded:
        assert run.returncode == 0, run.stderr
        dataset = Dataset(pd.DataFrame(FIT_FRAME), features=["a", "b"], targets=["y"])
        expected = sorted(r.to_string() for r in GULA.fit(dataset))
        assert run.stdout.splitlines() == expected
    else:
        assert run.returncode != 0
        assert "BrokenProcessPool" in run.stderr
        assert 'must call fit under `if __name__ == "__main__":`' in run.stderr
