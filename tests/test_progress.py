"""Tests for ``pylfit._progress``: tagged, throttled lines on stderr."""

import io
import sys

import pytest

from pylfit import _progress
from pylfit._progress import Progress, duration, log


def _clock(monkeypatch: pytest.MonkeyPatch, *times: float) -> None:
    """Makes ``time.perf_counter`` return ``times``, one per call."""
    ticks = iter(times)
    monkeypatch.setattr(_progress.time, "perf_counter", lambda: next(ticks))


# --- log ---


def test_log_tags_the_line_and_writes_to_stderr(capsys):
    log("GULA", "done")
    log("GULA", "step", indent=True)
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "[GULA] done\n[GULA]   step\n"


def test_log_flushes_each_line(monkeypatch):
    # A stream that buffers, as a notebook's does, shows each line as
    # it is logged.
    class Stream(io.StringIO):
        flushed = ""

        def flush(self):
            self.flushed = self.getvalue()

    stream = Stream()
    monkeypatch.setattr(sys, "stderr", stream)
    log("GULA", "done")
    assert stream.flushed == "[GULA] done\n"


def test_log_prints_nothing_without_stderr(capsys, monkeypatch):
    # Python sets `sys.stderr` to None when fd 2 is closed at startup
    # (`2>&-`); `print` would then write to stdout.
    monkeypatch.setattr(sys, "stderr", None)
    log("GULA", "done")
    assert capsys.readouterr().out == ""


# --- Progress ---


def test_progress_throttles_and_ends_a_shown_stage_at_100(capsys, monkeypatch):
    _clock(monkeypatch, 0.0, 0.1, 0.2, 1.5, 1.6, 1.7)
    progress = Progress("GULA", "stage", 5)  # starts at 0.0
    progress.update(1)  # 0.1: too soon
    progress.update(2)  # 0.2: too soon
    progress.update(3)  # 1.5: due
    progress.update(4)  # 1.6: too soon after 1.5
    progress.update(5)  # 1.7: last item, and a line was printed
    lines = capsys.readouterr().err.splitlines()
    assert [line.split(":")[1].split()[0] for line in lines] == ["3/5", "5/5"]
    assert lines[-1].startswith("[GULA]   stage: 5/5 (100%)")


def test_progress_prints_at_most_once_a_second(capsys, monkeypatch):
    # The interval itself: 0.9s after a line is too soon, 1.0s is due.
    _clock(monkeypatch, 0.0, 1.0, 1.9, 2.0)
    progress = Progress("GULA", "stage", 10)
    progress.update(1)  # 1.0: due
    progress.update(2)  # 1.9: too soon
    progress.update(3)  # 2.0: due
    lines = capsys.readouterr().err.splitlines()
    assert [line.split(":")[1].split()[0] for line in lines] == ["1/10", "3/10"]


def test_progress_estimates_from_the_rate_since_the_previous_line(capsys, monkeypatch):
    # Later items are slower, as GULA's negatives are: 5 items in the
    # first second, 1 in the next two. The mean rate since the start
    # (2 a second) would say ~2s for the 4 left; the recent one (0.5 a
    # second) says ~8s.
    _clock(monkeypatch, 0.0, 1.0, 3.0)
    progress = Progress("GULA", "stage", 10)
    progress.update(5)  # 1.0: 5 a second, 5 left
    progress.update(6)  # 3.0: 1 in 2 seconds, 4 left
    lines = capsys.readouterr().err.splitlines()
    assert lines[0].endswith("~1s remaining")
    assert lines[1].endswith("elapsed 3.0s, ~8s remaining")


def test_progress_shows_an_unknown_eta_before_any_item_is_done(capsys, monkeypatch):
    _clock(monkeypatch, 0.0, 2.0)
    Progress("PRIDE", "stage", 5).update(0)
    assert capsys.readouterr().err.endswith("~? remaining\n")


def test_progress_shows_a_minute_or_more_as_hours_minutes_seconds(capsys, monkeypatch):
    _clock(monkeypatch, 0.0, 123.4)
    Progress("GULA", "stage", 100).update(1)  # 99 left at 123.4s each
    assert capsys.readouterr().err.endswith("elapsed 0:02:03, ~3:23:37 remaining\n")


def test_progress_shows_seconds_under_a_minute(capsys, monkeypatch):
    _clock(monkeypatch, 0.0, 12.3)
    Progress("GULA", "stage", 5).update(1)  # 4 left at 12.3s each
    assert capsys.readouterr().err.endswith("elapsed 12.3s, ~49s remaining\n")


def test_progress_a_quick_stage_prints_nothing(capsys, monkeypatch):
    # The caller's summary line follows the loop with the same count
    # and the time; a 100% line alone would only repeat it.
    _clock(monkeypatch, 0.0, 0.1, 0.2, 0.3, 2.3)
    progress = Progress("GULA", "stage", 2)
    progress.update(1)  # 0.1: too soon
    progress.update(2)  # 0.2: last item, no line printed before
    assert capsys.readouterr().err == ""
    slow = Progress("GULA", "stage", 1)  # starts at 0.3
    slow.update(1)  # 2.3: due by time, but the last item
    assert capsys.readouterr().err == ""


def test_progress_prints_an_estimate_beyond_a_century(capsys, monkeypatch):
    # BruteForce on about 40 binary features: the estimate overflowed
    # timedelta at the first line, before any was printed.
    _clock(monkeypatch, 0.0, 1.0)
    Progress("BruteForce", "stage", 10**30).update(1)
    assert "~100+ years remaining" in capsys.readouterr().err


def test_progress_prints_an_estimate_beyond_any_float(capsys, monkeypatch):
    # BruteForce on 700 binary features: 3**700 candidates, too many to
    # divide as floats.
    _clock(monkeypatch, 0.0, 1.0)
    Progress("BruteForce", "stage", 3**700).update(1)
    assert "~100+ years remaining" in capsys.readouterr().err


def test_progress_prints_a_total_beyond_python_s_digit_limit(capsys, monkeypatch):
    # BruteForce on 9,100 binary features: 3**9100 candidates, more
    # digits than `str(int)` accepts (4300).
    _clock(monkeypatch, 0.0, 1.0)
    Progress("BruteForce", "stage", 3**9100).update(1)
    err = capsys.readouterr().err
    assert "stage: 1/6.4e+4341 (0%)" in err
    assert "~100+ years remaining" in err


def test_progress_writes_counts_in_full_up_to_fifteen_digits(capsys, monkeypatch):
    _clock(monkeypatch, 0.0, 1.0)
    Progress("BruteForce", "stage", 10**15 - 1).update(1)
    assert f"stage: 1/{10**15 - 1} (0%)" in capsys.readouterr().err


# --- duration ---


def test_duration_keeps_seconds_under_a_minute_and_h_mm_ss_from_one():
    assert duration(0.0123) == "0.01s"
    assert duration(59.4, 1) == "59.4s"
    assert duration(60) == "0:01:00"
    assert duration(199_998.4) == "2 days, 7:33:18"


def test_duration_rounding_up_to_a_minute_reads_as_one():
    assert duration(59.996) == "0:01:00"
    assert duration(59.6, 0) == "0:01:00"


def test_duration_reads_a_negative_estimate_as_zero():
    assert duration(-3.2, 0) == "0s"


def test_duration_writes_out_up_to_a_century():
    year = 365 * 24 * 3600
    assert duration(99 * year) == "36135 days, 0:00:00"
    assert duration(101 * year) == "100+ years"


@pytest.mark.parametrize("seconds", [1e15, float("inf"), float("nan")])
def test_duration_caps_what_timedelta_cannot_hold(seconds):
    assert duration(seconds, 0) == "100+ years"
