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


# --- Progress ---


def test_progress_throttles_but_always_reports_the_last_item(capsys, monkeypatch):
    _clock(monkeypatch, 0.0, 0.1, 0.2, 1.5, 1.6, 1.7)
    progress = Progress("GULA", "stage", 5)  # starts at 0.0
    progress.update(1)  # 0.1: too soon
    progress.update(2)  # 0.2: too soon
    progress.update(3)  # 1.5: due
    progress.update(4)  # 1.6: too soon after 1.5
    progress.update(5)  # 1.7: last item, always
    lines = capsys.readouterr().err.splitlines()
    assert [line.split(":")[1].split()[0] for line in lines] == ["3/5", "5/5"]
    assert lines[-1].startswith("[GULA]   stage: 5/5 (100%)")


def test_progress_reports_an_empty_stage_as_complete(capsys):
    Progress("GULA", "stage", 0).update(0)
    assert capsys.readouterr().err.startswith("[GULA]   stage: 0/0 (100%)")


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


def test_progress_survives_a_clock_that_has_not_moved(capsys, monkeypatch):
    _clock(monkeypatch, 0.0, 0.0)
    Progress("GULA", "stage", 1).update(1)  # the last item: always reported
    assert capsys.readouterr().err.endswith("elapsed 0.0s, ~? remaining\n")


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
