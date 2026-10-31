"""Progress lines on stderr for ``verbose=True`` runs.

Shared by the learning algorithms and ``WeightedRuleSetModel``: both
report long loops the same way, and ``pylfit.models`` and
``pylfit.algorithms`` stay independent of each other (a test checks that
neither imports the other), so the code lives here instead. Output goes
to stderr, one line per call, tagged with the reporting class's name.
Printing behind a ``verbose`` flag, rather than through ``logging``,
follows scikit-learn and the rest of the scientific Python stack:
``verbose=True`` shows output without any logging setup.
"""

import sys
import time
from datetime import timedelta
from typing import ClassVar


def log(tag: str, message: str, indent: bool = False) -> None:
    """Prints a ``[tag] message`` line to stderr.

    Args:
        tag: The name of the reporting class, shown in brackets.
        message: The text to print, after the tag.
        indent: If True, adds extra indentation for lines nested under a
            named stage.
    """
    prefix = f"[{tag}]   " if indent else f"[{tag}] "
    print(prefix + message, file=sys.stderr, flush=True)


#: The longest duration ``duration`` writes out, in seconds.
_CENTURY = 100 * 365.25 * 24 * 3600


def duration(seconds: float, digits: int = 2) -> str:
    """Formats a duration: ``12.30s`` under a minute, ``0:02:03`` from one.

    Args:
        seconds: The duration, in seconds.
        digits: Decimal places shown under a minute.
            Defaults to 2.

    Returns:
        The duration as text. From a minute up it is ``timedelta``'s
        text, rounded to the second, so a long run reads ``2 days,
        7:33:18`` rather than ``199998s``. A negative duration reads as
        zero, and one over a century, or infinite, as ``100+
        years``: ``timedelta`` can't hold the estimate of a fit that
        would never finish.
    """
    seconds = max(seconds, 0.0)
    if round(seconds, digits) < 60:
        return f"{seconds:.{digits}f}s"
    if not seconds < _CENTURY:  # also catches inf and NaN
        return "100+ years"
    return str(timedelta(seconds=round(seconds)))


class Progress:
    """Throttled progress for one stage of work: count, elapsed, time left.

    Create one before a loop over ``total`` items and call ``update``
    after each item. A line is printed at most once per
    ``INTERVAL_SECONDS``, and always for the last item. A stage of no
    items counts as complete: ``0/0 (100%)``.
    """

    #: Minimum time between two progress lines, in seconds. Throttled by
    #: time rather than item count, so the rate suits fast and slow
    #: loops alike without tuning.
    INTERVAL_SECONDS: ClassVar[float] = 1.0

    def __init__(self, tag: str, stage: str, total: int) -> None:
        """Starts the stage's clock.

        Args:
            tag: The name of the reporting class, shown in brackets.
            stage: What is being counted, shown before the count.
            total: The number of items in the stage.
        """
        self.tag = tag
        self.stage = stage
        self.total = total
        self.start = time.perf_counter()
        self._last_report = self.start

    def update(self, current: int) -> None:
        """Reports that ``current`` items are done, if a line is due.

        Args:
            current: Items processed so far, counting from 1.
        """
        now = time.perf_counter()
        if current != self.total and now - self._last_report < self.INTERVAL_SECONDS:
            return
        elapsed = now - self.start
        rate = current / elapsed if elapsed > 0 else 0.0
        if rate > 0:
            try:
                left = (self.total - current) / rate
            except OverflowError:  # a total beyond any float: BruteForce, 700 features
                left = float("inf")
            eta = duration(left, 0)
        else:
            eta = "?"
        fraction = current / self.total if self.total else 1.0
        log(
            self.tag,
            f"{self.stage}: {current}/{self.total} ({fraction:.0%}), "
            f"elapsed {duration(elapsed, 1)}, ~{eta} remaining",
            indent=True,
        )
        self._last_report = now
