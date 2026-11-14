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

import math
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
    if sys.stderr is None:
        # No stderr (fd 2 closed at startup, or pythonw.exe): drop the
        # line, as ``warnings`` does, since ``print`` would fall back to
        # stdout.
        return
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


def _count(n: int) -> str:
    """Formats an item count: in full up to 15 digits, ``3.4e+4342`` beyond.

    Python refuses to write out an ``int`` of more than 4300 digits, and
    ``BruteForce`` on 9,100 binary features has ``3**9100`` candidates;
    ``math.log10`` takes an ``int`` of any size.

    Args:
        n: The count.

    Returns:
        The count as text.
    """
    if n < 10**15:
        return str(n)
    exponent = math.log10(n)
    whole = math.floor(exponent)
    return f"{10 ** (exponent - whole):.1f}e+{whole}"


class Progress:
    """Throttled progress for one stage of work: count, elapsed, time left.

    Create one before a loop over ``total`` items and call ``update``
    after each item. A line is printed at most once per
    ``INTERVAL_SECONDS``. The last item gets a line only if the stage
    already printed one, so a stage whose progress was shown ends at
    100%, and a quick stage prints nothing: each caller prints its own
    summary line after the loop, with the count and the time.

    The time left is estimated from the rate since the previous line,
    not since the start: an item often costs more than the one before
    it (each negative ``GULA`` processes grows the rule set the next is
    checked against), and the mean rate since the start lags that,
    while the recent rate follows it. The estimate is still a lower
    bound when items keep getting slower, and it covers this stage only.
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
        self._printed = False
        self._last_count = 0

    def update(self, current: int) -> None:
        """Reports that ``current`` items are done, if a line is due.

        Args:
            current: Items processed so far, counting from 1.
        """
        now = time.perf_counter()
        if current == self.total:
            if not self._printed:
                return
        elif now - self._last_report < self.INTERVAL_SECONDS:
            return
        elapsed = now - self.start
        since = now - self._last_report
        rate = (current - self._last_count) / since if since > 0 else 0.0
        if rate > 0:
            try:
                left = (self.total - current) / rate
            except OverflowError:  # a total beyond any float: BruteForce, 700 features
                left = float("inf")
            eta = duration(left, 0)
        else:
            eta = "?"
        fraction = current / self.total
        log(
            self.tag,
            f"{self.stage}: {_count(current)}/{_count(self.total)} ({fraction:.0%}), "
            f"elapsed {duration(elapsed, 1)}, ~{eta} remaining",
            indent=True,
        )
        self._last_report = now
        self._last_count = current
        self._printed = True
