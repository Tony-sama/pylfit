"""The ``.githooks/commit-msg`` hook, run as git runs it.

The hook is POSIX shell, so these run it under ``sh`` on a message
file and read the exit code and stderr. Skipped where no ``sh`` is on
``PATH``.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".githooks" / "commit-msg"
SH = shutil.which("sh")

pytestmark = pytest.mark.skipif(SH is None, reason="no sh on PATH")

#: The line ``git commit -v`` writes above the diff it appends.
SCISSORS = "# ------------------------ >8 ------------------------"


def run_hook(tmp_path, message):
    """Runs the hook on ``message``; returns its exit code and stderr."""
    message_file = tmp_path / "COMMIT_EDITMSG"
    message_file.write_bytes(message.encode("utf-8"))
    result = subprocess.run(
        [SH, str(HOOK), str(message_file)],
        capture_output=True,
        check=False,
    )
    return result.returncode, result.stderr.decode("utf-8")


@pytest.mark.parametrize(("length", "code"), [(72, 0), (73, 1)])
def test_subject_cap(tmp_path, length, code):
    """A subject of 72 characters passes silently; 73 is rejected."""
    returncode, stderr = run_hook(tmp_path, "a" * length + "\n")
    assert returncode == code
    assert ("over the 72 cap" in stderr) == bool(code)


def test_subject_counts_characters_not_bytes(tmp_path):
    """60 accented characters are 120 bytes, and still pass."""
    returncode, stderr = run_hook(tmp_path, "é" * 60 + "\n")
    assert returncode == 0, stderr


def test_comment_lines_are_not_the_subject(tmp_path):
    """A comment over the cap, above the subject, is left out."""
    message = "# " + "x" * 80 + "\nshort subject\n"
    assert run_hook(tmp_path, message) == (0, "")


def test_the_subject_is_the_first_line_not_blank(tmp_path):
    """Blank lines above a subject over the cap don't hide it."""
    returncode, stderr = run_hook(tmp_path, "\n\n" + "a" * 73 + "\n")
    assert returncode == 1
    assert "over the 72 cap" in stderr


@pytest.mark.parametrize(
    "prefix", ["Merge ", "Revert ", "fixup! ", "squash! ", "amend! "]
)
def test_generated_subjects_are_exempt(tmp_path, prefix):
    """A subject git writes passes, whatever its length."""
    assert run_hook(tmp_path, prefix + "b" * 80 + "\n")[0] == 0


@pytest.mark.parametrize(
    "pointer",
    [
        "see PLAN.md for why",
        "Recorded in DECISIONS.md, with the numbers",
        "the numbers in RATIONALE.md section 4",
    ],
)
def test_planning_note_pointer_warns(tmp_path, pointer):
    """Sending a reader into a planning note warns, without rejecting."""
    returncode, stderr = run_hook(tmp_path, f"docs: x\n\n{pointer}\n")
    assert returncode == 0
    assert pointer in stderr


@pytest.mark.parametrize(
    "line",
    [
        "shrink PLAN.md's breakdown",  # no cue
        "the numbers living in PLAN.md moved",  # "in" with no section after
        "as you see it, PLAN.md is shorter",  # the cue in another clause
    ],
)
def test_naming_a_planning_note_is_quiet(tmp_path, line):
    """Describing a change to a planning note is not a pointer."""
    assert run_hook(tmp_path, f"docs: x\n\n{line}\n") == (0, "")


def test_diff_below_scissors_is_ignored(tmp_path):
    """``git commit -v``'s diff is not part of the message."""
    message = f"short subject\n{SCISSORS}\n+notes: see PLAN.md for the list\n"
    assert run_hook(tmp_path, message) == (0, "")
