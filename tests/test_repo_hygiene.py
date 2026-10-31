"""Repo-hygiene guards: conventions no other test would catch.

**The rule: don't cite what the reader won't have.** It has three
forms here, differing only in who the stranded reader is.

A pointer is also fragile in a way prose isn't: things get reorganized
(sections renamed, content moved, directories removed), and a pointer
written against the old layout rots silently because nothing resolves
it.

Scoped to references, not to the words themselves: a docstring is free
to discuss a decision, or to say what a benchmark measured, it just
can't tell the reader which file to go read about it. Put the substance
inline instead.

The three forms:

- **``AGENT_DOCS``**: the maintainers' planning notes, which are kept
  outside this repo. A planning-note filename has no legitimate use
  anywhere in this repo, so there is nothing to scope around.
- **``DEPARTED_DIRS``**: directories removed from this repo. The
  reference is dangling for every reader, not merely an installing
  one. Searching for the property (any path under a removed
  directory) rather than for one known string is what makes it catch
  every stale reference at once.
- **``UNSHIPPED_DIRS``**: directories that are here but are not inside
  the wheel. Checked under ``src/`` only: a *test* may freely name
  ``tests/data/``, but a shipped docstring naming it strands anyone
  reading from an install or from the generated doc site.

The first two are checked in **every tracked file**, not just ``.py``
ones: a config file or a Markdown page can carry a pointer as easily
as a docstring. ``_RULE_EXEMPT`` names the three files that describe
or test the rule rather than break it.

Three conventions outside that rule sit here as well, because they
need the same file lists: a DOI is written as a full
``https://doi.org/`` link (``OLD_DOI_FORMS``), a ``References:``
entry is a ``.. [Label]`` citation its file cites, and every file
keeps the whitespace ``.editorconfig`` asks for.
"""

import subprocess
from pathlib import Path

import pytest

#: Repo root: ``tests/`` sits directly under it.
REPO_ROOT = Path(__file__).resolve().parent.parent

#: The only directory that ends up inside the wheel. Hatchling
#: auto-detects it from the ``src/`` layout, so there is no explicit
#: include list to cross-check this against; if packaging ever gains
#: one, they must agree.
SHIPPED_DIR = "src"

#: Directories removed from this repo. Naming one is dangling for every
#: reader, so this is checked in every tracked file. Add to this list
#: when a directory is removed; that is cheaper than rediscovering the
#: references one at a time.
DEPARTED_DIRS = ("benchmarks/",)

#: Directories that exist here but are not packaged. A docstring under
#: ``SHIPPED_DIR`` naming one strands anyone reading from an install.
#: Not checked outside it: a test naming ``tests/data/`` is fine, since
#: anyone running it has the file.
UNSHIPPED_DIRS = ("tests/", "examples/", "docs/")

#: The planning notes, by filename. Matching on the bare name (not on a
#: directory prefix) is deliberate: a pointer usually names the file
#: with no path at all, and the notes have no path inside this repo to
#: match against. ``.githooks/commit-msg`` keeps its own copy of this
#: list; ``test_commit_hook_lists_the_same_docs`` keeps the two equal.
AGENT_DOCS = (
    "PLAN.md",
    "DECISIONS.md",
    "RATIONALE.md",
    "HANDOFF.md",
    "IDEAS.md",
    "TEACHING.md",
)

#: The three files that name a design doc or a departed directory in
#: order to *describe* or *test* the rule forbidding it. Exempting them
#: by path is deliberate: any other mechanism (a marker comment, a
#: heuristic on surrounding words) would be one a real violation could
#: also wear.
_RULE_EXEMPT = (
    Path("tests") / "test_repo_hygiene.py",
    Path(".githooks") / "commit-msg",
    Path("tests") / "test_commit_msg_hook.py",
)


def _tracked_files(exempt=_RULE_EXEMPT):
    """Every file ``git`` tracks here, exemptions and binaries removed.

    Tracked-ness is the right selector for a rule about what a reader
    will have: it is exactly what a clone gets, and it excludes the
    generated ``docs/_build/`` without this guard having to name it.

    Skipped rather than failed outside a checkout. A source export
    (``git archive``, a downloaded zip) has the files but no index, and
    a convention guard has no business failing a tree it cannot
    enumerate.

    Args:
        exempt: Repo-relative paths to leave out.

    Yields:
        Absolute paths to decodable text files.
    """
    try:
        listing = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout, so tracked files can't be listed")

    for name in listing.stdout.split("\0"):
        if not name:
            continue
        relative = Path(name)
        if relative in exempt:
            continue
        path = REPO_ROOT / relative
        try:
            path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # binary, or vanished between listing and read
        yield path


def _shipped_python_files():
    """Every ``.py`` file under ``SHIPPED_DIR``, sorted.

    Returns:
        A list of absolute paths.
    """
    return sorted((REPO_ROOT / SHIPPED_DIR).rglob("*.py"))


def _references(needle, files):
    """Every ``file:line: text`` in ``files`` containing ``needle``.

    Args:
        needle: Literal substring to search for.
        files: Iterable of paths to read.

    Returns:
        A list of formatted ``path:number: line`` strings, empty when
        nothing matches.
    """
    offenders = []
    for path in files:
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if needle in line:
                relative = path.relative_to(REPO_ROOT)
                offenders.append(f"{relative}:{number}: {line.strip()}")
    return offenders


def test_guard_sees_the_files():
    tracked = set(_tracked_files())
    init = REPO_ROOT / SHIPPED_DIR / "pylfit" / "__init__.py"
    assert REPO_ROOT / "pyproject.toml" in tracked
    assert REPO_ROOT / "README.md" in tracked
    assert init in tracked
    assert Path(__file__).resolve() not in tracked
    assert init in _shipped_python_files()


@pytest.mark.parametrize("doc", AGENT_DOCS)
def test_no_agent_doc_references_anywhere(doc):
    """No tracked file names a design doc.

    Args:
        doc: Filename of the design doc to search for.
    """
    offenders = _references(doc, _tracked_files())

    assert not offenders, (
        f"{len(offenders)} reference(s) to {doc}. That file is not part "
        f"of this repo or the package, and section "
        f"names move when the docs are reorganized. State the substance "
        f"inline, or point at the code that carries it:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize("prefix", DEPARTED_DIRS)
def test_no_departed_path_references_anywhere(prefix):
    """No tracked file names a directory removed from this repo.

    Args:
        prefix: Path prefix of the departed directory, with a trailing
            slash so it cannot match a bare word.
    """
    offenders = _references(prefix, _tracked_files())

    assert not offenders, (
        f"{len(offenders)} reference(s) to `{prefix}`. That "
        f"directory has been removed, so the path resolves for "
        f"nobody, not even someone with this repo fully cloned. Say "
        f"what it does inline instead of pointing at it:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize("prefix", UNSHIPPED_DIRS)
def test_no_unshipped_path_references_in_package(prefix):
    """No shipped file names a directory absent from the wheel.

    Args:
        prefix: Path prefix of the unshipped directory, with a trailing
            slash so it cannot match a bare word.
    """
    offenders = _references(prefix, _shipped_python_files())

    assert not offenders, (
        f"{len(offenders)} reference(s) to `{prefix}` under "
        f"`{SHIPPED_DIR}/`. Only `{SHIPPED_DIR}/` goes into the wheel, "
        f"so that path is dangling for anyone reading this from an "
        f"install or from the generated doc site. Tests and examples "
        f"may name it freely; shipped code may not:\n" + "\n".join(offenders)
    )


#: DOI forms the Crossref display guidelines (March 2017) retire: a DOI
#: is shown as a full ``https://doi.org/10.xxxx/xxxxx`` link, never
#: after ``doi:``, never through ``dx.doi.org`` and never over plain
#: ``http``. ``CITATION.cff``'s ``doi: 10.`` key, with its space, is
#: the format's own and passes.
OLD_DOI_FORMS = ("doi:10.", "DOI:10.", "dx.doi.org", "http://doi.org")


@pytest.mark.parametrize("form", OLD_DOI_FORMS)
def test_dois_shown_as_full_links(form):
    """No tracked file writes a DOI in a retired form.

    Args:
        form: The retired form to search for.
    """
    offenders = _references(form, _tracked_files())

    assert not offenders, (
        f"{len(offenders)} DOI(s) written as `{form}`. Write each as a "
        f"full https://doi.org/10.xxxx/xxxxx link, the form Crossref's "
        f"display guidelines ask for:\n" + "\n".join(offenders)
    )


def _reference_problems(text):
    """What is wrong with the ``References:`` entries in a file's text.

    An entry is the first line of each item one indent level under a
    ``References:`` line; deeper lines continue it, and the first line
    back at a shallower level ends the section.

    Args:
        text: A source file's text.

    Returns:
        One message per entry that is not a ``.. [Label]`` citation,
        or whose label the text never cites as ``[Label]_``.
    """
    problems = []
    lines = text.splitlines()
    for start, line in enumerate(lines):
        if line.strip() != "References:":
            continue
        indent = None
        for entry in lines[start + 1 :]:
            if not entry.strip():
                continue
            depth = len(entry) - len(entry.lstrip())
            if indent is None:
                indent = depth
            if depth < indent:
                break
            if depth > indent:
                continue
            entry = entry.strip()
            if not entry.startswith(".. [") or "]" not in entry:
                problems.append(f"not a `.. [Label]` entry: {entry}")
                continue
            label = entry[4 : entry.index("]")]
            if f"[{label}]_" not in text:
                problems.append(f"[{label}] is never cited as [{label}]_")
    return problems


@pytest.mark.parametrize(
    ("entries", "problems"),
    [
        ("    .. [A] x.", []),
        ("    Smith, A. 2020. x.", ["not a `.. [Label]` entry: Smith, A. 2020. x."]),
        ("    Smith [A]. x.", ["not a `.. [Label]` entry: Smith [A]. x."]),
        ("    .. [A x.", ["not a `.. [Label]` entry: .. [A x."]),
        ("    .. [A] x.\n    .. [B] y.", ["[B] is never cited as [B]_"]),
        ("    .. [A] x.\n\n    .. [B] y.", ["[B] is never cited as [B]_"]),
        ("    .. [A] x,\n      [B] y.", []),  # a continuation line
        ("    .. [A] x.\nNote:\n    [B] y.", []),  # the next section
    ],
)
def test_the_guard_reads_reference_entries(entries, problems):
    assert _reference_problems(f"[A]_.\nReferences:\n{entries}\n") == problems


def test_docstring_references_are_cited_by_label():
    """Every ``References:`` entry is a citation its file cites.

    The docs render ``[Label]_`` as a link to ``.. [Label] ...``; a
    plain entry or an uncited label leaves an in-text cite unlinked.
    """
    problems = [
        f"{path.relative_to(REPO_ROOT)}: {problem}"
        for path in _shipped_python_files()
        for problem in _reference_problems(path.read_text(encoding="utf-8"))
    ]

    assert not problems, "\n".join(problems)


def test_commit_hook_lists_the_same_docs():
    """``.githooks/commit-msg`` searches for exactly ``AGENT_DOCS``.

    The hook is shell, so it can't import the tuple; it keeps a copy as
    a ``docs='...'`` regex alternation, and this keeps the copy from
    drifting.
    """
    hook = (REPO_ROOT / ".githooks" / "commit-msg").read_text(encoding="utf-8")
    lines = [line for line in hook.splitlines() if line.startswith("docs='")]
    assert len(lines) == 1, "expected exactly one docs='...' line in the hook"
    hook_docs = lines[0][len("docs='") : -1].replace("\\.", ".").split("|")
    assert sorted(hook_docs) == sorted(AGENT_DOCS)


#: The extensions ``.editorconfig`` indents with spaces. Keep the two
#: in step.
SPACE_INDENTED = (
    ".py",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".cff",
    ".js",
    ".css",
    ".svg",
)


def _whitespace_problems(name, text):
    """What ``.editorconfig``'s whitespace rules find wrong in a file.

    Args:
        name: The file's name, whose extension decides whether a tab
            may indent.
        text: The file's text.

    Returns:
        One message per line ending in spaces or tabs, per line
        indented by a tab where ``SPACE_INDENTED`` forbids it, and one
        for a missing final newline (an empty file needs none).
    """
    problems = []
    for number, line in enumerate(text.splitlines(), start=1):
        if line != line.rstrip(" \t"):
            problems.append(f"line {number} ends in whitespace")
        if name.endswith(SPACE_INDENTED) and line.startswith("\t"):
            problems.append(f"line {number} is indented by a tab")
    if text and not text.endswith("\n"):
        problems.append("no final newline")
    return problems


@pytest.mark.parametrize(
    ("name", "text", "problems"),
    [
        ("a.md", "x\n", []),
        ("a.md", "", []),
        ("a.md", "x", ["no final newline"]),
        ("a.md", "x \ny\n", ["line 1 ends in whitespace"]),
        ("a.md", "x\ny\t\n", ["line 2 ends in whitespace"]),
        ("a.json", "{\n\t1\n}\n", ["line 2 is indented by a tab"]),
        ("a.json", '{"a":\t1}\n', []),  # a tab inside a line indents nothing
        ("a.tsv", "\tx\n", []),  # tabs are fine outside SPACE_INDENTED
    ],
)
def test_the_guard_reads_whitespace(name, text, problems):
    assert _whitespace_problems(name, text) == problems


def test_whitespace_as_editorconfig_asks():
    """Every tracked text file keeps ``.editorconfig``'s whitespace.

    Ruff already holds the Python files to it; this covers the rest
    (Markdown, YAML, data), where an editor that follows the config
    would otherwise slip unrelated whitespace changes into the next
    edit. The rule-exempt files are checked too: their exemption is
    from naming the planning notes, not from this.
    """
    problems = [
        f"{path.relative_to(REPO_ROOT)}: {problem}"
        for path in _tracked_files(exempt=())
        for problem in _whitespace_problems(path.name, path.read_text(encoding="utf-8"))
    ]

    assert not problems, "\n".join(problems)
