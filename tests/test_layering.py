"""``pylfit.models`` and ``pylfit.algorithms`` don't import each other.

The two packages stay independent: an algorithm returns rules, not a
model, and a model is built from rules wherever they came from. Code
both need, such as the progress lines, goes in a private top-level
module instead, which imports neither.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "src" / "pylfit"


def _imported_modules(source: str, package: str) -> set[str]:
    """Every module ``source`` imports, by its full dotted name.

    A relative import is resolved against ``package``, the dotted name
    of the file's package. ``from x import y`` counts ``x.y`` too, as
    ``y`` may be a module.
    """
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                parts = parts[: len(parts) - node.level + 1]
                base = ".".join([*parts, base] if base else parts)
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import pylfit.algorithms", {"pylfit.algorithms"}),
        (
            "from ..algorithms import GULA",
            {"pylfit.algorithms", "pylfit.algorithms.GULA"},
        ),
        ("from .. import algorithms", {"pylfit", "pylfit.algorithms"}),
        (
            "from ._helpers import x",
            {"pylfit.models._helpers", "pylfit.models._helpers.x"},
        ),
        ("def f():\n    from pylfit import GULA", {"pylfit", "pylfit.GULA"}),
    ],
)
def test_imports_are_read_by_their_full_name(source, expected):
    assert _imported_modules(source, "pylfit.models") == expected


def _offenders(paths: list[Path], others: list[str]) -> dict[str, list[str]]:
    """Each file's imports of ``pylfit`` or of the packages ``others``.

    ``pylfit`` itself counts: its ``__init__`` imports both packages.
    """
    forbidden = [f"pylfit.{other}" for other in others]
    offenders = {}
    for path in paths:
        package = ".".join(path.relative_to(PACKAGE.parent).parent.parts)
        found = sorted(
            name
            for name in _imported_modules(path.read_text(encoding="utf-8"), package)
            if name == "pylfit"
            or any(name == f or name.startswith(f + ".") for f in forbidden)
        )
        if found:
            offenders[str(path.relative_to(PACKAGE))] = found
    return offenders


@pytest.mark.parametrize(
    ("package", "other"), [("models", "algorithms"), ("algorithms", "models")]
)
def test_models_and_algorithms_do_not_import_each_other(package, other):
    paths = sorted((PACKAGE / package).rglob("*.py"))
    assert paths
    assert _offenders(paths, [other]) == {}


def test_the_shared_modules_import_neither_package():
    paths = sorted(PACKAGE.glob("_[!_]*.py"))
    assert paths
    assert _offenders(paths, ["models", "algorithms"]) == {}
