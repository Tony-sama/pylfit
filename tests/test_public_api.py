"""The public API's shape: one import path per name, private modules.

Users import from ``pylfit`` (or the ``pylfit.models`` and
``pylfit.algorithms`` subpackages); every module defining code is
``_``-prefixed, so it can be renamed or split without breaking anyone.
"""

import pkgutil
from importlib import metadata
from pathlib import Path

import pytest

import pylfit
import pylfit.algorithms
import pylfit.models

PUBLIC_PACKAGES = {"pylfit", "pylfit.models", "pylfit.algorithms"}


def test_every_module_outside_the_public_packages_is_private():
    modules = {m.name for m in pkgutil.walk_packages(pylfit.__path__, "pylfit.")}
    public = {
        name
        for name in modules - PUBLIC_PACKAGES
        if not name.rsplit(".", 1)[1].startswith("_")
    }
    assert public == set()


@pytest.mark.parametrize("package", [pylfit, pylfit.models, pylfit.algorithms])
def test_every_public_attribute_is_in_all(package):
    # A helper imported without an underscore would be importable, and
    # so part of the API, without anyone deciding it.
    public = {name for name in vars(package) if not name.startswith("_")}
    subpackages = {name for name in public if name in {"models", "algorithms"}}
    assert public - subpackages == set(package.__all__) - {"__version__"}
    # Once each: a name listed twice would pass the comparison above.
    assert len(package.__all__) == len(set(package.__all__))


@pytest.mark.parametrize("package", [pylfit.models, pylfit.algorithms])
def test_subpackage_names_are_also_at_the_top_level(package):
    for name in package.__all__:
        assert getattr(pylfit, name) is getattr(package, name)
    assert set(package.__all__) <= set(pylfit.__all__)


@pytest.mark.parametrize("package", [pylfit, pylfit.models, pylfit.algorithms])
def test_the_package_docstring_names_every_public_name(package):
    assert package.__doc__ is not None
    missing = [name for name in package.__all__ if f"``{name}``" not in package.__doc__]
    assert missing == []


def test_version_is_the_installed_distribution_version():
    assert pylfit.__version__ == metadata.version("pylfit")


def test_the_package_is_marked_typed():
    # PEP 561: without py.typed, type checkers ignore pylfit's hints,
    # whatever the Typing :: Typed classifier says.
    assert (Path(pylfit.__file__).parent / "py.typed").is_file()
