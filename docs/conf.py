"""Sphinx configuration for the pylfit documentation site."""

import pkgutil
import sys
from datetime import date
from importlib.metadata import version as _installed_version
from pathlib import Path

from sphinx.pycode import ModuleAnalyzer

# Document this checkout's source rather than whatever pylfit happens to
# be installed. The package still has to be importable with its
# dependencies, so the build needs `pip install -e ".[docs]"`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# And this folder, for `_gallery_order`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

project = "pylfit"
# From the installed metadata, so `pyproject.toml` stays the one place
# the version is written.
release = _installed_version("pylfit")
version = release
html_title = f"pylfit {release}"
# 2019 is the public repository's first commit; "the pylfit authors"
# are the people `AUTHORS.md` lists.
copyright = f"2019–{date.today().year}, the pylfit authors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
]


def _defining_module_docs(app, what, name, obj, options, lines):
    """Gives a constant ``pylfit`` re-exports its own ``#:`` comment.

    The API pages document constants under ``pylfit``, where users import
    them, so autodoc looks for their ``#:`` comments in
    ``pylfit/__init__.py``, finds none, and falls back to the docstring
    of the value's type (``tuple``'s, ``str``'s). The comments are in the
    private module defining each constant; read them from there.
    """
    if what != "data" or not name.startswith("pylfit."):
        return
    attr = name.removeprefix("pylfit.")
    import pylfit  # importable only once `src` is on `sys.path` above

    for module in pkgutil.walk_packages(pylfit.__path__, "pylfit."):
        docs = ModuleAnalyzer.for_module(module.name).find_attr_docs()
        if ("", attr) in docs:
            lines[:] = list(docs[("", attr)])
            return


def setup(app):
    app.connect("autodoc-process-docstring", _defining_module_docs)


# Runs every script in `examples/` at build time and renders its code
# and output, so a broken example fails the build. `_bnet.py` is a
# helper the Boolean-network scripts import, not an example. Nothing
# plots, so no image scraper (and no matplotlib dependency).
# `examples/data/` holds a README, which sphinx-gallery would read as a
# sub-gallery (any `README.<ext>` in `source_suffix` counts, and MyST
# adds `.md`); it holds no example, so the gallery has no subsections.
# If a sphinx-gallery release renames this function, the folder
# reappears and `-W` fails the build.
import sphinx_gallery.gen_gallery as _gallery  # noqa: E402

_gallery.get_subsections = lambda *args, **kwargs: []

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/[^_][^/]*\.py$",
    "ignore_pattern": r"^_",
    "within_subsection_order": "_gallery_order.GalleryOrder",
    "image_scrapers": (),
    "reset_modules": (),
    "download_all_examples": False,
    "remove_config_comments": True,
}

exclude_patterns = ["_build", "auto_examples/*.ipynb", "auto_examples/*.py"]

# Links `pandas.DataFrame` and the standard library in signatures. The
# inventories are fetched over the network at build time, and a failed
# fetch is a warning, so an offline `-W` build fails: drop `-W` to
# build offline.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

# `sphinx-build -b linkcheck`, run monthly by CI. ACM (10.1145) and
# Oxford University Press (10.1093) answer robots with 403 Forbidden, so
# their DOIs would always read as broken; the others are still checked.
linkcheck_ignore = [r"https://doi\.org/10\.(1145|1093)/"]

# A single-backtick span means code, and links when it names a
# documented object; RST's own default would render it as italics.
default_role = "py:obj"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
# Render `Attributes:` sections as a field list rather than as separate
# attribute entries: those would duplicate the properties of the same
# name that autodoc documents anyway, and fail `-W` on it.
napoleon_use_ivar = True

# autodoc already skips _-prefixed members by default, and that is
# relied on: the site is the public API surface, so anything _-prefixed
# is off it by definition.
autodoc_default_options = {"members": True}
# Merge the __init__ docstring into the class's, under one signature.
autoclass_content = "both"
autodoc_typehints = "both"
autodoc_typehints_description_target = "documented"
# `State` is a union, expanded in each signature that takes one; its
# last member is the private protocol for "any object with a `get`",
# which has no page to link to.
nitpick_ignore = [("py:class", "pylfit._rule._Gettable")]

# Slug every heading down to h3, so links to a section, such as
# `getting-started.md#score-the-model`, resolve.
myst_heading_anchors = 3

html_theme = "pydata_sphinx_theme"
# A GitHub icon in the top bar, linking to the repository.
html_theme_options = {"github_url": "https://github.com/Tony-sama/pylfit"}

# The theme's code fonts include Cascadia Code, whose ligatures draw
# `!=` as `≠`; `_static/custom.css` turns them off in code.
# `_static/stepper.js` shows a `pylfit-stepper` container's figures one
# at a time, with Previous and Next buttons; deferred like the theme's
# own scripts, so it doesn't hold up the page.
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = [("stepper.js", {"defer": "defer"})]
