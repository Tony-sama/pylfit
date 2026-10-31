"""The examples gallery's order, as sphinx-gallery's ``within_subsection_order``.

sphinx-gallery sorts a gallery by code length unless told otherwise,
which put a benchmark first and the starter fifth. A key named by its
import path, rather than a function in ``conf.py``, keeps the
configuration picklable, which Sphinx needs to cache it.
"""

#: First steps, then the tutorials, then the benchmarks.
ORDER = [
    "starter.py",
    "tutorial_heart_disease.py",
    "dynamics.py",
    "migrate_from_0x.py",
    "benchmark_classic.py",
    "benchmark_bnet.py",
    "benchmark_influence.py",
    "benchmark_min_support.py",
    "benchmark_negative_order.py",
    "benchmark_n_jobs.py",
    "benchmark_unknowns.py",
]


class GalleryOrder:
    """Sort key: an example's place in ``ORDER``.

    Args:
        src_dir: The gallery's source folder, which sphinx-gallery passes
            to every sort key; unused.
    """

    def __init__(self, src_dir: str) -> None:
        self.src_dir = src_dir

    def __call__(self, filename: str) -> int:
        """Returns ``filename``'s place in ``ORDER``.

        A file missing from ``ORDER`` makes ``list.index`` raise
        ``ValueError`` and stops the build, so a new example can't land
        in the gallery unplaced.

        Args:
            filename: An example's file name.

        Returns:
            Its index in ``ORDER``.
        """
        return ORDER.index(filename)

    def __repr__(self) -> str:
        return "GalleryOrder"
