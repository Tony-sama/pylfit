"""pylfit's learning algorithms: ``BruteForce``, ``GULA`` and ``PRIDE``.

Each has a static ``fit(dataset)`` that returns the learned rules as a
list. The same names are importable from ``pylfit`` itself, which is
the import the docs use.
"""

from pylfit.algorithms._brute_force import BruteForce
from pylfit.algorithms._gula import GULA
from pylfit.algorithms._pride import PRIDE

__all__ = ["BruteForce", "GULA", "PRIDE"]
