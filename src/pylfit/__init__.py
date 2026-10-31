"""pylfit: learn logic rules from data.

The public API, importable from ``pylfit`` directly:

- data: ``Dataset`` and ``Rule``;
- models: ``RuleSetModel``, ``WeightedRuleSetModel``;
- learning algorithms: ``BruteForce``, ``GULA``, ``PRIDE``;
- ``generalize``, which drops the conditions a rule's data does not
  need;
- metrics: ``accuracy_score``, ``accuracy_by_target``, ``coverage``,
  ``rule_recovery``;
- a check to run on your data first: ``find_missing_stand_ins``, with
  its default list ``DEFAULT_STAND_INS``;
- warning classes, to filter pylfit's warnings by class:
  ``PylfitWarning``, the base of all of them, and ``ConflictWarning``;
- ``__version__``, the installed version.

The modules that define them are private (``_``-prefixed), as in
scikit-learn: import from ``pylfit``, or from the ``pylfit.models``
and ``pylfit.algorithms`` subpackages, never from a module.

An algorithm's ``fit`` returns a plain list of rules; wrap it in a model
to predict, for example ``RuleSetModel(dataset, GULA.fit(dataset))``.
"""

from pylfit._checks import (
    DEFAULT_STAND_INS,
    find_missing_stand_ins,
)
from pylfit._dataset import Dataset
from pylfit._exceptions import ConflictWarning, PylfitWarning
from pylfit._generalize import generalize
from pylfit._metrics import (
    accuracy_by_target,
    accuracy_score,
    coverage,
    rule_recovery,
)
from pylfit._rule import Rule
from pylfit.algorithms import GULA, PRIDE, BruteForce
from pylfit.models import RuleSetModel, WeightedRuleSetModel
from pylfit.models._helpers import pylfit_version as _pylfit_version

__version__ = _pylfit_version()

__all__ = [
    # data
    "Dataset",
    "Rule",
    # models
    "RuleSetModel",
    "WeightedRuleSetModel",
    # learning algorithms
    "BruteForce",
    "GULA",
    "PRIDE",
    # rules
    "generalize",
    # metrics
    "accuracy_score",
    "accuracy_by_target",
    "coverage",
    "rule_recovery",
    # the data check
    "find_missing_stand_ins",
    "DEFAULT_STAND_INS",
    # warnings
    "PylfitWarning",
    "ConflictWarning",
    "__version__",
]
