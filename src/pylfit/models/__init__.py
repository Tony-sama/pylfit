"""pylfit's models: ``RuleSetModel`` and ``WeightedRuleSetModel``.

``RuleSetModel`` holds an unordered set of rules: every matching rule's
conclusion is a candidate, none ranked above another. Its subclass
``WeightedRuleSetModel`` adds each rule's positive and negative counts,
and weighs the candidates by them. The same names are importable from
``pylfit`` itself, which is the import the docs use.
"""

from pylfit.models._rule_set_model import RuleSetModel
from pylfit.models._weighted_rule_set_model import WeightedRuleSetModel

__all__ = ["RuleSetModel", "WeightedRuleSetModel"]
