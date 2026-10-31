"""User-style code that ``mypy`` checks against the package's hints.

Never run: ``mypy`` lists this directory in ``files``, and pytest
doesn't collect it (no ``test_`` prefix). Each ``assert_type`` fails
the ``mypy`` check if a public method's return type stops naming the
class a user called it on, or a method stops returning a list for a
DataFrame and a mapping for one state; each call fails it if a hint
refuses an argument the docs accept.
"""

from typing import assert_type

import pandas as pd

from pylfit import Rule, RuleSetModel, WeightedRuleSetModel


def loaded_models_keep_their_class(path: str) -> None:
    assert_type(WeightedRuleSetModel.from_file(path), WeightedRuleSetModel)
    assert_type(RuleSetModel.from_file(path), RuleSetModel)


def a_dict_built_from_a_row_is_a_state(rule: Rule, df: pd.DataFrame) -> None:
    rule.matches(df.iloc[0].to_dict())
    rule.matches(df.to_dict("records")[0])
    rule.matches(df.iloc[0])


def predictions_take_the_shape_of_their_input(
    model: RuleSetModel, weighted: WeightedRuleSetModel, df: pd.DataFrame
) -> None:
    state = df.iloc[0].to_dict()
    assert_type(model.predict_proba(df), list[dict[str, dict[str, float]]])
    assert_type(model.predict_proba(state), dict[str, dict[str, float]])
    assert_type(weighted.predict_proba(df), list[dict[str, dict[str, float]]])
    assert_type(weighted.predict_proba(state), dict[str, dict[str, float]])
    assert_type(weighted.predict_likelihood(df), list[dict[str, dict[str, float]]])
    assert_type(weighted.predict_likelihood(state), dict[str, dict[str, float]])
