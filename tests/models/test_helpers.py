"""Tests for the model helpers that the models can't reach on their own.

The models check a state before ``match_weight`` sees it and always
pass shares for every feature, so its own refusals and defaults are
tested here directly. So is ``pylfit_version``'s fallback, which an
installed package never takes. The last three check that error
messages list names sorted, so a set's order never reaches them.
"""

from importlib import metadata
from unittest import mock

import pandas as pd
import pytest

from pylfit import Dataset, Rule, RuleSetModel
from pylfit.models import _helpers
from pylfit.models._helpers import (
    check_columns,
    check_domains,
    match_weight,
    pylfit_version,
    validate_domain,
)

RULE = Rule.from_string("y(1) :- a(1), b(0).")


def test_match_weight_refuses_a_state_without_a_tested_feature():
    with pytest.raises(KeyError, match="'b'"):
        match_weight(RULE, {"a": None})


def test_match_weight_refuses_a_known_value_that_is_not_a_str():
    with pytest.raises(TypeError, match="state value for 'b' is 0"):
        match_weight(RULE, {"a": None, "b": 0})


def test_match_weight_without_shares_counts_an_unknown_as_possible():
    assert match_weight(RULE, {"a": None, "b": "0"}) == 1.0
    assert match_weight(RULE, {"a": None, "b": "0"}, {"b": {"0": 0.5}}) == 1.0


def test_match_weight_gives_a_value_without_a_share_no_weight():
    assert match_weight(RULE, {"a": None, "b": "0"}, {"a": {"0": 1.0}}) == 0.0
    assert match_weight(RULE, {"a": None, "b": None}, {"a": {"1": 0.5}}) == 0.5


def test_pylfit_version_falls_back_without_installed_metadata(tmp_path):
    with mock.patch.object(
        _helpers.metadata,
        "version",
        side_effect=metadata.PackageNotFoundError("pylfit"),
    ):
        assert pylfit_version() == "unknown"
        dataset = Dataset(
            pd.DataFrame({"a": ["0", "1"], "y": ["0", "1"]}),
            features=["a"],
            targets=["y"],
        )
        RuleSetModel(dataset, []).save(tmp_path / "model.json")
    assert '"pylfit_version": "unknown"' in (tmp_path / "model.json").read_text()


NAMES = [f"v{i}" for i in range(12)]


def test_check_columns_lists_unexpected_columns_sorted():
    frame = pd.DataFrame(columns=["a", *NAMES[::-1]])
    with pytest.raises(ValueError, match=", ".join(sorted(NAMES))):
        check_columns(frame, ["a"], [])


def test_check_domains_lists_out_of_domain_values_sorted():
    frame = pd.DataFrame({"a": NAMES[::-1]})
    with pytest.raises(ValueError, match=", ".join(sorted(NAMES))):
        check_domains(frame, ["a"], {"a": set()})


def test_validate_domain_lists_missing_and_extra_variables_sorted():
    quoted = ", ".join(repr(name) for name in sorted(NAMES))
    with pytest.raises(ValueError, match=f"missing variable\\(s\\): {quoted}"):
        validate_domain(NAMES[::-1], [], {})
    with pytest.raises(ValueError, match=f"unexpected variable\\(s\\): {quoted}"):
        validate_domain([], [], {name: [] for name in NAMES[::-1]})
