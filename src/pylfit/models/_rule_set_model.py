"""``RuleSetModel``: an unweighted set of rules.

A ``RuleSetModel`` predicts target values from a feature state with a
set of rules of equal weight: if a rule's body matches the state, its
head value is a possible value for that target. It counts no support
and has no notion of confidence; for that, use
``WeightedRuleSetModel``.

Build one from any rule set: a learning algorithm's output
(``RuleSetModel(dataset, GULA.fit(dataset))``), rules written by hand,
or a file written by ``save``.

Why it works this way: :doc:`/explanation/about-models`.
"""

import json
import math
import numbers
import os
import random
from collections.abc import Iterable, Mapping
from functools import cached_property
from typing import Any, Self, overload

import pandas as pd

from pylfit._dataset import Dataset, _check_dataset
from pylfit._rule import Rule, State
from pylfit.models._helpers import (
    Unknowns,
    check_rules,
    check_state,
    check_unknowns,
    dataframe_states,
    match_weight,
    pylfit_version,
    validate_domain,
)

#: The classes whose files ``from_file`` reads, by the name a file
#: records.
_MODEL_CLASSES = ("RuleSetModel", "WeightedRuleSetModel")


class RuleSetModel:
    """An unweighted set of rules: a match makes a value a candidate.

    Two models are equal (``==``) when their class, features, targets,
    rules and domain are. A model is not hashable, since its ``domain``
    can be reassigned: it can't go in a ``set`` or be a ``dict`` key.

    Attributes:
        features: The names of the feature (input) variables. Read-only;
            each access returns a new ``list``.
        targets: The names of the target (output) variables. Read-only;
            each access returns a new ``list``.
        rules: The rules of the model, as a ``tuple`` sorted by text,
            with no duplicates. Fixed once the model is built: for other
            rules, build another model, for example
            ``RuleSetModel(dataset, rules)``.
        domain: For each variable, features and targets alike, the
            sorted list of values it may take. Captured from the dataset
            at construction; you can assign another mapping afterward,
            as long as every rule's values stay in it. The prediction
            methods reject any input value outside its feature's domain
            rather than read it as unknown: to predict a value the
            dataset lacked, assign a domain that includes it.

    Examples:
        >>> import pandas as pd
        >>> from pylfit import GULA, Dataset, RuleSetModel
        >>> df = pd.DataFrame(
        ...     {
        ...         "a": ["0", "0", "1", "1"],
        ...         "b": ["0", "1", "0", "1"],
        ...         "y": ["0", "1", "1", "1"],
        ...     }
        ... )
        >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
        >>> model = RuleSetModel(dataset, GULA.fit(dataset))
        >>> print(model)
        features: ['a', 'b']
        targets: ['y']
        rules:
          y(0) :- a(0), b(0).
          y(1) :- a(1).
          y(1) :- b(1).
    """

    _SCHEMA_VERSION = 2

    def __init__(self, dataset: Dataset, rules: Iterable[Rule]) -> None:
        """Initializes a model from a dataset and a set of rules.

        Args:
            dataset: The dataset the rules come from, used only for its
                ``features``, ``targets`` and ``domain``.
            rules: The rules of the model. Duplicates are collapsed.
                Pass an empty iterable for a model without rules; there
                is no default.

        Raises:
            TypeError: If ``dataset`` is not a ``Dataset``, ``rules`` is
                not iterable or is a single ``str``, or a rule is not a
                ``Rule``.
            ValueError: If a rule names a variable or value outside the
                dataset's features, targets and domain.
        """
        _check_dataset(dataset)
        self._features: list[str] = list(dataset.features)
        self._targets: list[str] = list(dataset.targets)
        given: object = rules  # checked as the caller may pass it
        if isinstance(given, str):
            raise TypeError(
                "rules must be a collection of Rule, not a str; read a rule "
                "with Rule.from_string and pass [rule]"
            )
        given_rules = list(rules)
        # Checked in the order given, before the set: a set's order
        # would make the rule named below vary between runs.
        others = [rule for rule in given_rules if not isinstance(rule, Rule)]
        if others:
            raise TypeError(
                f"rules must be Rule objects, got {others[0]!r} "
                f"({type(others[0]).__name__}); Rule.from_string reads a "
                "rule from its text."
            )
        self._rules: set[Rule] = set(given_rules)
        self.domain = {
            variable: dataset.domain(variable)
            for variable in self._features + self._targets
        }

    # --- Read-only views ---

    @property
    def features(self) -> list[str]:
        """The names of the feature variables, as a new ``list``."""
        return list(self._features)

    @property
    def targets(self) -> list[str]:
        """The names of the target variables, as a new ``list``."""
        return list(self._targets)

    @property
    def rules(self) -> tuple[Rule, ...]:
        """The rules of the model, as a ``tuple`` sorted by text.

        Read-only: the rules are checked against the model's variables
        and domain once, at construction, and never change after. The
        class itself works on an internal ``set``.
        """
        return self._ordered_rules

    @cached_property
    def _ordered_rules(self) -> tuple[Rule, ...]:
        """``rules`` sorted by text, for ``rules``, ``__str__`` and ``save``.

        Sorted on first use and kept, as the rules never change: a model
        that is never printed, saved or asked for its rules pays
        nothing. The key is computed once per rule rather than at every
        comparison.
        """
        return tuple(sorted(self._rules, key=Rule._sort_key))

    # --- Domain ---

    @property
    def domain(self) -> dict[str, list[str]]:
        """For each variable, the sorted list of values it may take, as a copy.

        Changing the copy changes nothing in the model: assign a new
        mapping instead, which is validated first; see the setter.
        """
        return {variable: sorted(values) for variable, values in self._domain.items()}

    @domain.setter
    def domain(self, value: Mapping[str, Iterable[str]]) -> None:
        """Validates and stores a new domain mapping.

        Any iterable of values is accepted; the getter returns each
        sorted.

        Args:
            value: A mapping from variable name to an iterable of
                allowed values, with exactly one entry per variable in
                ``features + targets``.

        Raises:
            ValueError: If ``value``'s keys don't exactly match
                ``features + targets``, with every missing and/or extra
                variable named in the message; if a variable's values
                are a single ``str``; or if a rule of the model uses a
                value the new domain lacks.
            TypeError: If ``value`` is not a mapping, or a variable's
                values include a value that is not a ``str``.
        """
        domain = validate_domain(self._features, self._targets, value)
        check_rules(self._rules, self._features, self._targets, domain)
        self._domain: dict[str, set[str]] = domain

    # --- Prediction ---

    def predict_candidates(
        self, state: State, *, unknowns: Unknowns = "fallback"
    ) -> dict[str, list[str]]:
        """Returns, for each target, the values some matching rule concludes.

        Every matching rule counts equally, and how many rules agree is
        not counted. A target that no rule matches gets its whole
        domain: no value was ruled out.

        An unknown feature value (``None``) never satisfies a condition,
        so a rule testing it doesn't match. With
        ``unknowns="fallback"``, the default, a target no rule matches
        that way gets instead the values of the rules that *could*
        match: each of their conditions holds or falls on an unknown.
        Only if none could does it get its whole domain.
        ``unknowns="strict"`` skips that step. On a state without
        unknowns both give the same result.

        Args:
            state: A mapping of feature to ``str`` value, ``None`` where
                unknown, for example a ``dict`` or a row of a
                ``DataFrame``. Keys that are not features, a target's
                included, are ignored, so a full row of the training
                data can be passed.
            unknowns: ``"fallback"`` or ``"strict"``, as above; anything
                else raises ``ValueError``.

        Returns:
            A mapping of target to its candidate values, sorted, never
            empty for a target with a non-empty domain.

        Raises:
            KeyError: If ``state`` lacks a feature.
            TypeError: If ``state`` is not a mapping, or a known value
                of ``state`` is not a ``str``.
            ValueError: If a value of ``state`` is outside its feature's
                domain, or ``unknowns`` is not a known mode.

        See Also:
            WeightedRuleSetModel.predict_likelihood: The scored version.

        Examples:
            >>> import pandas as pd
            >>> from pylfit import GULA, Dataset, RuleSetModel
            >>> df = pd.DataFrame(
            ...     {
            ...         "a": ["0", "0", "1", "1"],
            ...         "b": ["0", "1", "0", "1"],
            ...         "y": ["0", "1", "1", "1"],
            ...     }
            ... )
            >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
            >>> model = RuleSetModel(dataset, GULA.fit(dataset))
            >>> model.predict_candidates({"a": "1", "b": "1"})
            {'y': ['1']}
        """
        check_state(state, self._features, self._domain)
        return {
            target: sorted(values)
            for target, values in self._candidates(state, unknowns).items()
        }

    def _candidates(self, state: State, unknowns: Unknowns) -> dict[str, set[str]]:
        """``predict_candidates`` on a state already checked."""
        check_unknowns(unknowns)
        candidates: dict[str, set[str]] = {target: set() for target in self._targets}
        for rule in self._rules:
            variable, value = rule.head
            if variable in candidates and rule.matches(state):
                candidates[variable].add(value)
        unmatched = {target for target, values in candidates.items() if not values}
        if unknowns == "fallback" and unmatched:
            for rule in self._rules:
                variable, value = rule.head
                if variable in unmatched and match_weight(rule, state) > 0:
                    candidates[variable].add(value)
        for target, values in candidates.items():
            if not values:
                candidates[target] = set(self._domain.get(target, set()))
        return candidates

    @overload
    def predict_proba(
        self,
        state: pd.DataFrame,
        *,
        unknowns: Unknowns = "fallback",
    ) -> list[dict[str, dict[str, float]]]: ...

    @overload
    def predict_proba(
        self,
        state: State,
        *,
        unknowns: Unknowns = "fallback",
    ) -> dict[str, dict[str, float]]: ...

    def predict_proba(
        self,
        state: State | pd.DataFrame,
        *,
        unknowns: Unknowns = "fallback",
    ) -> dict[str, dict[str, float]] | list[dict[str, dict[str, float]]]:
        """Predicts a uniform distribution over each target's candidates.

        Each value ``predict_candidates`` returns for a target gets an
        equal share. A target that no rule matches, nor under
        ``"fallback"`` could match, is uniform over its whole domain.

        Args:
            state: One state, as a mapping of feature to ``str`` value,
                where keys that are not features are ignored as in
                ``predict_candidates``; or a DataFrame with one row per
                state, checked and converted as ``predict`` checks its
                input, which refuses a column that is not a feature.
            unknowns: ``"fallback"`` or ``"strict"``, passed to
                ``predict_candidates``.

        Returns:
            A mapping of target to (value -> probability), summing to
            ``1`` for each target, with every value of the target's
            domain, in sorted order, ``0.0`` for a value no matching
            rule concludes. Empty only for a target whose domain is
            empty. For a DataFrame, a list of those, one per row, in row
            order.

        Raises:
            KeyError: If ``state`` is a mapping that lacks a feature.
            TypeError: If ``state`` is neither a mapping nor a
                DataFrame, or a mapping with a known value that is not a
                ``str``.
            ValueError: If ``unknowns`` is not a known mode, a value of
                ``state`` is outside its feature's domain, or ``state``
                is a DataFrame that ``predict`` would reject: a missing,
                target or extra column, or a column of a type
                ``predict`` refuses.

        See Also:
            WeightedRuleSetModel.predict_proba: Weighted by support.

        Examples:
            >>> import pandas as pd
            >>> from pylfit import GULA, Dataset, RuleSetModel
            >>> df = pd.DataFrame(
            ...     {
            ...         "a": ["0", "0", "1", "1"],
            ...         "b": ["0", "1", "0", "1"],
            ...         "y": ["0", "1", "1", "1"],
            ...     }
            ... )
            >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
            >>> model = RuleSetModel(dataset, GULA.fit(dataset))
            >>> model.predict_proba({"a": "0", "b": "0"})
            {'y': {'0': 1.0, '1': 0.0}}
        """
        check_unknowns(unknowns)
        if isinstance(state, pd.DataFrame):
            return [self._proba(row, unknowns) for row in self._states(state)]
        check_state(state, self._features, self._domain)
        return self._proba(state, unknowns)

    def _proba(self, state: State, unknowns: Unknowns) -> dict[str, dict[str, float]]:
        """``predict_proba`` on one state already checked.

        Overridden by ``WeightedRuleSetModel``; ``predict`` and both
        forms of ``predict_proba`` go through it, after
        ``check_unknowns``.
        """
        candidates = self._candidates(state, unknowns)
        # Every domain value, 0.0 for a non-candidate, sorted: a set's
        # order varies between processes under hash randomization, and
        # printed or saved output should not.
        return {
            target: {
                value: 1 / len(values) if value in values else 0.0
                for value in sorted(self._domain.get(target, set()))
            }
            for target, values in candidates.items()
        }

    def _states(self, dataframe: pd.DataFrame) -> list[dict[str, str | None]]:
        """Validates prediction input and converts it to one state per row.

        ``dataframe_states`` with this model's attributes, for
        ``predict`` and the ``DataFrame`` form of ``predict_proba`` (and
        ``WeightedRuleSetModel.predict_likelihood``).

        Args:
            dataframe: One row per state, with exactly ``self.features``
                as columns.

        Returns:
            One ``dict`` of feature to ``str`` value per row, in row
            order, with ``None`` for a missing value.

        Raises:
            ValueError: As ``dataframe_states``: a missing, target,
                extra or repeated column, a column of a refused type, or
                a value outside its feature's domain.
        """
        return dataframe_states(dataframe, self._features, self._targets, self._domain)

    def predict(
        self,
        dataframe: pd.DataFrame,
        *,
        random_state: int | None = None,
        unknowns: Unknowns = "fallback",
    ) -> list[dict[str, str | None]]:
        """Predicts the most likely value of each target, for each row.

        For each row and target, returns the value with the highest
        ``predict_proba``. Ties are broken at random, so on this class,
        where every candidate is equally likely, a row with several
        candidates gets a random one. Probabilities equal to a relative
        ``1e-9`` tie, so a tie doesn't depend on rounding. Each tie
        takes the next random draw; a target with one best value takes
        none. Pass ``random_state`` for reproducible output: a seed
        fixes the whole call.

        Args:
            dataframe: One row per state to predict, with exactly the
                columns in ``self.features`` (no target column, no other
                column), and every value within its feature's domain.
            random_state: Seed for the tie-breaking draw, an integer
                ``0`` or more. ``None`` draws unseeded, so results can
                differ between calls.
            unknowns: How a row with unknown values (``NaN`` or
                ``None``) is predicted, forwarded to ``predict_proba``:
                ``"fallback"`` (default) or ``"strict"``; see
                ``predict_candidates``.

        Returns:
            One dict per row of ``dataframe``, in row order, mapping
            each target to its predicted value. A value is ``None`` only
            for a target whose domain is empty.

        Raises:
            ValueError: If ``dataframe`` is missing a feature column,
                has a target column, any other extra column or a
                repeated one, or holds a value outside its feature's
                domain, a float that is not a whole number, ``bytes`` or
                a ``Decimal``, or a complex, date or duration column; if
                ``unknowns`` is not a known mode; or if ``random_state``
                is negative.
            TypeError: If ``dataframe`` is not a ``DataFrame``, or
                ``random_state`` is neither an integer (a NumPy integer
                counts as one) nor ``None``.

        Examples:
            >>> import pandas as pd
            >>> from pylfit import GULA, Dataset, RuleSetModel
            >>> df = pd.DataFrame(
            ...     {
            ...         "a": ["0", "0", "1", "1"],
            ...         "b": ["0", "1", "0", "1"],
            ...         "y": ["0", "1", "1", "1"],
            ...     }
            ... )
            >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
            >>> model = RuleSetModel(dataset, GULA.fit(dataset))
            >>> model.predict(df[["a", "b"]])
            [{'y': '0'}, {'y': '1'}, {'y': '1'}, {'y': '1'}]

            A tie is broken at random; a seed makes the call
            reproducible:

            >>> from pylfit import Rule
            >>> undecided = RuleSetModel(
            ...     dataset,
            ...     [Rule.from_string("y(0) :- ."), Rule.from_string("y(1) :- .")],
            ... )
            >>> first = undecided.predict(df[["a", "b"]], random_state=0)
            >>> first == undecided.predict(df[["a", "b"]], random_state=0)
            True
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"predict takes a DataFrame of states, got "
                f"{type(dataframe).__name__}; for one state, call "
                "predict_proba or predict_candidates"
            )
        check_unknowns(unknowns)
        seed: object = random_state  # checked as the caller may pass it
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
                hint = (
                    "; pylfit draws with Python's random module, so pass "
                    "an integer seed instead"
                    if type(seed).__module__.startswith("numpy")
                    else ""
                )
                raise TypeError(
                    "random_state must be an int 0 or more, or None, got "
                    f"{seed!r} ({type(seed).__name__}){hint}"
                )
            seed = int(seed)
            if seed < 0:
                raise ValueError(
                    f"random_state must be 0 or more, got {seed}; random.Random "
                    "would seed from its absolute value"
                )
        generator = random.Random(seed)

        predictions: list[dict[str, str | None]] = []
        for state in self._states(dataframe):
            probabilities = self._proba(state, unknowns)
            row_predictions: dict[str, str | None] = {}
            for target, value_probabilities in probabilities.items():
                if not value_probabilities:
                    row_predictions[target] = None
                else:
                    best = max(value_probabilities.values())
                    tied = [
                        value
                        for value, probability in value_probabilities.items()
                        if math.isclose(probability, best, rel_tol=1e-9)
                    ]
                    row_predictions[target] = (
                        tied[0] if len(tied) == 1 else generator.choice(tied)
                    )
            predictions.append(row_predictions)
        return predictions

    # --- Persistence ---

    def _rule_payload(self, rule: Rule) -> dict[str, Any]:
        """Returns one rule's saved JSON entry: ``display``/``head``/``body``.

        Overridden (extended) by ``WeightedRuleSetModel`` to add a
        ``score`` key.
        """
        return {
            "display": str(rule),
            "head": list(rule.head),
            "body": dict(rule.body),
        }

    def _extra_save_payload(self) -> dict[str, Any]:
        """Extra top-level keys for this class's file format.

        Empty here; overridden by ``WeightedRuleSetModel`` to add
        ``value_frequencies``.
        """
        return {}

    def save(self, path: str | os.PathLike[str]) -> None:
        """Saves this model to a JSON file.

        The file records ``features``, ``targets``, ``domain`` and every
        rule, plus the class that saved it, so ``from_file`` on a
        different class refuses it. Each rule also gets a readable
        ``display`` string, which ``from_file`` ignores. The file is
        UTF-8, with names and values written as they are rather than
        escaped. A name or value UTF-8 can't write, such as a lone
        surrogate from a file decoded with ``surrogateescape``, raises
        ``UnicodeEncodeError`` and leaves an existing file as it was.

        Args:
            path: Path of the JSON file to write. An existing file is
                overwritten.

        Examples:
            >>> import pandas as pd
            >>> from pylfit import GULA, Dataset, RuleSetModel
            >>> df = pd.DataFrame(
            ...     {
            ...         "a": ["0", "0", "1", "1"],
            ...         "b": ["0", "1", "0", "1"],
            ...         "y": ["0", "1", "1", "1"],
            ...     }
            ... )
            >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
            >>> import os, tempfile
            >>> model = RuleSetModel(dataset, GULA.fit(dataset))
            >>> with tempfile.TemporaryDirectory() as folder:
            ...     path = os.path.join(folder, "model.json")
            ...     model.save(path)
            ...     loaded = RuleSetModel.from_file(path)
            >>> loaded == model
            True
        """
        rules_payload = [self._rule_payload(rule) for rule in self._ordered_rules]

        payload = {
            "schema_version": self._SCHEMA_VERSION,
            "model_class": type(self).__name__,
            "pylfit_version": pylfit_version(),
            "features": list(self._features),
            "targets": list(self._targets),
            "domain": {
                variable: sorted(self._domain[variable])
                for variable in (*self._features, *self._targets)
            },
            "rules": rules_payload,
        }
        payload.update(self._extra_save_payload())

        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        text.encode("utf-8")  # before the open, which empties the file
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)

    def _init_extra_state(self) -> None:
        """Initializes subclass-specific mutable state before a load.

        Called once, right after ``__new__``, before any rule entry is
        parsed — so a hook like ``WeightedRuleSetModel._load_rule`` has
        somewhere (``self._rule_scores``) to write into as it goes.
        No-op here.
        """

    def _load_rule(self, entry: dict[str, Any]) -> Rule:
        """Parses one saved rule entry into a ``Rule``.

        Overridden (extended) by ``WeightedRuleSetModel`` to also
        attach the entry's score, if any, as a side effect on ``self``
        — see ``_init_extra_state``.
        """
        head = entry["head"]
        if not (
            isinstance(head, list)
            and len(head) == 2
            and all(isinstance(part, str) for part in head)
        ):
            raise TypeError(
                f"a rule entry's head must be a [variable, value] list of str, "
                f"got {head!r}"
            )
        body = entry["body"]
        if not isinstance(body, dict):
            raise TypeError(
                "a rule entry's body must be an object of variable to value, "
                f"as save writes it, got {body!r}"
            )
        return Rule((head[0], head[1]), body)

    def _load_extra(self, payload: dict[str, Any]) -> None:
        """Populates subclass-specific top-level state from a loaded payload.

        No-op here; overridden by ``WeightedRuleSetModel`` to set
        ``value_frequencies``.
        """

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> Self:
        """Loads a model written by ``save``.

        The file must have been saved by this same class: loading a
        ``WeightedRuleSetModel`` file through
        ``RuleSetModel.from_file``, or the reverse, raises instead of
        dropping or missing data.

        Args:
            path: Path of a JSON file written by this class's ``save``.

        Returns:
            A new instance of ``cls`` with the file's ``features``,
            ``targets``, ``rules`` and ``domain``, plus any state the
            subclass saves.

        Raises:
            ValueError: If the file does not hold a JSON object, its
                ``schema_version`` is not one this class reads,
                ``model_class`` is not this class, a name appears twice
                across ``features`` and ``targets``, ``domain`` doesn't
                cover exactly ``features + targets``, or a rule names a
                variable or value outside them, or ``domain`` gives a
                variable a single ``str``; on ``WeightedRuleSetModel``,
                also if a score is null or negative, a rule is listed
                twice with different scores, or ``value_frequencies``
                names a variable that is not a feature, or a share that
                is not a number in ``[0, 1]``.
            TypeError: If the file's ``features`` or ``targets`` is not
                a list of ``str``, its ``rules`` is not a list, its
                ``domain`` is not a mapping or holds a value that is not
                a ``str``, a rule entry's ``head`` is not a list of two
                ``str``, or its ``body`` is not an object of variable to
                ``str`` value, or, on ``WeightedRuleSetModel``, a rule's
                ``score`` is not a two-item list of ``int``, or its
                ``value_frequencies`` is not a mapping of mappings.
            KeyError: If the file lacks ``features``, ``targets``,
                ``domain`` or ``rules``, a rule entry lacks ``head`` or
                ``body``, or, on ``WeightedRuleSetModel``, a rule entry
                lacks ``score``.

        Examples:
            >>> import pandas as pd
            >>> from pylfit import GULA, Dataset, RuleSetModel
            >>> df = pd.DataFrame(
            ...     {
            ...         "a": ["0", "0", "1", "1"],
            ...         "b": ["0", "1", "0", "1"],
            ...         "y": ["0", "1", "1", "1"],
            ...     }
            ... )
            >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
            >>> import os, tempfile
            >>> with tempfile.TemporaryDirectory() as folder:
            ...     path = os.path.join(folder, "model.json")
            ...     RuleSetModel(dataset, GULA.fit(dataset)).save(path)
            ...     loaded = RuleSetModel.from_file(path)
            >>> len(loaded.rules)
            3
        """
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(
                f"not a model file: it holds a JSON {type(payload).__name__}, "
                "not an object"
            )

        found_version = payload.get("schema_version")
        if found_version != cls._SCHEMA_VERSION:
            raise ValueError(
                f"unsupported {cls.__name__} file schema_version: found "
                f"{found_version!r}, expected {cls._SCHEMA_VERSION!r}"
            )

        found_class = payload.get("model_class")
        if found_class != cls.__name__:
            hint = (
                f": load it via {found_class}.from_file(...) instead"
                if found_class in _MODEL_CLASSES
                else ""
            )
            raise ValueError(
                f"file was saved by {found_class!r}, not {cls.__name__!r}{hint}"
            )

        for key in ("features", "targets"):
            names = payload[key]
            if not isinstance(names, list) or not all(
                isinstance(name, str) for name in names
            ):
                raise TypeError(f"file's {key!r} must be a list of str, got {names!r}")
        named = payload["features"] + payload["targets"]
        repeated = sorted({name for name in named if named.count(name) > 1})
        if repeated:
            raise ValueError(
                f"file names {repeated} more than once across features and targets"
            )
        if not isinstance(payload["rules"], list):
            raise TypeError(f"file's 'rules' must be a list, got {payload['rules']!r}")

        model = cls.__new__(cls)
        model._features = list(payload["features"])
        model._targets = list(payload["targets"])
        model._init_extra_state()
        model._rules = {model._load_rule(entry) for entry in payload["rules"]}
        # Assigned as read: the setter refuses a str where a collection
        # of values belongs, rather than splitting it into characters.
        model.domain = payload["domain"]
        model._load_extra(payload)
        return model

    # --- Comparison operators ---

    def __eq__(self, other: object) -> bool:
        """Checks whether two models are equal in every attribute.

        Compares ``features``, ``targets``, ``rules``, and ``domain``.
        Models of different classes are never equal: a
        ``WeightedRuleSetModel`` never equals a plain ``RuleSetModel``,
        even with identical rules, since one carries scores the other
        lacks.
        """
        if not isinstance(other, RuleSetModel):
            return NotImplemented
        return (
            type(other) is type(self)
            and self._features == other._features
            and self._targets == other._targets
            and self._rules == other._rules
            and self._domain == other._domain
        )

    # --- String representation ---

    def _rule_line(self, rule: Rule) -> str:
        """Returns one rule's line in ``str(model)``.

        Overridden by ``WeightedRuleSetModel`` to append its
        ``(positives, negatives)`` score, if attached.
        """
        return f"  {rule}"

    def __str__(self) -> str:
        """Lists the model's features, targets and rules, one rule a line.

        Returns:
            A multi-line string: the features, the targets, then the
            rules sorted by text, one per line.

        Examples:
            >>> import pandas as pd
            >>> from pylfit import GULA, Dataset, RuleSetModel
            >>> df = pd.DataFrame(
            ...     {
            ...         "a": ["0", "0", "1", "1"],
            ...         "b": ["0", "1", "0", "1"],
            ...         "y": ["0", "1", "1", "1"],
            ...     }
            ... )
            >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
            >>> model = RuleSetModel(dataset, GULA.fit(dataset))
            >>> print(model)
            features: ['a', 'b']
            targets: ['y']
            rules:
              y(0) :- a(0), b(0).
              y(1) :- a(1).
              y(1) :- b(1).
        """
        lines = [f"features: {self._features}", f"targets: {self._targets}", "rules:"]
        lines.extend(self._rule_line(rule) for rule in self._ordered_rules)
        return "\n".join(lines)

    def __repr__(self) -> str:
        """A one-line summary; ``str`` lists the rules."""
        rules = f"{len(self._rules)} rule" + ("" if len(self._rules) == 1 else "s")
        return (
            f"<{type(self).__name__}: {rules}, "
            f"features {self._features}, targets {self._targets}>"
        )
