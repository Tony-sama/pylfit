"""Minimal ``.bnet`` (BoolNet) format reader and transition-table generator.

Support code for the examples that read the networks under
``tests/data/bnet/``, and for the test suite's
``tests/algorithms/test_boolean_networks.py``, which loads this file
rather than keeping a copy. Not part of the ``pylfit`` package and not
the point of any example: underscored so it reads as plumbing, letting
the examples stay about ``pylfit`` rather than about file parsing.

If ``.bnet`` loading ever moves into ``pylfit`` as a real loader, this
module goes with it.

Format: each non-comment, non-blank line is ``target, factor``, where
``factor`` is a Boolean expression over other target names using
``&`` (and), ``|`` (or), ``!`` (not, prefix), parens, and the literals
``0``/``1``. A ``targets, factors`` header line, if present, is
skipped; six vendored networks omit it (``n12c5.bnet`` among them).
``#``-to-end-of-line comments and blank lines are skipped anywhere.

Two kinds of node need no special case. A self-referencing line
(``CycD, CycD``) is a free input, keeping its value: evaluation reads
the current value like any other identifier. A literal factor
(``AUXINS, 1``) is a constant node, which only ever takes that value.
Both are common in PyBoolNet's model repository: self-loops in
``arellano_rootstem``, ``faure_cellcycle``, ``multivalued``,
``tournier_apoptosis`` and ``xiao_wnt5a``, constants in
``arellano_rootstem``, ``davidich_yeast`` and ``multivalued``.
"""

from __future__ import annotations

import ast
import itertools
import re
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

# Only these characters are permitted in a factor expression before it
# is handed to `eval` — anything else raises, rather than silently
# evaluating attacker-controlled or malformed text. Vendored data is
# trusted, but this keeps `parse_bnet_text` honest about the grammar
# it actually accepts.
_ALLOWED_CHARS = set("&|!() \t") | set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)


def parse_bnet_text(text: str) -> dict[str, str]:
    """Parse `.bnet` text into `{target: factor_expression}`.

    Strips the optional `targets, factors` header, `#` comments, and
    blank lines. Raises `ValueError` on a data line that doesn't split
    into exactly `target, factor` or whose factor contains a character
    outside the supported grammar.
    """
    factors: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.lower().replace(" ", "").startswith("targets,factors"):
            continue
        if "," not in line:
            raise ValueError(f"malformed .bnet line (no comma): {raw_line!r}")
        target, factor = line.split(",", 1)
        target = target.strip()
        factor = factor.strip()
        bad_chars = set(factor) - _ALLOWED_CHARS
        if bad_chars:
            raise ValueError(
                f"factor for {target!r} has unsupported character(s) "
                f"{sorted(bad_chars)!r}: {factor!r}"
            )
        factors[target] = factor
    return factors


def parse_bnet_file(path: str | Path) -> dict[str, str]:
    """`parse_bnet_text`, reading from a `.bnet` file on disk."""
    return parse_bnet_text(Path(path).read_text())


#: Identifier tokens in a factor expression, used to find which
#: variables a node actually depends on.
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def to_python(expr: str) -> str:
    """Rewrite a `.bnet` factor expression as a Python expression.

    A straight token-level substitution (`!`/`&`/`|` -> `not`/`and`/
    `or`), valid because none of the three operator characters can
    appear inside a `.bnet` identifier and Python's `not`/`and`/`or`
    precedence already matches their standard reading.

    Shared by `compile_factor`, which evaluates the result, and
    `monotone_regulators`, which parses it. They must agree on the
    translation or the second would be reasoning about a different
    expression than the first evaluates.
    """
    return expr.replace("!", " not ").replace("&", " and ").replace("|", " or ").strip()


def compile_factor(expr: str) -> Callable[[Mapping[str, bool]], bool]:
    """Compile a `.bnet` factor expression into `state -> bool`.

    Translated by `to_python`. `eval` runs with an empty `__builtins__`
    and only the state mapping as locals, so a malformed or unexpected
    identifier raises `NameError` rather than resolving to a builtin.
    """
    compiled = compile(to_python(expr), filename="<bnet-factor>", mode="eval")

    def evaluate(state: Mapping[str, bool]) -> bool:
        return bool(eval(compiled, {"__builtins__": {}}, dict(state)))

    return evaluate


def generate_transitions(path: str | Path) -> pd.DataFrame:
    """Enumerate every state of a `.bnet` network and its successor.

    Returns a `DataFrame` with `f"{var}_t"` feature columns and
    `f"{var}_t+1"` target columns, one row per one of the network's
    `2**n` states, variables in sorted order rather than file order.
    The `_t`/`_t+1` naming is the one the docs pages use for the state
    at `t` and at `t+1`. Values are the strings `"True"` and `"False"`,
    the form `influences` names them in.
    """
    factors = parse_bnet_file(path)
    variables = sorted(factors)
    compiled = {var: compile_factor(factors[var]) for var in variables}

    rows = []
    for combo in itertools.product([False, True], repeat=len(variables)):
        state = dict(zip(variables, combo, strict=True))
        next_state = {var: compiled[var](state) for var in variables}
        row = {f"{var}_t": str(state[var]) for var in variables}
        row.update({f"{var}_t+1": str(next_state[var]) for var in variables})
        rows.append(row)

    columns = [f"{var}_t" for var in variables] + [f"{var}_t+1" for var in variables]
    return pd.DataFrame(rows, columns=columns)


def sample_transitions(path: str | Path, rows: int, seed: int = 0) -> pd.DataFrame:
    """Sample distinct states of a `.bnet` network, each with its successor.

    For networks too large for `generate_transitions`: `rows` distinct
    states drawn uniformly at random, the same columns and values.
    Twice `rows` random states are drawn, the duplicates dropped, and
    `rows` of the rest kept in a random order.

    Args:
        path: Path to a `.bnet` file.
        rows: Number of distinct states to keep.
        seed: Seed of the NumPy generator; the same seed gives the same
            rows.

    Returns:
        A `DataFrame` like `generate_transitions`', with `rows` rows.

    Raises:
        ValueError: If fewer than `rows` distinct states came up, as on
            a network with few states; use `generate_transitions` there.
    """
    factors = parse_bnet_file(path)
    variables = sorted(factors)
    compiled = {var: compile_factor(factors[var]) for var in variables}

    rng = np.random.default_rng(seed)
    states = np.unique(rng.random((rows * 2, len(variables))) < 0.5, axis=0)
    if len(states) < rows:
        raise ValueError(
            f"{Path(path).name}: {len(states)} distinct states came up, "
            f"not {rows}; use generate_transitions"
        )
    states = states[rng.permutation(len(states))[:rows]]

    records = []
    for combo in states.tolist():
        state = dict(zip(variables, combo, strict=True))
        next_state = {var: compiled[var](state) for var in variables}
        row = {f"{var}_t": str(state[var]) for var in variables}
        row.update({f"{var}_t+1": str(next_state[var]) for var in variables})
        records.append(row)

    columns = [f"{var}_t" for var in variables] + [f"{var}_t+1" for var in variables]
    return pd.DataFrame(records, columns=columns)


def num_variables(path: str | Path) -> int:
    """Variable count of a `.bnet` file.

    Read from the file rather than hardcoded beside the network's name,
    so the benchmark's reported size cannot drift from the data.
    """
    return len(parse_bnet_file(path))


def feature_target_columns(path: str | Path) -> tuple[list[str], list[str]]:
    """Return `generate_transitions`' column names, without the table.

    Args:
        path: Path to a `.bnet` file.

    Returns:
        ``([f"{var}_t", ...], [f"{var}_t+1", ...])``, variables sorted.
    """
    variables = sorted(parse_bnet_file(path))
    return [f"{v}_t" for v in variables], [f"{v}_t+1" for v in variables]


def build_dataset_arguments(
    path: str | Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Everything `Dataset` needs for one network, in one call.

    Args:
        path: Path to a `.bnet` file.

    Returns:
        ``(table, features, targets)``, ready for
        ``Dataset(table, features=features, targets=targets)``.
    """
    table = generate_transitions(path)
    features, targets = feature_target_columns(path)
    return table, features, targets


def regulators(expression: str, variables: list[str]) -> list[str]:
    """Which of `variables` actually appear in `expression`.

    Args:
        expression: A node's raw `.bnet` update expression.
        variables: All variable names, sorted.

    Returns:
        The subset appearing in `expression`, in `variables`' order.
    """
    tokens = set(_IDENTIFIER_RE.findall(expression))
    return [variable for variable in variables if variable in tokens]


def monotone_regulators(expression: str, node_regulators: list[str]) -> dict[str, str]:
    """Which regulators the node's function is monotone in.

    Read off the expression's syntax rather than from a truth table. A
    `.bnet` expression only combines literals with `not`/`and`/`or`,
    and `and`/`or` never flip a literal's sign — only `not` does. So a
    regulator appearing under the same negation parity everywhere can
    only push the output one way: always under an even number of
    `not`s (bare counts as zero) is positive, always under an odd
    number is negative. Walking the parsed expression while tracking parity is
    therefore enough, with nothing exponential in regulator count.

    A regulator appearing under *different* parities (both ``v`` and
    ``!v``) is left out. It may be genuinely non-monotone, or merely
    written in a form this check cannot untangle; either way, omitting
    it costs only completeness of the background knowledge, never
    correctness of what `GULA` then learns.

    Args:
        expression: A node's raw `.bnet` update expression.
        node_regulators: That node's regulators, from `regulators`.

    Returns:
        ``{regulator: "positive" | "negative"}`` for the subset under a
        single consistent parity.
    """
    if not node_regulators:
        return {}

    regulator_set = set(node_regulators)
    parities: dict[str, set[int]] = {name: set() for name in regulator_set}

    def walk(node: ast.AST, negated: bool) -> None:
        if isinstance(node, ast.Name):
            if node.id in regulator_set:
                parities[node.id].add(1 if negated else 0)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            walk(node.operand, not negated)
        elif isinstance(node, ast.BoolOp):  # `and`/`or` never flip sign
            for value in node.values:
                walk(value, negated)
        elif isinstance(node, ast.Constant):
            pass
        else:
            # Outside the grammar this module documents. Mark every
            # regulator underneath ambiguous rather than guess.
            for descendant in ast.walk(node):
                if isinstance(descendant, ast.Name) and descendant.id in regulator_set:
                    parities[descendant.id].update({0, 1})

    walk(ast.parse(to_python(expression), mode="eval").body, negated=False)

    directions: dict[str, str] = {}
    for regulator, seen in parities.items():
        if seen == {0}:
            directions[regulator] = "positive"
        elif seen == {1}:
            directions[regulator] = "negative"
        # Mixed parity: left unrestricted rather than guessed at.
    return directions


def influences(
    path: str | Path, *, with_values: bool = False
) -> dict[tuple[str, str], list[str | tuple[str, str]]]:
    """A `.bnet` file's wiring as `GULA.fit`'s ``allowed_conditions``.

    This is the network's own wiring handed to the learner: which
    variables can possibly affect which, and — optionally — in which
    direction. Since the file *is* the ground truth here, the knowledge
    is exact, which is what makes the restricted run recover the same
    rules as the unrestricted one rather than merely a subset of them.

    Args:
        path: Path to a `.bnet` file.
        with_values: When `False`, restrict each head to its regulators
            as bare feature names — "these features can matter", no
            claim about values. When `True`, additionally pin each
            monotone regulator to the value its sign supports for that
            head. Non-monotone regulators stay bare either way.

    Returns:
        ``{(f"{variable}_t+1", value): [...]}`` for every variable and
        each value its factor can take, ready to pass to `GULA.fit`: a
        constant node has one value, and `GULA.fit` rejects a head
        whose value the data never shows. Values are
        ``"True"``/``"False"`` to match `generate_transitions`.
    """
    factors = parse_bnet_file(path)
    variables = sorted(factors)

    allowed: dict[tuple[str, str], list[str | tuple[str, str]]] = {}
    for variable in variables:
        expression = factors[variable]
        node_regulators = regulators(expression, variables)
        directions = (
            monotone_regulators(expression, node_regulators) if with_values else {}
        )
        factor = compile_factor(expression)
        produced = {
            str(factor(dict(zip(node_regulators, combo, strict=True))))
            for combo in itertools.product([False, True], repeat=len(node_regulators))
        }

        for head_value in ("True", "False"):
            if head_value not in produced:
                continue
            conditions: list[str | tuple[str, str]] = []
            for regulator in node_regulators:
                direction = directions.get(regulator)
                if direction is None:
                    conditions.append(f"{regulator}_t")  # both values allowed
                elif direction == "positive":
                    conditions.append((f"{regulator}_t", head_value))
                else:  # negative: the head's value flipped
                    opposite = "False" if head_value == "True" else "True"
                    conditions.append((f"{regulator}_t", opposite))
            allowed[(f"{variable}_t+1", head_value)] = conditions

    return allowed
