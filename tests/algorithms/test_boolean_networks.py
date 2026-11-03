"""Regression tests on Boolean networks: the three algorithms agree.

Correctness, completeness, minimality and agreement between the
algorithms, checked over 16 of the 21 networks in ``tests/data/bnet/``;
the other five have 15 variables or more.

Data is generated on the fly (``examples/_bnet.py``, loaded by path)
from the vendored ``.bnet`` files in ``tests/data/bnet/`` rather than
committed as CSVs, since exhaustively enumerating a network's ``2**n``
states and evaluating each variable's own update function is cheap at
this scale.

There's no external expected-rule source for these networks (only
``.bnet`` Boolean update functions) — ground truth is cross-algorithm
agreement instead: ``BruteForce``/``GULA``/``PRIDE`` are each
documented (module docstrings) to be correct and minimal, and
``BruteForce``/``GULA`` additionally documented to compute the exact
same unique, complete rule set. Since every network here is generated
exhaustively (every one of its ``2**n`` states is a row), ``PRIDE``'s
documented non-exhaustive completeness gap doesn't apply either — see
``_pride.py``'s module docstring — so all three algorithms' rule sets
are checked for correctness/completeness/minimality directly, not just
against each other.

**Two tiers, split by measured cost rather than variable count alone.**
``GULA`` dominates the cost, not the property checks: final rule
counts on exhaustive data stay modest (tens to ~90 rules), so
``assert_not_dominated``'s O(rules²) loop and ``assert_complete``'s
state-product scan are cheap.

- **Fast (default pytest tier), <=6 vars, 5 networks:** all three
  algorithms run, including ``BruteForce``. Comfortably under a
  second combined.
- **Slow (``@pytest.mark.slow``), 7-13 vars, 11 networks:**
  ``GULA``/``PRIDE`` only. ``BruteForce`` takes ~4s at 9 vars and
  ~26s at 10, so the cross-check against it stays in the fast tier,
  and this tier relies on ``GULA``'s own documented
  correctness/completeness on exhaustive data plus ``PRIDE``'s subset
  check. The 7-11 var networks take under a second each, the 12-13
  var ones 1.5-3.5s, about 11s for the whole tier.

**The two 15-var networks (``dinwoodie_life``, ``randomnet_n15k3``)
are not tested here.** Fitting and checking one takes 18-21s, so
together they would add about 40s to the slow tier's 11s. Their
``.bnet`` files are vendored with the rest under
``tests/data/bnet/``, and their correctness/completeness/minimality is
not verified by ``tests/``.
"""

import importlib.util
from pathlib import Path

import pytest

from pylfit import GULA, PRIDE, BruteForce, Dataset
from tests.algorithms._properties import (
    assert_complete,
    assert_correct,
    assert_irreducible,
    assert_not_dominated,
)

_DATA_DIR = Path(__file__).parent.parent / "data" / "bnet"


def _load_bnet_module():
    """Loads ``examples/_bnet.py``, the one ``.bnet`` reader, by path.

    ``examples/`` is not a package and not on ``sys.path``; loading the
    file directly avoids changing either for the whole test session.
    """
    path = Path(__file__).resolve().parents[2] / "examples" / "_bnet.py"
    spec = importlib.util.spec_from_file_location("_bnet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_bnet = _load_bnet_module()

# fmt: off
_FAST_TIER = ["raf", "n3s1c1a", "n3s1c1b", "n5s3", "n6s1c2"]
_SLOW_TIER = [
    "n7s3", "randomnet_n7k3", "xiao_wnt5a", "arellano_rootstem",
    "davidich_yeast", "faure_cellcycle", "krumsiek_myeloid",
    "n12c5", "tournier_apoptosis", "multivalued", "saadatpour_guardcell",
]
# fmt: on


def _load_dataset(name: str) -> Dataset:
    """Generates a vendored network's full transition table.

    One ``_t`` feature and one ``_t+1`` target per network variable,
    the state at ``t`` and at ``t+1``.
    """
    df = _bnet.generate_transitions(_DATA_DIR / f"{name}.bnet")
    features = [column for column in df.columns if column.endswith("_t")]
    targets = [column for column in df.columns if column.endswith("_t+1")]
    return Dataset(df, features=features, targets=targets)


def _run_full_cross_check(name: str) -> None:
    """Fits ``BruteForce``, ``GULA`` and ``PRIDE``, cross-checks them.

    Each is fitted once, not once per property, since fitting — not
    checking — is where the cost is. Fast tier only: ``BruteForce``
    is too slow for the 7-13 var tier; see the module docstring.
    """
    dataset = _load_dataset(name)
    brute_force_rules = BruteForce.fit(dataset)
    gula_rules = GULA.fit(dataset)
    pride_rules = PRIDE.fit(dataset)

    assert_correct(dataset, brute_force_rules)
    assert_complete(dataset, brute_force_rules)
    assert_not_dominated(dataset, brute_force_rules)
    assert_irreducible(dataset, brute_force_rules)

    assert gula_rules == brute_force_rules, (
        f"{name}: GULA and BruteForce learned different rule sets"
    )

    # Exhaustive data (every state is a row), so PRIDE's non-exhaustive
    # completeness gap (_pride.py's module docstring) doesn't apply —
    # checked directly, not just via the subset relationship below.
    assert_correct(dataset, pride_rules)
    assert_complete(dataset, pride_rules)
    assert_not_dominated(dataset, pride_rules)
    assert_irreducible(dataset, pride_rules)
    assert set(pride_rules) <= set(gula_rules), (
        f"{name}: PRIDE produced a rule GULA didn't"
    )


def _run_gula_pride_check(name: str) -> None:
    """Fits ``GULA``/``PRIDE`` only (no ``BruteForce``), checks both.

    ``GULA``'s correctness/completeness/minimality on exhaustive data
    is documented, not just assumed (module docstring), so it stands
    in for ``BruteForce`` as ground truth here, and ``PRIDE``'s rules
    must be a subset of ``GULA``'s.
    """
    dataset = _load_dataset(name)
    gula_rules = GULA.fit(dataset)
    pride_rules = PRIDE.fit(dataset)

    assert_correct(dataset, gula_rules)
    assert_complete(dataset, gula_rules)
    assert_not_dominated(dataset, gula_rules)
    assert_irreducible(dataset, gula_rules)

    assert_correct(dataset, pride_rules)
    assert_complete(dataset, pride_rules)
    assert_not_dominated(dataset, pride_rules)
    assert_irreducible(dataset, pride_rules)
    assert set(pride_rules) <= set(gula_rules), (
        f"{name}: PRIDE produced a rule GULA didn't"
    )


@pytest.mark.parametrize("name", _FAST_TIER)
def test_fast_tier_networks(name):
    """<=6 var networks: all three algorithms, fully cross-checked.

    Each takes under 0.1s, all five about 0.15s.
    """
    _run_full_cross_check(name)


@pytest.mark.slow
@pytest.mark.parametrize("name", _SLOW_TIER)
def test_slow_tier_networks(name):
    """7-13 var networks: GULA/PRIDE only, no BruteForce.

    ``BruteForce`` would take ~4s at 9 vars and ~26s at 10, more
    than this whole tier's 11s; see the module docstring.
    """
    _run_gula_pride_check(name)


# Pruning leaves positive rows uncovered, as it should; the warning
# itself is tested in test_coverage_warning.py.
@pytest.mark.filterwarnings("ignore:.*matched by no rule of their head")
@pytest.mark.parametrize("name", _FAST_TIER)
@pytest.mark.parametrize("fraction", [1.0, 0.5, 0.2])
@pytest.mark.parametrize("min_support", [1, 2, 0.3])
def test_gula_min_support_is_the_unrestricted_rules_filtered_by_support(
    name, fraction, min_support
):
    """``GULA``'s support pruning keeps the unrestricted rules over it.

    Exactly the unrestricted rules whose support clears the threshold,
    no more, no fewer.

    Holds because the unrestricted rules are the minimal consistent
    ones, and a more general rule never has less support than a more
    specific one: pruning below threshold can't drop a rule's
    dominator without also dropping the rule. That makes a plain
    rescan of each unrestricted rule against its head's positives an
    oracle independent of how ``GULA`` tracks support internally.
    Partial samples are where pruning bites hardest — fewer negatives
    leave more low-support rules standing.
    """
    df = _bnet.generate_transitions(_DATA_DIR / f"{name}.bnet")
    dataset = Dataset(
        df.sample(frac=fraction, random_state=0),
        features=[column for column in df.columns if column.endswith("_t")],
        targets=[column for column in df.columns if column.endswith("_t+1")],
    )

    expected = set()
    for rule in GULA.fit(dataset):
        positives = dataset.positives(*rule.head)
        threshold = GULA._resolve_min_support(min_support, len(positives))
        if sum(rule.matches(positive) for positive in positives) >= threshold:
            expected.add(rule)

    assert set(GULA.fit(dataset, min_support=min_support)) == expected


@pytest.mark.parametrize("name", ["arellano_rootstem", "davidich_yeast", "raf"])
@pytest.mark.parametrize("with_values", [False, True])
def test_influences_restrict_without_changing_the_rules(name, with_values):
    """``GULA`` limited to the network's wiring learns the same rules.

    The wiring, as ``_bnet.influences`` reads it, is exact background
    knowledge. The first two networks have constant nodes, which take
    one value only, so ``influences`` must leave out the head of the
    other.
    """
    dataset = _load_dataset(name)
    allowed = _bnet.influences(_DATA_DIR / f"{name}.bnet", with_values=with_values)
    assert GULA.fit(dataset, allowed_conditions=allowed) == GULA.fit(dataset)


def test_manifest_matches_real_variable_counts():
    """The tier lists above match the vendored ``.bnet`` files.

    A future file swap that changes a network's variable count would
    otherwise leave it in a tier priced for another size.
    """
    for name in _FAST_TIER:
        n = _bnet.num_variables(_DATA_DIR / f"{name}.bnet")
        assert n <= 6, f"{name} has {n} vars, expected <=6 for the fast tier"
    for name in _SLOW_TIER:
        n = _bnet.num_variables(_DATA_DIR / f"{name}.bnet")
        assert 7 <= n <= 13, f"{name} has {n} vars, expected 7-13 for the slow tier"
