"""Rule-set properties the algorithms' tests check on every fixture.

Correctness, completeness and minimality, as the algorithms' module
docstrings define them, written once for ``test_learners.py`` and
``test_boolean_networks.py``.
Each takes the dataset and the rules learned from it and fails with
the first offending rule or state.
"""

from itertools import product


def assert_heads_in_domain(dataset, rules):
    """Every rule concludes a target on a value of its domain.

    Args:
        dataset: The dataset the rules were learned from.
        rules: The learned rules.
    """
    for rule in rules:
        variable, value = rule.head
        assert variable in dataset.targets
        assert value in dataset.domain(variable)


def assert_correct(dataset, rules):
    """No rule matches a negative example of its own head.

    Args:
        dataset: The dataset the rules were learned from.
        rules: The learned rules.
    """
    for target in dataset.targets:
        for value in dataset.domain(target):
            negatives = dataset.negatives(target, value)
            for rule in rules:
                if rule.head != (target, value):
                    continue
                for negative in negatives:
                    assert not rule.matches(negative), (
                        f"{rule} incorrectly matches negative state {negative}"
                    )


def assert_complete(dataset, rules):
    """Every feature-domain combination is matched, for every target.

    Only a guarantee of ``PRIDE``'s on exhaustive data, where every
    combination is a row.

    Args:
        dataset: The dataset the rules were learned from.
        rules: The learned rules.
    """
    domains = [dataset.domain(feature) for feature in dataset.features]
    for combination in product(*domains):
        state = dict(zip(dataset.features, combination, strict=True))
        for target in dataset.targets:
            assert any(
                rule.head[0] == target and rule.matches(state) for rule in rules
            ), f"no rule concludes {target} for state {state}"


def assert_not_dominated(dataset, rules):
    """No rule is dominated by a more general rule for the same head.

    Args:
        dataset: The dataset the rules were learned from.
        rules: The learned rules.
    """
    for target in dataset.targets:
        for value in dataset.domain(target):
            same_head = [rule for rule in rules if rule.head == (target, value)]
            for rule in same_head:
                for other in same_head:
                    if other != rule:
                        assert not other.is_more_general_than(rule), (
                            f"{other} dominates {rule}, should have been pruned"
                        )


def assert_irreducible(dataset, rules):
    """No condition of any rule can be dropped without matching a negative.

    Non-domination alone passes a rule with a needless condition when
    no other rule is more general than it; this checks each condition.

    Args:
        dataset: The dataset the rules were learned from.
        rules: The learned rules.
    """
    for rule in rules:
        negatives = dataset.negatives(*rule.head)
        for variable in rule.body:
            general = rule.without_condition(variable)
            assert any(general.matches(negative) for negative in negatives), (
                f"{rule} matches no negative without its condition on "
                f"{variable}, so that condition is not needed"
            )
