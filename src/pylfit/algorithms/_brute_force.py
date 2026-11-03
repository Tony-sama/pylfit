"""BruteForce, the exhaustive reference rule learner.

``BruteForce`` enumerates every possible rule for each (target, value)
pair and keeps those that are consistent and minimal: they match no
negative example, and dropping any one of their conditions would make
them match one. Each candidate is judged on its own, so only the kept
rules are ever held in memory.

The learned rule set is correct, complete and minimal in the same
sense as ``GULA``'s, and is identical to it: ``GULA`` is tested against
this class. The cost is exponential in the number of features and the
size of their domains, so it suits small problems and testing, not
everyday use.

Unknown feature values (``None`` in ``Dataset``'s states) never
satisfy a condition, so a candidate is consistent when it matches no
negative example as recorded. The rules are again ``GULA``'s. A
positive row that no known feature separates from some negative can be
matched by no consistent rule; ``fit`` warns when that happens.

Why it works this way: :doc:`/explanation/about-algorithms`.
"""

import math
import time
from collections.abc import Iterator, Mapping, Sequence
from functools import partial
from itertools import product

from pylfit._dataset import Dataset
from pylfit._progress import duration
from pylfit._rule import Rule
from pylfit.algorithms._algorithm import Algorithm, Heads, Uncovered


class BruteForce(Algorithm):
    """Brute-force enumeration algorithm for learning classification rules."""

    @staticmethod
    def fit(
        dataset: Dataset,
        *,
        verbose: bool = False,
        heads: Heads | None = None,
        warn_uncovered: bool = True,
        n_jobs: int | None = 1,
    ) -> list[Rule]:
        """Learns rules from a dataset.

        Args:
            dataset: The dataset to learn from.
            verbose: If True, prints progress for each (target, value)
                pair to stderr: candidate, consistent and kept rule
                counts, and elapsed time.
            heads: The (target, value) pairs to learn rules for, as in
                ``GULA.fit``, with the same ``ValueError`` if invalid.
                ``None`` learns every head.
            warn_uncovered: If True, warns when unknown values leave
                some positive rows matched by no rule of their head, as
                ``GULA.fit`` does. With ``verbose``, lists each row.
                Checked only when the dataset has unknown feature
                values.
            n_jobs: The number of processes heads are learned in at
                once: ``1`` (default) or ``None`` learns them one after
                another in this process, ``-1`` uses every CPU, ``-2``
                all but one, as in scikit-learn; ``0`` raises
                ``ValueError``, and any other value that is not an
                integer ``TypeError``. The rules are the same whatever
                the value. No more processes are started than there are
                heads to learn, nor more than 61 on Windows, the most it
                allows. Starting processes takes time, so it pays off
                only when a fit takes seconds or more; see
                :doc:`/explanation/about-algorithms`. With ``verbose``
                and more than one process, prints a line per head as it
                finishes, instead of each head's progress. A script
                fitting with ``n_jobs`` other than 1 must do so under
                ``if __name__ == "__main__":``, or the fit fails: on
                macOS, on Windows and on Linux from Python 3.14, each
                worker process imports the script again.

        Returns:
            The learned rules, sorted by text, without duplicates. Wrap
            them in a model to predict, for example
            ``RuleSetModel(dataset, rules)``.

        Raises:
            ValueError: If ``heads`` is invalid, as in ``GULA.fit``, or
                ``n_jobs`` is ``0``.
            TypeError: If ``dataset`` is not a ``Dataset``, ``heads`` is
                not iterable or holds a value that is not a ``str``,
                ``verbose`` or ``warn_uncovered`` is not a ``bool``, or
                ``n_jobs`` is neither an integer nor ``None``.
            ~concurrent.futures.process.BrokenProcessPool: If a worker
                process dies or can't start, as when a script fitting
                with ``n_jobs`` other than 1 lacks the
                ``if __name__ == "__main__":`` guard (see ``n_jobs``).

        See Also:
            GULA: Learns the same rules, much faster.

        Examples:
            >>> import pandas as pd
            >>> from pylfit import Dataset, BruteForce
            >>> df = pd.DataFrame(
            ...     {
            ...         "a": ["0", "0", "1", "1"],
            ...         "b": ["0", "1", "0", "1"],
            ...         "y": ["0", "1", "1", "1"],
            ...     }
            ... )
            >>> dataset = Dataset(df, features=["a", "b"], targets=["y"])
            >>> rules = BruteForce.fit(dataset)
            >>> [rule.to_string() for rule in rules]
            ['y(0) :- a(0), b(0).', 'y(1) :- a(1).', 'y(1) :- b(1).']
        """
        learn = partial(BruteForce._learn_head, warn_uncovered=warn_uncovered)
        return BruteForce._fit_heads(
            dataset, learn, heads, verbose, warn_uncovered, n_jobs
        )

    @staticmethod
    def _learn_head(
        dataset: Dataset,
        target: str,
        value: str,
        verbose: bool,
        *,
        warn_uncovered: bool,
    ) -> tuple[list[Rule], list[Uncovered]]:
        """Learns one head for ``fit``'s shared loop, with its uncovered rows.

        Args:
            dataset: The dataset to learn from.
            target: The head's target.
            value: The head's value.
            verbose: If True, prints this head's progress to stderr.
            warn_uncovered: Whether to look for uncovered rows at all.

        Returns:
            The head's rules, and its positive rows none of them
            matches, each with why; looked for only on data with unknown
            values.
        """
        rules = BruteForce._fit_target_value(dataset, target, value, verbose=verbose)
        uncovered = (
            BruteForce._uncovered(dataset, target, value, rules, [])
            if warn_uncovered and dataset._has_unknowns
            else []
        )
        return rules, uncovered

    @staticmethod
    def _fit_target_value(
        dataset: Dataset, target: str, value: str, verbose: bool = False
    ) -> list[Rule]:
        """Learns the minimal consistent rules concluding target=value.

        Candidates are generated lazily and judged one at a time against
        the negatives, so neither all candidates nor all consistent
        rules are ever held in memory: only the rules kept.

        Args:
            dataset: The classification dataset to learn from.
            target: The target variable to conclude on.
            value: The value of ``target`` the learned rules conclude.
            verbose: If True, prints progress for this (target, value)
                pair to stderr.

        Returns:
            The candidate rules that are consistent and minimal.
        """
        start = time.perf_counter()
        head = f"{target}={value}"
        # Each candidate is judged against every negative: a repeated
        # row would only repeat the work.
        negatives = dataset._distinct_negative_states(target, value)
        total = BruteForce._candidate_count(dataset)
        progress = BruteForce._progress(f"{head}: checking candidates", total)
        kept = []
        consistent = 0
        for i, rule in enumerate(BruteForce._candidates(dataset, target, value), 1):
            is_consistent, is_minimal = BruteForce._judge(rule, negatives)
            consistent += is_consistent
            if is_minimal:
                kept.append(rule)
            if verbose:
                progress.update(i)

        if verbose:
            elapsed = time.perf_counter() - start
            BruteForce._log(
                f"{head}: {total} candidates, {consistent} consistent, "
                f"{len(kept)} rules kept ({duration(elapsed)} total)"
            )

        return kept

    @staticmethod
    def _judge(
        rule: Rule, negatives: Sequence[Mapping[str, str | None]]
    ) -> tuple[bool, bool]:
        """Tells whether a rule is consistent, and whether it is also minimal.

        One pass over the negatives answers both
        (``_needed_conditions``). A consistent rule is minimal when
        every condition is needed.

        Checking single drops is enough. Dropping conditions only widens
        what a rule matches, so if some more general rule is consistent,
        so is every rule between the two, the one-condition drop
        included.

        Args:
            rule: The candidate rule.
            negatives: The feature states where the target never has the
                rule's head value.

        Returns:
            ``(consistent, minimal)``; ``minimal`` implies
            ``consistent``.
        """
        needed = BruteForce._needed_conditions(rule, negatives)
        if needed is None:
            return False, False
        return True, len(needed) == len(rule.body)

    @staticmethod
    def _candidate_count(dataset: Dataset) -> int:
        """Counts the rules ``_candidates`` generates for one head.

        Each feature is unconstrained or fixed to one of its values, so
        the count is the product of ``domain size + 1`` over features.

        Args:
            dataset: The classification dataset to learn from.

        Returns:
            The number of candidate rules per head.
        """
        return math.prod(
            len(dataset.domain(feature)) + 1 for feature in dataset.features
        )

    @staticmethod
    def _candidates(dataset: Dataset, target: str, value: str) -> Iterator[Rule]:
        """Generates every possible rule concluding target=value, lazily.

        Each feature is either left unconstrained or fixed to one value
        of its domain, producing every rule from the fully general one
        (empty body) to the fully specific ones.

        Args:
            dataset: The classification dataset to learn from.
            target: The target variable to conclude on.
            value: The value of ``target`` the candidate rules conclude.

        Yields:
            Each candidate rule, whether consistent with the dataset or
            not.
        """
        options = [
            [None] + [(feature, val) for val in dataset.domain(feature)]
            for feature in dataset.features
        ]
        for combination in product(*options):
            body = dict(condition for condition in combination if condition is not None)
            yield Rule(head=(target, value), body=body)
