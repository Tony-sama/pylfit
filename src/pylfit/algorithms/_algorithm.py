"""The interface every pylfit learning algorithm implements.

An algorithm is a class with a stateless static ``fit`` that takes a
``Dataset`` and returns a list of ``Rule``, with no dependency on
``pylfit.models``: callers wrap the rules in whichever model they
choose. ``Algorithm`` declares that ``fit``, so type checkers know that
``BruteForce``, ``GULA`` and ``PRIDE`` share it, and gives subclasses
the loop over heads (``_fit_heads``, with ``n_jobs``) and the progress
helpers their ``verbose`` output goes through.

Each head, a ``(target, value)`` pair, is learned independently of the
others, so ``n_jobs`` learns several at once, each worker process
taking the next head when it finishes one. Results are joined in head
order, so the rules and any warning are the same as a sequential
run's.
"""

import math
import multiprocessing
import numbers
import os
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from warnings import warn

import numpy as np

from pylfit._checks import _check_collection
from pylfit._dataset import Dataset, _check_dataset, _check_flag
from pylfit._exceptions import PylfitWarning
from pylfit._progress import Progress, duration, log
from pylfit._rule import Rule

#: ``fit``'s ``heads``: each a ``(target, value)`` pair, or a bare
#: target name standing for every value of that target.
Heads = Collection[str | tuple[str, str]]

#: ``fit``'s ``allowed_conditions``: per head, the conditions its rules
#: may use, or ``None`` for unrestricted.
AllowedConditions = Mapping[tuple[str, str], Collection[str | tuple[str, str]] | None]

#: A positive row no rule of its head matches: the head, the row's
#: feature state, and why.
Uncovered = tuple[tuple[str, str], Mapping[str, str | None], str]

#: Learns one head: ``(dataset, target, value, verbose)`` to the head's
#: rules and its uncovered rows. Must be picklable for ``n_jobs``: a
#: ``functools.partial`` of a static method is.
HeadLearner = Callable[[Dataset, str, str, bool], tuple[list[Rule], list[Uncovered]]]


def _is_pair(entry: object) -> bool:
    """Tells whether ``entry`` is a ``(name, value)`` pair of two ``str``."""
    return (
        isinstance(entry, tuple)
        and len(entry) == 2
        and isinstance(entry[0], str)
        and isinstance(entry[1], str)
    )


def _not_a_pair(entry: object, what: str) -> str:
    """Says why ``entry`` is not a ``what`` name or ``(what, value)`` pair.

    Args:
        entry: The entry that is neither.
        what: ``"target"`` or ``"feature"``.

    Returns:
        The problem, for the caller's list of them.

    Raises:
        TypeError: If ``entry`` is a pair whose value is not a ``str``,
            a wrong type rather than a wrong name.
    """
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str):
        raise TypeError(
            f"{entry!r}: the value {entry[1]!r} is not a str; pylfit compares "
            "values as strings, convert it with str()"
        )
    return f"{entry!r}: not a {what} or a ({what}, value) pair"


# The dataset a worker process learns from, set once per worker by
# `_init_worker` rather than sent with every head.
_worker_dataset: Dataset | None = None


def _init_worker(dataset: Dataset) -> None:
    """Stores the dataset in a worker process, for ``_learn_in_worker``.

    Args:
        dataset: The dataset every head of the ``fit`` is learned from.
    """
    global _worker_dataset
    _worker_dataset = dataset


def _learn_in_worker(
    learn: HeadLearner, target: str, value: str
) -> tuple[list[Rule], list[Uncovered]]:
    """Learns one head in a worker process, without progress output.

    Args:
        learn: The per-head function.
        target: The head's target.
        value: The head's value.

    Returns:
        ``learn``'s result.
    """
    assert _worker_dataset is not None
    return learn(_worker_dataset, target, value, False)


def _broken_pool_message() -> str:
    """Says why a worker process may have died or failed to start.

    Where workers start afresh rather than as copies of this process
    (``spawn`` or ``forkserver``: macOS, Windows, Linux from Python
    3.14), each imports the main script again, and a script that fits
    outside ``if __name__ == "__main__":`` makes it fit again there,
    which ``multiprocessing`` refuses. The message names that cause
    first. Running out of memory kills a worker under any start method.

    Returns:
        The message.
    """
    guard = ""
    if multiprocessing.get_start_method() != "fork":
        guard = (
            " If a script runs this fit, it must call fit under "
            '`if __name__ == "__main__":`, as each worker process imports '
            "the script again."
        )
    return (
        f"a worker process of n_jobs died or could not start.{guard} A worker "
        "that runs out of memory dies too; a smaller n_jobs uses less."
    )


class Algorithm(ABC):
    """Base class of the learning algorithms: one static ``fit`` method.

    Subclasses implement ``fit`` and may add options after ``verbose``,
    as ``GULA`` does. Every argument but ``dataset`` is keyword-only, so
    a call means the same whichever algorithm it goes to, though their
    options differ. ``Algorithm`` itself can't be instantiated.
    """

    @staticmethod
    @abstractmethod
    def fit(dataset: Dataset, *, verbose: bool = False) -> list[Rule]:
        """Learns rules from a dataset.

        Args:
            dataset: The dataset to learn from.
            verbose: If True, prints progress to stderr.

        Returns:
            The learned rules, sorted by text, without duplicates. Wrap
            them in a model to predict, for example
            ``RuleSetModel(dataset, rules)``.

        Raises:
            NotImplementedError: Always: a subclass that doesn't define
                ``fit`` fails loudly when called, not with ``None``.
        """
        raise NotImplementedError

    # --- The loop over heads ---

    @classmethod
    def _fit_heads(
        cls,
        dataset: Dataset,
        learn: HeadLearner,
        heads: Heads | None = None,
        verbose: bool = False,
        warn_uncovered: bool = True,
        n_jobs: int | None = 1,
    ) -> list[Rule]:
        """Learns every selected head with ``learn`` and joins the results.

        Heads are taken target by target, in ``dataset.targets`` order,
        and each target's values sorted, so the order is the same on
        every run. With ``n_jobs`` other than 1, heads are learned in
        worker processes, each given the dataset once; each worker
        learns silently, and with ``verbose`` this process prints a line
        per head as it finishes. Under the ``fork`` start method, the
        dataset's row caches are built here first, so that the workers
        inherit them and the next fit finds them.

        Args:
            dataset: The dataset to learn from.
            learn: The per-head function, options already bound.
            heads: ``fit``'s ``heads``; ``None`` for every head.
            verbose: If True, prints progress to stderr.
            warn_uncovered: If True, warns about the uncovered rows
                ``learn`` reports.
            n_jobs: ``fit``'s ``n_jobs``.

        Returns:
            The rules of every selected head, without duplicates, sorted
            as ``sorted(rules)`` sorts them: by text.

        Raises:
            BrokenProcessPool: If a worker process dies or can't start,
                with a message naming the likely causes.
            BaseException: Whatever learning a head raised, or a Ctrl-C
                while waiting, once the heads still queued are
                cancelled.
        """
        _check_dataset(dataset)
        _check_flag(verbose, "verbose")
        _check_flag(warn_uncovered, "warn_uncovered")
        cls._check_n_jobs(n_jobs)
        selected = None if heads is None else cls._selected_heads(dataset, heads)
        todo = [
            (target, value)
            for target in dataset.targets
            for value in dataset.domain(target)
            if selected is None or (target, value) in selected
        ]
        workers = min(cls._resolve_n_jobs(n_jobs), len(todo))
        start = time.perf_counter()
        if workers <= 1:
            results = [learn(dataset, target, value, verbose) for target, value in todo]
        else:
            if multiprocessing.get_start_method() == "fork":
                # Workers copied from this process inherit the row
                # caches built here, and this process keeps them for
                # the next fit; built in the workers, they would be
                # lost with the pool. Other start methods send the
                # dataset without its caches, so there it would only
                # add time.
                for target, value in todo:
                    dataset._positive_states(target, value)
                    dataset._negative_states(target, value)
            with ProcessPoolExecutor(
                workers, initializer=_init_worker, initargs=(dataset,)
            ) as pool:
                try:
                    futures = {}
                    for target, value in todo:
                        future = pool.submit(_learn_in_worker, learn, target, value)
                        futures[future] = (target, value)
                except (OSError, EOFError) as error:
                    # Under `forkserver` a worker starts from a server
                    # that imports the script first; when that import
                    # fails, as it does without the `__main__` guard,
                    # the server closes its connection and `submit`
                    # raises.
                    raise BrokenProcessPool(_broken_pool_message()) from error
                try:
                    if verbose:
                        for done, future in enumerate(as_completed(futures), 1):
                            target, value = futures[future]
                            cls._log(
                                f"{target}={value}: {len(future.result()[0])} rules "
                                f"({done}/{len(todo)} heads, "
                                f"{duration(time.perf_counter() - start)})"
                            )
                    results = [future.result() for future in futures]
                except BaseException as error:
                    # Ctrl-C or a failed head: leaving the `with` would
                    # otherwise wait for every queued head to be
                    # learned.
                    pool.shutdown(cancel_futures=True)
                    if isinstance(error, BrokenProcessPool):
                        raise BrokenProcessPool(_broken_pool_message()) from error
                    raise
        rules = [rule for head_rules, _ in results for rule in head_rules]
        uncovered = [row for _, head_uncovered in results for row in head_uncovered]

        if verbose:
            cls._log(
                f"all (target, value) pairs done: {len(rules)} rules "
                f"in {duration(time.perf_counter() - start)}"
                + (f" on {workers} processes" if workers > 1 else "")
            )
        if warn_uncovered and uncovered:
            total = sum(len(dataset._positive_states(*head)) for head in todo)
            cls._warn_uncovered(uncovered, total, verbose)
        # The key computed once per rule rather than at every
        # comparison.
        return sorted(set(rules), key=Rule._sort_key)

    @staticmethod
    def _check_n_jobs(n_jobs: object) -> None:
        """Validates ``fit``'s ``n_jobs``.

        Args:
            n_jobs: The value to check.

        Raises:
            TypeError: If ``n_jobs`` is neither an integer (a NumPy
                integer counts as one) nor ``None``.
            ValueError: If ``n_jobs`` is zero.
        """
        if n_jobs is None:
            return
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, numbers.Integral):
            raise TypeError(
                f"n_jobs must be an int or None, got {n_jobs!r} "
                f"({type(n_jobs).__name__})"
            )
        if n_jobs == 0:
            raise ValueError("n_jobs must be nonzero, got 0")

    @staticmethod
    def _resolve_n_jobs(n_jobs: int | None) -> int:
        """Turns ``n_jobs`` into a process count, as scikit-learn does.

        ``None`` is 1. A positive value is the count itself; ``-1`` is
        every CPU this process may use, ``-2`` all but one, and so on,
        never below 1.
        On Windows the count is capped at 61, the most
        ``ProcessPoolExecutor`` accepts there.

        Args:
            n_jobs: A nonzero ``int``, or ``None``.

        Returns:
            The number of processes, at least 1.
        """
        if n_jobs is None:
            return 1
        limit = 61 if sys.platform == "win32" else None
        if n_jobs > 0:
            return int(n_jobs) if limit is None else min(int(n_jobs), limit)
        if hasattr(os, "process_cpu_count"):  # Python 3.13+
            cpus = os.process_cpu_count() or 1
        elif hasattr(os, "sched_getaffinity"):  # Linux
            cpus = len(os.sched_getaffinity(0))
        else:
            cpus = os.cpu_count() or 1
        processes = max(1, cpus + 1 + n_jobs)
        return processes if limit is None else min(processes, limit)

    @staticmethod
    def _needed_conditions(
        rule: Rule, negatives: Sequence[Mapping[str, str | None]]
    ) -> set[str] | None:
        """Finds a rule's needed conditions, or ``None`` if it is inconsistent.

        A negative that violates none of the rule's conditions is
        matched by it: the rule is inconsistent. A negative that
        violates exactly one condition is matched once that condition is
        dropped: the condition is needed. It stays needed in every rule
        obtained by dropping other conditions, since that negative still
        violates nothing else there.

        Args:
            rule: The rule to check.
            negatives: The feature states where the target never has the
                rule's head value.

        Returns:
            The variables of the needed conditions, or ``None`` if the
            rule matches a negative.
        """
        conditions = list(rule.body.items())
        needed: set[str] = set()
        for negative in negatives:
            # The one condition this negative violates, if exactly one:
            # a second violation ends the scan, since such a negative
            # says nothing about either condition.
            violated = None
            for variable, value in conditions:
                if negative[variable] != value:
                    if violated is not None:
                        break
                    violated = variable
            else:
                if violated is None:
                    return None
                needed.add(violated)
        return needed

    @staticmethod
    def _selected_heads(dataset: Dataset, heads: Heads) -> set[tuple[str, str]]:
        """Checks ``fit``'s ``heads``, expanded into (target, value) pairs.

        Every problem is reported at once, as for
        ``allowed_conditions``: a misspelt head would otherwise be
        skipped without a word.

        Args:
            dataset: The dataset ``fit`` learns from.
            heads: ``fit``'s argument of the same name.

        Returns:
            The (target, value) pairs to learn.

        Raises:
            TypeError: If ``heads`` is a single string, a mapping or
                not iterable, or holds a ``(target, value)`` pair whose
                value is not a ``str``.
            ValueError: If ``heads`` holds an entry that is neither a
                target name nor a
                ``(target, value)`` pair, or names a target or value
                ``dataset`` lacks.
        """
        _check_collection("heads", heads)
        if isinstance(heads, Mapping):
            raise TypeError(
                "heads must be a collection of targets or (target, value) "
                "pairs, such as ['y'] or [('y', '1')], not a mapping"
            )
        if not isinstance(heads, Iterable):
            raise TypeError(
                "heads must be a collection of targets or (target, value) "
                f"pairs, such as ['y'] or [('y', '1')], got {type(heads).__name__}"
            )
        selected: set[tuple[str, str]] = set()
        problems: list[str] = []
        for head in heads:
            if not (isinstance(head, str) or _is_pair(head)):
                problems.append(_not_a_pair(head, "target"))
                continue
            target = head if isinstance(head, str) else head[0]
            if target not in dataset.targets:
                problems.append(f"{head!r}: {target!r} is not a target")
            elif isinstance(head, str):
                selected.update((target, value) for value in dataset.domain(target))
            elif head[1] not in dataset.domain(target):
                problems.append(f"{head!r}: {head[1]!r} is not a value of {target!r}")
            else:
                selected.add((target, head[1]))
        if problems:
            raise ValueError("invalid heads: " + "; ".join(problems))
        return selected

    @staticmethod
    def _check_allowed_conditions(
        dataset: Dataset,
        allowed_conditions: AllowedConditions,
    ) -> AllowedConditions:
        """Rejects an ``allowed_conditions`` naming what ``dataset`` lacks.

        Without this check a misspelt head would be skipped and its head
        learned unrestricted, and an unknown value would become a rule
        condition no state can meet. Every problem is reported at once.

        Args:
            dataset: The dataset ``fit`` learns from.
            allowed_conditions: ``fit``'s argument of the same name.

        Returns:
            The same mapping with each head's conditions read into a
            tuple, so a one-pass iterable such as a generator, used up
            by this check, still reaches the learner whole.

        Raises:
            TypeError: If ``allowed_conditions`` is not a mapping, a
                head's conditions are neither ``None`` nor iterable, or
                a ``(feature, value)`` pair's value is not a ``str``.
            ValueError: If a key is not a ``(target, value)`` pair, a
                condition is neither a feature name nor a
                ``(feature, value)`` pair, any head, feature or value is
                unknown, or a head is given a ``str`` or a mapping
                rather than a collection of conditions.
        """
        if not isinstance(allowed_conditions, Mapping):
            raise TypeError(
                "allowed_conditions must be a mapping of head to conditions, "
                "such as {('y', '1'): ['a']}, got "
                f"{type(allowed_conditions).__name__}"
            )
        problems: list[str] = []
        checked: dict[tuple[str, str], tuple[str | tuple[str, str], ...] | None] = {}
        for head, given in allowed_conditions.items():
            if not _is_pair(head):
                problems.append(
                    f"head {head!r}: not a (target, value) pair, such as ('y', '1')"
                )
                continue
            target, value = head
            if given is not None and not isinstance(given, Iterable):
                raise TypeError(
                    f"allowed_conditions: head {head!r}: conditions must be a "
                    "collection of features or (feature, value) pairs, such as "
                    f"['a'] or [('a', '0')], got {type(given).__name__}"
                )
            refused = isinstance(given, str | Mapping)
            conditions = None if given is None or refused else tuple(given)
            checked[head] = conditions
            if target not in dataset.targets:
                problems.append(f"head {head!r}: {target!r} is not a target")
                continue
            if value not in dataset.domain(target):
                problems.append(
                    f"head {head!r}: {value!r} is not a value of {target!r}"
                )
            if isinstance(given, str):
                problems.append(
                    f"head {head!r}: conditions must be a collection, such as "
                    f"[{given!r}], not a single string"
                )
                continue
            if isinstance(given, Mapping):
                problems.append(
                    f"head {head!r}: conditions must be a collection of "
                    "features or (feature, value) pairs, such as "
                    f"{list(given.items())!r}, not a mapping"
                )
                continue
            if conditions is None:
                continue
            for condition in conditions:
                if not (isinstance(condition, str) or _is_pair(condition)):
                    problems.append(
                        f"head {head!r}: " + _not_a_pair(condition, "feature")
                    )
                    continue
                feature = condition if isinstance(condition, str) else condition[0]
                if feature not in dataset.features:
                    problems.append(f"head {head!r}: {feature!r} is not a feature")
                elif not isinstance(condition, str) and (
                    condition[1] not in dataset.domain(feature)
                ):
                    problems.append(
                        f"head {head!r}: {condition[1]!r} is not a value of {feature!r}"
                    )
        if problems:
            raise ValueError("invalid allowed_conditions: " + "; ".join(problems))
        return checked

    @staticmethod
    def _distinct(
        negatives: Sequence[Mapping[str, str | None]], features: list[str]
    ) -> list[Mapping[str, str | None]]:
        """Drops repeated feature states, keeping each one's first occurrence.

        ``Dataset.negatives`` returns one entry per row; a repeated
        state would be processed again for nothing, since every kept
        rule is already consistent with it.

        Args:
            negatives: The head's negatives, one per row.
            features: The feature names a state is keyed on.

        Returns:
            Each distinct state once, in first-seen order.
        """
        seen: set[tuple[str | None, ...]] = set()
        distinct = []
        for negative in negatives:
            key = tuple(negative[feature] for feature in features)
            if key not in seen:
                seen.add(key)
                distinct.append(negative)
        return distinct

    @staticmethod
    def _expand_allowed(
        allowed_conditions: Collection[str | tuple[str, str]], dataset: Dataset
    ) -> list[tuple[str, str]]:
        """Expands one head's allowed conditions into sorted pairs, once.

        A bare feature name stands for every value of that feature.
        Sorted, and deduplicated in case both forms name a pair, so the
        specializations come in the same order on every run.

        Args:
            allowed_conditions: One head's entry of ``fit``'s mapping.
            dataset: Used to expand a bare feature name to its domain.

        Returns:
            Every allowed ``(feature, value)`` pair, sorted.
        """
        expanded: set[tuple[str, str]] = set()
        for entry in allowed_conditions:
            if isinstance(entry, str):
                expanded.update((entry, v) for v in dataset.domain(entry))
            else:
                expanded.add(entry)
        return sorted(expanded)

    @staticmethod
    def _state_key(
        features: list[str], unknowns: bool = True
    ) -> Callable[[Mapping[str, str | None]], tuple[object, ...]]:
        """Makes a sort key: a state's values in ``features`` order.

        An unknown sorts before every value. ``None`` can't be compared
        with a ``str``, so with ``unknowns`` each value is paired with
        whether it is known. That pairing makes a sort about three times
        slower, so without unknowns the key is the plain tuple of
        values, which orders complete states the same way.

        Args:
            features: The feature order to read the values in.
            unknowns: Whether a state may hold ``None``.

        Returns:
            A function from a state to its sort key.
        """
        if not unknowns:
            return lambda state: tuple(state[feature] for feature in features)
        return lambda state: tuple(
            (value is not None, value or "")
            for value in (state[feature] for feature in features)
        )

    @staticmethod
    def _check_min_support(min_support: object, name: str = "min_support") -> None:
        """Rejects a support threshold that is not a valid count or fraction.

        Args:
            min_support: ``fit``'s argument ``name``, not ``None``.
            name: The argument's name, for the error message.

        Raises:
            ValueError: If ``min_support`` is a negative ``int`` or a
                ``float`` outside ``(0, 1]``.
            TypeError: If ``min_support`` is neither an integer nor a
                float, NumPy's included; ``bool`` is refused.
        """
        if isinstance(min_support, bool) or not isinstance(
            min_support, numbers.Integral | float | np.floating
        ):
            raise TypeError(
                f"{name} must be an int or float, got {type(min_support).__name__}"
            )
        if isinstance(min_support, float | np.floating) and not (0 < min_support <= 1):
            raise ValueError(f"{name} fraction must be in (0, 1], got {min_support}")
        if isinstance(min_support, numbers.Integral) and min_support < 0:
            raise ValueError(f"{name} count must be >= 0, got {min_support}")

    @staticmethod
    def _resolve_min_support(min_support: int | float, total_positives: int) -> int:
        """Converts a valid ``min_support`` into an absolute positive count.

        Args:
            min_support: An absolute count (``int``) or a fraction of
                ``total_positives`` (``float`` in ``(0, 1]``), already
                checked by ``_check_min_support``.
            total_positives: The head's total number of positives.

        Returns:
            The minimum number of positives a rule must match to be
            kept, rounded up when ``min_support`` is a fraction. The
            product is rounded to 9 decimals first: in binary floating
            point it can land just above a whole number (``0.28 * 25``
            and ``0.07 * 100`` are both ``7.000000000000001``), which
            would demand one row too many.
        """
        if isinstance(min_support, float | np.floating):
            return math.ceil(round(float(min_support) * total_positives, 9))
        return int(min_support)

    # --- Coverage of the positive rows ---

    #: Why a positive row can be left uncovered, as ``_warn_uncovered``
    #: words it.
    _UNKNOWN_CAUSE = "unknown values"

    @staticmethod
    def _inseparable(
        state: Mapping[str, str | None],
        negatives: Sequence[Mapping[str, str | None]],
        features: list[str],
    ) -> bool:
        """Tells whether a negative equals ``state`` on its known features.

        Then no rule of the state's head can match it: a rule matching
        it tests only features it knows, with its values, so it also
        matches that negative. Only a state with unknowns can be
        inseparable, since ``Dataset.negatives`` never holds a fully
        known positive.

        Args:
            state: A positive example, ``None`` where unknown.
            negatives: The negatives of the same head.
            features: The features to compare on.

        Returns:
            True if no known feature of ``state`` separates it from some
            negative.
        """
        known = [feature for feature in features if state[feature] is not None]
        return any(
            all(negative[feature] == state[feature] for feature in known)
            for negative in negatives
        )

    @staticmethod
    def _uncovered(
        dataset: Dataset,
        target: str,
        value: str,
        rules: Sequence[Rule],
        restrictions: Sequence[str],
    ) -> list[Uncovered]:
        """Finds a head's positive rows no rule of that head matches, with why.

        Only the rows that can be uncovered are checked: those with an
        unknown value when nothing restricts the search, since every
        other row is then covered, and every positive row otherwise.

        Args:
            dataset: The dataset learned from.
            target: The head's target.
            value: The head's value.
            rules: The rules learned for the head.
            restrictions: The names of the options that restricted this
                head's search, such as ``"min_support"``; empty if none.

        Returns:
            Each uncovered row as its head, its feature state and its
            cause: ``_UNKNOWN_CAUSE`` when no known feature separates it
            from a negative, otherwise the restrictions, joined by
            ``" or "``.
        """
        positives = dataset._positive_states(target, value)
        if not restrictions:
            positives = [p for p in positives if None in p.values()]
        if not positives:
            return []
        features = dataset.features
        negatives = Algorithm._distinct(
            dataset._negative_states(target, value), features
        )
        restricted = " or ".join(restrictions)
        # Per distinct state, since repeated rows get the same answer.
        causes: dict[tuple[str | None, ...], str | None] = {}
        uncovered: list[Uncovered] = []
        for state in positives:
            key = tuple(state[feature] for feature in features)
            if key not in causes:
                if any(rule.matches(state) for rule in rules):
                    causes[key] = None
                elif not restrictions or (
                    None in key and Algorithm._inseparable(state, negatives, features)
                ):
                    causes[key] = Algorithm._UNKNOWN_CAUSE
                else:
                    causes[key] = restricted
            cause = causes[key]
            if cause is not None:
                uncovered.append(((target, value), state, cause))
        return uncovered

    @classmethod
    def _warn_uncovered(
        cls,
        uncovered: Sequence[Uncovered],
        total: int,
        verbose: bool,
    ) -> None:
        """Warns that some positive rows are matched by no rule of their head.

        One warning per ``fit``, after every head is learned: the share
        of positive rows left uncovered, split by cause. With
        ``verbose``, it also lists each row.

        Args:
            uncovered: Each uncovered row as its head, its feature state
                and its cause: ``_UNKNOWN_CAUSE``, or the options that
                left it out, such as ``"min_support"``.
            total: The number of positive rows of the heads learned.
            verbose: If True, lists every uncovered row.
        """
        causes: dict[str, int] = {}
        for _, _, cause in uncovered:
            causes[cause] = causes.get(cause, 0) + 1
        parts = []
        for cause, count in sorted(causes.items()):
            if cause == cls._UNKNOWN_CAUSE:
                parts.append(
                    f"{count} because of unknown values: no known feature "
                    "separates them from a negative example, so no rule can "
                    "match them"
                )
            else:
                parts.append(f"{count} left out by {cause}")
        message = (
            f"{cls.__name__}: {len(uncovered)} of {total} positive rows "
            f"({len(uncovered) / total:.1%}) are matched by no rule of their "
            f"head: {'; '.join(parts)}. Pass warn_uncovered=False to silence "
            "this warning."
        )
        if verbose:
            message += "".join(
                f"\n  {target}={value}: {dict(state)} ({cause})"
                for (target, value), state, cause in uncovered
            )
        else:
            message += " Pass verbose=True to list them."
        # stacklevel 4: past this helper, `_fit_heads` and `fit`, to the
        # caller's line.
        warn(message, PylfitWarning, stacklevel=4)

    @classmethod
    def _log(cls, message: str, indent: bool = False) -> None:
        """Prints a ``[ClassName] message`` line to stderr.

        Args:
            message: The text to print, after the tag.
            indent: If True, adds extra indentation for lines nested
                under a named stage.
        """
        log(cls.__name__, message, indent=indent)

    @classmethod
    def _progress(cls, stage: str, total: int) -> Progress:
        """Starts throttled progress reporting for one stage of work.

        Args:
            stage: What is being counted, shown before the count.
            total: The number of items in the stage.

        Returns:
            A ``Progress`` tagged with the class name; call its
            ``update(current)`` after each item.
        """
        return Progress(cls.__name__, stage, total)
