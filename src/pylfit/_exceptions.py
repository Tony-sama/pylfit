"""The warning classes pylfit emits, so callers can filter them by class.

Every warning pylfit emits is a ``PylfitWarning``, itself a
``UserWarning``, so a filter on ``UserWarning`` still catches them all.
A kind of warning gets a class of its own when a caller has a reason to
silence it alone and no option of pylfit's does it.
"""


class PylfitWarning(UserWarning):
    """Base class of every warning pylfit emits.

    Examples:
        Silence every pylfit warning in a block:

        >>> import warnings
        >>> import pylfit
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore", pylfit.PylfitWarning)
    """


class ConflictWarning(PylfitWarning):
    """Two rows with the same features, all known, differ on a target.

    Rows with an unknown feature are not compared, since the unknown
    could hide a difference between their states;
    ``Dataset.conflicting_targets`` lists the targets that conflict.

    ``Dataset`` emits it unless built with
    ``assume_nondeterministic=True``. Filter it to keep the default on
    data whose conflicts are noise, where ``assume_nondeterministic``
    would describe the system wrongly.

    Examples:
        >>> import warnings
        >>> import pandas as pd
        >>> from pylfit import ConflictWarning, Dataset
        >>> df = pd.DataFrame({"a": ["0", "0"], "y": ["0", "1"]})
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore", ConflictWarning)
        ...     dataset = Dataset(df, features=["a"], targets=["y"])
        >>> dataset.conflicting_targets
        ['y']
    """
