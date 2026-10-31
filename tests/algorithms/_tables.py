"""Random tables with unknown values, shared by the algorithms' tests."""

import random

import pandas as pd

from pylfit import Dataset


def blanked_frame(seed, rate=0.2, deterministic=False):
    """A random 40-row table over three features, some values blanked.

    Features ``a`` and ``c`` take ``0``-``2``, ``b`` takes ``0``-``1``;
    each feature value is blanked (``None``) with probability ``rate``,
    ``y`` never. ``y`` is random, so rows can conflict, unless
    ``deterministic``: then ``y`` is ``1`` when ``b=1`` or when ``a=0``
    and ``c`` is not ``2``, computed before blanking.
    """
    rng = random.Random(seed)
    domains = {"a": "012", "b": "01", "c": "012"}
    records = []
    for _ in range(40):
        record = {f: rng.choice(values) for f, values in domains.items()}
        record["y"] = rng.choice("01")
        if deterministic:
            hit = (record["a"] == "0" and record["c"] != "2") or record["b"] == "1"
            record["y"] = "1" if hit else "0"
        for feature in domains:
            if rng.random() < rate:
                record[feature] = None
        records.append(record)
    return pd.DataFrame(records)


def as_question_marks(frame):
    """The table as a ``Dataset`` with each unknown written as ``"?"``.

    Returns that dataset and the ``allowed_conditions`` that leave every
    condition on ``"?"`` out. The ``"?"`` is a value here on purpose, so
    the stand-in check is off.
    """
    dataset = Dataset(
        frame.fillna("?"),
        features=["a", "b", "c"],
        targets=["y"],
        check_stand_ins=False,
    )
    known = [
        (feature, value)
        for feature in dataset.features
        for value in dataset.domain(feature)
        if value != "?"
    ]
    allowed = {("y", value): known for value in dataset.domain("y")}
    return dataset, allowed
