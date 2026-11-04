"""Everything pylfit returns, prints or writes is the same on every run.

Python seeds its string hashing per process (``PYTHONHASHSEED``), so a
``set`` of strings iterates in a different order in each process. One
script runs a whole workflow, writing every output pylfit gives, its
errors and warnings included, in a subprocess per hash seed; the
outputs must be byte for byte the same.
SymPy runs its own suite under random hash seeds for the same reason.

The script's error cases name three of each kind of problem where it
can: two names come out in the same order under half the hash seeds,
three under one in six.

Elapsed times and the order of ``verbose`` lines under ``n_jobs`` are
exempt, so the script prints neither.
"""

import difflib

WORKFLOW = """
import json
import sys
import warnings

import pandas as pd

from pylfit import (
    GULA,
    PRIDE,
    BruteForce,
    Dataset,
    Rule,
    RuleSetModel,
    WeightedRuleSetModel,
    accuracy_by_target,
    coverage,
    find_missing_stand_ins,
    generalize,
    rule_recovery,
)


def main():
    colors = ["red", "green", "blue", "cyan", "grey", "pink"]
    rows = [
        {
            "a": colors[i % 6],
            "b": colors[(i * 5) % 6],
            "c": colors[(i * i) % 4],
            "y": colors[(i // 2) % 3],
            "z": colors[(i * 7) % 4],
        }
        for i in range(30)
    ]
    frame = pd.DataFrame(rows)
    frame.loc[3, "b"] = None
    frame.loc[5, "c"] = "?"
    frame.loc[7, "c"] = "n/a"
    out = []

    def error(call):
        try:
            call()
        except (TypeError, ValueError) as raised:
            out.append(f"{type(raised).__name__}: {raised}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out.append(repr(find_missing_stand_ins(frame)))
        dataset = Dataset(
            frame, features=["a", "b", "c"], targets=["y", "z"],
            assume_nondeterministic=True,
        )
        out.append(repr(dataset))
        out.append(repr(dataset.domain("a")))
        out.append(repr(dataset.domains()))
        out.append(repr(dataset.negatives("y", "red")))
        out.append(repr(dataset.positives("y", "red")))
        out.append(repr(dataset.conflicting_targets))
        out.append(repr(dataset.to_dataframe()))
        allowed = {("y", "red"): ["a", ("b", "blue"), ("b", "red")]}
        learned = {
            "gula": GULA.fit(dataset),
            "gula_jobs": GULA.fit(dataset, n_jobs=2),
            "gula_support": GULA.fit(dataset, min_support=2),
            "gula_allowed": GULA.fit(dataset, allowed_conditions=allowed),
            "pride": PRIDE.fit(dataset),
            "pride_allowed": PRIDE.fit(dataset, allowed_conditions=allowed),
            "brute_force": BruteForce.fit(dataset, heads=["y"]),
        }
        for name, rules in learned.items():
            out.append(name + ": " + repr(rules))
        states = frame[["a", "b", "c"]]
        # The second pair has no rule on z, so its z comes from the
        # fallback for a state no rule matches.
        for model_class, rules in (
            (RuleSetModel, learned["pride"]),
            (WeightedRuleSetModel, learned["pride"]),
            (RuleSetModel, learned["brute_force"]),
            (WeightedRuleSetModel, learned["brute_force"]),
        ):
            model = model_class(dataset, rules)
            out.append(repr(model))
            out.append(repr(model.rules))
            out.append(repr(model.domain))
            out.append(str(model))
            path = sys.argv[1] + model_class.__name__
            model.save(path)
            with open(path, encoding="utf-8") as file:
                saved = file.read()
            out.append(saved)
            out.append(str(model_class.from_file(path)))
            out.append(repr(model.predict_candidates(states.iloc[3])))
            out.append(repr(model.predict_proba(states)))
            predicted = model.predict(states, random_state=0)
            out.append(repr(predicted))
            out.append(repr(accuracy_by_target(frame, predicted, ["y", "z"])))
            if model_class is WeightedRuleSetModel:
                out.append(repr(model.predict_likelihood(states)))
                out.append(repr(model.value_frequencies))
                out.append(repr([model.get_score(rule) for rule in rules]))
            columns = frame.assign(q=1, r=2, s=3).drop(columns=["a", "b", "c"])
            error(lambda: model.predict(columns))
            repeated = pd.concat([states, states], axis=1)
            error(lambda: model.predict(repeated))
            unseen = states.assign(
                a=["black", "white", "mauve"] * 10,
                b=["black", "white", "mauve"] * 10,
            )
            error(lambda: model.predict(unseen))
            payload = json.loads(saved)
            for name in ("a", "b", "c"):
                payload["domain"][name.upper()] = payload["domain"].pop(name)
            with open(path, "w", encoding="utf-8") as file:
                json.dump(payload, file)
            error(lambda: model_class.from_file(path))
            payload = json.loads(saved)
            payload["targets"] += ["a", "b", "c"]
            with open(path, "w", encoding="utf-8") as file:
                json.dump(payload, file)
            error(lambda: model_class.from_file(path))
        out.append(repr(coverage(dataset, learned["gula_support"])))
        out.append(repr(generalize(learned["gula_support"], dataset, alpha=0.5)))
        out.append(repr(rule_recovery(learned["gula"], learned["pride"])))
        out.append(
            repr(rule_recovery(learned["gula"], learned["pride"], states.dropna()))
        )
        error(
            lambda: coverage(
                dataset, [Rule.from_string("y(red) :- p(x), q(x), r(x).")]
            )
        )
        error(
            lambda: RuleSetModel(
                dataset,
                [Rule.from_string(f"y({value}) :- a(black).") for value in colors[:3]],
            )
        )
    out.extend(str(warning.message) for warning in caught)
    print("\\n".join(out))


# `n_jobs` starts worker processes that import this script again;
# without the guard each would run the workflow itself. `fork`, the
# Linux default before Python 3.14, hid that.
if __name__ == "__main__":
    main()
"""


# SPEC: RUN-1
def test_every_output_is_the_same_whatever_the_hash_seed(
    outputs_per_hash_seed, tmp_path
):
    outputs = outputs_per_hash_seed(WORKFLOW, str(tmp_path / "model-"))
    first, *others = sorted(outputs)
    changed = [
        line
        for other in others
        for line in difflib.unified_diff(
            first.splitlines(), other.splitlines(), lineterm="", n=0
        )
    ]
    assert not changed, "\n".join(changed[:40])
