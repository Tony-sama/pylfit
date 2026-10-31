"""A whole experiment, start to finish
===================================

Load, split, learn, score, save: copy this and change it.

Run it:

    python3 examples/starter.py

This is the shape of a real run: load data, hold some back, learn from
the rest, check the result, keep the model. Seven steps, in the order
you will always do them.

**Things worth changing first**, all at the top of the file: which
dataset (``DATASET``, and its ``TARGET`` column), which algorithm
(``ALGORITHM``), how much evidence a rule needs (``MIN_SUPPORT``), how
much data is held back (``TEST_FRACTION``), and how good the result has
to be (``MIN_ACCURACY``).

**Three choices here are less obvious than they look**, and each is the
kind of thing that quietly ruins a first experiment:

- **A weighted model:** ``WeightedRuleSetModel``, not ``RuleSetModel``.
  ``GULA`` returns *every* minimal rule consistent with the data, which
  means every class has a rule matching any row the training data
  doesn't hold. A plain ``RuleSetModel`` gives each matching rule one
  vote, so they tie, and the answer is a coin toss — about 0.23 here
  instead of 0.92 (0.72 with ``min_support=1``, below, which drops the
  rules behind most ties). Weighting by how much evidence each rule
  carries is what breaks the tie. Run ``benchmark_classic.py`` to see
  both numbers side by side.
- **The domain comes from the whole dataset, not the training split.**
  Otherwise a value that happens to appear only in the held-out rows
  is unknown to the model, and predicting on it fails — your score
  would then depend on how the shuffle fell rather than on what was
  learned. On ``car.csv`` it changes nothing, since every value is in
  the training split; on your own data it matters as soon as a value is
  rare.
- **Every rule backed by an example:** ``min_support=1``, which you
  usually want.
  Without it ``GULA`` keeps every rule merely *consistent* with the
  data, including ones no training example actually backs. Turning it
  on here drops 1122 rules to 337 while held-out accuracy stays at
  0.9220, and on ``soybean-small.csv`` it drops 12176 to 1897, also with
  identical accuracy and about ten times faster. 70% to 84% of a
  complete rule set is evidence-free, and a rule set you can read beats
  one you cannot.

  **A support of 1 is a different kind of change from any higher
  value**, which is why it is the one this script uses. It removes
  exactly the rules backed by *no* training example — and a rule
  backed by nothing cannot tell you anything. Measured here: three
  held-out rows end up matched by no rule after the filter. All three
  are feature *combinations* absent from training (no out-of-domain
  values — every individual value appears in the training split), and
  unrestricted they were covered by four rules apiece, one per class,
  all with support 0, all saying "nothing contradicts this". That is a
  four-way tie, decided by a coin. Predictions for those rows come out
  byte-identical with and without the filter, across every seed tried.
  The coverage was real in the sense that a rule matched; it was worth
  nothing.

  Raising it further is a genuine trade, and not the same thing at
  all: at 2 you start discarding rules that *do* carry evidence, and
  rows matched by nothing go to 14, then 47 at 3. Treat 1 as the
  default and anything above it as a claim to measure. See
  ``benchmark_min_support.py`` for what the filter does to runtime, and
  for the check that it keeps exactly the well-supported rules.

**The assertion at the end is the point of the script.** Printing an
accuracy tells you nothing on its own; you have to already know what
number would be bad. Stating the threshold up front turns the run into
a check you can leave in place and notice failing later.
"""

from __future__ import annotations

# sphinx_gallery_start_ignore
# The docs build runs this file from its own directory, without `__file__`.
if "__file__" not in globals():
    __file__ = "starter.py"
# sphinx_gallery_end_ignore

import tempfile
from pathlib import Path

import pandas as pd

from pylfit import GULA, Dataset, WeightedRuleSetModel, accuracy_score

HERE = Path(__file__).resolve().parent

# --- things to change ------------------------------------------------

#: Any CSV of categorical columns. See `data/README.md` for what ships
#: here; your own file works just as well.
DATASET = HERE / "data" / "car.csv"

#: The column to predict. Everything else becomes a feature.
TARGET = "class"

#: `GULA` learns every minimal rule; `PRIDE` stops once every example
#: is covered, so it is much faster and returns far fewer rules. To
#: swap, change the name here; the `fit` call below works as it is.
#: `PRIDE` takes `min_support` too, though at 1 it changes nothing for
#: it: it builds every rule out from a real example, which the rule
#: always matches.
ALGORITHM = GULA

#: Discard rules backed by fewer than this many training examples.
#: See the module docstring for why 1 is a good default and why
#: raising it is a decision rather than a knob: each step up leaves
#: more unseen feature states matched by no rule at all.
MIN_SUPPORT = 1

#: Held back from training, and never seen until scoring.
TEST_FRACTION = 0.2

#: Fixed so two runs are comparable. Remove it and every run reshuffles.
RANDOM_SEED = 0

#: What counts as a good enough result. Fail loudly below it.
MIN_ACCURACY = 0.85

# ---------------------------------------------------------------------


def main() -> None:
    """Run the whole thing and report."""
    # 1. Load. Everything becomes `str`: pylfit is categorical, so a
    #    column of 1s and 2s means two labels, not two numbers.
    frame = pd.read_csv(DATASET).astype(str)
    features = [column for column in frame.columns if column != TARGET]
    print(f"{DATASET.name}: {len(frame)} rows, {len(features)} features")

    # 2. Hold out a test set. Shuffle first — many datasets arrive
    #    sorted by class, and slicing one of those without shuffling
    #    gives a test set that is entirely one label.
    shuffled = frame.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    cut = int(len(shuffled) * (1 - TEST_FRACTION))
    train, test = shuffled.iloc[:cut], shuffled.iloc[cut:].reset_index(drop=True)
    print(f"  {len(train)} train / {len(test)} test")

    # 3. Learn. `fit` returns a plain list of rules, not a model —
    #    what you do with them is a separate decision.
    train_dataset = Dataset(train, features=features, targets=[TARGET])
    rules = ALGORITHM.fit(train_dataset, min_support=MIN_SUPPORT)
    print(f"  {ALGORITHM.__name__} learned {len(rules)} rules")

    # 4. Wrap the rules in something that can predict. See the module
    #    docstring for why weighted, and why the domain is widened.
    model = WeightedRuleSetModel(train_dataset, rules)
    model.domain = Dataset(frame, features=features, targets=[TARGET]).domains()

    # 5. Predict on data the model has never seen, and score it.
    predictions = model.predict(test[features], random_state=RANDOM_SEED)
    accuracy = accuracy_score(test, predictions, TARGET)
    print(f"  held-out accuracy: {accuracy:.4f}")

    # 6. Say what you expected. A number you have not judged is not a
    #    result yet.
    assert accuracy >= MIN_ACCURACY, (
        f"held-out accuracy {accuracy:.4f} is below the {MIN_ACCURACY} "
        f"this script expects. Either the change you just made hurt, or "
        f"the threshold needs rethinking; decide which before moving on."
    )

    # 7. Save it and load it back. Reloading and comparing is how you
    #    find out the file is actually usable, rather than merely
    #    written. The folder is deleted on exit; to keep the model, save
    #    it to a path of your own.
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "starter_model.json"
        model.save(path)
        reloaded = WeightedRuleSetModel.from_file(path)
    assert reloaded == model, "the reloaded model differs from the one saved"
    print("  saved and reloaded intact")

    print("\nPassed. Now change something at the top and run it again.")


if __name__ == "__main__":
    main()
