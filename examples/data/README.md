# `examples/data/`

Datasets the example scripts read. Small on purpose: on a laptop most
scripts finish in a few seconds, and the slowest,
`benchmark_unknowns.py`, in about half a minute.

| file | rows | features | target | used by |
| --- | --- | --- | --- | --- |
| `car.csv` | 1728 | 6 categorical | `class` (4 values) | `benchmark_classic.py`, `benchmark_min_support.py`, `benchmark_negative_order.py`, `benchmark_unknowns.py`, `starter.py` |
| `soybean-small.csv` | 47 | 35 categorical | `class` (4 values) | `benchmark_min_support.py`, `benchmark_negative_order.py` |
| `heart-disease.csv` | 303 | 8 coded, 5 numeric | `num` (0 to 4) | `tutorial_heart_disease.py` |

Boolean networks are not here — `benchmark_bnet.py`,
`benchmark_influence.py`, `benchmark_n_jobs.py`,
`benchmark_negative_order.py`, `benchmark_unknowns.py` and
`dynamics.py` read the `.bnet` files vendored under `tests/data/bnet/`,
so there is one copy of that corpus rather than one per consumer.

## `car.csv`

The **Car Evaluation** dataset — Bohanec & Rajkovič, 1988, via the UCI
Machine Learning Repository. A hierarchical decision model for
acceptability of a car, flattened to six categorical features
(`buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`) over a
four-valued `class`.

Complete: 1728 rows is the full Cartesian product of the six feature
domains, and there are no missing values. That completeness is why it
is the dataset used here — the held-out score measures the rules
rather than an imputation policy, and no missing-value handling is
needed to read the script.

Note it originates as a *decision model demonstration*, not from a
rule-induction paper, so published accuracy figures for it come from
later work rather than from Bohanec & Rajkovič themselves.

Licensed CC BY 4.0 by the UCI repository. Cite it as: Bohanec, M.
(1988). Car Evaluation [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5JP48>.

## `soybean-small.csv`

The small **Soybean** diagnosis dataset, via the UCI Machine Learning
Repository: a four-disease subset of Michalski's soybean data
(Michalski & Chilausky, 1980), here for its *shape* rather than its
history. 35 features over only 47 rows: many candidate conditions and
few rows to back them, which is what makes `min_support` worth having.
14 of the 35 features take a single value in this subset. No missing
values.

Licensed CC BY 4.0 by the UCI repository. Cite it as: Michalski, R.
(1980). Soybean (Small) [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5DS3P>.

## `heart-disease.csv`

The Cleveland part of the **Heart Disease** dataset — Janosi,
Steinbrunn, Pfisterer & Detrano, 1989, via the UCI Machine Learning
Repository: UCI's `processed.cleveland.data`, values unchanged, with a
header row added. The 14 columns are the subset every published
experiment on it uses. `num` is the diagnosis of coronary artery
disease, 0 for none and 1 to 4 for its severity; experiments on this
data predict presence (1 to 4) against absence (0).

Here because it is real data that needs preparing: five measurements
to bin (age, resting blood pressure, cholesterol, maximum heart rate,
ST depression), integer codes to name, and six patients with a value
nobody recorded, written `?` (four in `ca`, two in `thal`).

Licensed CC BY 4.0 by the UCI repository. Cite it as: Janosi, A.,
Steinbrunn, W., Pfisterer, M. & Detrano, R. (1989). Heart Disease
[Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C52P4X>.
