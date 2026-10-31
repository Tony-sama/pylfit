# `tests/data/bnet/`

Boolean networks in PyBoolNet's `.bnet` format, used as regression
fixtures by `tests/algorithms/test_boolean_networks.py` and read by
six examples: `benchmark_bnet.py`, `benchmark_influence.py`,
`benchmark_n_jobs.py`, `benchmark_negative_order.py`,
`benchmark_unknowns.py` and `dynamics.py`, all in `examples/`.

## Source and license

All 18 files are taken from the model repository of **PyBoolNet**
([hklarner/pyboolnet](https://github.com/hklarner/pyboolnet)):

> Klarner, H., Streck, A. & Siebert, H. "PyBoolNet: a python package
> for the generation, analysis and visualization of boolean networks."
> *Bioinformatics* 33.5 (2017): 770–772.
> <https://doi.org/10.1093/bioinformatics/btw682>

PyBoolNet declares itself LGPL-licensed in its package metadata (no
version is stated, and its repository carries no license file). These
files keep that license; they are not covered by this repository's
GPL-3.0 `LICENSE`. They are redistributed unmodified apart from
whitespace.

## Citations

PyBoolNet collects published models, and each file's leading comment
names the paper it comes from. Cite that paper, not this repository,
if you use a network.

Eight files have no external paper — they are PyBoolNet's own
illustrative examples, credited to the PyBoolNet paper above:
`raf.bnet`, `n3s1c1a.bnet`, `n3s1c1b.bnet`, `n5s3.bnet`, `n6s1c2.bnet`,
`n7s3.bnet`, `n12c5.bnet` and `multivalued.bnet`; the last uses the
Van Ham encoding of multivalued variables, from Didier, G., Remy, E. &
Chaouiya, C. "Mapping multivalued onto Boolean dynamics." *Journal of
Theoretical Biology* 270.1 (2011): 177–184.
