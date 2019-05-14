# pylfit
Python implementation of the main algorithms of the Learning From Interpretation Transitions (LFIT) framework.
	- LF1T: Learning From 1-step transitions
	- LFkT: Learning From k-step Transtions
	- LUST: Learning From Uncertain State Transtions
	- GULA: General Usage LFIT Algorithm
	- ACEDIA: Abstraction-free Continuum Environment Dynamics Inference Algorithm

Example of the usage of the different algorithms can be found in the examples/ folder
use the following command from the root of the repository:

python3 examples/example_lf1t.py

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3
- only tested on Ubuntu 18.04 but should be multi-platform

### Installing

Use the example folder to create your own scripts and run it from the repository root folder.
For example: examples/example_lf1t.py
```
python3 examples/example_lf1t.py
```

Add the following folders to the path:
	- src/
	- src/algorithms
	- src/objects

```
import sys
sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')
```

Import the necessary scripts
```
from utils import eprint
from logicProgram import LogicProgram
from lf1t import LF1T
```
- eprint is for debugging purpose, print to sdterr
- LogicProgram is the object class representing a multi-valued logic program
 	- load benchmark from text file
	- generate random instance of logic program
	- generate corresponding transitions to feed a learning algorithm
- LF1T is the class implementing the LF1T algorithm
	- the learning algorithm to use for inference of the logic program from its transitions

Extract a logic program benchmark from a file:
```
benchmark = LogicProgram.load_from_file("benchmarks/logic_programs/repressilator.lp")
```

Generate all its transitions:
```
input = benchmark.generate_all_transitions()
```

Use LF1T to learn an equivalent logic program:
```
model = LF1T.fit(benchmark.get_variables(), benchmark.get_values(), input)
```

Check the dynamics of original versus learned program:
```
expected = benchmark.generate_all_transitions()
predicted = model.generate_all_transitions()
precision = LogicProgram.precision(expected, predicted) * 100
eprint("Model accuracy: ", precision, "%")
```

Perform a next state prediction:
```
state = [1,1,1]
next = model.next(state)

eprint("Next state of ", state, " is ", next, " according to learned model")
```

Transitions can also be extracted directly from a csv file:
```
input = LF1T.load_input_from_csv("benchmarks/transitions/repressilator.csv")
```

Using the previous code you get more or less the example file examles/example_lf1t.py.
Its expected output is as follows.
Expected output from examples/example_lf1t.py

```
Example using logic program definition file:
----------------------------------------------
Original logic program:
VAR p 0 1
VAR q 0 1
VAR r 0 1

p(0,T) :- q(0,T-1).
p(1,T) :- q(1,T-1).
q(0,T) :- p(0,T-1).
q(0,T) :- r(0,T-1).
q(1,T) :- p(1,T-1), r(1,T-1).
r(0,T) :- p(1,T-1).
r(1,T) :- p(0,T-1).

Generating transitions...
LF1T input:
[[[0, 0, 0], [0, 0, 1]], [[0, 0, 1], [0, 0, 1]], [[0, 1, 0], [1, 0, 1]], [[0, 1, 1], [1, 0, 1]], [[1, 0, 0], [0, 0, 0]], [[1, 0, 1], [0, 1, 0]], [[1, 1, 0], [1, 0, 0]], [[1, 1, 1], [1, 1, 0]]]
LF1T output:
VAR p 0 1
VAR q 0 1
VAR r 0 1

p(0,T) :- q(0,T-1).
p(1,T) :- q(1,T-1).
q(0,T) :- p(0,T-1).
q(0,T) :- r(0,T-1).
q(1,T) :- p(1,T-1), r(1,T-1).
r(0,T) :- p(1,T-1).
r(1,T) :- p(0,T-1).

Model accuracy: 100.0%
Next state of [1, 1, 1] is [1, 1, 0] according to learned model
----------------------------------------------

Example using transition from csv file:
----------------------------------------------
LF1T input:
[[[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [0, 0, 0]], [[0, 1, 0], [1, 0, 1]], [[0, 0, 1], [0, 0, 1]], [[1, 1, 0], [1, 0, 0]], [[1, 0, 1], [0, 1, 0]], [[0, 1, 1], [1, 0, 1]], [[1, 1, 1], [1, 1, 0]]]
LF1T output:
VAR p 0 1
VAR q 0 1
VAR r 0 1

p(0,T) :- q(0,T-1).
p(1,T) :- q(1,T-1).
q(0,T) :- p(0,T-1).
q(0,T) :- r(0,T-1).
q(1,T) :- p(1,T-1), r(1,T-1).
r(0,T) :- p(1,T-1).
r(1,T) :- p(0,T-1).

Model accuracy: 100.0%
Next state of [1, 1, 1] is [1, 1, 0] according to learned model
----------------------------------------------
```

#### LFkT

```
from lfkt import LFkT
```

LFkT algorithm learns delayed influences and thus takes as input time series of state transitions.
LFkT input can be generated as follows:

```
time_serie_size = 10
input = benchmark.generate_all_time_series(time_serie_size)
```
See examples/example_lfkt.py for more details.

#### LUST

```
from lust import LUST
```

LUST algorithm learned non-determistic systems in the form of a set of deterministic logic programs.
See examples/example_lust.py for the details of how to deal with its output.

#### GULA

```
from gula import GULA
```

GULA algorithm can be used exactly like LF1T. The difference is that it is semantic free where LF1T can only learn from sycnhronous deterministic transitions.

#### ACEDIA

Import the necessary scripts
```
from utils import eprint
from continuum import Continuum
from continuumLogicProgram import ContinuumLogicProgram
from acedia import ACEDIA
```
- eprint is for debugging purpose, print to sdterr
- Continuum is the object class representing a continuum, a continuous set of real values
- ContinuumLogicProgram is the object class representing a Continuum logic program
	- generate random instance of logic program
	- generate corresponding transitions to feed a learning algorithm
- ACEDIA is the class implementing the ACEDIA algorithm
	- the learning algorithm to use for inference of the continuum logic program from its transitions

Generate a random Continuum Logic Program:
```
variables = ["a", "b", "c"]
domains = [ Continuum(0.0,1.0,True,True) for v in variables ]
rule_min_size = 0
rule_max_size = 3
epsilon = 0.5
random.seed(9999)

benchmark = ContinuumLogicProgram.random(variables, domains, rule_min_size, rule_max_size, epsilon, delay=1)
```

Generate all epsilon transitions of the continuum logic program
```
input = benchmark.generate_all_transitions(epsilon)
```

Learn a continuum logic program whose dynamics capture the transitions.
```
model = ACEDIA.fit(benchmark.get_variables(), benchmark.get_domains(), input)
```

Check the validity of the program learned:
```
expected = benchmark.generate_all_transitions(epsilon)
predicted = [(s1,model.next(s1)) for s1,s2 in expected]

precision = ContinuumLogicProgram.precision(expected, predicted) * 100

eprint("Model accuracy: ", precision, "%")
```

Predict the next continuum of value of each variable from a given state:
```
state = [0.75,0.33,0.58]
next = model.next(state)

eprint("Next state of ", state, " is ", next, " according to learned model")
```

ACEDIA input can also be directly extracted from csv file.
```
input = ACEDIA.load_input_from_csv("benchmarks/transitions/repressilator_continuous.csv")
```

In this case the variables and their respective domains must be known or generated to be feed to the algorithm together with the transitions.
```
variables = ["p", "q", "r"]
domains = [ Continuum(0.0,1.0,True,True) for v in variables ]
model = ACEDIA.fit(variables, domains, input)
```

See examples/example_acedia.py for more details.

## Running the tests

From the repository folder run the following comands:

For each algorithm example:
```
python3 examples/example_lf1t.py
```
```
python3 examples/example_lfkt.py
```
```
python3 examples/example_lust.py
```
```
python3 examples/example_gula.py
```
```
python3 examples/example_acedia.py
```

For complete unit tests
```
python3 src/unit_tests/unit_tests_all.py
```

For specific unit tests
```
python3 src/unit_tests/unit_tests_<script_name>
```

## Built With

* [Python 3](https://docs.python.org/3/) - The language used
* [Atom](https://atom.io/) - Source code editor

## Contributing

Please send a mail to tonyribeiro.contact@gmail.com if you want to add your own contribution to LFIT framework to the repository.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/Tony-sama/pylfit/tags).

## Authors

* **Tony Ribeiro** - *Initial work* - [Tony-sama](https://github.com/Tony-sama)

See also the list of [contributors](https://github.com/Tony-sama/pylfit/contributors) who participated in this project.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

More material about the LFIT framework and its applications can be found at http://www.tonyribeiro.fr/.
- General explanation of the framework: http://www.tonyribeiro.fr/research_main.html
- Biofinformatics applications: http://www.tonyribeiro.fr/research_bioinformatics.html
- Robotics applications: http://www.tonyribeiro.fr/research_robotics.html
- Publications: http://www.tonyribeiro.fr/index.html#publications

Main related scientifics publications:

	- LF1T:
		- MLJ 2014: Learning from Interpretation Transition
			http://link.springer.com/article/10.1007%2Fs10994-013-5353-8
		- ILP 2014: Learning Prime Implicant Conditions From Interpretation Transition
			http://link.springer.com/chapter/10.1007%2F978-3-319-23708-4_8
		- PhD Thesis 2015: Studies on Learning Dynamics of Systems from State Transitions
			http://www.tonyribeiro.fr/material/thesis/phd_thesis.pdf

	- LFkT:
		- Frontiers 2015: Learning delayed influences of biological systems
			http://www.frontiersin.org/Journal/Abstract.aspx?s=1267&name=bioinformatics_and_computational_biology&ART_DOI=10.3389/fbioe.2014.00081
		- ILP 2015: Learning Multi-Valued Biological Models with Delayed Influence from Time-Series Observations
			http://www.ilp2015.jp/papers/ILP2015_submission_44.pdf
		- ICMLA 2015: Learning Multi-Valued Biological Models with Delayed Influence from Time-Series Observations
			https://ieeexplore.ieee.org/document/7424281
		- PhD Thesis 2015: Studies on Learning Dynamics of Systems from State Transitions
			http://www.tonyribeiro.fr/material/thesis/phd_thesis.pdf

	- LUST:
		- ICLP 2015: Learning probabilistic action models from interpretation transitions
			http://www.tonyribeiro.fr/material/publications/iclp_2015.pdf
		- PhD Thesis 2015: Studies on Learning Dynamics of Systems from State Transitions
			http://www.tonyribeiro.fr/material/thesis/phd_thesis.pdf

	- GULA:
		- ILP 2018: Learning Dynamics with Synchronous, Asynchronous and General Semantics
			https://hal.archives-ouvertes.fr/hal-01826564

	- ACEDIA:
		- ILP 2017: Inductive Learning from State Transitions over Continuous Domains
			https://hal.archives-ouvertes.fr/hal-01655644
