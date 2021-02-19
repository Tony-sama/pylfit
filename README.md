# pylfit
Python implementation of the main algorithms of the Learning From Interpretation Transitions (LFIT) framework.
- GULA: General Usage LFIT Algorithm
- PRIDE: Polynomial Relational Inference of Dynamic Environnement
- Synchronizer

Example of the usage of the different algorithms can be found in the pylfit_package/tests/examples/ folder
use the following command from the tests/ directory:
```
python3 examples/api_gula_and_pride_example.py
```
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3
- only tested on Ubuntu 18.04 but should be multi-platform

### Installing

Use pip to install the last realease version of the pylfit library.
```
pip install pylfit
```

Import the library in your script to use it.

```
import pylfit

```

Format your data into states transitions: list of pair (list of string, list of string)s
```
data = [ \
(["0","0","0"],["0","0","1"]), \
(["1","0","0"],["0","0","0"]), \
(["0","1","0"],["1","0","1"]), \
(["0","0","1"],["0","0","1"]), \
(["1","1","0"],["1","0","0"]), \
(["1","0","1"],["0","1","0"]), \
(["0","1","1"],["1","0","1"]), \
(["1","1","1"],["1","1","0"])]
```
Use the pylfit.preprocessing api to load your data into the dataset format.
```
dataset = pylfit.preprocessing.transitions_dataset_from_array(data=data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])
```

Use the summary() method to get a look at your formated data.
```
dataset.summary()
```
summary() print:
```
StateTransitionsDataset summary:
 Features:
  p_t_1: ['0', '1']
  q_t_1: ['0', '1']
  r_t_1: ['0', '1']
 Targets:
  p_t: ['0', '1']
  q_t: ['0', '1']
  r_t: ['0', '1']
 Data:
  (['0', '0', '0'], ['0', '0', '1'])
  (['1', '0', '0'], ['0', '0', '0'])
  (['0', '1', '0'], ['1', '0', '1'])
  (['0', '0', '1'], ['0', '0', '1'])
  (['1', '1', '0'], ['1', '0', '0'])
  (['1', '0', '1'], ['0', '1', '0'])
  (['0', '1', '1'], ['1', '0', '1'])
  (['1', '1', '1'], ['1', '1', '0'])
```

Use the pylfit.models api to initialize a Dynamic Multi-valued Logic Program (DMVLP) model with the features/targets variables of the dataset.
Use compile(algorithm="gula") or compile(algorithm="pride") to prepare the model to be trained using GULA or PRIDE algorithm.
GULA has exponential complexity but guaranty all possible minimal rules to be learned.
PRIDE has polynomial complexity but only learn enough minimal rules to explain the dataset.
PRIDE is adviced in practice and GULA for small tests (< 10 variables, < 3 domain values).
```
model = pylfit.models.DMVLP(features=dataset.features, targets=dataset.targets)
model.compile(algorithm="pride") # model.compile(algorithm="gula")
model.summary()
```
summary() print:
```
DMVLP summary:
 Algorithm: GULA (<class 'pylfit.algorithms.gula.GULA'>)
 Features:
  p_t_1: ['0', '1']
  q_t_1: ['0', '1']
  r_t_1: ['0', '1']
 Targets:
  p_t: ['0', '1']
  q_t: ['0', '1']
  r_t: ['0', '1']
 Rules: []
```


Train the model on the dataset using the fit() method
```
model.fit(dataset=dataset)
model.summary()
```
summary() print:
```
DMVLP summary:
 Algorithm: GULA (<class 'pylfit.algorithms.gula.GULA'>)
 Features:
  p_t_1: ['0', '1']
  q_t_1: ['0', '1']
  r_t_1: ['0', '1']
 Targets:
  p_t: ['0', '1']
  q_t: ['0', '1']
  r_t: ['0', '1']
 Rules:
  p_t(0) :- q_t_1(0).
  p_t(1) :- q_t_1(1).
  q_t(0) :- p_t_1(0).
  q_t(0) :- r_t_1(0).
  q_t(1) :- p_t_1(1), r_t_1(1).
  r_t(0) :- p_t_1(1).
  r_t(1) :- p_t_1(0).
```

Use predict(feature_state) to make the model generate the possible targets states following a given feature states according to the model rules.
Default semantics is synchronous but you can request asynchronous or general transitions using predict(feature_state,semantics) as follows.
```
print("Predict from ['0','0','0'] (default: synchronous): ", end='')
prediction = model.predict(["0","0","0"])
print(prediction)

print("Predict from ['1','0','1'] (synchronous): ", end='')
prediction = model.predict(["1","0","1"])
print(prediction)

print("Predict from ['1','0','1'] (asynchronous): ", end='')
prediction = model.predict(["1","0","1"], semantics="asynchronous")
print(prediction)

print("Predict from ['1','0','1'] (general): ", end='')
prediction = model.predict(["1","0","1"], semantics="general")
print(prediction)
```

print:
```
Predict from ['0','0','0'] (default: synchronous): [['0', '0', '1']]
Predict from ['1','0','1'] (synchronous): [['0', '1', '0']]
Predict from ['1','0','1'] (asynchronous): [['0', '0', '1'], ['1', '1', '1'], ['1', '0', '0']]
Predict from ['1','0','1'] (general): [['0', '0', '0'], ['0', '0', '1'], ['0', '1', '0'], ['0', '1', '1'], ['1', '0', '0'], ['1', '0', '1'], ['1', '1', '0'], ['1', '1', '1']]
```

Using the previous code you get more or less the example file pylfit_package/tests/examles/api_gula_and_pride_example.py.
Its expected output is as follows.

```
Convert array data as a StateTransitionsDataset using pylfit.preprocessing
StateTransitionsDataset summary:
 Features:
  p_t_1: ['0', '1']
  q_t_1: ['0', '1']
  r_t_1: ['0', '1']
 Targets:
  p_t: ['0', '1']
  q_t: ['0', '1']
  r_t: ['0', '1']
 Data:
  (['0', '0', '0'], ['0', '0', '1'])
  (['1', '0', '0'], ['0', '0', '0'])
  (['0', '1', '0'], ['1', '0', '1'])
  (['0', '0', '1'], ['0', '0', '1'])
  (['1', '1', '0'], ['1', '0', '0'])
  (['1', '0', '1'], ['0', '1', '0'])
  (['0', '1', '1'], ['1', '0', '1'])
  (['1', '1', '1'], ['1', '1', '0'])

Initialize a DMVLP with the dataset variables and set GULA as learning algorithm
DMVLP summary:
 Algorithm: GULA (<class 'pylfit.algorithms.gula.GULA'>)
 Features:
  p_t_1: ['0', '1']
  q_t_1: ['0', '1']
  r_t_1: ['0', '1']
 Targets:
  p_t: ['0', '1']
  q_t: ['0', '1']
  r_t: ['0', '1']
 Rules: []

Fit the DMVLP on the dataset
Trained model:
DMVLP summary:
 Algorithm: GULA (<class 'pylfit.algorithms.gula.GULA'>)
 Features:
  p_t_1: ['0', '1']
  q_t_1: ['0', '1']
  r_t_1: ['0', '1']
 Targets:
  p_t: ['0', '1']
  q_t: ['0', '1']
  r_t: ['0', '1']
 Rules:
  p_t(0) :- q_t_1(0).
  p_t(1) :- q_t_1(1).
  q_t(0) :- p_t_1(0).
  q_t(0) :- r_t_1(0).
  q_t(1) :- p_t_1(1), r_t_1(1).
  r_t(0) :- p_t_1(1).
  r_t(1) :- p_t_1(0).
Predict from ['0','0','0'] (default: synchronous): [['0', '0', '1']]
Predict from ['1','0','1'] (synchronous): [['0', '1', '0']]
Predict from ['1','0','1'] (asynchronous): [['0', '0', '1'], ['1', '1', '1'], ['1', '0', '0']]
Predict from ['1','0','1'] (general): [['0', '0', '0'], ['0', '0', '1'], ['0', '1', '0'], ['0', '1', '1'], ['1', '0', '0'], ['1', '0', '1'], ['1', '1', '0'], ['1', '1', '1']]
All states transitions of the model (synchronous):
[(['0', '0', '0'], ['0', '0', '1']), (['0', '0', '1'], ['0', '0', '1']), (['0', '1', '0'], ['1', '0', '1']), (['0', '1', '1'], ['1', '0', '1']), (['1', '0', '0'], ['0', '0', '0']), (['1', '0', '1'], ['0', '1', '0']), (['1', '1', '0'], ['1', '0', '0']), (['1', '1', '1'], ['1', '1', '0'])]
Saving transitions to csv...
Saved to tmp/output.csv
```

#### Synchronizer

```
import pylfit
```

Dataset is same as previously for DMVLP model but to learn constraint we will use another type of model: Constrained Dynamic Multi-valued Logic Program (CDMVLP)

```
model = pylfit.models.CDMVLP(features=dataset.features, targets=dataset.targets)
model.compile(algorithm=ALGORITHM)
model.summary()
```

print:
```
CDMVLP summary:
 Algorithm: Synchronizer (<class 'pylfit.algorithms.synchronizer.Synchronizer'>)
 Features:
  p_t_1: ['0', '1']
  q_t_1: ['0', '1']
  r_t_1: ['0', '1']
 Targets:
  p_t: ['0', '1']
  q_t: ['0', '1']
  r_t: ['0', '1']
 Rules: []
 Constraints: []
```

Basically a DMVLP with an additional set of constraints rules preventing some combinations of feature/target variable values to occur in a transition.
CDMVLP api is the same as DMVLP, use fit() to train the model on the dataset and summary() to have a look to the model.

```
model.fit(dataset=dataset) # optional targets
print("Trained model:")
model.summary()
```

print:

```
CDMVLP summary:
 Algorithm: Synchronizer (<class 'pylfit.algorithms.synchronizer.Synchronizer'>)
 Features:
  p_t_1: ['0', '1']
  q_t_1: ['0', '1']
  r_t_1: ['0', '1']
 Targets:
  p_t: ['0', '1']
  q_t: ['0', '1']
  r_t: ['0', '1']
 Rules:
  p_t(0) :- q_t_1(0).
  p_t(1) :- q_t_1(1).
  p_t(1) :- p_t_1(0), r_t_1(0).
  q_t(0) :- p_t_1(0).
  q_t(0) :- r_t_1(0).
  q_t(1) :- p_t_1(1), r_t_1(1).
  r_t(0) :- p_t_1(1).
  r_t(0) :- q_t_1(0), r_t_1(0).
  r_t(1) :- p_t_1(0).
 Constraints:
  :- q_t_1(0), p_t(1), r_t(1).
  :- p_t_1(0), p_t(0), r_t(0).
```

Prediction are obtained the same way as for DMVLP, but no semantics option.
Use predict(feature_state) to get the list of possible target states according to the model rules and constraints

```
print("Predict from ['0','0','0']: ", end='')
prediction = model.predict(["0","0","0"])
print(prediction)

print("Predict from ['1','1','1']: ", end='')
prediction = model.predict(["1","1","1"])
print(prediction)
```

print:
```
Predict from ['0','0','0']: [['0', '0', '1'], ['1', '0', '0']]
Predict from ['1','1','1']: [['1', '1', '0']]

```

## Running the tests

From the tests/ folder run the following comands:

For each algorithm example:
```
python3 examples/api_gula_and_pride_example.py
```
```
python3 examples/api_synchronizer_example.py
```
```
python3 examples/api_weighted_prediction_and_explanation_example.py
```

For complete regression tests
```
python3 regression_tests/all_tests.py
```

For specific regression tests
```
python3 regression_tests/.../<script_name>
```
For example
```
python3 regression_tests/algorithms/gula_benchmark_tests.py
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

More material about the LFIT framework and its applications can be found at http://www.tonyribeiro.fr/
- General explanation of the framework: http://www.tonyribeiro.fr/research_main.html
- Biofinformatics applications: http://www.tonyribeiro.fr/research_bioinformatics.html
- Robotics applications: http://www.tonyribeiro.fr/research_robotics.html
- Publications: http://www.tonyribeiro.fr/index.html#publications

Main related scientifics publications:

- LF1T:
	- MLJ 2014: Learning from Interpretation Transition
		- http://link.springer.com/article/10.1007%2Fs10994-013-5353-8
	- ILP 2014: Learning Prime Implicant Conditions From Interpretation Transition
		- http://link.springer.com/chapter/10.1007%2F978-3-319-23708-4_8
	- PhD Thesis 2015: Studies on Learning Dynamics of Systems from State Transitions
		- http://www.tonyribeiro.fr/material/thesis/phd_thesis.pdf

- LFkT:
	- Frontiers 2015: Learning delayed influences of biological systems
		- http://www.frontiersin.org/Journal/Abstract.aspx?s=1267&name=bioinformatics_and_computational_biology&ART_DOI=10.3389/fbioe.2014.00081
	- ILP 2015: Learning Multi-Valued Biological Models with Delayed Influence from Time-Series Observations
		- http://www.ilp2015.jp/papers/ILP2015_submission_44.pdf
	- ICMLA 2015: Learning Multi-Valued Biological Models with Delayed Influence from Time-Series Observations
		- https://ieeexplore.ieee.org/document/7424281
	- PhD Thesis 2015: Studies on Learning Dynamics of Systems from State Transitions
		- http://www.tonyribeiro.fr/material/thesis/phd_thesis.pdf

- LUST:
	- ICLP 2015: Learning probabilistic action models from interpretation transitions
		- http://www.tonyribeiro.fr/material/publications/iclp_2015.pdf
	- PhD Thesis 2015: Studies on Learning Dynamics of Systems from State Transitions
		- http://www.tonyribeiro.fr/material/thesis/phd_thesis.pdf

- GULA:
	- ILP 2018: Learning Dynamics with Synchronous, Asynchronous and General Semantics
		- https://hal.archives-ouvertes.fr/hal-01826564

- ACEDIA:
	- ILP 2017: Inductive Learning from State Transitions over Continuous Domains
		- https://hal.archives-ouvertes.fr/hal-01655644
