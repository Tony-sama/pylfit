#-----------------------
# @author: Tony Ribeiro
# @created: 2022/08/29
# @updated: 2022/08/30
#
# @desc: clp class unit test script
# done:
#  - __init__
# - compile
# - fit
# - predict
# - summary
# - to_string
# - feature_states
# - target_states
#
# Todo:
#
#-----------------------

import unittest
import random
import sys
import numpy as np
import itertools
import io
import contextlib

from io import StringIO

import pylfit
from pylfit.objects import Continuum, ContinuumRule
from pylfit.models import CLP
from pylfit.algorithms import Algorithm, ACEDIA
from pylfit.datasets import Dataset
from pylfit.semantics import Synchronous, Asynchronous, General

from pylfit.utils import eprint

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from tests_generator import random_ContinuousStateTransitionsDataset, random_ContinuumRule, random_CLP, random_continuous_state

#random.seed(0)

class CLP_tests(unittest.TestCase):
    """
        Unit test of module clp.py
    """
    _nb_tests = 10

    _nb_transitions = 10

    _nb_features = 2

    _nb_targets = 2

    _min_epsilon = 0.3

    """ must be < _max_value"""
    _min_value = -100.0

    """ must be > _min_value"""
    _max_value = 100.0

    _min_domain_size = 1.0

    _min_continuum_size = 1

    _nb_rules = 10

    _body_size = 10


    _SUPPORTED_ALGORITHMS = ["acedia"]

    def test_constructor(self):
        print(">> CLP(features, targets, rules)")

        features = [("x0", Continuum(0,1,True, True)), ("x1", Continuum(0,1,True, True)), ("x2", Continuum(0,1,True, True))]
        targets = [("y0", Continuum(0,1,True, True)), ("y1", Continuum(0,1,True, True)), ("y2", Continuum(0,1,True, True))]
        model = CLP(features=features, targets=targets)

        self.assertEqual(model.features, features)
        self.assertEqual(model.targets, targets)
        self.assertEqual(model.rules, [])
        self.assertEqual(model.algorithm, None)

        # Exceptions:
        #-------------

        # Features format
        features = '[("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]' # not list
        self.assertRaises(TypeError, CLP, features, targets)

        features = [["x0", ["0","1"]], ("x1", ["0","1"]), ("x2", ["0","1"])] # not tuple
        self.assertRaises(TypeError, CLP, features, targets)

        features = [("x0", "0","1"), ("x1", "0","1"), ("x2", ["0","1"])] # not tuple of size 2
        self.assertRaises(TypeError, CLP, features, targets)

        features = [("x0", ["0","1"]), ("x1", '0","1"'), ("x2", ["0","1"])] # domain is not Continuum
        self.assertRaises(TypeError, CLP, features, targets)

        # Targets format
        features = [("x0", Continuum(0,1,True, True)), ("x1", Continuum(0,1,True, True)), ("x2", Continuum(0,1,True, True))]

        targets = '[("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]' # not list
        self.assertRaises(TypeError, CLP, features, targets)

        targets = [["x0", ["0","1"]], ("x1", ["0","1"]), ("x2", ["0","1"])] # not tuple
        self.assertRaises(TypeError, CLP, features, targets)

        targets = [("x0", "0","1"), ("x1", "0","1"), ("x2", ["0","1"])] # not tuple of size 2
        self.assertRaises(TypeError, CLP, features, targets)

        targets = [("x0", ["0","1"]), ("x1", '0","1"'), ("x2", ["0","1"])] # domain is not Continuum
        self.assertRaises(TypeError, CLP, features, targets)

        model = random_CLP( \
        nb_features=random.randint(1,self._nb_features), \
        nb_targets=random.randint(1,self._nb_targets), \
        algorithm="acedia")

        rules = ""
        self.assertRaises(TypeError, CLP, model.features, model.targets, rules)
        rules = model.rules + [""]
        self.assertRaises(TypeError, CLP, model.features, model.targets, rules)

    def test_copy(self):
        print(">> CLP.copy()")

        features = [("x0", Continuum(0,1,True, True)), ("x1", Continuum(0,1,True, True)), ("x2", Continuum(0,1,True, True))]
        targets = [("y0", Continuum(0,1,True, True)), ("y1", Continuum(0,1,True, True)), ("y2", Continuum(0,1,True, True))]
        model = CLP(features=features, targets=targets)

        copy = model.copy()

        self.assertEqual(model.features, copy.features)
        self.assertEqual(model.targets, copy.targets)
        self.assertEqual(model.rules, copy.rules)
        self.assertEqual(model.algorithm, copy.algorithm)

        for i in range(self._nb_tests):
            for algorithm in self._SUPPORTED_ALGORITHMS:
                model = random_CLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                algorithm=algorithm)

                copy = model.copy()

                self.assertEqual(model.features, copy.features)
                self.assertEqual(model.targets, copy.targets)
                self.assertEqual(model.rules, copy.rules)
                self.assertEqual(model.algorithm, copy.algorithm)

    def test___str__(self):
        print(">> CLP.__str__()")
        for i in range(self._nb_tests):
            for algorithm in self._SUPPORTED_ALGORITHMS:
                model = random_CLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                algorithm=algorithm)

                self.assertEqual(model.__str__(), model.to_string())

    def test___repr__(self):
        print(">> CLP.__repr__()")
        for i in range(self._nb_tests):
            for algorithm in self._SUPPORTED_ALGORITHMS:
                model = random_CLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                algorithm=algorithm)

                self.assertEqual(model.__repr__(), model.to_string())

    def test_compile(self):
        print(">> CLP.compile(algorithm)")
        features = [("x0", Continuum(0,1,True, True)), ("x1", Continuum(0,1,True, True)), ("x2", Continuum(0,1,True, True))]
        targets = [("y0", Continuum(0,1,True, True)), ("y1", Continuum(0,1,True, True)), ("y2", Continuum(0,1,True, True))]
        model = CLP(features=features, targets=targets)

        model.compile(algorithm="acedia")
        self.assertEqual(model.algorithm, "acedia")

        backup = CLP._ALGORITHMS.copy()
        CLP._ALGORITHMS = CLP._ALGORITHMS+["lol"]
        self.assertRaises(NotImplementedError, model.compile, "lol")
        CLP._ALGORITHMS = backup

        self.assertRaises(ValueError, model.compile, "lol")


    def test_fit(self):
        print(">> CLP.fit(dataset, verbose)")

        for test in range(0,self._nb_tests):
            for verbose in [0,1]:

                dataset = random_ContinuousStateTransitionsDataset( \
                nb_transitions=random.randint(1, self._nb_transitions), \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                min_value=self._min_value, \
                max_value=self._max_value, \
                min_continuum_size=self._min_continuum_size)

                model = CLP(features=dataset.features, targets=dataset.targets)
                model.compile(algorithm="acedia")
                f = io.StringIO()
                with contextlib.redirect_stderr(f):
                    model.fit(dataset=dataset, verbose=verbose)

                expected_rules = ACEDIA.fit(dataset)
                self.assertEqual(expected_rules, model.rules)

                # Exceptions
                #------------

                for algorithm in self._SUPPORTED_ALGORITHMS:
                    backup = CLP._ALGORITHMS.copy()
                    CLP._ALGORITHMS = CLP._ALGORITHMS+["lol"]
                    model.algorithm = "lol"
                    self.assertRaises(NotImplementedError, model.fit, dataset)
                    CLP._ALGORITHMS = backup

                    model = CLP(features=dataset.features, targets=dataset.targets)
                    model.compile(algorithm=algorithm)

                    self.assertRaises(ValueError, model.fit, None, []) # dataset is not of valid type

                    model.algorithm = "acediaaaa"
                    self.assertRaises(ValueError, model.fit, dataset, None, verbose) # algorithm is not valid
                    model.algorithm = algorithm

                    self.assertRaises(ValueError, model.fit, dataset, "", verbose) # targets_to_learn is not valid
                    self.assertRaises(ValueError, model.fit, dataset, [var for var, vals in model.targets+model.features], verbose) # targets_to_learn is not valid
                    bad_targets = model.targets.copy()
                    bad_targets[0] = ("lol", bad_targets[0][1])
                    self.assertRaises(ValueError, model.fit, dataset, [var for var, vals in  bad_targets], verbose) # targets_to_learn is not valid

                    original = CLP._COMPATIBLE_DATASETS.copy()
                    class newdataset(Dataset):
                        def __init__(self, data, features, targets):
                            x = ""
                        @property
                        def data(self):
                            return ""

                        @data.setter
                        def data(self, value):
                            self._data = value

                        @property
                        def features(self):
                            return ""

                        @features.setter
                        def features(self, value):
                            self._features = value

                        @property
                        def targets(self):
                            return ""

                        @targets.setter
                        def targets(self, value):
                            self._targets = value

                    CLP._COMPATIBLE_DATASETS = [newdataset]
                    self.assertRaises(ValueError, model.fit, newdataset([],[],[]), None, verbose) # dataset not supported by the algo
                    CLP._COMPATIBLE_DATASETS = original

    def test_summary(self):
        print(">> CLP.summary()")
        for i in range(self._nb_tests):
            for algorithm in self._SUPPORTED_ALGORITHMS:
                # Empty CLP
                model = random_CLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                algorithm=algorithm)

                model.rules = []

                expected_print = \
                "CLP summary:\n"+\
                " Algorithm: "+ algorithm +"\n"
                expected_print += " Features: \n"
                for var,vals in model.features:
                    expected_print += "  " + var + ": " + str(vals) + "\n"
                expected_print += " Targets: \n"
                for var,vals in model.targets:
                    expected_print += "  " + var + ": " + str(vals) + "\n"
                expected_print +=" Rules: []\n"

                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                model.summary()
                sys.stdout = old_stdout

                self.assertEqual(mystdout.getvalue(), expected_print)

                # Usual CLP
                model = random_CLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                algorithm=algorithm)

                expected_print = \
                "CLP summary:\n"+\
                " Algorithm: "+ algorithm +"\n"
                expected_print += " Features: \n"
                for var,vals in model.features:
                    expected_print += "  " + var + ": " + str(vals) + "\n"
                expected_print += " Targets: \n"
                for var,vals in model.targets:
                    expected_print += "  " + var + ": " + str(vals) + "\n"
                if len(model.rules) == 0:
                    expected_print +=" Rules: []\n"
                else:
                    expected_print +=" Rules:\n"
                    for r in model.rules:
                        expected_print += "  "+r.logic_form(model.features, model.targets)+"\n"

                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                model.summary()
                sys.stdout = old_stdout

                self.assertEqual(mystdout.getvalue(), expected_print)

                # Exceptions
                #------------

                model = CLP(features=model.features, targets=model.targets)
                self.assertRaises(ValueError, model.summary) # compile not called

    def test_to_string(self):
        print(">> pylfit.models.CLP.to_string()")
        for i in range(self._nb_tests):
            for algorithm in self._SUPPORTED_ALGORITHMS:
                model = random_CLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                algorithm=algorithm)

                expected = \
                 "{\n"+\
                 "Algorithm: " + str(model.algorithm)+\
                 "\nFeatures: "+ str(model.features)+\
                 "\nTargets: "+ str(model.targets)+\
                 "\nRules:\n"
                for r in model.rules:
                    expected += r.logic_form(model.features, model.targets) + "\n"
                expected += "}"

                self.assertEqual(model.to_string(), expected)

    def test_predict(self):
        print(">> pylfit.models.CLP.predict(feature_states)")
        for i in range(self._nb_tests):
            for algorithm in self._SUPPORTED_ALGORITHMS:
                model = random_CLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                algorithm=algorithm)

                feature_state = random_continuous_state(model.features)

                output = model.predict([feature_state])

                # Exceptions
                self.assertRaises(TypeError, model.predict, "")
                self.assertRaises(TypeError, model.predict, [[],""])
                self.assertRaises(TypeError, model.predict, [["" for var in feature_state]])
                self.assertRaises(TypeError, model.predict, [[0.0 for var in feature_state]+[0.0]])

    def test_feature_states(self):
        print(">> pylfit.models.CLP.feature_states(epsilon)")
        for i in range(self._nb_tests):
            for algorithm in self._SUPPORTED_ALGORITHMS:
                model = random_CLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                algorithm=algorithm)

                epsilon = random.uniform(0.1, 1)

                output = model.feature_states(epsilon)

                # Exceptions
                self.assertRaises(ValueError, model.feature_states, -1)
                self.assertRaises(ValueError, model.feature_states, 1.1)
                self.assertRaises(ValueError, model.feature_states, 0.01)

                model.features = [(var, Continuum()) for var,val in model.features]
                output = model.feature_states(epsilon)

    def test_target_states(self):
        print(">> pylfit.models.CLP.target_states(epsilon)")
        for i in range(self._nb_tests):
            for algorithm in self._SUPPORTED_ALGORITHMS:
                model = random_CLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                algorithm=algorithm)

                epsilon = random.uniform(0.1, 1)

                output = model.target_states(epsilon)

                # Exceptions
                self.assertRaises(ValueError, model.target_states, -1)
                self.assertRaises(ValueError, model.target_states, 1.1)
                self.assertRaises(ValueError, model.target_states, 0.01)

                model.targets = [(var, Continuum()) for var,val in model.targets]
                output = model.target_states(epsilon)
'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
