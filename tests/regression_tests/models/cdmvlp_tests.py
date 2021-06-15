#-----------------------
# @author: Tony Ribeiro
# @created: 2021/02/05
# @updated: 2021/06/15
#
# @desc: CCDMVLP class regression test script
# done:
#  - __init__
# - compile
# - fit
# - predict
# - summary
# - to_string
#
# Todo:
#
#-----------------------

import unittest
import random
import sys
import numpy as np
import itertools
import contextlib

import io
from io import StringIO

import pylfit
from pylfit.objects import Rule
from pylfit.models import CDMVLP
from pylfit.algorithms import Synchronizer
from pylfit.semantics import SynchronousConstrained
from pylfit.datasets import Dataset

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from tests_generator import random_CDMVLP, random_StateTransitionsDataset

random.seed(0)

class CDMVLP_tests(unittest.TestCase):
    """
        Unit test of module tabular_dataset.py
    """
    _nb_tests = 10

    _nb_transitions = 100

    _nb_features = 4

    _nb_targets = 3

    _nb_feature_values = 3

    _nb_target_values = 3

    _SUPPORTED_ALGORITHMS = ["synchronizer","synchronizer-pride"]

    def test_constructor(self):
        print(">> CDMVLP(features, targets, rules)")
        for i in range(self._nb_tests):
            dataset = random_StateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            model = CDMVLP(features=dataset.features, targets=dataset.targets)
            features = dataset.features
            targets = dataset.targets

            self.assertEqual(model.features, features)
            self.assertEqual(model.targets, targets)
            self.assertEqual(model.rules, [])
            self.assertEqual(model.constraints, [])
            self.assertEqual(model.algorithm, None)

            # Exceptions:
            #-------------

            # Features format
            features = '[("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]' # not list
            self.assertRaises(TypeError, CDMVLP, features, targets)

            features = [["x0", ["0","1"]], ("x1", ["0","1"]), ("x2", ["0","1"])] # not tuple
            self.assertRaises(TypeError, CDMVLP, features, targets)

            features = [("x0", "0","1"), ("x1", "0","1"), ("x2", ["0","1"])] # not tuple of size 2
            self.assertRaises(TypeError, CDMVLP, features, targets)

            features = [("x0", ["0","1"]), ("x1", '0","1"'), ("x2", ["0","1"])] # domain is not list
            self.assertRaises(TypeError, CDMVLP, features, targets)

            features = [("x0", ["0","1"]), ("x1", [0,"1"]), ("x2", ["0","1"])] # domain values are not string
            self.assertRaises(ValueError, CDMVLP, features, targets)

            # Targets format
            features = [("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]

            targets = '[("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]' # not list
            self.assertRaises(TypeError, CDMVLP, features, targets)

            targets = [["x0", ["0","1"]], ("x1", ["0","1"]), ("x2", ["0","1"])] # not tuple
            self.assertRaises(TypeError, CDMVLP, features, targets)

            targets = [("x0", "0","1"), ("x1", "0","1"), ("x2", ["0","1"])] # not tuple of size 2
            self.assertRaises(TypeError, CDMVLP, features, targets)

            targets = [("x0", ["0","1"]), ("x1", '0","1"'), ("x2", ["0","1"])] # domain is not list
            self.assertRaises(TypeError, CDMVLP, features, targets)

            targets = [("x0", ["0","1"]), ("x1", [0,"1"]), ("x2", ["0","1"])] # domain values are not string
            self.assertRaises(ValueError, CDMVLP, features, targets)

    def test_compile(self):
        print(">> CDMVLP.compile(algorithm)")

        for i in range(self._nb_tests):
            for algorithm in self._SUPPORTED_ALGORITHMS:
                dataset = random_StateTransitionsDataset( \
                nb_transitions=random.randint(1, self._nb_transitions), \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values)

                model = CDMVLP(features=dataset.features, targets=dataset.targets)

                model.compile()

                self.assertEqual(model.algorithm, "synchronizer") # default algorithm

                model.compile(algorithm=algorithm)

                self.assertEqual(model.algorithm, algorithm)

                self.assertRaises(ValueError, model.compile, "lol")
                self.assertRaises(ValueError, model.compile, "gula")
                #self.assertRaises(NotImplementedError, model.compile, "pride")
                #self.assertRaises(NotImplementedError, model.compile, "synchronizer-pride")

                original = CDMVLP._ALGORITHMS.copy()
                CDMVLP._ALGORITHMS = ["gula"]
                self.assertRaises(NotImplementedError, model.compile, "gula") # dataset not supported yet
                CDMVLP._ALGORITHMS = original

    def test_fit(self):
        print(">> CDMVLP.fit(dataset)")
        for i in range(self._nb_tests):
            dataset = random_StateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            for algorithm in self._SUPPORTED_ALGORITHMS:
                for verbose in [0,1]:

                    model = CDMVLP(features=dataset.features, targets=dataset.targets)
                    model.compile(algorithm=algorithm)
                    f = io.StringIO()
                    with contextlib.redirect_stderr(f):
                        model.fit(dataset=dataset, verbose=verbose)

                    expected_rules, expected_constraints = Synchronizer.fit(dataset, complete=(algorithm == "synchronizer"))
                    self.assertEqual(expected_rules, model.rules)
                    self.assertEqual(expected_constraints, model.constraints)

                    # Exceptions
                    #------------

                    model = CDMVLP(features=dataset.features, targets=dataset.targets)
                    model.compile(algorithm=algorithm)
                    self.assertRaises(ValueError, model.fit, [], verbose) # dataset is not of valid type

                    model.algorithm = "bad_value"
                    self.assertRaises(ValueError, model.fit, dataset, verbose) # algorithm not supported

                    model.algorithm = algorithm
                    original = CDMVLP._COMPATIBLE_DATASETS.copy()
                    class newdataset(Dataset):
                        def __init__(self, data, features, targets):
                            x = ""
                    CDMVLP._COMPATIBLE_DATASETS = [newdataset]
                    self.assertRaises(ValueError, model.fit, newdataset([],[],[]), verbose) # dataset not supported by the algo
                    CDMVLP._COMPATIBLE_DATASETS = original

                    model.algorithm = "gula"
                    original = CDMVLP._ALGORITHMS.copy()
                    class newdataset(Dataset):
                        def __init__(self, data, features, targets):
                            x = ""
                    CDMVLP._ALGORITHMS = ["gula"]
                    self.assertRaises(NotImplementedError, model.fit, dataset, verbose) # dataset not supported yet
                    CDMVLP._ALGORITHMS = original


    def test_predict(self):
        print(">> CDMVLP.predict()")
        for i in range(self._nb_tests):

            dataset = random_StateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            for algorithm in self._SUPPORTED_ALGORITHMS:
                model = CDMVLP(features=dataset.features, targets=dataset.targets)
                model.compile(algorithm=algorithm)
                model.fit(dataset=dataset)

                feature_states = list(set(tuple(s1) for s1,s2 in dataset.data))

                prediction = model.predict(feature_states)

                for state_id, s1 in enumerate(feature_states):
                    feature_state_encoded = []
                    for var_id, val in enumerate(s1):
                        val_id = model.features[var_id][1].index(str(val))
                        feature_state_encoded.append(val_id)

                    #eprint(feature_state_encoded)

                    target_states = SynchronousConstrained.next(feature_state_encoded, model.targets, model.rules, model.constraints)
                    output = []
                    for s in target_states:
                        target_state = []
                        for var_id, val_id in enumerate(s):
                            #eprint(var_id, val_id)
                            if val_id == -1:
                                target_state.append("?")
                            else:
                                target_state.append(model.targets[var_id][1][val_id])
                        output.append(target_state)
                    self.assertEqual(prediction[state_id][0], list(s1))
                    self.assertEqual(prediction[state_id][1], output)

                # Force missing value
                model.rules = [r for r in model.rules if r.head_variable != random.randint(0,len(model.targets))]

                prediction = model.predict(feature_states)
                for state_id, s1 in enumerate(feature_states):
                    feature_state_encoded = []
                    for var_id, val in enumerate(s1):
                        val_id = model.features[var_id][1].index(str(val))
                        feature_state_encoded.append(val_id)

                    #eprint(feature_state_encoded)

                    target_states = SynchronousConstrained.next(feature_state_encoded, model.targets, model.rules, model.constraints)
                    output = []
                    for s in target_states:
                        target_state = []
                        for var_id, val_id in enumerate(s):
                            #eprint(var_id, val_id)
                            if val_id == -1:
                                target_state.append("?")
                            else:
                                target_state.append(model.targets[var_id][1][val_id])
                        output.append(target_state)

                    self.assertEqual(prediction[state_id][1], output)

                # Exceptions:
                self.assertRaises(TypeError, model.predict, "") # Feature_states bad format: is not a list
                self.assertRaises(TypeError, model.predict, [["0","1"],0,10]) # Feature_states bad format: is not a list of list
                self.assertRaises(TypeError, model.predict, [["0","1"],[0,10]]) # Feature_states bad format: is not a list of list of string

                feature_states = [list(s) for s in set(tuple(s1) for s1,s2 in dataset.data)]
                state_id = random.randint(0,len(feature_states)-1)
                original = feature_states[state_id].copy()

                feature_states[state_id] = feature_states[state_id][:-random.randint(1,len(dataset.features))]
                self.assertRaises(TypeError, model.predict, feature_states) # Feature_states bad format: size of state not correspond to model features <
                feature_states[state_id] = original.copy()

                feature_states[state_id].extend(["0" for i in range(random.randint(1,10))])
                self.assertRaises(TypeError, model.predict, feature_states) # Feature_states bad format: size of state not correspond to model features >
                feature_states[state_id] = original.copy()

                var_id = random.randint(0,len(dataset.features)-1)
                feature_states[state_id][var_id] = "bad_value"
                self.assertRaises(ValueError, model.predict, feature_states) # Feature_states bad format: value out of domain
                feature_states[state_id] = original.copy()

    def test_summary(self):
        print(">> CDMVLP.summary()")
        for i in range(self._nb_tests):
            for algorithm in self._SUPPORTED_ALGORITHMS:
                # Empty CDMVLP
                model = random_CDMVLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values, \
                algorithm=algorithm)

                model.rules = []
                model.constraints = []

                expected_print = \
                "CDMVLP summary:\n"+\
                " Algorithm: "+ algorithm +"\n"
                expected_print += " Features: \n"
                for var,vals in model.features:
                    expected_print += "  " + var + ": " + str(vals) + "\n"
                expected_print += " Targets: \n"
                for var,vals in model.targets:
                    expected_print += "  " + var + ": " + str(vals) + "\n"
                expected_print +=" Rules: []\n"
                expected_print +=" Constraints: []\n"

                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                model.summary()
                sys.stdout = old_stdout

                self.assertEqual(mystdout.getvalue(), expected_print)

                # Usual CDMVLP
                model = random_CDMVLP( \
                nb_features=random.randint(2,self._nb_features), \
                nb_targets=random.randint(2,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values, \
                algorithm=algorithm)

                expected_print = \
                "CDMVLP summary:\n"+\
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
                if len(model.constraints) == 0:
                    expected_print +=" Constraints: []\n"
                else:
                    expected_print +=" Constraints:\n"
                    for r in model.constraints:
                        expected_print += "  "+r.logic_form(model.features, model.targets)+"\n"

                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                model.summary()
                sys.stdout = old_stdout

                self.assertEqual(mystdout.getvalue(), expected_print)

            # Exceptions
            #------------

            model = CDMVLP(features=model.features, targets=model.targets)
            self.assertRaises(ValueError, model.summary) # compile not called

    def test_to_string(self):
        print(">> pylfit.models.CDMVLP.to_string()")
        for i in range(self._nb_tests):
            for algorithm in self._SUPPORTED_ALGORITHMS:
                model = random_CDMVLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values, \
                algorithm=algorithm)

                expected = \
                 "{\n"+\
                 "Algorithm: " + algorithm +\
                 "\nFeatures: "+ str(model.features)+\
                 "\nTargets: "+ str(model.targets)+\
                 "\nRules:\n"
                for r in model.rules:
                    expected += r.logic_form(model.features, model.targets) + "\n"
                for r in model.constraints:
                    expected += r.logic_form(model.features, model.targets) + "\n"
                expected += "}"

                self.assertEqual(model.to_string(), expected)
'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
