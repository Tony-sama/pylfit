#-----------------------
# @author: Tony Ribeiro
# @created: 2022/08/16
# @updated: 2022/08/16
#
# @desc: dataset class unit test script
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
from pylfit.objects import Rule
from pylfit.models import PDMVLP
from pylfit.algorithms import Algorithm, GULA, PRIDE, Probalizer
from pylfit.datasets import Dataset
from pylfit.semantics import SynchronousConstrained

from pylfit.utils import eprint

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from tests_generator import random_PDMVLP, random_DiscreteStateTransitionsDataset, random_symmetric_DiscreteStateTransitionsDataset

random.seed(0)

class PDMVLP_tests(unittest.TestCase):
    """
        Unit test of module tabular_dataset.py
    """
    _nb_tests = 10

    _nb_transitions = 100

    _nb_features = 3

    _nb_targets = 3

    _nb_feature_values = 3

    _nb_target_values = 3

    _SUPPORTED_ALGORITHMS = ["gula","pride","synchronizer"] #,"lf1t"

    def test_constructor(self):
        print(">> PDMVLP(features, targets, rules)")


        features = [("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]
        targets = [("y0", ["0","1"]), ("y1", ["0","1"]), ("y2", ["0","1"])]
        model = PDMVLP(features=features, targets=targets)

        self.assertEqual(model.features, features)
        self.assertEqual(model.targets, targets)
        self.assertEqual(model.rules, [])
        self.assertEqual(model.algorithm, None)

        # Exceptions:
        #-------------

        # Features format
        features = '[("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]' # not list
        self.assertRaises(TypeError, PDMVLP, features, targets)

        features = [["x0", ["0","1"]], ("x1", ["0","1"]), ("x2", ["0","1"])] # not tuple
        self.assertRaises(TypeError, PDMVLP, features, targets)

        features = [("x0", "0","1"), ("x1", "0","1"), ("x2", ["0","1"])] # not tuple of size 2
        self.assertRaises(TypeError, PDMVLP, features, targets)

        features = [("x0", ["0","1"]), ("x1", '0","1"'), ("x2", ["0","1"])] # domain is not list
        self.assertRaises(TypeError, PDMVLP, features, targets)

        features = [("x0", ["0","1"]), ("x1", [0,"1"]), ("x2", ["0","1"])] # domain values are not string
        self.assertRaises(ValueError, PDMVLP, features, targets)

        # Targets format
        features = [("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]

        targets = '[("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]' # not list
        self.assertRaises(TypeError, PDMVLP, features, targets)

        targets = [["x0", ["0","1"]], ("x1", ["0","1"]), ("x2", ["0","1"])] # not tuple
        self.assertRaises(TypeError, PDMVLP, features, targets)

        targets = [("x0", "0","1"), ("x1", "0","1"), ("x2", ["0","1"])] # not tuple of size 2
        self.assertRaises(TypeError, PDMVLP, features, targets)

        targets = [("x0", ["0","1"]), ("x1", '0","1"'), ("x2", ["0","1"])] # domain is not list
        self.assertRaises(TypeError, PDMVLP, features, targets)

        targets = [("x0", ["0","1"]), ("x1", [0,"1"]), ("x2", ["0","1"])] # domain values are not string
        self.assertRaises(ValueError, PDMVLP, features, targets)

    def test_copy(self):
        print(">> PDMVLP.copy()")

        features = [("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]
        targets = [("y0", ["0","1"]), ("y1", ["0","1"]), ("y2", ["0","1"])]
        model = PDMVLP(features=features, targets=targets)

        copy = model.copy()

        self.assertEqual(model.features, copy.features)
        self.assertEqual(model.targets, copy.targets)
        self.assertEqual(model.rules, copy.rules)
        self.assertEqual(model.algorithm, copy.algorithm)

        for i in range(self._nb_tests):
            for algorithm in self._SUPPORTED_ALGORITHMS:
                model = random_PDMVLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values, \
                algorithm=algorithm)

                copy = model.copy()

                self.assertEqual(model.features, copy.features)
                self.assertEqual(model.targets, copy.targets)
                self.assertEqual(model.rules, copy.rules)
                self.assertEqual(model.algorithm, copy.algorithm)

    def test_compile(self):
        print(">> PDMVLP.compile(algorithm)")
        features = [("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]
        targets = [("y0", ["0","1"]), ("y1", ["0","1"]), ("y2", ["0","1"])]
        model = PDMVLP(features=features, targets=targets)

        model.compile(algorithm="gula")
        self.assertEqual(model.algorithm, "gula")

        model.compile(algorithm="pride")
        self.assertEqual(model.algorithm, "pride")

        self.assertRaises(ValueError, model.compile, "lol")
        #self.assertRaises(NotImplementedError, model.compile, "pride")

    def test_fit(self):
        print(">> PDMVLP.fit(dataset, verbose)")

        for test in range(0,self._nb_tests):
            for verbose in [0,1]:

                dataset = random_DiscreteStateTransitionsDataset( \
                nb_transitions=random.randint(1, self._nb_transitions), \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values)

                model = PDMVLP(features=dataset.features, targets=dataset.targets)
                model.compile(algorithm="gula")
                f = io.StringIO()
                with contextlib.redirect_stderr(f):
                    model.fit(dataset=dataset, verbose=verbose)

                expected_targets, expected_rules, expected_constraints = Probalizer.fit(dataset, complete=True)
                self.assertEqual(expected_rules, model.rules)
                self.assertEqual(expected_constraints, [])

                model = PDMVLP(features=dataset.features, targets=dataset.targets)
                model.compile(algorithm="pride")
                f = io.StringIO()
                with contextlib.redirect_stderr(f):
                    model.fit(dataset=dataset, verbose=verbose)

                expected_targets, expected_rules, expected_constraints = Probalizer.fit(dataset, complete=False)
                self.assertEqual(expected_rules, model.rules)
                self.assertEqual(expected_constraints, [])

                # Exceptions
                #------------

                for algorithm in self._SUPPORTED_ALGORITHMS:
                    model = PDMVLP(features=dataset.features, targets=dataset.targets)
                    model.compile(algorithm=algorithm)

                    self.assertRaises(ValueError, model.fit, []) # dataset is not of valid type

                    model.algorithm = "gulaaaaa"
                    self.assertRaises(ValueError, model.fit, dataset, verbose) # algorithm is not of valid
                    model.algorithm = algorithm

                    original = PDMVLP._COMPATIBLE_DATASETS.copy()
                    class newdataset(Dataset):
                        def __init__(self, data, features, targets):
                            x = ""
                    PDMVLP._COMPATIBLE_DATASETS = [newdataset]
                    self.assertRaises(ValueError, model.fit, newdataset([],[],[]), verbose) # dataset not supported by the algo
                    PDMVLP._COMPATIBLE_DATASETS = original

    def test_predict(self):
        print(">> PDMVLP.predict()")

        data = [ \
        ([0,0,0],[0,0,1]), \
        ([0,0,0],[1,0,0]), \
        ([1,0,0],[0,0,0]), \
        ([0,1,0],[1,0,1]), \
        ([0,0,1],[0,0,1]), \
        ([1,1,0],[1,0,0]), \
        ([1,0,1],[0,1,0]), \
        ([0,1,1],[1,0,1]), \
        ([1,1,1],[1,1,0])]
        feature_names=["p_t-1","q_t-1","r_t-1"]
        target_names=["p_t","q_t","r_t"]

        dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=data, feature_names=feature_names, target_names=target_names)

        model = PDMVLP(features=dataset.features, targets=dataset.targets)
        model.compile(algorithm="gula")
        model.fit(dataset=dataset)

        self.assertEqual(set([tuple(s) for s in model.predict([["0","0","0"]])[("0","0","0")]]), set([('1','0','0'), ('0','0','0'), ('0', '0', '1'), ('1','0','1')]))
        self.assertEqual(set(tuple(s) for s in model.predict([["1","1","1"]])[("1","1","1")]), set([('1', '1', '0')]))
        #self.assertEqual(set(tuple(s) for s in model.predict([["0","0","0"]], semantics="asynchronous")[("0","0","0")]), set([('1','0','0'), ('0', '0', '1')]))
        #self.assertEqual(set([tuple(s) for s in model.predict([['1','1','1']], semantics="general")[("1","1","1")]]), set([('1', '1', '0'), ('1','1','1')]))
        #self.assertEqual(set([tuple(s) for s in model.predict([["0","0","0"]], semantics="general")[("0","0","0")]]), set([('1','0','0'), ('0','0','0'), ('0', '0', '1'), ('1','0','1')]))

        for i in range(self._nb_tests):
            dataset = random_DiscreteStateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            for algorithm in self._SUPPORTED_ALGORITHMS:
                model = PDMVLP(features=dataset.features, targets=dataset.targets)
                model.compile(algorithm=algorithm)
                model.fit(dataset=dataset)

                feature_states = [list(s) for s in set(tuple(s1) for s1,s2 in dataset.data)]

                prediction = model.predict(feature_states)

                for state_id, s1 in enumerate(feature_states):
                    feature_state_encoded = []
                    for var_id, val in enumerate(s1):
                        val_id = model.features[var_id][1].index(str(val))
                        feature_state_encoded.append(val_id)

                    #eprint(feature_state_encoded)

                    target_states = SynchronousConstrained.next(feature_state_encoded, model.targets, model.rules, model.constraints)
                    output = dict()
                    for s, rules in target_states.items():
                        target_state = []
                        for var_id, val_id in enumerate(s):
                            #eprint(var_id, val_id)
                            if val_id == -1:
                                target_state.append("?")
                            else:
                                target_state.append(model.targets[var_id][1][val_id])

                        # proba of target state
                        if algorithm == "synchronizer":
                            val_proba = target_state[0].split(",")[1]
                            target_state_proba =  int(val_proba.split("/")[0]) / int(val_proba.split("/")[1])
                        else:
                            target_state_proba = 1.0
                            for var_id, val in enumerate(target_state):
                                val_label = val.split(",")[0]
                                val_proba = val.split(",")[1]
                                val_proba = int(val_proba.split("/")[0]) / int(val_proba.split("/")[1])
                                target_state_proba *= val_proba
                                target_state[var_id] = val_label

                        output[tuple(target_state)] = (target_state_proba, rules)

                    self.assertEqual(prediction[tuple(s1)], output)

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

'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
