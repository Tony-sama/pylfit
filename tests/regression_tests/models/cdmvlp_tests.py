#-----------------------
# @author: Tony Ribeiro
# @created: 2021/02/05
# @updated: 2021/02/05
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

from io import StringIO

import pylfit
from pylfit.objects import Rule
from pylfit.models import CDMVLP
from pylfit.algorithms import Synchronizer
from pylfit.semantics import SynchronousConstrained

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

    _nb_features = 5

    _nb_targets = 5

    _nb_feature_values = 3

    _nb_target_values = 3

    def test_constructor(self):
        print(">> CDMVLP(features, targets, rules)")


        features = [("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]
        targets = [("y0", ["0","1"]), ("y1", ["0","1"]), ("y2", ["0","1"])]
        model = CDMVLP(features=features, targets=targets)

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
        features = [("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]
        targets = [("y0", ["0","1"]), ("y1", ["0","1"]), ("y2", ["0","1"])]
        model = CDMVLP(features=features, targets=targets)

        model.compile(algorithm="synchronizer")

        self.assertEqual(model.algorithm, pylfit.algorithms.Synchronizer)

        self.assertRaises(ValueError, model.compile, "lol")
        #self.assertRaises(NotImplementedError, model.compile, "pride")
        self.assertRaises(NotImplementedError, model.compile, "synchronizer-pride")

    def test_fit(self):
        print(">> CDMVLP.fit(dataset)")

        dataset = random_StateTransitionsDataset( \
        nb_transitions=random.randint(1, self._nb_transitions), \
        nb_features=random.randint(1,self._nb_features), \
        nb_targets=random.randint(1,self._nb_targets), \
        max_feature_values=self._nb_feature_values, \
        max_target_values=self._nb_target_values)

        model = CDMVLP(features=dataset.features, targets=dataset.targets)
        model.compile(algorithm="synchronizer")
        model.fit(dataset=dataset)

        expected_rules, expected_constraints = Synchronizer.fit(dataset)
        self.assertEqual(expected_rules, model.rules)
        self.assertEqual(expected_constraints, model.constraints)

        # Exceptions
        #------------

        self.assertRaises(ValueError, model.fit, []) # dataset is not of valid type

    def test_predict(self):
        print(">> CDMVLP.predict()")

        dataset = random_StateTransitionsDataset( \
        nb_transitions=random.randint(1, self._nb_transitions), \
        nb_features=random.randint(1,self._nb_features), \
        nb_targets=random.randint(1,self._nb_targets), \
        max_feature_values=self._nb_feature_values, \
        max_target_values=self._nb_target_values)

        model = CDMVLP(features=dataset.features, targets=dataset.targets)
        model.compile(algorithm="synchronizer")
        model.fit(dataset=dataset)

        for s1, s2 in dataset.data:
            prediction = model.predict(s1)

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

            self.assertEqual(prediction, output)


    def test_summary(self):
        print(">> CDMVLP.summary()")

        # Empty CDMVLP
        model = random_CDMVLP( \
        nb_features=random.randint(1,self._nb_features), \
        nb_targets=random.randint(1,self._nb_targets), \
        max_feature_values=self._nb_feature_values, \
        max_target_values=self._nb_target_values, \
        algorithm="synchronizer")

        model.rules = []
        model.constraints = []

        expected_print = \
        "CDMVLP summary:\n"+\
        " Algorithm: Synchronizer (<class 'pylfit.algorithms.synchronizer.Synchronizer'>)\n"
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
        nb_features=random.randint(1,self._nb_features), \
        nb_targets=random.randint(1,self._nb_targets), \
        max_feature_values=self._nb_feature_values, \
        max_target_values=self._nb_target_values, \
        algorithm="synchronizer")

        expected_print = \
        "CDMVLP summary:\n"+\
        " Algorithm: Synchronizer (<class 'pylfit.algorithms.synchronizer.Synchronizer'>)\n"
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
            model = random_CDMVLP( \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values, \
            algorithm="synchronizer")

            expected = \
             "{\n"+\
             "Algorithm: " + str(model.algorithm.__name__)+\
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
