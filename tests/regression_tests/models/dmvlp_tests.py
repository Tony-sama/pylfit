#-----------------------
# @author: Tony Ribeiro
# @created: 2020/12/23
# @updated: 2020/12/23
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

from io import StringIO

import pylfit
from pylfit.objects import Rule
from pylfit.models import DMVLP
from pylfit.algorithms import GULA
from pylfit.algorithms import PRIDE

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from tests_generator import random_DMVLP, random_StateTransitionsDataset

random.seed(0)

class DMVLP_tests(unittest.TestCase):
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
        print(">> DMVLP(features, targets, rules)")


        features = [("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]
        targets = [("y0", ["0","1"]), ("y1", ["0","1"]), ("y2", ["0","1"])]
        model = DMVLP(features=features, targets=targets)

        self.assertEqual(model.features, features)
        self.assertEqual(model.targets, targets)
        self.assertEqual(model.rules, [])
        self.assertEqual(model.algorithm, None)

        # Exceptions:
        #-------------

        # Features format
        features = '[("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]' # not list
        self.assertRaises(TypeError, DMVLP, features, targets)

        features = [["x0", ["0","1"]], ("x1", ["0","1"]), ("x2", ["0","1"])] # not tuple
        self.assertRaises(TypeError, DMVLP, features, targets)

        features = [("x0", "0","1"), ("x1", "0","1"), ("x2", ["0","1"])] # not tuple of size 2
        self.assertRaises(TypeError, DMVLP, features, targets)

        features = [("x0", ["0","1"]), ("x1", '0","1"'), ("x2", ["0","1"])] # domain is not list
        self.assertRaises(TypeError, DMVLP, features, targets)

        features = [("x0", ["0","1"]), ("x1", [0,"1"]), ("x2", ["0","1"])] # domain values are not string
        self.assertRaises(ValueError, DMVLP, features, targets)

        # Targets format
        features = [("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]

        targets = '[("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]' # not list
        self.assertRaises(TypeError, DMVLP, features, targets)

        targets = [["x0", ["0","1"]], ("x1", ["0","1"]), ("x2", ["0","1"])] # not tuple
        self.assertRaises(TypeError, DMVLP, features, targets)

        targets = [("x0", "0","1"), ("x1", "0","1"), ("x2", ["0","1"])] # not tuple of size 2
        self.assertRaises(TypeError, DMVLP, features, targets)

        targets = [("x0", ["0","1"]), ("x1", '0","1"'), ("x2", ["0","1"])] # domain is not list
        self.assertRaises(TypeError, DMVLP, features, targets)

        targets = [("x0", ["0","1"]), ("x1", [0,"1"]), ("x2", ["0","1"])] # domain values are not string
        self.assertRaises(ValueError, DMVLP, features, targets)

    def test_compile(self):
        print(">> DMVLP.compile(algorithm)")
        features = [("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]
        targets = [("y0", ["0","1"]), ("y1", ["0","1"]), ("y2", ["0","1"])]
        model = DMVLP(features=features, targets=targets)

        model.compile(algorithm="gula")

        self.assertEqual(model.algorithm, pylfit.algorithms.GULA)

        self.assertRaises(ValueError, model.compile, "lol")
        #self.assertRaises(NotImplementedError, model.compile, "pride")
        self.assertRaises(NotImplementedError, model.compile, "lf1t")

    def test_fit(self):
        print(">> DMVLP.fit(dataset, verbose)")

        for test in range(0,self._nb_tests):
            for verbose in [0,1]:

                dataset = random_StateTransitionsDataset( \
                nb_transitions=random.randint(1, self._nb_transitions), \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values)

                model = DMVLP(features=dataset.features, targets=dataset.targets)
                model.compile(algorithm="gula")
                model.fit(dataset=dataset, verbose=verbose)

                expected_rules = GULA.fit(dataset)
                self.assertEqual(expected_rules, model.rules)

                model = DMVLP(features=dataset.features, targets=dataset.targets)
                model.compile(algorithm="pride")
                model.fit(dataset=dataset)

                expected_rules = PRIDE.fit(dataset)
                self.assertEqual(expected_rules, model.rules)

                # Exceptions
                #------------

                self.assertRaises(ValueError, model.fit, []) # dataset is not of valid type

    def test_predict(self):
        print(">> DMVLP.predict()")

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

        dataset = pylfit.preprocessing.transitions_dataset_from_array(data=data, feature_names=feature_names, target_names=target_names)

        model = DMVLP(features=dataset.features, targets=dataset.targets)
        model.compile(algorithm="gula")
        model.fit(dataset=dataset)

        self.assertEqual(set([tuple(s) for s in model.predict([0,0,0])]), set([('1','0','0'), ('0','0','0'), ('0', '0', '1'), ('1','0','1')]))
        self.assertEqual(model.predict(["1","1","1"]), [['1', '1', '0']])
        self.assertEqual(model.predict([0,0,0], semantics="asynchronous"), [['1','0','0'], ['0', '0', '1']])
        self.assertEqual(set([tuple(s) for s in model.predict(['1','1','1'], semantics="general")]), set([('1', '1', '0'), ('1','1','1')]))
        self.assertEqual(set([tuple(s) for s in model.predict([0,0,0], semantics="general")]), set([('1','0','0'), ('0','0','0'), ('0', '0', '1'), ('1','0','1')]))

    def test_summary(self):
        print(">> DMVLP.summary()")

        # Empty DMVLP
        model = random_DMVLP( \
        nb_features=random.randint(1,self._nb_features), \
        nb_targets=random.randint(1,self._nb_targets), \
        max_feature_values=self._nb_feature_values, \
        max_target_values=self._nb_target_values, \
        algorithm="gula")

        model.rules = []

        expected_print = \
        "DMVLP summary:\n"+\
        " Algorithm: GULA (<class 'pylfit.algorithms.gula.GULA'>)\n"
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

        # Usual DMVLP
        model = random_DMVLP( \
        nb_features=random.randint(1,self._nb_features), \
        nb_targets=random.randint(1,self._nb_targets), \
        max_feature_values=self._nb_feature_values, \
        max_target_values=self._nb_target_values, \
        algorithm="gula")

        expected_print = \
        "DMVLP summary:\n"+\
        " Algorithm: GULA (<class 'pylfit.algorithms.gula.GULA'>)\n"
        expected_print += " Features: \n"
        for var,vals in model.features:
            expected_print += "  " + var + ": " + str(vals) + "\n"
        expected_print += " Targets: \n"
        for var,vals in model.targets:
            expected_print += "  " + var + ": " + str(vals) + "\n"
        expected_print +=" Rules:\n"
        for r in model.rules:
            expected_print += "  "+r.logic_form(model.features, model.targets)+"\n"

        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        model.summary()
        sys.stdout = old_stdout

        self.assertEqual(mystdout.getvalue(), expected_print)

        # Usual DMVLP
        model = random_DMVLP( \
        nb_features=random.randint(1,self._nb_features), \
        nb_targets=random.randint(1,self._nb_targets), \
        max_feature_values=self._nb_feature_values, \
        max_target_values=self._nb_target_values, \
        algorithm="pride")

        expected_print = \
        "DMVLP summary:\n"+\
        " Algorithm: PRIDE (<class 'pylfit.algorithms.pride.PRIDE'>)\n"
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

        model = DMVLP(features=model.features, targets=model.targets)
        self.assertRaises(ValueError, model.summary) # compile not called

    def test_to_string(self):
        print(">> pylfit.models.DMVLP.to_string()")
        for i in range(self._nb_tests):
            model = random_DMVLP( \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values, \
            algorithm="gula")

            expected = \
             "{\n"+\
             "Algorithm: " + str(model.algorithm.__name__)+\
             "\nFeatures: "+ str(model.features)+\
             "\nTargets: "+ str(model.targets)+\
             "\nRules:\n"
            for r in model.rules:
                expected += r.logic_form(model.features, model.targets) + "\n"
            expected += "}"

            self.assertEqual(model.to_string(), expected)


    def test_feature_states(self):
        print(">> pylfit.models.DMVLP.feature_states()")
        for i in range(self._nb_tests):
            model = random_DMVLP( \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values, \
            algorithm="gula")

            values_ids = [[j for j in range(0,len(model.features[i][1]))] for i in range(0,len(model.features))]
            expected = [list(i) for i in list(itertools.product(*values_ids))]

            output = model.feature_states(True)

            for state in output:
                self.assertTrue(state in expected)
            for state in expected:
                self.assertTrue(state in output)

            values_ids = [[model.features[i][1][j] for j in range(0,len(model.features[i][1]))] for i in range(0,len(model.features))]
            expected = [list(i) for i in list(itertools.product(*values_ids))]

            output = model.feature_states(False)

            for state in output:
                self.assertTrue(state in expected)
            for state in expected:
                self.assertTrue(state in output)


    def test_target_states(self):
        print(">> pylfit.models.DMVLP.target_states()")
        for i in range(self._nb_tests):
            model = random_DMVLP( \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values, \
            algorithm="gula")

            values_ids = [[j for j in range(0,len(model.targets[i][1]))] for i in range(0,len(model.targets))]
            expected = [list(i) for i in list(itertools.product(*values_ids))]

            output = model.target_states(True)

            for state in output:
                self.assertTrue(state in expected)
            for state in expected:
                self.assertTrue(state in output)

            values_ids = [[model.targets[i][1][j] for j in range(0,len(model.targets[i][1]))] for i in range(0,len(model.targets))]
            expected = [list(i) for i in list(itertools.product(*values_ids))]

            output = model.target_states(False)

            for state in output:
                self.assertTrue(state in expected)
            for state in expected:
                self.assertTrue(state in output)
'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
