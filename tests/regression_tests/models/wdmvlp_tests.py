#-----------------------
# @author: Tony Ribeiro
# @created: 2021/02/16
# @updated: 2021/02/16
#
# @desc: dataset class unit test script
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
from pylfit.models import WDMVLP
from pylfit.algorithms import GULA
from pylfit.algorithms import PRIDE
from pylfit.utils import eprint

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from tests_generator import random_WDMVLP, random_StateTransitionsDataset

random.seed(0)

class WDMVLP_tests(unittest.TestCase):
    """
        Unit test of module wdmvlp.py
    """
    _nb_tests = 10

    _nb_transitions = 100

    _nb_features = 2

    _nb_targets = 2

    _nb_feature_values = 3

    _nb_target_values = 3

    def test_constructor(self):
        print(">> WDMVLP(features, targets, rules, unlikeliness_rules)")


        features = [("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]
        targets = [("y0", ["0","1"]), ("y1", ["0","1"]), ("y2", ["0","1"])]
        model = WDMVLP(features=features, targets=targets)

        self.assertEqual(model.features, features)
        self.assertEqual(model.targets, targets)
        self.assertEqual(model.rules, [])
        self.assertEqual(model.unlikeliness_rules, [])
        self.assertEqual(model.algorithm, None)

        # Exceptions:
        #-------------

        # Features format
        features = '[("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]' # not list
        self.assertRaises(TypeError, WDMVLP, features, targets)

        features = [["x0", ["0","1"]], ("x1", ["0","1"]), ("x2", ["0","1"])] # not tuple
        self.assertRaises(TypeError, WDMVLP, features, targets)

        features = [("x0", "0","1"), ("x1", "0","1"), ("x2", ["0","1"])] # not tuple of size 2
        self.assertRaises(TypeError, WDMVLP, features, targets)

        features = [("x0", ["0","1"]), ("x1", '0","1"'), ("x2", ["0","1"])] # domain is not list
        self.assertRaises(TypeError, WDMVLP, features, targets)

        features = [("x0", ["0","1"]), ("x1", [0,"1"]), ("x2", ["0","1"])] # domain values are not string
        self.assertRaises(ValueError, WDMVLP, features, targets)

        # Targets format
        features = [("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]

        targets = '[("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]' # not list
        self.assertRaises(TypeError, WDMVLP, features, targets)

        targets = [["x0", ["0","1"]], ("x1", ["0","1"]), ("x2", ["0","1"])] # not tuple
        self.assertRaises(TypeError, WDMVLP, features, targets)

        targets = [("x0", "0","1"), ("x1", "0","1"), ("x2", ["0","1"])] # not tuple of size 2
        self.assertRaises(TypeError, WDMVLP, features, targets)

        targets = [("x0", ["0","1"]), ("x1", '0","1"'), ("x2", ["0","1"])] # domain is not list
        self.assertRaises(TypeError, WDMVLP, features, targets)

        targets = [("x0", ["0","1"]), ("x1", [0,"1"]), ("x2", ["0","1"])] # domain values are not string
        self.assertRaises(ValueError, WDMVLP, features, targets)

    def test_compile(self):
        print(">> WDMVLP.compile(algorithm)")
        features = [("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]
        targets = [("y0", ["0","1"]), ("y1", ["0","1"]), ("y2", ["0","1"])]
        model = WDMVLP(features=features, targets=targets)

        for algo in ["gula","pride"]:
            model.compile(algorithm=algo)

            if algo == "gula":
                self.assertEqual(model.algorithm, pylfit.algorithms.GULA)
            else:
                self.assertEqual(model.algorithm, pylfit.algorithms.PRIDE)

            self.assertRaises(ValueError, model.compile, "lol")
            #self.assertRaises(NotImplementedError, model.compile, "pride")
            self.assertRaises(NotImplementedError, model.compile, "lf1t")

    def test_fit(self):
        print(">> WDMVLP.fit(dataset)")

        for test in range(0,self._nb_tests):

            dataset = random_StateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            for algo in ["gula","pride"]:
                for verbose in [0,1]:

                    model = WDMVLP(features=dataset.features, targets=dataset.targets)
                    model.compile(algorithm=algo)
                    model.fit(dataset=dataset,verbose=verbose)

                    weighted_rules = {}
                    train_init = [GULA.encode_state(s1, dataset.features) for s1,s2 in dataset.data]
                    for w,r in model.rules:
                        weight = 0
                        for s1 in train_init:
                            if r.matches(s1):
                                weight += 1
                        self.assertEqual(w,weight)

                    for w,r in model.unlikeliness_rules:
                        weight = 0
                        for s1 in train_init:
                            if r.matches(s1):
                                weight += 1
                        self.assertEqual(w,weight)

                    # TODO: check no missing rules

                    #model = WDMVLP(features=dataset.features, targets=dataset.targets)
                    #model.compile(algorithm="pride")
                    #model.fit(dataset=dataset)

                    #expected_rules = PRIDE.fit(dataset)
                    #self.assertEqual(expected_rules, model.rules)

                    # Exceptions
                    #------------

                    self.assertRaises(ValueError, model.fit, []) # dataset is not of valid type

    def test_predict(self):
        print(">> WDMVLP.predict()")

        for test in range(0,self._nb_tests):

            dataset = random_StateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            for algo in ["gula", "pride"]:
                for raw_rules in [True, False]:
                    model = WDMVLP(features=dataset.features, targets=dataset.targets)
                    model.compile(algorithm=algo)
                    model.fit(dataset=dataset)

                    feature_state = random.choice(model.feature_states())
                    output = model.predict(feature_state, raw_rules)
                    self.assertEqual(len(output.items()), len(model.targets))

                    feature_state = GULA.encode_state(feature_state, model.features)

                    for var_id, (var, vals) in enumerate(model.targets):
                        self.assertEqual(len(output[var]), len(model.targets[var_id][1]))
                        for val_id, val in enumerate(vals):
                            best_rule = None
                            max_rule_weight = 0
                            for w,r in model.rules:
                                if r.head_variable == var_id and r.head_value == val_id:
                                    if r.matches(feature_state) and w > max_rule_weight:
                                        best_rule = r
                                        max_rule_weight = w

                            best_anti_rule = None
                            max_anti_rule_weight = 0
                            for w,r in model.unlikeliness_rules:
                                if r.head_variable == var_id and r.head_value == val_id:
                                    if r.matches(feature_state) and w > max_anti_rule_weight:
                                        best_anti_rule = r
                                        max_anti_rule_weight = w

                            if not raw_rules:
                                if best_rule is not None:
                                    best_rule = best_rule.logic_form(model.features, model.targets)
                                if best_anti_rule is not None:
                                    best_anti_rule = best_anti_rule.logic_form(model.features, model.targets)

                            prediction = round(0.5 + 0.5*(max_rule_weight - max_anti_rule_weight) / max(1,(max_rule_weight+max_anti_rule_weight)),3)

                            self.assertEqual(output[var][val], (prediction, (max_rule_weight, best_rule), (max_anti_rule_weight, best_anti_rule)) )

                #self.assertEqual(model.predict([0,0,0]), [['0', '0', '0'], ['0', '0', '1'], ['1', '0', '0'], ['1', '0', '1']])

    def test_summary(self):
        print(">> WDMVLP.summary()")

        # Empty WDMVLP
        model = random_WDMVLP( \
        nb_features=random.randint(1,self._nb_features), \
        nb_targets=random.randint(1,self._nb_targets), \
        max_feature_values=self._nb_feature_values, \
        max_target_values=self._nb_target_values, \
        algorithm="gula")

        model.rules = []

        expected_print = \
        "WDMVLP summary:\n"+\
        " Algorithm: GULA (<class 'pylfit.algorithms.gula.GULA'>)\n"
        expected_print += " Features: \n"
        for var,vals in model.features:
            expected_print += "  " + var + ": " + str(vals) + "\n"
        expected_print += " Targets: \n"
        for var,vals in model.targets:
            expected_print += "  " + var + ": " + str(vals) + "\n"
        expected_print +=" Likeliness rules: []\n"
        expected_print +=" Unlikeliness rules: []\n"

        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        model.summary()
        sys.stdout = old_stdout

        self.assertEqual(mystdout.getvalue(), expected_print)

        # Usual WDMVLP
        for algo in ["gula","pride"]:
            model = random_WDMVLP( \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values, \
            algorithm=algo)

            expected_print = "WDMVLP summary:\n"
            if algo == "gula":
                expected_print += " Algorithm: GULA (<class 'pylfit.algorithms.gula.GULA'>)\n"
            else:
                expected_print += " Algorithm: PRIDE (<class 'pylfit.algorithms.pride.PRIDE'>)\n"
            expected_print += " Features: \n"
            for var,vals in model.features:
                expected_print += "  " + var + ": " + str(vals) + "\n"
            expected_print += " Targets: \n"
            for var,vals in model.targets:
                expected_print += "  " + var + ": " + str(vals) + "\n"
            expected_print +=" Likeliness rules:\n"
            if len(model.rules) == 0:
                expected_print +=" Likeliness rules: []\n"
            else:
                for w,r in model.rules:
                    expected_print += "  "+ str(w) + ", " +r.logic_form(model.features, model.targets)+ "\n"
            if len(model.unlikeliness_rules) == 0:
                expected_print +=" Unlikeliness rules: []\n"
            else:
                expected_print +=" Unlikeliness rules:\n"
                for w,r in model.unlikeliness_rules:
                    expected_print += "  "+ str(w) + ", " +r.logic_form(model.features, model.targets)+ "\n"

            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            model.summary()
            sys.stdout = old_stdout

            self.assertEqual(mystdout.getvalue(), expected_print)

            # Exceptions
            #------------

            model = WDMVLP(features=model.features, targets=model.targets)
            self.assertRaises(ValueError, model.summary) # compile not called

    def test_to_string(self):
        print(">> pylfit.models.DMVLP.to_string()")
        for i in range(self._nb_tests):
            for algo in ["gula","pride"]:
                model = random_WDMVLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values, \
                algorithm=algo)

                expected = \
                 "{\n"+\
                 "Algorithm: " + str(model.algorithm.__name__)+\
                 "\nFeatures: "+ str(model.features)+\
                 "\nTargets: "+ str(model.targets)+\
                 "\nLikeliness rules:\n"
                for w,r in model.rules:
                    expected += "(" + str(w) + ", " + r.logic_form(model.features, model.targets) + ")\n"
                expected += "\nUnlikeliness rules:\n"
                for w,r in model.unlikeliness_rules:
                    expected += "(" + str(w) + ", " + r.logic_form(model.features, model.targets) + ")\n"
                expected += "}"

                self.assertEqual(model.to_string(), expected)

'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
