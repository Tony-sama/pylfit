#-----------------------
# @author: Tony Ribeiro
# @created: 2021/02/16
# @updated: 2021/06/15
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
import io
import contextlib

from io import StringIO

import pylfit
from pylfit.objects import Rule
from pylfit.objects import LegacyAtom
from pylfit.models import WDMVLP
from pylfit.algorithms import Algorithm
from pylfit.algorithms import GULA
from pylfit.algorithms import PRIDE
from pylfit.utils import eprint
from pylfit.datasets import Dataset

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from tests_generator import random_WDMVLP, random_DiscreteStateTransitionsDataset

#random.seed(0)

class WDMVLP_tests(unittest.TestCase):
    """
        Unit test of module wdmvlp.py
    """
    _nb_tests = 100

    _nb_transitions = 100

    _nb_features = 3

    _nb_targets = 2

    _nb_feature_values = 3

    _nb_target_values = 3

    _ALGORITHMS = ["gula", "pride", "brute-force"]

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

    def test___str__(self):
        print(">> pylfit.models.DMVLP.__str__()")
        for i in range(self._nb_tests):
            for algo in self._ALGORITHMS:
                model = random_WDMVLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values, \
                algorithm=algo)

                self.assertEqual(model.__str__(), model.to_string())
    def test___repr__(self):
        print(">> pylfit.models.DMVLP.__repr__()")
        for i in range(self._nb_tests):
            for algo in self._ALGORITHMS:
                model = random_WDMVLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values, \
                algorithm=algo)

                self.assertEqual(model.__repr__(), model.to_string())

    def test_compile(self):
        print(">> WDMVLP.compile(algorithm)")
        features = [("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]
        targets = [("y0", ["0","1"]), ("y1", ["0","1"]), ("y2", ["0","1"])]
        model = WDMVLP(features=features, targets=targets)

        for algo in self._ALGORITHMS:
            model.compile(algorithm=algo)

            self.assertEqual(model.algorithm, algo)

            self.assertRaises(ValueError, model.compile, "lol")
            #self.assertRaises(NotImplementedError, model.compile, "pride")
            #self.assertRaises(NotImplementedError, model.compile, "lf1t")

    def test_fit(self):
        print(">> WDMVLP.fit(dataset)")

        for test in range(0,self._nb_tests):

            dataset = random_DiscreteStateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            for algo in self._ALGORITHMS:
                for verbose in [0,1]:

                    model = WDMVLP(features=dataset.features, targets=dataset.targets)
                    model.compile(algorithm=algo)
                    f = io.StringIO()
                    with contextlib.redirect_stderr(f):
                        model.fit(dataset=dataset,verbose=verbose)

                    weighted_rules = {}
                    train_init = set(tuple(s1) for s1,s2 in dataset.data)

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
                    #self.assertEqual(expected_rules, model.rules)s

                    # Exceptions
                    #------------

                    self.assertRaises(ValueError, model.fit, []) # dataset is not of valid type
                    original = WDMVLP._COMPATIBLE_DATASETS.copy()
                    class newdataset(Dataset):
                        def __init__(self, data, features, targets):
                            x = ""
                    WDMVLP._COMPATIBLE_DATASETS = [newdataset]
                    self.assertRaises(ValueError, model.fit, newdataset([],[],[]), verbose) # dataset not supported by the algo
                    WDMVLP._COMPATIBLE_DATASETS = original

                    #model.algorithm = "lf1t"
                    #self.assertRaises(NotImplementedError, model.fit, dataset, verbose) # algorithm is not of valid)

    def test_extend(self):
        print(">> WDMVLP.extend(dataset, feature_states)")

        for test in range(0,self._nb_tests):

            dataset = random_DiscreteStateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            for algo in self._ALGORITHMS:
                for verbose in [0,1]:

                    model = WDMVLP(features=dataset.features, targets=dataset.targets)
                    model.compile(algorithm=algo)
                    f = io.StringIO()
                    with contextlib.redirect_stderr(f):
                        model.fit(dataset=dataset,verbose=verbose)

                    original_rules = model.rules.copy()
                    original_unlikeliness_rules = model.unlikeliness_rules.copy()

                    values_ids = [[j for j in dataset.features[i][1]] for i in range(0,len(dataset.features))]
                    feature_states = [list(i) for i in list(itertools.product(*values_ids))]

                    feature_states_to_match = [random.choice(feature_states) for i in range(10)]
                    #eprint(feature_states_to_match)
                    #eprint(model.features)

                    model.extend(dataset, feature_states_to_match)

                    # No rule disapear
                    for (w,r) in original_rules:
                        self.assertTrue((w,r) in model.rules)
                    for (w,r) in original_unlikeliness_rules:
                        self.assertTrue((w,r) in model.unlikeliness_rules)

                    # atmost one aditional rule per feature state for each var/val
                    for var_id, (var,vals) in enumerate(dataset.targets):
                        for val_id, val in enumerate(vals):
                            self.assertTrue(len([(w,r) for (w,r) in model.rules if r.head.variable == var if r.head.value == val if (w,r) not in original_rules]) <= len(feature_states))
                            self.assertTrue(len([(w,r) for (w,r) in model.unlikeliness_rules if r.head.variable == var if r.head.value == val if (w,r) not in original_unlikeliness_rules]) <= len(feature_states))


                    for feature_state in feature_states_to_match:
                        for var_id, (var,vals) in enumerate(dataset.targets):
                            for val_id, val in enumerate(vals):
                                #eprint("var: ", var_id)
                                #eprint("val: ", val_id)
                                head = LegacyAtom(var, set(dataset.targets[var_id][1]), val, var_id)
                                pos, neg = PRIDE.interprete(dataset, head)

                                # Only way to not match is no rule can be find
                                new_rule = PRIDE.find_one_optimal_rule_of(head, dataset, pos, neg, feature_state, 0)
                                matched = False
                                for w,r in model.rules:
                                    if r.head.variable == var and r.head.value == val and r.matches(feature_state):
                                        matched = True
                                        break

                                if not matched:
                                    self.assertTrue(new_rule is None)

                                # Only way to not match is no unlikeliness rule can be find
                                new_unlikeliness_rule = PRIDE.find_one_optimal_rule_of(head, dataset, neg, pos, feature_state, 0)
                                matched = False
                                for w,r in model.unlikeliness_rules:
                                    if r.head.variable == var and r.head.value == val and r.matches(feature_state):
                                        matched = True
                                        break

                                if not matched:
                                    self.assertTrue(new_unlikeliness_rule is None)

                    # check rules
                    for var_id, (var,vals) in enumerate(dataset.targets):
                        for val_id, val in enumerate(vals):
                            head = LegacyAtom(var, set(dataset.targets[var_id][1]), val, var_id)
                            pos, neg = PRIDE.interprete(dataset, head)
                            new_likely_rules = [x for x in model.rules if x not in original_rules]
                            new_unlikeliness_rules = [x for x in model.unlikeliness_rules if x not in original_unlikeliness_rules]
                            unlikely_check = False
                            for new_rules in [new_likely_rules, new_unlikeliness_rules]:

                                if unlikely_check:
                                    pos_ = pos
                                    pos = neg
                                    neg = pos_

                                for w,r in [(w,r) for (w,r) in new_rules if r.head.variable==var if r.head.value==val]:
                                    # Cover at least a positive
                                    cover = False
                                    for s in pos:
                                        if r.matches(s):
                                            cover = True
                                            break

                                    self.assertTrue(cover)

                                    # No negative is covered
                                    cover = False
                                    for s in neg:
                                        if r.matches(s):
                                            cover = True
                                            break
                                    self.assertFalse(cover)

                                    # Rules is minimal
                                    for var in r.body:
                                        r_ = r.copy()
                                        r_.remove_condition(var) # Try remove condition

                                        conflict = False
                                        for s in neg:
                                            if r_.matches(s): # Cover a negative example
                                                conflict = True
                                                break
                                        self.assertTrue(conflict)

                                unlikely_check = True

                    # Check weights
                    feature_states = set(tuple(s1) for s1,s2 in dataset.data)
                    for w,r in model.rules:
                        expected_weight = 0
                        for s in feature_states:
                            if r.matches(s):
                                expected_weight += 1
                        self.assertEqual(w,expected_weight)

                    for w,r in model.unlikeliness_rules:
                        expected_weight = 0
                        for s in feature_states:
                            if r.matches(s):
                                expected_weight += 1
                        self.assertEqual(w,expected_weight)

                    # Check feature state cannot be matched
                    for var_id, (var,vals) in enumerate(dataset.targets):
                        for val_id, val in enumerate(vals):
                            head = LegacyAtom(var, set(dataset.targets[var_id][1]), val, var_id)
                            pos, neg = PRIDE.interprete(dataset, head)
                            if len(neg) > 0:
                                f = io.StringIO()
                                with contextlib.redirect_stderr(f):
                                    model.extend(dataset, [neg[0]], verbose)

                    # exceptions
                    self.assertRaises(TypeError, model.extend, dataset, "", verbose)
                    self.assertRaises(TypeError, model.extend, dataset, [""], verbose)
                    self.assertRaises(TypeError, model.extend, dataset, [["0","1","0"], [0,"0"]], verbose)
                    self.assertRaises(TypeError, model.extend, dataset, [["0","1","0"], ["0","0"]], verbose)

    def test_predict(self):
        print(">> WDMVLP.predict()")

        # TODO: unit tests

        for test in range(0,self._nb_tests):

            dataset = random_DiscreteStateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            for algo in self._ALGORITHMS:
                for raw_rules in [True, False]:
                    model = WDMVLP(features=dataset.features, targets=dataset.targets)
                    model.compile(algorithm=algo)
                    f = io.StringIO()
                    with contextlib.redirect_stderr(f):
                        model.fit(dataset=dataset)

                    feature_state = random.choice(model.feature_states())
                    output = model.predict([list(feature_state)], raw_rules)[tuple(feature_state)]
                    self.assertEqual(len(output.items()), len(model.targets))

                    for var_id, (var, vals) in enumerate(model.targets):
                        self.assertEqual(len(output[var]), len(model.targets[var_id][1]))
                        for val_id, val in enumerate(vals):
                            best_rule = None
                            max_rule_weight = 0
                            for w,r in model.rules:
                                if r.head.variable == var and r.head.value == val:
                                    if w > max_rule_weight and r.matches(feature_state):
                                        max_rule_weight = w
                                        best_rule = r
                                    elif w == max_rule_weight and r.matches(feature_state):
                                        if best_rule == None or r.size() < best_rule.size():
                                            max_rule_weight = w
                                            best_rule = r

                            best_anti_rule = None
                            max_anti_rule_weight = 0
                            for w,r in model.unlikeliness_rules:
                                if r.head.variable == var and r.head.value == val:
                                    if w > max_anti_rule_weight and r.matches(feature_state):
                                        max_anti_rule_weight = w
                                        best_anti_rule = r
                                    elif w == max_anti_rule_weight and r.matches(feature_state):
                                        if best_anti_rule == None or r.size() < best_anti_rule.size():
                                            max_anti_rule_weight = w
                                            best_anti_rule = r

                            if not raw_rules:
                                if best_rule is not None:
                                    best_rule = best_rule.to_string()
                                if best_anti_rule is not None:
                                    best_anti_rule = best_anti_rule.to_string()

                            prediction = round(0.5 + 0.5*(max_rule_weight - max_anti_rule_weight) / max(1,(max_rule_weight+max_anti_rule_weight)),3)

                            self.assertEqual(output[var][val], (prediction, (max_rule_weight, best_rule), (max_anti_rule_weight, best_anti_rule)) )

                    # exceptions
                    self.assertRaises(TypeError, model.predict, "")
                    self.assertRaises(TypeError, model.predict, [""])
                    self.assertRaises(TypeError, model.predict, [["0","1","0"], [0,"0"]])
                    self.assertRaises(TypeError, model.predict, [["0","1","0"], ["0","0"]])

                #self.assertEqual(model.predict([0,0,0]), [['0', '0', '0'], ['0', '0', '1'], ['1', '0', '0'], ['1', '0', '1']])

    def test_summary(self):
        print(">> WDMVLP.summary()")
        for test in range(0,self._nb_tests):
            # Empty WDMVLP
            model = random_WDMVLP( \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values, \
            algorithm="gula")

            model.rules = []
            model.unlikeliness_rules = []

            expected_print = \
            "WDMVLP summary:\n"+\
            " Algorithm: gula\n"
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
            for algo in self._ALGORITHMS:
                model = random_WDMVLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values, \
                algorithm=algo)

                expected_print = "WDMVLP summary:\n"
                expected_print += " Algorithm: "+algo+"\n"
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
                        expected_print += "  "+ str(w) + ", " +r.to_string()+ "\n"
                if len(model.unlikeliness_rules) == 0:
                    expected_print +=" Unlikeliness rules: []\n"
                else:
                    expected_print +=" Unlikeliness rules:\n"
                    for w,r in model.unlikeliness_rules:
                        expected_print += "  "+ str(w) + ", " +r.to_string()+ "\n"

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
            for algo in self._ALGORITHMS:
                model = random_WDMVLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values, \
                algorithm=algo)

                expected = \
                 "{\n"+\
                 "Algorithm: " + str(model.algorithm)+\
                 "\nFeatures: "+ str(model.features)+\
                 "\nTargets: "+ str(model.targets)+\
                 "\nLikeliness rules:\n"
                for w,r in model.rules:
                    expected += "(" + str(w) + ", " + r.to_string() + ")\n"
                expected += "\nUnlikeliness rules:\n"
                for w,r in model.unlikeliness_rules:
                    expected += "(" + str(w) + ", " + r.to_string() + ")\n"
                expected += "}"

                self.assertEqual(model.to_string(), expected)


PRIDE
'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
