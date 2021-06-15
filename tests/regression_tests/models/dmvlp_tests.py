#-----------------------
# @author: Tony Ribeiro
# @created: 2020/12/23
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
from pylfit.models import DMVLP
from pylfit.algorithms import Algorithm, GULA, PRIDE
from pylfit.datasets import Dataset
from pylfit.semantics import Synchronous, Asynchronous, General

from pylfit.utils import eprint

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from tests_generator import random_DMVLP, random_StateTransitionsDataset, random_symmetric_StateTransitionsDataset

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

    _SUPPORTED_ALGORITHMS = ["gula","pride"] #,"lf1t"

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

    def test_copy(self):
        print(">> DMVLP.copy()")

        features = [("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]
        targets = [("y0", ["0","1"]), ("y1", ["0","1"]), ("y2", ["0","1"])]
        model = DMVLP(features=features, targets=targets)

        copy = model.copy()

        self.assertEqual(model.features, copy.features)
        self.assertEqual(model.targets, copy.targets)
        self.assertEqual(model.rules, copy.rules)
        self.assertEqual(model.algorithm, copy.algorithm)

        for i in range(self._nb_tests):
            for algorithm in self._SUPPORTED_ALGORITHMS:
                model = random_DMVLP( \
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

    def test___str__(self):
        print(">> DMVLP.__str__()")
        for i in range(self._nb_tests):
            for algorithm in self._SUPPORTED_ALGORITHMS:
                model = random_DMVLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values, \
                algorithm=algorithm)

                self.assertEqual(model.__str__(), model.to_string())

    def test___repr__(self):
        print(">> DMVLP.__repr__()")
        for i in range(self._nb_tests):
            for algorithm in self._SUPPORTED_ALGORITHMS:
                model = random_DMVLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values, \
                algorithm=algorithm)

                self.assertEqual(model.__repr__(), model.to_string())

    def test_compile(self):
        print(">> DMVLP.compile(algorithm)")
        features = [("x0", ["0","1"]), ("x1", ["0","1"]), ("x2", ["0","1"])]
        targets = [("y0", ["0","1"]), ("y1", ["0","1"]), ("y2", ["0","1"])]
        model = DMVLP(features=features, targets=targets)

        model.compile(algorithm="gula")
        self.assertEqual(model.algorithm, "gula")

        model.compile(algorithm="pride")
        self.assertEqual(model.algorithm, "pride")

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
                f = io.StringIO()
                with contextlib.redirect_stderr(f):
                    model.fit(dataset=dataset, verbose=verbose)

                expected_rules = GULA.fit(dataset)
                self.assertEqual(expected_rules, model.rules)

                model = DMVLP(features=dataset.features, targets=dataset.targets)
                model.compile(algorithm="pride")
                f = io.StringIO()
                with contextlib.redirect_stderr(f):
                    model.fit(dataset=dataset, verbose=verbose)

                expected_rules = PRIDE.fit(dataset)
                self.assertEqual(expected_rules, model.rules)

                # Exceptions
                #------------

                for algorithm in self._SUPPORTED_ALGORITHMS:
                    model = DMVLP(features=dataset.features, targets=dataset.targets)
                    model.compile(algorithm=algorithm)

                    self.assertRaises(ValueError, model.fit, []) # dataset is not of valid type

                    model.algorithm = "gulaaaaa"
                    self.assertRaises(ValueError, model.fit, dataset, verbose) # algorithm is not of valid
                    model.algorithm = algorithm

                    original = DMVLP._COMPATIBLE_DATASETS.copy()
                    class newdataset(Dataset):
                        def __init__(self, data, features, targets):
                            x = ""
                    DMVLP._COMPATIBLE_DATASETS = [newdataset]
                    self.assertRaises(ValueError, model.fit, newdataset([],[],[]), verbose) # dataset not supported by the algo
                    DMVLP._COMPATIBLE_DATASETS = original

                    #self.assertRaises(ValueError, model.fit, dataset, verbose) # algorithm is not of valid
                    model.algorithm = "lf1t"
                    self.assertRaises(NotImplementedError, model.fit, dataset, verbose) # algorithm is not of valid

    def test_extend(self):
        print(">> DMVLP.extend(dataset, feature_states)")

        for test in range(0,self._nb_tests):

            dataset = random_StateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            for algo in self._SUPPORTED_ALGORITHMS:
                for verbose in [0,1]:

                    model = DMVLP(features=dataset.features, targets=dataset.targets)
                    model.compile(algorithm=algo)
                    f = io.StringIO()
                    with contextlib.redirect_stderr(f):
                        model.fit(dataset=dataset,verbose=verbose)

                    original_rules = model.rules.copy()

                    # Encode data with StateTransitionsDataset
                    data_encoded = []
                    for (s1,s2) in dataset.data:
                        s1_encoded = [domain.index(s1[var_id]) for var_id, (var,domain) in enumerate(dataset.features)]
                        s2_encoded = [domain.index(s2[var_id]) for var_id, (var,domain) in enumerate(dataset.targets)]
                        data_encoded.append((s1_encoded,s2_encoded))

                    values_ids = [[j for j in dataset.features[i][1]] for i in range(0,len(dataset.features))]
                    feature_states = [list(i) for i in list(itertools.product(*values_ids))]

                    feature_states_to_match = [random.choice(feature_states) for i in range(10)]
                    #eprint(feature_states_to_match)
                    #eprint(model.features)

                    model.extend(dataset, feature_states_to_match)

                    # No rule disapear
                    for r in original_rules:
                        self.assertTrue(r in model.rules)

                    # atmost one aditional rule per feature state for each var/val
                    for var_id, (var,vals) in enumerate(dataset.targets):
                        for val_id, val in enumerate(vals):
                            self.assertTrue(len([r for r in model.rules if r.head_variable == var_id if r.head_value == val_id if r not in original_rules]) <= len(feature_states))


                    for feature_state in feature_states_to_match:
                        encoded_feature_state = Algorithm.encode_state(feature_state, dataset.features)
                        for var_id, (var,vals) in enumerate(dataset.targets):
                            for val_id, val in enumerate(vals):
                                #eprint("var: ", var_id)
                                #eprint("val: ", val_id)
                                pos, neg = PRIDE.interprete(data_encoded, var_id, val_id)

                                # Only way to not match is no rule can be find
                                new_rule = PRIDE.find_one_optimal_rule_of(var_id, val_id, len(dataset.features), pos, neg, encoded_feature_state, 0)
                                matched = False
                                for r in model.rules:
                                    if r.head_variable == var_id and r.head_value == val_id and r.matches(encoded_feature_state):
                                        matched = True
                                        break

                                if not matched:
                                    self.assertTrue(new_rule is None)

                    # check rules
                    for var_id, (var,vals) in enumerate(dataset.targets):
                        for val_id, val in enumerate(vals):
                            pos, neg = PRIDE.interprete(data_encoded, var_id, val_id)
                            new_rules = [x for x in model.rules if x not in original_rules]

                            for r in [r for r in new_rules if r.head_variable==var_id if r.head_value==val_id]:
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
                                for (var_id_, val_id_) in r.body:
                                    r.remove_condition(var_id_) # Try remove condition

                                    conflict = False
                                    for s in neg:
                                        if r.matches(s): # Cover a negative example
                                            conflict = True
                                            break
                                    self.assertTrue(conflict)
                                    r.add_condition(var_id_,val_id_) # Cancel removal

                    # Check feature state cannot be matched
                    for var_id, (var,vals) in enumerate(dataset.targets):
                        for val_id, val in enumerate(vals):
                            pos, neg = PRIDE.interprete(data_encoded, var_id, val_id)
                            if len(neg) > 0:
                                state_raw = neg[0]
                                state_string = []
                                for var_id_, val_id_ in enumerate(state_raw):
                                    #eprint(var_id, val_id)
                                    state_string.append(model.features[var_id_][1][val_id_])

                                f = io.StringIO()
                                with contextlib.redirect_stderr(f):
                                    model.extend(dataset, [state_string], verbose)

                    # exceptions
                    self.assertRaises(TypeError, model.extend, dataset, "", verbose)
                    self.assertRaises(TypeError, model.extend, dataset, [""], verbose)
                    self.assertRaises(TypeError, model.extend, dataset, [["0","1","0"], [0,"0"]], verbose)
                    self.assertRaises(TypeError, model.extend, dataset, [["0","1","0"], ["0","0"]], verbose)

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

        self.assertEqual(set([tuple(s) for s in model.predict([["0","0","0"]])[("0","0","0")]]), set([('1','0','0'), ('0','0','0'), ('0', '0', '1'), ('1','0','1')]))
        self.assertEqual(set(tuple(s) for s in model.predict([["1","1","1"]])[("1","1","1")]), set([('1', '1', '0')]))
        self.assertEqual(set(tuple(s) for s in model.predict([["0","0","0"]], semantics="asynchronous")[("0","0","0")]), set([('1','0','0'), ('0', '0', '1')]))
        self.assertEqual(set([tuple(s) for s in model.predict([['1','1','1']], semantics="general")[("1","1","1")]]), set([('1', '1', '0'), ('1','1','1')]))
        self.assertEqual(set([tuple(s) for s in model.predict([["0","0","0"]], semantics="general")[("0","0","0")]]), set([('1','0','0'), ('0','0','0'), ('0', '0', '1'), ('1','0','1')]))

        for i in range(self._nb_tests):
            for semantics in [None, "synchronous", "asynchronous", "general"]:
                semantics_class = Synchronous
                if semantics == "asynchronous":
                    semantics_class = Asynchronous
                if semantics == "general":
                    semantics_class = General

            dataset = random_StateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            # Need same features/targets for some semantics
            if semantics == "asynchronous" or semantics == "general":
                dataset = random_symmetric_StateTransitionsDataset(nb_transitions=random.randint(1, self._nb_transitions), \
                nb_variables=random.randint(1,self._nb_features), \
                max_variable_values=self._nb_feature_values)

            for algorithm in self._SUPPORTED_ALGORITHMS:
                model = DMVLP(features=dataset.features, targets=dataset.targets)
                model.compile(algorithm=algorithm)
                model.fit(dataset=dataset)

                feature_states = [list(s) for s in set(tuple(s1) for s1,s2 in dataset.data)]

                if semantics is None:
                    prediction = model.predict(feature_states)
                else:
                    prediction = model.predict(feature_states, semantics=semantics)

                for state_id, s1 in enumerate(feature_states):
                    feature_state_encoded = []
                    for var_id, val in enumerate(s1):
                        val_id = model.features[var_id][1].index(str(val))
                        feature_state_encoded.append(val_id)

                    #eprint(feature_state_encoded)

                    target_states = semantics_class.next(feature_state_encoded, model.targets, model.rules)
                    output = dict()
                    for s in target_states:
                        target_state = []
                        for var_id, val_id in enumerate(s):
                            #eprint(var_id, val_id)
                            if val_id == -1:
                                target_state.append("?")
                            else:
                                target_state.append(model.targets[var_id][1][val_id])
                        output[tuple(target_state)] = target_states[s]

                    self.assertEqual(prediction[tuple(s1)], output)

                # Force missing value
                rules = model.rules
                model.rules = [r for r in model.rules if r.head_variable != random.randint(0,len(model.targets))]

                prediction = model.predict(feature_states, semantics=semantics)
                for state_id, s1 in enumerate(feature_states):
                    feature_state_encoded = []
                    for var_id, val in enumerate(s1):
                        val_id = model.features[var_id][1].index(str(val))
                        feature_state_encoded.append(val_id)

                    #eprint(feature_state_encoded)

                    target_states = semantics_class.next(feature_state_encoded, model.targets, model.rules)
                    output = dict()
                    for s in target_states:
                        target_state = []
                        for var_id, val_id in enumerate(s):
                            #eprint(var_id, val_id)
                            if val_id == -1:
                                target_state.append("?")
                            else:
                                target_state.append(model.targets[var_id][1][val_id])
                        output[tuple(target_state)] = target_states[s]

                    self.assertEqual(prediction[tuple(s1)], output)
                model.rules = rules

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

                # Semantics restriction
                model_2 = model.copy()
                model_2.targets = model_2.targets + model_2.targets
                self.assertRaises(ValueError, model_2.predict, feature_states, "asynchronous")
                self.assertRaises(ValueError, model_2.predict, feature_states, "general")
                self.assertRaises(ValueError, model_2.predict, feature_states, "badvalue")

    def test_summary(self):
        print(">> DMVLP.summary()")
        for i in range(self._nb_tests):
            for algorithm in self._SUPPORTED_ALGORITHMS:
                # Empty DMVLP
                model = random_DMVLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values, \
                algorithm=algorithm)

                model.rules = []

                expected_print = \
                "DMVLP summary:\n"+\
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

                # Usual DMVLP
                model = random_DMVLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values, \
                algorithm=algorithm)

                expected_print = \
                "DMVLP summary:\n"+\
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

                model = DMVLP(features=model.features, targets=model.targets)
                self.assertRaises(ValueError, model.summary) # compile not called

    def test_to_string(self):
        print(">> pylfit.models.DMVLP.to_string()")
        for i in range(self._nb_tests):
            for algorithm in self._SUPPORTED_ALGORITHMS:
                model = random_DMVLP( \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values, \
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
