#-----------------------
# @author: Tony Ribeiro
# @created: 2020/12/23
# @updated: 2021/06/15
#
# @desc: dataset class unit test script
# done:
# - __init__
# - summary
# - to_csv
# - to_string
#
# Todo:
#
#-----------------------

import unittest
import random
import sys
import os
import csv

import numpy as np

from io import StringIO

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from tests_generator import random_StateTransitionsDataset

from pylfit.utils import eprint
from pylfit.datasets import StateTransitionsDataset

random.seed(0)

class StateTransitionsDataset_tests(unittest.TestCase):
    """
        Unit test of class Datatset from dataset.py
    """
    _nb_tests = 100

    _nb_transitions = 100

    _nb_features = 5

    _nb_targets = 5

    _nb_feature_values = 3

    _nb_target_values = 3

    #------------------
    # Constructors
    #------------------

    def test_constructor(self):
        print(">> pylfit.datasets.StateTransitionsDataset.__init__(self, data, feature_domains, targets_domains)")
        for i in range(self._nb_tests):
            features = [("x_"+str(var), ["val_"+str(val) for val in range(0, random.randint(1,self._nb_feature_values))]) for var in range(0, random.randint(1,self._nb_features))]
            targets = [("y_"+str(var), ["val_"+str(val) for val in range(0, random.randint(1,self._nb_target_values))]) for var in range(0, random.randint(1,self._nb_targets))]
            data = [(np.array([random.choice(features[var][1]) for var in range(0,len(features))]),np.array([random.choice(targets[var][1]) for var in range(0,len(targets))])) for i in range(0, random.randint(0, self._nb_transitions))]


            # Constructor data/features/targets
            dataset = StateTransitionsDataset(data, features, targets)


            self.assertEqual(dataset.features, features)
            self.assertEqual(dataset.targets, targets)

            # Exceptions:
            #-------------
            features = [(str(var), [str(val) for val in range(5)]) for var in range(3)]
            targets = [(str(var), [str(val) for val in range(3)]) for var in range(4)]


            # data is not list
            data = "[ \
            ([0,0,0],[0.1,0,1]), \
            ([0,0.6,0],[1,0,0]), \
            ([1,0,0],[0,0,0])]"

            self.assertRaises(TypeError, StateTransitionsDataset, data, features, targets)

            # data is not list of pairs
            data = [ \
            ([0,0,0],[0,0,1],[0,0,0],[1,0,0]), \
            ([1,0,0],[0,0,0])]

            #data = [([str(val) for val in s1],[str(val) for val in s2]) for s1,s2 in data]
            self.assertRaises(TypeError, StateTransitionsDataset, data, features, targets)

            # Not same size for features
            data = [ \
            ([0,0,0],[0,0,1,0]), \
            ([0,0,0],[1,0,0,0]), \
            ([1,0],[0,0,0,1])]

            data = [([str(val) for val in s1], [str(val) for val in s2]) for s1,s2 in data]
            self.assertRaises(ValueError, StateTransitionsDataset, data, features, targets)

            # Not same size for targets
            data = [ \
            ([0,0,0],[0,0,1,0]), \
            ([0,0,0],[1,0]), \
            ([1,0,0],[0,0,0,0])]

            data = [([str(val) for val in s1], [str(val) for val in s2]) for s1,s2 in data]
            self.assertRaises(ValueError, StateTransitionsDataset, data, features, targets)

            # Not only string in features
            data = [ \
            (["0","0","0"],["0","0","1","1"]), \
            ([0,0.3,0],["1","0","0","1"]), \
            (["1","0",0],["0","0","0","1"])]

            self.assertRaises(ValueError, StateTransitionsDataset, data, features, targets)

            # Not only int/string in targets
            data = [ \
            (["0","0","0"],["0",0.11,"1","1"]), \
            (["0","0","0"],[1,0,0,"1"]), \
            (["1","0","0"],[0,0,0,"1"])]

            self.assertRaises(ValueError, StateTransitionsDataset, data, features, targets)

            # Value not in features domain
            data = [ \
            (["0","0","-1"],["0","0","1","1"]), \
            (["0","0","0"],["1","0","0","1"]), \
            (["1","0","0"],["0","0","0","1"])]

            self.assertRaises(ValueError, StateTransitionsDataset, data, features, targets)

            # Value not in features domain
            data = [ \
            (["0","0","0"],["0","0","1","1"]), \
            (["0","0","0"],["1","0","-1","1"]), \
            (["1","0","0"],["0","0","0","1"])]

            self.assertRaises(ValueError, StateTransitionsDataset, data, features, targets)

            # features is not list of string
            data = [ \
            ([0,0,0],[0,0,1]), \
            ([0,0,0],[1,0,0])]
            data = [([str(val) for val in s1], [str(val) for val in s2]) for s1,s2 in data]

            features = [("p_t-1",["0","1"]),("q_t-1",["0","1"]),("r_t-1",["0","1"])]
            targets = [("p_t",["0","1"]),("q_t",["0","1"]),("r_t",["0","1"])]

            features = "" # not list
            self.assertRaises(TypeError, StateTransitionsDataset, data, features, targets)
            features = [("p_t-1",["0","1"]),("q_t-1",["0","1"],"r_t-1",["0","1"])] # not tuple var/vals
            self.assertRaises(TypeError, StateTransitionsDataset, data, features, targets)
            features = [("p_t-1","1"),("q_t-1",["0","1"]),("r_t-1",["0","1"])] # vals not list
            self.assertRaises(TypeError, StateTransitionsDataset, data, features, targets)
            features = [("p_t-1",["0","1"]),("q_t-1",["0",1]),("r_t-1",[0.2,"1"])] # vals not only string
            self.assertRaises(TypeError, StateTransitionsDataset, data, features, targets)

            # targets is not list of string
            data = [ \
            ([0,0,0],[0,0,1]), \
            ([0,0,0],[1,0,0])]
            data = [([str(val) for val in s1], [str(val) for val in s2]) for s1,s2 in data]

            features = [("p_t-1",["0","1"]),("q_t-1",["0","1"]),("r_t-1",["0","1"])]
            targets = [("p_t",["0","1"]),("q_t",["0","1"]),("r_t",["0","1"])]

            targets = "" # not list
            self.assertRaises(TypeError, StateTransitionsDataset, data, features, targets)
            targets = [("p_t-1",["0","1"]),("q_t-1",["0","1"],"r_t-1",["0","1"])] # not tuple var/vals
            self.assertRaises(TypeError, StateTransitionsDataset, data, features, targets)
            targets = [("p_t-1","1"),("q_t-1",["0","1"]),("r_t-1",["0","1"])] # vals not list
            self.assertRaises(TypeError, StateTransitionsDataset, data, features, targets)
            targets = [("p_t-1",["0","1"]),("q_t-1",["0",1]),("r_t-1",[0.2,"1"])] # vals not only string
            self.assertRaises(TypeError, StateTransitionsDataset, data, features, targets)

    def test_summary(self):
        print(">> pylfit.datasets.StateTransitionsDataset.summary()")
        for i in range(self._nb_tests):
            # Empty dataset
            dataset = random_StateTransitionsDataset( \
            nb_transitions=0, \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            expected_print = "StateTransitionsDataset summary:\n"
            expected_print += " Features: \n"
            for var,vals in dataset.features:
                expected_print += "  " + var + ": " + str(vals) + "\n"
            expected_print += " Targets: \n"
            for var,vals in dataset.targets:
                expected_print += "  " + var + ": " + str(vals) + "\n"
            expected_print += " Data: []\n"

            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            dataset.summary()
            sys.stdout = old_stdout

            self.assertEqual(mystdout.getvalue(), expected_print)

            # Usual dataset
            dataset = random_StateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            expected_print = "StateTransitionsDataset summary:\n"
            expected_print += " Features: \n"
            for var,vals in dataset.features:
                expected_print += "  " + var + ": " + str(vals) + "\n"
            expected_print += " Targets: \n"
            for var,vals in dataset.targets:
                expected_print += "  " + var + ": " + str(vals) + "\n"
            expected_print += " Data:\n"
            for s1,s2 in dataset.data:
                expected_print += "  " + str( (list(s1), list(s2) )) + "\n"

            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            dataset.summary()
            sys.stdout = old_stdout

            self.assertEqual(mystdout.getvalue(), expected_print)

    def test_to_string(self):
        print(">> pylfit.datasets.StateTransitionsDataset.to_string()")
        for i in range(self._nb_tests):
            dataset = random_StateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            expected= \
             "{"+\
             "Features: "+ str(dataset.features)+\
             "\nTargets: "+ str(dataset.targets)+\
             "\nData: "+ str(dataset.data)+\
             "}"

            self.assertEqual(dataset.to_string(), expected)

    def test_to_csv(self):
        print(">> pylfit.datasets.StateTransitionsDataset.to_csv(path_to_file)")
        for i in range(self._nb_tests):
            dataset = random_StateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            path = "tmp/StateTransitionsDataset_test.csv"
            dataset.to_csv(path)

            self.assertTrue(os.path.isfile(path))

            with open(path, newline='') as f:
                reader = csv.reader(f)
                data = list(reader)

            self.assertEqual(data[0], [var for var,vals in dataset.features+dataset.targets])
            for id, line in enumerate(data[1:]):
                self.assertEqual(line, [val for val in list(dataset.data[id][0]) + list(dataset.data[id][1]) ] )

'''
@desc: main
'''
if __name__ == '__main__':
    unittest.main()
