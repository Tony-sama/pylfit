#-----------------------
# @author: Tony Ribeiro
# @created: 2020/12/23
# @updated: 2022/08/17
#
# @desc: DiscreteStateTransitionsDataset class unit test script
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

from tests_generator import random_DiscreteStateTransitionsDataset, random_unknown_values_dataset

from pylfit.utils import eprint
from pylfit.datasets import DiscreteStateTransitionsDataset

random.seed(0)

class DiscreteStateTransitionsDataset_tests(unittest.TestCase):
    """
        Unit test of class Datatset from dataset.py
    """
    _nb_tests = 100

    _nb_transitions = 100

    _nb_features = 5

    _nb_targets = 5

    _nb_feature_values = 3

    _nb_target_values = 3

    _UNKNOWN_VALUES = [DiscreteStateTransitionsDataset._UNKNOWN_VALUE, "*", "", "NONE", "null"]

    #------------------
    # Constructors
    #------------------

    def test_constructor(self):
        print(">> pylfit.datasets.DiscreteStateTransitionsDataset.__init__(self, data, feature_domains, targets_domains)")

        for i in range(self._nb_tests):
            for partial_states in [True,False]:
                features = [("x_"+str(var), ["val_"+str(val) for val in range(0, random.randint(1,self._nb_feature_values))]) for var in range(0, random.randint(1,self._nb_features))]
                targets = [("y_"+str(var), ["val_"+str(val) for val in range(0, random.randint(1,self._nb_target_values))]) for var in range(0, random.randint(1,self._nb_targets))]
                data = [(np.array([random.choice(features[var][1]) for var in range(0,len(features))]),np.array([random.choice(targets[var][1]) for var in range(0,len(targets))])) for i in range(0, random.randint(0, self._nb_transitions))]


                # Constructor data/features/targets/unknown_values
                if partial_states:
                    data = random_unknown_values_dataset(data)
                    dataset = DiscreteStateTransitionsDataset(data, features, targets)
                    
                    if len(data) > 0:
                        self.assertTrue(dataset.has_unknown_values())

                    nb_unknown_values = 0
                    for (i,j) in data:
                        nb_unknown_values += np.count_nonzero(i == DiscreteStateTransitionsDataset._UNKNOWN_VALUE)
                        nb_unknown_values += np.count_nonzero(j == DiscreteStateTransitionsDataset._UNKNOWN_VALUE)
                    self.assertEqual(dataset.nb_unknown_values, nb_unknown_values)
                else:
                    dataset = DiscreteStateTransitionsDataset(data, features, targets)


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

                self.assertRaises(TypeError, DiscreteStateTransitionsDataset, data, features, targets)

                # data is not list of pairs
                data = [ \
                ([0,0,0],[0,0,1],[0,0,0],[1,0,0]), \
                ([1,0,0],[0,0,0])]

                #data = [([str(val) for val in s1],[str(val) for val in s2]) for s1,s2 in data]
                self.assertRaises(TypeError, DiscreteStateTransitionsDataset, data, features, targets)

                # Not same size for features
                data = [ \
                ([0,0,0],[0,0,1,0]), \
                ([0,0,0],[1,0,0,0]), \
                ([1,0],[0,0,0,1])]

                data = [([str(val) for val in s1], [str(val) for val in s2]) for s1,s2 in data]
                self.assertRaises(ValueError, DiscreteStateTransitionsDataset, data, features, targets)

                # Not same size for targets
                data = [ \
                ([0,0,0],[0,0,1,0]), \
                ([0,0,0],[1,0]), \
                ([1,0,0],[0,0,0,0])]

                data = [([str(val) for val in s1], [str(val) for val in s2]) for s1,s2 in data]
                self.assertRaises(ValueError, DiscreteStateTransitionsDataset, data, features, targets)

                # Not only string in features
                data = [ \
                (["0","0","0"],["0","0","1","1"]), \
                ([0,0.3,0],["1","0","0","1"]), \
                (["1","0",0],["0","0","0","1"])]

                self.assertRaises(ValueError, DiscreteStateTransitionsDataset, data, features, targets)

                # Not only int/string in targets
                data = [ \
                (["0","0","0"],["0",0.11,"1","1"]), \
                (["0","0","0"],[1,0,0,"1"]), \
                (["1","0","0"],[0,0,0,"1"])]

                self.assertRaises(ValueError, DiscreteStateTransitionsDataset, data, features, targets)

                # Value not in features domain
                data = [ \
                (["0","0","-1"],["0","0","1","1"]), \
                (["0","0","0"],["1","0","0","1"]), \
                (["1","0","0"],["0","0","0","1"])]

                self.assertRaises(ValueError, DiscreteStateTransitionsDataset, data, features, targets)

                # Value not in features domain
                data = [ \
                (["0","0","0"],["0","0","1","1"]), \
                (["0","0","0"],["1","0","-1","1"]), \
                (["1","0","0"],["0","0","0","1"])]

                self.assertRaises(ValueError, DiscreteStateTransitionsDataset, data, features, targets)

                # features is not list of string
                data = [ \
                ([0,0,0],[0,0,1]), \
                ([0,0,0],[1,0,0])]
                data = [([str(val) for val in s1], [str(val) for val in s2]) for s1,s2 in data]

                features = [("p_t-1",["0","1"]),("q_t-1",["0","1"]),("r_t-1",["0","1"])]
                targets = [("p_t",["0","1"]),("q_t",["0","1"]),("r_t",["0","1"])]

                features = "" # not list
                self.assertRaises(TypeError, DiscreteStateTransitionsDataset, data, features, targets)
                features = [("p_t-1",["0","1"]),("q_t-1",["0","1"],"r_t-1",["0","1"])] # not tuple var/vals
                self.assertRaises(TypeError, DiscreteStateTransitionsDataset, data, features, targets)
                features = [("p_t-1","1"),("q_t-1",["0","1"]),("r_t-1",["0","1"])] # vals not list
                self.assertRaises(TypeError, DiscreteStateTransitionsDataset, data, features, targets)
                features = [("p_t-1",["0","1"]),("q_t-1",["0",1]),("r_t-1",[0.2,"1"])] # vals not only string
                self.assertRaises(TypeError, DiscreteStateTransitionsDataset, data, features, targets)
                features = [("p_t-1",["0","1"]),("q_t-1",["0","1"]),("p_t-1",["0","1"])] # duplicate var name
                self.assertRaises(ValueError, DiscreteStateTransitionsDataset, data, features, targets)

                # targets is not list of string
                data = [ \
                ([0,0,0],[0,0,1]), \
                ([0,0,0],[1,0,0])]
                data = [([str(val) for val in s1], [str(val) for val in s2]) for s1,s2 in data]

                features = [("p_t-1",["0","1"]),("q_t-1",["0","1"]),("r_t-1",["0","1"])]
                targets = [("p_t",["0","1"]),("q_t",["0","1"]),("r_t",["0","1"])]

                targets = "" # not list
                self.assertRaises(TypeError, DiscreteStateTransitionsDataset, data, features, targets)
                targets = [("p_t-1",["0","1"]),("q_t-1",["0","1"],"r_t-1",["0","1"])] # not tuple var/vals
                self.assertRaises(TypeError, DiscreteStateTransitionsDataset, data, features, targets)
                targets = [("p_t-1","1"),("q_t-1",["0","1"]),("r_t-1",["0","1"])] # vals not list
                self.assertRaises(TypeError, DiscreteStateTransitionsDataset, data, features, targets)
                targets = [("p_t-1",["0","1"]),("q_t-1",["0",1]),("r_t-1",[0.2,"1"])] # vals not only string
                self.assertRaises(TypeError, DiscreteStateTransitionsDataset, data, features, targets)
                targets = [("p_t-1",["0","1"]),("q_t-1",["0","1"]),("p_t-1",["0","1"])] # duplicate var name
                self.assertRaises(ValueError, DiscreteStateTransitionsDataset, data, features, targets)

    def test_summary(self):
        print(">> pylfit.datasets.DiscreteStateTransitionsDataset.summary()")
        for i in range(self._nb_tests):
            for partial_states in [True,False]:

                # Empty dataset
                dataset = random_DiscreteStateTransitionsDataset( \
                nb_transitions=0, \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values)

                expected_print = "DiscreteStateTransitionsDataset summary:\n"
                expected_print += " Features: \n"
                for var,vals in dataset.features:
                    expected_print += "  " + var + ": " + str(vals) + "\n"
                expected_print += " Targets: \n"
                for var,vals in dataset.targets:
                    expected_print += "  " + var + ": " + str(vals) + "\n"
                expected_print += " Data: []\n"
                expected_print += " Unknown values: "+str(dataset.nb_unknown_values)+"\n"

                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                dataset.summary()
                sys.stdout = old_stdout

                self.assertEqual(mystdout.getvalue(), expected_print)

                # Usual dataset
                dataset = random_DiscreteStateTransitionsDataset( \
                nb_transitions=random.randint(1, self._nb_transitions), \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values)

                if partial_states:
                    data = random_unknown_values_dataset(dataset.data)
                    dataset = DiscreteStateTransitionsDataset(data, dataset.features, dataset.targets)

                expected_print = "DiscreteStateTransitionsDataset summary:\n"
                expected_print += " Features: \n"
                for var,vals in dataset.features:
                    expected_print += "  " + var + ": " + str(vals) + "\n"
                expected_print += " Targets: \n"
                for var,vals in dataset.targets:
                    expected_print += "  " + var + ": " + str(vals) + "\n"
                expected_print += " Data:\n"
                for s1,s2 in dataset.data:
                    expected_print += "  " + str( (list(s1), list(s2) )) + "\n"
                expected_print += " Unknown values: "+str(dataset.nb_unknown_values)+"\n"

                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                dataset.summary()
                sys.stdout = old_stdout

                self.assertEqual(mystdout.getvalue(), expected_print)

    def test_to_string(self):
        print(">> pylfit.datasets.DiscreteStateTransitionsDataset.to_string()")
        for i in range(self._nb_tests):
            for partial_states in [True,False]:
                dataset = random_DiscreteStateTransitionsDataset( \
                nb_transitions=random.randint(1, self._nb_transitions), \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values)

                if partial_states:
                    data = random_unknown_values_dataset(dataset.data)
                    dataset = DiscreteStateTransitionsDataset(data, dataset.features, dataset.targets)

                expected= \
                "{"+\
                "Features: "+ str(dataset.features)+\
                "\nTargets: "+ str(dataset.targets)+\
                "\nData: "
                for d in dataset.data:
                    expected += "\n"+str( ([i for i in d[0]],
                                [i for i in d[1]] ))
                expected += "\n}"

                self.assertEqual(dataset.to_string(), expected)

    def test_to_csv(self):
        print(">> pylfit.datasets.DiscreteStateTransitionsDataset.to_csv(path_to_file)")
        for i in range(self._nb_tests):
            for partial_states in [True,False]:
                dataset = random_DiscreteStateTransitionsDataset( \
                nb_transitions=random.randint(1, self._nb_transitions), \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values)

                if partial_states:
                    data = random_unknown_values_dataset(dataset.data)
                    dataset = DiscreteStateTransitionsDataset(data, dataset.features, dataset.targets)

                path = "tmp/DiscreteStateTransitionsDataset_test.csv"
                dataset.to_csv(path)

                self.assertTrue(os.path.isfile(path))

                with open(path, newline='') as f:
                    reader = csv.reader(f)
                    data = list(reader)

                self.assertEqual(data[0], [var for var,vals in dataset.features+dataset.targets])
                for id, line in enumerate(data[1:]):
                    self.assertEqual(line, [val for val in list(dataset.data[id][0]) + list(dataset.data[id][1]) ] )

    def test___eq__(self):
        print(">> pylfit.datasets.DiscreteStateTransitionsDataset.__eq__(path_to_file)")
        for i in range(self._nb_tests):
            for partial_states in [True,False]:
                dataset = random_DiscreteStateTransitionsDataset( \
                nb_transitions=random.randint(1, self._nb_transitions), \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values)

                if partial_states:
                    data = random_unknown_values_dataset(dataset.data)
                    dataset = DiscreteStateTransitionsDataset(data, dataset.features, dataset.targets)

                self.assertTrue(dataset == dataset)
                self.assertTrue(dataset == dataset.copy())


                dataset1 = random_DiscreteStateTransitionsDataset( \
                nb_transitions=random.randint(1, 10), \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values)

                dataset2 = random_DiscreteStateTransitionsDataset( \
                nb_transitions=random.randint(11, 20), \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values)

                self.assertTrue(dataset1 != dataset2)

                dataset3 = dataset1.copy()
                dataset3.data = dataset3.data[:-1]
                self.assertTrue(dataset1 != dataset3)

                dataset4 = dataset1.copy()
                dataset4.data[-1] = (np.array(["lol" for var in dataset4.data[-1][0]]),np.array(["lol" for var in dataset4.data[-1][1]]))
                self.assertTrue(dataset1 != dataset4)
'''
@desc: main
'''
if __name__ == '__main__':
    unittest.main()
