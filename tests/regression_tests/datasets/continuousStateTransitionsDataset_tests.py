#-----------------------
# @author: Tony Ribeiro
# @created: 2022/08/29
# @updated: 2023/12/26
#
# @desc: ContinuousStateTransitionsDataset class unit test script
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

from tests_generator import random_ContinuousStateTransitionsDataset

from pylfit.utils import eprint
from pylfit.datasets import ContinuousStateTransitionsDataset
from pylfit.objects import Continuum

random.seed(0)

class ContinuousStateTransitionsDataset_tests(unittest.TestCase):
    """
        Unit test of class Datatset from dataset.py
    """
    _nb_tests = 100

    _nb_transitions = 100

    _nb_features = 5

    _nb_targets = 5

    _min_value = -100.0

    _max_value = 100.0

    _min_continuum_size = 1

    #------------------
    # Constructors
    #------------------

    def test_constructor(self):
        print(">> pylfit.datasets.ContinuousStateTransitionsDataset.__init__(self, data, feature_domains, targets_domains)")
        for i in range(self._nb_tests):
            features = [("x_"+str(var), Continuum(self._min_value,self._max_value,True,True)) for var in range(0, random.randint(1,self._nb_features))]
            targets = [("y_"+str(var), Continuum(self._min_value,self._max_value,True,True)) for var in range(0, random.randint(1,self._nb_targets))]
            data = [([random.uniform(features[var][1].min_value, features[var][1].max_value) for var in range(0,len(features))], \
            [random.uniform(targets[var][1].min_value, targets[var][1].max_value) for var in range(0,len(targets))]) \
            for i in range(0, random.randint(0, self._nb_transitions))]


            # Constructor data/features/targets
            dataset = ContinuousStateTransitionsDataset(data, features, targets)


            self.assertEqual(dataset.features, features)
            self.assertEqual(dataset.targets, targets)

            # Exceptions:
            #-------------

            # data is not list
            data = "[ \
            ([0,0,0],[0.1,0,1]), \
            ([0,0.6,0],[1,0,0]), \
            ([1,0,0],[0,0,0])]"

            self.assertRaises(TypeError, ContinuousStateTransitionsDataset, data, features, targets)

            # data is not list of pairs
            data = [ \
            ([0,0,0],[0,0,1],[0,0,0],[1,0,0]), \
            ([1,0,0],[0,0,0])]

            #data = [([str(val) for val in s1],[str(val) for val in s2]) for s1,s2 in data]
            self.assertRaises(TypeError, ContinuousStateTransitionsDataset, data, features, targets)

            # Not same size for features
            data = [ \
            ([0,0,0],[0,0,1,0]), \
            ([0,0,0],[1,0,0,0]), \
            ([1,0],[0,0,0,1])]

            data = [([str(val) for val in s1], [str(val) for val in s2]) for s1,s2 in data]
            self.assertRaises(ValueError, ContinuousStateTransitionsDataset, data, features, targets)

            # Not same size for targets
            data = [ \
            ([0,0,0],[0,0,1,0]), \
            ([0,0,0],[1,0]), \
            ([1,0,0],[0,0,0,0])]

            data = [([str(val) for val in s1], [str(val) for val in s2]) for s1,s2 in data]
            self.assertRaises(ValueError, ContinuousStateTransitionsDataset, data, features, targets)

            # Not only float/int in features
            data = [ \
            (["0","0","0"],["0","0","1","1"]), \
            ([0,0.3,0],["1","0","0","1"]), \
            (["1","0",0],["0","0","0","1"])]

            self.assertRaises(ValueError, ContinuousStateTransitionsDataset, data, features, targets)

            # Not only float/int in targets
            data = [ \
            (["0","0","0"],["0",0.11,"1","1"]), \
            (["0","0","0"],[1,0,0,"1"]), \
            (["1","0","0"],[0,0,0,"1"])]

            self.assertRaises(ValueError, ContinuousStateTransitionsDataset, data, features, targets)

            # Value not in features domain
            data = [ \
            ([0,0,-1000],[0,0,1,1]), \
            ([0,0,0],[1,0,0,1]), \
            ([1,0,0],[0,0,0,1])]

            self.assertRaises(ValueError, ContinuousStateTransitionsDataset, data, features, targets)

            # Value not in targets domain
            data = [ \
            ([0,0,0],[0,0,1,1]), \
            ([0,0,0],[1,0,-1000,1]), \
            ([1,0,0],[0,0,0,1])]

            self.assertRaises(ValueError, ContinuousStateTransitionsDataset, data, features, targets)

            # features is not list of (string,Continuum)
            data = [ \
            ([0,0,0],[0,0,1]), \
            ([0,0,0],[1,0,0])]

            features = [("p_t-1",Continuum(0,1,True,True)),("q_t-1",Continuum(0,1,True,True)),("r_t-1",Continuum(0,1,True,True))]
            targets = [("p_t",Continuum(0,1,True,True)),("q_t",Continuum(0,1,True,True)),("r_t",Continuum(0,1,True,True))]

            features = "" # not list
            self.assertRaises(TypeError, ContinuousStateTransitionsDataset, data, features, targets)
            features = [("p_t-1",["0","1"]),("q_t-1",["0","1"],"r_t-1",["0","1"])] # not tuple var/vals
            self.assertRaises(TypeError, ContinuousStateTransitionsDataset, data, features, targets)
            features = [("p_t-1","1"),("q_t-1",["0","1"]),("r_t-1",["0","1"])] # vals not Continuum
            self.assertRaises(TypeError, ContinuousStateTransitionsDataset, data, features, targets)
            features = [("p_t-1",Continuum(0,1,True,True)),("q_t-1",Continuum(0,1,True,True)),("p_t-1",Continuum(0,1,True,True))] # duplicate var name
            self.assertRaises(ValueError, ContinuousStateTransitionsDataset, data, features, targets)

            # targets is not list of string
            data = [ \
            ([0,0,0],[0,0,1]), \
            ([0,0,0],[1,0,0])]

            features = [("p_t-1",Continuum(0,1,True,True)),("q_t-1",Continuum(0,1,True,True)),("r_t-1",Continuum(0,1,True,True))]
            targets = [("p_t",Continuum(0,1,True,True)),("q_t",Continuum(0,1,True,True)),("r_t",Continuum(0,1,True,True))]

            targets = "" # not list
            self.assertRaises(TypeError, ContinuousStateTransitionsDataset, data, features, targets)
            targets = [("p_t",["0","1"]),("q_t",["0","1"],"r_t",["0","1"])] # not tuple var/vals
            self.assertRaises(TypeError, ContinuousStateTransitionsDataset, data, features, targets)
            targets = [("p_t","1"),("q_t",["0","1"]),("r_t",["0","1"])] # vals not Continuum
            self.assertRaises(TypeError, ContinuousStateTransitionsDataset, data, features, targets)
            targets = [("p_t",Continuum(0,1,True,True)),("q_t",Continuum(0,1,True,True)),("p_t",Continuum(0,1,True,True))] # duplicate var name
            self.assertRaises(ValueError, ContinuousStateTransitionsDataset, data, features, targets)

    def test_summary(self):
        print(">> pylfit.datasets.ContinuousStateTransitionsDataset.summary()")
        for i in range(self._nb_tests):
            # Empty dataset
            dataset = random_ContinuousStateTransitionsDataset( \
            nb_transitions=0, \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            min_value=self._min_value, \
            max_value=self._max_value, \
            min_continuum_size=self._min_continuum_size)

            expected_print = "ContinuousStateTransitionsDataset summary:\n"
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
            dataset = random_ContinuousStateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            min_value=self._min_value, \
            max_value=self._max_value, \
            min_continuum_size=self._min_continuum_size)

            expected_print = "ContinuousStateTransitionsDataset summary:\n"
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
        print(">> pylfit.datasets.ContinuousStateTransitionsDataset.to_string()")
        for i in range(self._nb_tests):
            dataset = random_ContinuousStateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            min_value=self._min_value, \
            max_value=self._max_value, \
            min_continuum_size=self._min_continuum_size)

            expected= \
             "{"+\
             "Features: "+ str(dataset.features)+\
             "\nTargets: "+ str(dataset.targets)+\
             "\nData: "+ str(dataset.data)+\
             "}"

            self.assertEqual(dataset.to_string(), expected)

    def test_to_csv(self):
        print(">> pylfit.datasets.ContinuousStateTransitionsDataset.to_csv(path_to_file)")
        for i in range(self._nb_tests):
            dataset = random_ContinuousStateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            min_value=self._min_value, \
            max_value=self._max_value, \
            min_continuum_size=self._min_continuum_size)

            path = "tmp/ContinuousStateTransitionsDataset_test.csv"
            dataset.to_csv(path)

            self.assertTrue(os.path.isfile(path))

            with open(path, newline='') as f:
                reader = csv.reader(f)
                data = list(reader)

            self.assertEqual(data[0], [var for var,vals in dataset.features+dataset.targets])
            for id, line in enumerate(data[1:]):
                self.assertEqual(line, [str(val) for val in list(dataset.data[id][0]) + list(dataset.data[id][1]) ] )

    def test___eq__(self):
        print(">> pylfit.datasets.ContinuousStateTransitionsDataset.__eq__(path_to_file)")
        for i in range(self._nb_tests):
            dataset = random_ContinuousStateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            min_value=self._min_value, \
            max_value=self._max_value, \
            min_continuum_size=self._min_continuum_size)

            self.assertTrue(dataset == dataset)
            self.assertTrue(dataset == dataset.copy())


            dataset1 = random_ContinuousStateTransitionsDataset( \
            nb_transitions=random.randint(1, 10), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            min_value=self._min_value, \
            max_value=self._max_value, \
            min_continuum_size=self._min_continuum_size)

            dataset2 = random_ContinuousStateTransitionsDataset( \
            nb_transitions=random.randint(11, 20), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            min_value=self._min_value, \
            max_value=self._max_value, \
            min_continuum_size=self._min_continuum_size)

            self.assertTrue(dataset1 != dataset2)

            dataset3 = dataset1.copy()
            dataset3.data = dataset1.data[:-1]
            self.assertTrue(dataset1 != dataset3)

            dataset3 = dataset1.copy()
            dataset3.features = dataset3.features+dataset3.targets
            self.assertTrue(dataset1 != dataset3)

            dataset3 = dataset1.copy()
            dataset3.targets = dataset3.features+dataset3.targets
            self.assertTrue(dataset1 != dataset3)

            dataset4 = dataset1.copy()
            dataset4.data[-1] = (np.array([self._max_value+100 for var in dataset4.data[-1][0]]),np.array([self._max_value+100 for var in dataset4.data[-1][1]]))
            self.assertTrue(dataset1 != dataset4)
'''
@desc: main
'''
if __name__ == '__main__':
    unittest.main()
