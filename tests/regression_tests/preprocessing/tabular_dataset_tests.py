#-----------------------
# @author: Tony Ribeiro
# @created: 2020/12/23
# @updated: 2021/06/15
#
# @desc: dataset class unit test script
#
#-----------------------

import unittest
import random
import sys
import numpy as np

import pylfit

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from pylfit.datasets import StateTransitionsDataset
from pylfit.preprocessing.tabular_dataset import transitions_dataset_from_array
from tests_generator import random_StateTransitionsDataset

random.seed(0)

class tabular_dataset_tests(unittest.TestCase):
    """
        Unit test of module tabular_dataset.py
    """
    _nb_random_tests = 100

    _nb_transitions = 100

    _nb_features = 5

    _nb_targets = 5

    _nb_feature_values = 3

    _nb_target_values = 3

    def test_transitions_dataset_from_csv(self):
        print(">> pylfit.preprocessing.tabular_dataset.transitions_dataset_from_csv(path, features, targets, feature_names, target_names)")

        for i in range(self._nb_random_tests):
            # Full dataset
            #--------------
            dataset_filepath = "datasets/repressilator.csv"
            features_col_header = ["p_t_1","q_t_1","r_t_1"]
            targets_col_header = ["p_t","q_t","r_t"]

            dataset = pylfit.preprocessing.transitions_dataset_from_csv(path=dataset_filepath, feature_names=features_col_header, target_names=targets_col_header)

            self.assertEqual(dataset.features, [("p_t_1", ["0","1"]), ("q_t_1", ["0","1"]), ("r_t_1", ["0","1"])])
            self.assertEqual(dataset.targets, [("p_t", ["0","1"]), ("q_t", ["0","1"]), ("r_t", ["0","1"])])

            data = [ \
            ([0,0,0],[0,0,1]), \
            ([1,0,0],[0,0,0]), \
            ([0,1,0],[1,0,1]), \
            ([0,0,1],[0,0,1]), \
            ([1,1,0],[1,0,0]), \
            ([1,0,1],[0,1,0]), \
            ([0,1,1],[1,0,1]), \
            ([1,1,1],[1,1,0])]

            data = [(np.array([str(i) for i in s1]), np.array([str(i) for i in s2])) for (s1,s2) in data]

            self.assertEqual(len(data), len(dataset.data))

            for i in range(len(data)):
                self.assertTrue( (dataset.data[i][0]==data[i][0]).all() )
                self.assertTrue( (dataset.data[i][1]==data[i][1]).all() )


            # Partial dataset
            #-----------------
            dataset_filepath = "datasets/repressilator.csv"
            features_col_header = ["p_t_1","r_t_1"]
            targets_col_header = ["q_t"]

            dataset = pylfit.preprocessing.transitions_dataset_from_csv(path=dataset_filepath, feature_names=features_col_header, target_names=targets_col_header)

            self.assertEqual(dataset.features, [("p_t_1", ["0","1"]), ("r_t_1", ["0","1"])])
            self.assertEqual(dataset.targets, [("q_t", ["0","1"])])

            data = [ \
            ([0,0],[0]), \
            ([1,0],[0]), \
            ([0,0],[0]), \
            ([0,1],[0]), \
            ([1,0],[0]), \
            ([1,1],[1]), \
            ([0,1],[0]), \
            ([1,1],[1])]

            data = [(np.array([str(i) for i in s1]), np.array([str(i) for i in s2])) for (s1,s2) in data]

            self.assertEqual(len(data), len(dataset.data))

            for i in range(len(data)):
                self.assertTrue( (dataset.data[i][0]==data[i][0]).all() )
                self.assertTrue( (dataset.data[i][1]==data[i][1]).all() )


    def test_transitions_dataset_from_array(self):
        print(">> pylfit.preprocessing.tabular_dataset.transitions_dataset_from_csv(path, feature_names, target_names)")

        # unit tests
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

        feature_names = ["p_t-1","q_t-1","r_t-1"]
        target_names = ["p_t","q_t","r_t"]

        dataset = pylfit.preprocessing.transitions_dataset_from_array(data=data, feature_names=feature_names, target_names=target_names)

        data = [(np.array([str(i) for i in s1]), np.array([str(i) for i in s2])) for (s1,s2) in data]

        self.assertEqual(dataset.features, [("p_t-1", ["0","1"]), ("q_t-1", ["0","1"]), ("r_t-1", ["0","1"])])
        self.assertEqual(dataset.targets, [("p_t", ["0","1"]), ("q_t", ["0","1"]), ("r_t", ["0","1"])])

        data = [(np.array([str(i) for i in s1]), np.array([str(i) for i in s2])) for (s1,s2) in data]

        self.assertEqual(len(data), len(dataset.data))

        for i in range(len(data)):
            self.assertTrue( (dataset.data[i][0]==data[i][0]).all() )
            self.assertTrue( (dataset.data[i][1]==data[i][1]).all() )

        # exceptions
        #------------

        # data is not list
        data = "[ \
        ([0,0,0],[0.1,0,1]), \
        ([0,0.6,0],[1,0,0]), \
        ([1,0,0],[0,0,0])]"

        self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, feature_names, target_names)

        # data is not list of tuples
        data = [ \
        ([0,0,0],[0,0,1],[0,0,0],[1,0,0]), \
        [[1,0,0],[0,0,0]]]

        self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, feature_names, target_names)

        # data is not list of pairs
        data = [ \
        ([0,0,0],[0,0,1],[0,0,0],[1,0,0]), \
        ([1,0,0],[0,0,0])]

        self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, feature_names, target_names)

        # Not same size for features
        data = [ \
        ([0,0,0],[0,0,1]), \
        ([0,0,0],[1,0,0]), \
        ([1,0],[0,0,0])]

        self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, None, None, feature_names, target_names)

        # Not same size for targets
        data = [ \
        ([0,0,0],[0,0,1]), \
        ([0,0,0],[1,0]), \
        ([1,0,0],[0,0,0])]

        self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, None, None, feature_names, target_names)

        # Not only int/string in features
        data = [ \
        ([0,0,0],[0,0,1]), \
        ([0,0.3,0],[1,0,0]), \
        ([1,0,0],[0,0,0])]

        self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, None, None, feature_names, target_names)

        # Not only int/string in targets
        data = [ \
        ([0,0,0],[0,0.11,1]), \
        ([0,0,0],[1,0,0]), \
        ([1,0,0],[0,0,0])]

        self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, None, None, feature_names, target_names)

        # features is not a list of (string, list of string)
        data = [ \
        ([0,0,0],[0,0,1]), \
        ([0,0,0],[1,0,0])]

        features = "" # not list
        self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, features, None, None, None)
        features = [1,(1,2)] # not list of tuples
        self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, features, None, None, None)
        features = [(1,1),(1,2,4),(1,2)] # not list of pair
        self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, features, None, None, None)
        features = [("p_t",["1","2"]),(1, ["0"]),("r_t",["1","3"])] # not list of pair (string,_)
        self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, features, None, None, None)
        features = [("p_t",["1","2"]),("q_t", "0"),("r_t",["1","2"])] # not list of pair (string,list of _)
        self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, features, None, None, None)
        features = [("p_t",["1","2"]),("q_t", ["0"]),("r_t",["1",2])] # not list of pair (string,list of string)
        self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, features, None, None, None)
        features = [("p_t",["1","2"]),("q_t", ["0","1"]),("p_t",["1","3"])] # not all different variables
        self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, features, None, None, None)
        features = [("p_t",["1","2"]),("q_t", ["0","0"]),("r_t",["1","3"])] # not all different values
        self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, features, None, None, None)

        # targets is not a list of (string, list of string)
        targets = "" # not list
        self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, None, targets, None, None)
        targets = [1,(1,2)] # not list of tuples
        self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, None, targets, None, None)
        targets = [(1,1),(1,2,4),(1,2)] # not list of pair
        self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, None, targets, None, None)
        targets = [("p_t",["1","2"]),(1, ["0"]),("r_t",["1","3"])] # not list of pair (string,_)
        self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, None, targets, None, None)
        targets = [("p_t",["1","2"]),("q_t", "0"),("r_t",["1","2"])] # not list of pair (string,list of _)
        self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, None, targets, None, None)
        targets = [("p_t",["1","2"]),("q_t", ["0"]),("r_t",["1",2])] # not list of pair (string,list of string)
        self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, None, targets, None, None)
        targets = [("p_t",["1","2"]),("q_t", ["0","1"]),("p_t",["1","3"])] # not all different values
        self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, None, targets, None, None)
        targets = [("p_t",["1","2"]),("q_t", ["0","0"]),("r_t",["1","3"])] # not all different values
        self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, None, targets, None, None)


        # Both features/feature_names or targets/target_names given
        features = [("p_t",["1","2"]),("q_t", ["0","1"]),("r_t",["1","3"])]
        targets = [("p_t",["1","2"]),("q_t", ["0","1"]),("r_t",["1","2"])]
        feature_names = ["p_t-1","q_t-1","r_t-1"]
        target_names = ["p_t","q_t","r_t"]
        self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, features, None, feature_names, None)
        self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, None, targets, None, target_names)
        self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, features, targets, feature_names, target_names)

        # target_names is not list of string
        data = [ \
        ([0,0,0],[0,0,1]), \
        ([0,0,0],[1,0,0])]
        feature_names = ["p_t-1","q_t-1","r_t-1"]
        target_names = ["p_t","q_t","r_t"]

        feature_names = ""
        self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, None, None, feature_names, target_names)
        feature_names = [1,0.5,"lol"]
        self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, None, None, feature_names, target_names)

        # target_names is not list of string
        data = [ \
        ([0,0,0],[0,0,1]), \
        ([0,0,0],[1,0,0])]
        feature_names = ["p_t-1","q_t-1","r_t-1"]
        target_names = ["p_t","q_t","r_t"]

        target_names = ""
        self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, None, None, feature_names, target_names)
        target_names = [1,0.5,"lol"]
        self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, None, None, feature_names, target_names)

        # Random tests
        for i in range(self._nb_random_tests):

            original_dataset = random_StateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)


            data = original_dataset.data
            features = original_dataset.features
            targets = original_dataset.targets
            feature_names = [var for var, vals in features]
            target_names = [var for var, vals in targets]

            # empty dataset
            self.assertEqual(transitions_dataset_from_array(data=[], feature_domains=features, target_domains=targets), StateTransitionsDataset(data=[], features=features, targets=targets))

            # Only data given
            dataset = pylfit.preprocessing.transitions_dataset_from_array(data=data)

            # Only names given
            dataset = pylfit.preprocessing.transitions_dataset_from_array(data=data, feature_names=feature_names, target_names=target_names)

            self.assertEqual([var for var, vals in dataset.features], feature_names)
            self.assertEqual([var for var, vals in dataset.targets], target_names)

            # all domain value appear in data
            for var_id, (var, vals) in enumerate(dataset.features):
                for val_id, val in enumerate(vals):
                    appear = False
                    for s1,s2 in data:
                        if s1[var_id] == val:
                            appear = True
                    self.assertTrue(appear)

            for var_id, (var, vals) in enumerate(dataset.targets):
                for val_id, val in enumerate(vals):
                    appear = False
                    for s1,s2 in data:
                        if s2[var_id] == val:
                            appear = True
                    self.assertTrue(appear)

            #data = [(np.array([str(i) for i in s1]), np.array([str(i) for i in s2])) for (s1,s2) in data]

            self.assertEqual(len(dataset.data), len(data))

            for i in range(len(data)):
                self.assertTrue( (dataset.data[i][0]==data[i][0]).all() )
                self.assertTrue( (dataset.data[i][1]==data[i][1]).all() )

            # Domains given
            dataset = pylfit.preprocessing.transitions_dataset_from_array(data=data, feature_domains=features, target_domains=targets)

            self.assertEqual(dataset.features, features)
            self.assertEqual(dataset.targets, targets)
            self.assertEqual(len(dataset.data), len(data))
            for i in range(len(data)):
                self.assertTrue( (dataset.data[i][0]==data[i][0]).all() )
                self.assertTrue( (dataset.data[i][1]==data[i][1]).all() )

            # feature domains only
            dataset = pylfit.preprocessing.transitions_dataset_from_array(data=data, feature_domains=features, target_names=target_names)

            self.assertEqual(dataset.features, features)
            self.assertEqual(len(dataset.data), len(data))
            for i in range(len(data)):
                self.assertTrue( (dataset.data[i][0]==data[i][0]).all() )
                self.assertTrue( (dataset.data[i][1]==data[i][1]).all() )

            # target domains only
            dataset = pylfit.preprocessing.transitions_dataset_from_array(data=data, target_domains=targets, feature_names=feature_names)

            self.assertEqual(dataset.targets, targets)
            self.assertEqual(len(dataset.data), len(data))
            for i in range(len(data)):
                self.assertTrue( (dataset.data[i][0]==data[i][0]).all() )
                self.assertTrue( (dataset.data[i][1]==data[i][1]).all() )

            # Exceptions

            # empty dataset
            self.assertRaises(ValueError, transitions_dataset_from_array, [], None, targets)
            self.assertRaises(ValueError, transitions_dataset_from_array, [], features, None)

            # Wrong data format
            data = [(list(s1)+[0],list(s2)) for s1,s2 in original_dataset.data]
            self.assertRaises(ValueError, transitions_dataset_from_array, data, features, targets)


            data = [(list(s1),list(s2)+[0]) for s1,s2 in original_dataset.data]
            self.assertRaises(ValueError, transitions_dataset_from_array, data, features, targets)

'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
