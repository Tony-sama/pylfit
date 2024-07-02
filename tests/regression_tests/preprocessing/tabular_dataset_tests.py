#-----------------------
# @author: Tony Ribeiro
# @created: 2020/12/23
# @updated: 2023/12/26
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

from pylfit.datasets import DiscreteStateTransitionsDataset, ContinuousStateTransitionsDataset
from pylfit.objects import Continuum
from tests_generator import random_DiscreteStateTransitionsDataset, random_ContinuousStateTransitionsDataset

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

    _min_value = -100.0

    _max_value = 100.0

    _min_domain_size = 1.0

    _min_continuum_size = 1

    def test_discrete_state_transitions_dataset_from_csv(self):
        print(">> pylfit.preprocessing.tabular_dataset.discrete_state_transitions_dataset_from_csv(path, features, targets, feature_names, target_names)")

        for i in range(self._nb_random_tests):
            # Full dataset
            #--------------
            dataset_filepath = "datasets/repressilator.csv"
            features_col_header = ["p_t_1","q_t_1","r_t_1"]
            targets_col_header = ["p_t","q_t","r_t"]

            dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_csv(path=dataset_filepath, feature_names=features_col_header, target_names=targets_col_header)

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

            dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_csv(path=dataset_filepath, feature_names=features_col_header, target_names=targets_col_header)

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

            # exceptions
            #------------
            self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_csv, 10, features_col_header, targets_col_header)
        
            # features_names is not list of string
            feature_names = ["p_t-1","q_t-1","r_t-1"]
            target_names = ["p_t","q_t","r_t"]

            feature_names = ""
            self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_csv, dataset_filepath,feature_names, target_names)
            feature_names = [1,0.5,"lol"]
            self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_csv, dataset_filepath,feature_names, target_names)

            # target_names is not list of string
            feature_names = ["p_t-1","q_t-1","r_t-1"]
            target_names = ["p_t","q_t","r_t"]

            target_names = ""
            self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_csv, dataset_filepath, feature_names, target_names)
            target_names = [1,0.5,"lol"]
            self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_csv, dataset_filepath, feature_names, target_names)

            # Unknown values must be a list
            feature_names = ["p_t-1","q_t-1","r_t-1"]
            target_names = ["p_t","q_t","r_t"]
            self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_csv, dataset_filepath, feature_names, target_names, "?")
            #self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_csv, dataset_filepath, feature_names, target_names, ["?",1])



    def test_discrete_state_transitions_dataset_from_array(self):
        print(">> pylfit.preprocessing.tabular_dataset.discrete_state_transitions_dataset_from_csv(path, feature_names, target_names)")

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

        dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=data, feature_names=feature_names, target_names=target_names)

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

        self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, feature_names, target_names)

        # data is not list of tuples
        data = [ \
        ([0,0,0],[0,0,1],[0,0,0],[1,0,0]), \
        [[1,0,0],[0,0,0]]]

        self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, feature_names, target_names)

        # data is not list of pairs
        data = [ \
        ([0,0,0],[0,0,1],[0,0,0],[1,0,0]), \
        ([1,0,0],[0,0,0])]

        self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, feature_names, target_names)

        # Not same size for features
        data = [ \
        ([0,0,0],[0,0,1]), \
        ([0,0,0],[1,0,0]), \
        ([1,0],[0,0,0])]

        self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, None, feature_names, target_names)

        # Not same size for targets
        data = [ \
        ([0,0,0],[0,0,1]), \
        ([0,0,0],[1,0]), \
        ([1,0,0],[0,0,0])]

        self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, None, feature_names, target_names)

        # Not only int/string in features
        data = [ \
        ([0,0,0],[0,0,1]), \
        ([0,0.3,0],[1,0,0]), \
        ([1,0,0],[0,0,0])]

        self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, None, feature_names, target_names)

        # Not only int/string in targets
        data = [ \
        ([0,0,0],[0,0.11,1]), \
        ([0,0,0],[1,0,0]), \
        ([1,0,0],[0,0,0])]

        self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, None, feature_names, target_names)

        # features is not a list of (string, list of string)
        data = [ \
        ([0,0,0],[0,0,1]), \
        ([0,0,0],[1,0,0])]

        features = "" # not list
        self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, features, None, None, None)
        features = [1,(1,2)] # not list of tuples
        self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, features, None, None, None)
        features = [(1,1),(1,2,4),(1,2)] # not list of pair
        self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, features, None, None, None)
        features = [("p_t",["1","2"]),(1, ["0"]),("r_t",["1","3"])] # not list of pair (string,_)
        self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, features, None, None, None)
        features = [("p_t",["1","2"]),("q_t", "0"),("r_t",["1","2"])] # not list of pair (string,list of _)
        self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, features, None, None, None)
        features = [("p_t",["1","2"]),("q_t", ["0"]),("r_t",["1",2])] # not list of pair (string,list of string)
        self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, features, None, None, None)
        features = [("p_t",["1","2"]),("q_t", ["0","1"]),("p_t",["1","3"])] # not all different variables
        self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, features, None, None, None)
        features = [("p_t",["1","2"]),("q_t", ["0","0"]),("r_t",["1","3"])] # not all different values
        self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, features, None, None, None)

        # targets is not a list of (string, list of string)
        targets = "" # not list
        self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, targets, None, None)
        targets = [1,(1,2)] # not list of tuples
        self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, targets, None, None)
        targets = [(1,1),(1,2,4),(1,2)] # not list of pair
        self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, targets, None, None)
        targets = [("p_t",["1","2"]),(1, ["0"]),("r_t",["1","3"])] # not list of pair (string,_)
        self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, targets, None, None)
        targets = [("p_t",["1","2"]),("q_t", "0"),("r_t",["1","2"])] # not list of pair (string,list of _)
        self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, targets, None, None)
        targets = [("p_t",["1","2"]),("q_t", ["0"]),("r_t",["1",2])] # not list of pair (string,list of string)
        self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, targets, None, None)
        targets = [("p_t",["1","2"]),("q_t", ["0","1"]),("p_t",["1","3"])] # not all different values
        self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, targets, None, None)
        targets = [("p_t",["1","2"]),("q_t", ["0","0"]),("r_t",["1","3"])] # not all different values
        self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, targets, None, None)


        # Both features/feature_names or targets/target_names given
        features = [("p_t",["1","2"]),("q_t", ["0","1"]),("r_t",["1","3"])]
        targets = [("p_t",["1","2"]),("q_t", ["0","1"]),("r_t",["1","2"])]
        feature_names = ["p_t-1","q_t-1","r_t-1"]
        target_names = ["p_t","q_t","r_t"]
        self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, features, None, feature_names, None)
        self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, targets, None, target_names)
        self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, features, targets, feature_names, target_names)

        # features_names is not list of string
        data = [ \
        ([0,0,0],[0,0,1]), \
        ([0,0,0],[1,0,0])]
        feature_names = ["p_t-1","q_t-1","r_t-1"]
        target_names = ["p_t","q_t","r_t"]

        feature_names = ""
        self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, None, feature_names, target_names)
        feature_names = [1,0.5,"lol"]
        self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, None, feature_names, target_names)

        # target_names is not list of string
        data = [ \
        ([0,0,0],[0,0,1]), \
        ([0,0,0],[1,0,0])]
        feature_names = ["p_t-1","q_t-1","r_t-1"]
        target_names = ["p_t","q_t","r_t"]

        target_names = ""
        self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, None, feature_names, target_names)
        target_names = [1,0.5,"lol"]
        self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, None, feature_names, target_names)

        # Unknown values must be a list
        data = [ \
        ([0,"?",0],[0,0,1]), \
        ([0,0,0],[1,"?",0])]
        feature_names = ["p_t-1","q_t-1","r_t-1"]
        target_names = ["p_t","q_t","r_t"]
        self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, None, feature_names, target_names, "?")
        #self.assertRaises(TypeError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, None, None, feature_names, target_names, ["?",1])

        # Random tests
        for i in range(self._nb_random_tests):

            original_dataset = random_DiscreteStateTransitionsDataset( \
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
            self.assertEqual(pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=[], feature_domains=features, target_domains=targets), DiscreteStateTransitionsDataset(data=[], features=features, targets=targets))

            # Only data given
            dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=data)

            # Only names given
            dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=data, feature_names=feature_names, target_names=target_names)

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
            dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=data, feature_domains=features, target_domains=targets)

            self.assertEqual(dataset.features, features)
            self.assertEqual(dataset.targets, targets)
            self.assertEqual(len(dataset.data), len(data))
            for i in range(len(data)):
                self.assertTrue( (dataset.data[i][0]==data[i][0]).all() )
                self.assertTrue( (dataset.data[i][1]==data[i][1]).all() )

            # feature domains only
            dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=data, feature_domains=features, target_names=target_names)

            self.assertEqual(dataset.features, features)
            self.assertEqual(len(dataset.data), len(data))
            for i in range(len(data)):
                self.assertTrue( (dataset.data[i][0]==data[i][0]).all() )
                self.assertTrue( (dataset.data[i][1]==data[i][1]).all() )

            # target domains only
            dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=data, target_domains=targets, feature_names=feature_names)

            self.assertEqual(dataset.targets, targets)
            self.assertEqual(len(dataset.data), len(data))
            for i in range(len(data)):
                self.assertTrue( (dataset.data[i][0]==data[i][0]).all() )
                self.assertTrue( (dataset.data[i][1]==data[i][1]).all() )

            # Exceptions

            # empty dataset
            self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, [], None, targets)
            self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, [], features, None)

            # Wrong data format
            data = [(list(s1)+[0],list(s2)) for s1,s2 in original_dataset.data]
            self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, features, targets)


            data = [(list(s1),list(s2)+[0]) for s1,s2 in original_dataset.data]
            self.assertRaises(ValueError, pylfit.preprocessing.discrete_state_transitions_dataset_from_array, data, features, targets)

    def test_continuous_state_transitions_dataset_from_csv(self):
        print(">> pylfit.preprocessing.tabular_dataset.continous_state_transitions_dataset_from_csv(path, feature_names, target_names)")

        for i in range(self._nb_random_tests):
            # Full dataset
            #--------------
            dataset_filepath = "datasets/repressilator.csv"
            features_col_header = ["p_t_1","q_t_1","r_t_1"]
            targets_col_header = ["p_t","q_t","r_t"]

            dataset = pylfit.preprocessing.continuous_state_transitions_dataset_from_csv(path=dataset_filepath, feature_names=features_col_header, target_names=targets_col_header)

            self.assertEqual(dataset.features, [("p_t_1", Continuum(0,1,True,True)), ("q_t_1", Continuum(0,1,True,True)), ("r_t_1", Continuum(0,1,True,True))])
            self.assertEqual(dataset.targets, [("p_t", Continuum(0,1,True,True)), ("q_t", Continuum(0,1,True,True)), ("r_t", Continuum(0,1,True,True))])

            data = [ \
            ([0,0,0],[0,0,1]), \
            ([1,0,0],[0,0,0]), \
            ([0,1,0],[1,0,1]), \
            ([0,0,1],[0,0,1]), \
            ([1,1,0],[1,0,0]), \
            ([1,0,1],[0,1,0]), \
            ([0,1,1],[1,0,1]), \
            ([1,1,1],[1,1,0])]

            data = [(np.array([float(i) for i in s1]), np.array([float(i) for i in s2])) for (s1,s2) in data]

            self.assertEqual(len(data), len(dataset.data))

            for i in range(len(data)):
                self.assertTrue( (dataset.data[i][0]==data[i][0]).all() )
                self.assertTrue( (dataset.data[i][1]==data[i][1]).all() )


            # Partial dataset
            #-----------------
            dataset_filepath = "datasets/repressilator.csv"
            features_col_header = ["p_t_1","r_t_1"]
            targets_col_header = ["q_t"]

            dataset = pylfit.preprocessing.continuous_state_transitions_dataset_from_csv(path=dataset_filepath, feature_names=features_col_header, target_names=targets_col_header)

            self.assertEqual(dataset.features, [("p_t_1", Continuum(0,1,True,True)), ("r_t_1", Continuum(0,1,True,True))])
            self.assertEqual(dataset.targets, [("q_t", Continuum(0,1,True,True))])

            data = [ \
            ([0,0],[0]), \
            ([1,0],[0]), \
            ([0,0],[0]), \
            ([0,1],[0]), \
            ([1,0],[0]), \
            ([1,1],[1]), \
            ([0,1],[0]), \
            ([1,1],[1])]

            data = [(np.array([float(i) for i in s1]), np.array([float(i) for i in s2])) for (s1,s2) in data]

            self.assertEqual(len(data), len(dataset.data))

            for i in range(len(data)):
                self.assertTrue( (dataset.data[i][0]==data[i][0]).all() )
                self.assertTrue( (dataset.data[i][1]==data[i][1]).all() )


    def test_continuous_state_transitions_dataset_from_array(self):
        print(">> pylfit.preprocessing.tabular_dataset.continuous_state_transitions_dataset_from_array(data, feature_domains, target_domains, feature_names, target_names)")

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

        dataset = pylfit.preprocessing.continuous_state_transitions_dataset_from_array(data=data, feature_names=feature_names, target_names=target_names)

        data = [(np.array([float(i) for i in s1]), np.array([float(i) for i in s2])) for (s1,s2) in data]

        self.assertEqual(dataset.features, [("p_t-1", Continuum(0,1,True,True)), ("q_t-1", Continuum(0,1,True,True)), ("r_t-1", Continuum(0,1,True,True))])
        self.assertEqual(dataset.targets, [("p_t", Continuum(0,1,True,True)), ("q_t", Continuum(0,1,True,True)), ("r_t", Continuum(0,1,True,True))])

        data = [(np.array([float(i) for i in s1]), np.array([float(i) for i in s2])) for (s1,s2) in data]

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

        self.assertRaises(TypeError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, feature_names, target_names)

        # data is not list of tuples
        data = [ \
        ([0,0,0],[0,0,1],[0,0,0],[1,0,0]), \
        [[1,0,0],[0,0,0]]]

        self.assertRaises(TypeError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, feature_names, target_names)

        # data is not list of pairs
        data = [ \
        ([0,0,0],[0,0,1],[0,0,0],[1,0,0]), \
        ([1,0,0],[0,0,0])]

        self.assertRaises(TypeError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, feature_names, target_names)

        # Not same size for features
        data = [ \
        ([0,0,0],[0,0,1]), \
        ([0,0,0],[1,0,0]), \
        ([1,0],[0,0,0])]

        self.assertRaises(ValueError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, None, None, feature_names, target_names)

        # Not same size for targets
        data = [ \
        ([0,0,0],[0,0,1]), \
        ([0,0,0],[1,0]), \
        ([1,0,0],[0,0,0])]

        self.assertRaises(ValueError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, None, None, feature_names, target_names)

        # Not only int/float in features
        data = [ \
        ([0,"0",0],[0,0,1]), \
        ([0,0.3,0],[1,0,0]), \
        ([1,0,0],[0,0,0])]

        self.assertRaises(ValueError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, None, None, feature_names, target_names)

        # Not only int/float in targets
        data = [ \
        ([0,0,0],[0,0.11,1]), \
        ([0,0,0],[1,"0",0]), \
        ([1,0,0],[0,0,0])]

        self.assertRaises(ValueError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, None, None, feature_names, target_names)

        # features is not a list of (string, Continuum)
        data = [ \
        ([0,0,0],[0,0,1]), \
        ([0,0,0],[1,0,0])]

        features = "" # not list
        self.assertRaises(TypeError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, features, None, None, None)
        features = [1,(1,2)] # not list of tuples
        self.assertRaises(TypeError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, features, None, None, None)
        features = [(1,1),(1,2,4),(1,2)] # not list of pair
        self.assertRaises(TypeError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, features, None, None, None)
        features = [("p_t",["1","2"]),(1, ["0"]),("r_t",["1","3"])] # not list of pair (string,_)
        self.assertRaises(TypeError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, features, None, None, None)
        features = [("p_t",["1","2"]),("q_t", "0"),("r_t",["1","2"])] # not list of pair (string, Continuum)
        self.assertRaises(TypeError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, features, None, None, None)
        features = [("p_t",Continuum(0,1,True,True)),("q_t", Continuum(0,1,True,True)),("p_t",Continuum(0,1,True,True))] # not all different variables
        self.assertRaises(ValueError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, features, None, None, None)

        # targets is not a list of (string, list of string)
        targets = "" # not list
        self.assertRaises(TypeError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, None, targets, None, None)
        targets = [1,(1,2)] # not list of tuples
        self.assertRaises(TypeError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, None, targets, None, None)
        targets = [(1,1),(1,2,4),(1,2)] # not list of pair
        self.assertRaises(TypeError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, None, targets, None, None)
        targets = [("p_t",["1","2"]),(1, ["0"]),("r_t",["1","3"])] # not list of pair (string, Continuum)
        self.assertRaises(TypeError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, None, targets, None, None)
        targets = [("p_t",["1","2"]),("q_t", "0"),("r_t",["1","2"])] # not list of pair (string, Continuum)
        self.assertRaises(TypeError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, None, targets, None, None)
        targets = [("p_t",Continuum(0,1,True,True)),("q_t", Continuum(0,1,True,True)),("p_t",Continuum(0,1,True,True))] # not all different variables
        self.assertRaises(ValueError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, None, targets, None, None)


        # Both features/feature_names or targets/target_names given
        features = [("p_t-1", Continuum(0,1,True,True)),("q_t-1",  Continuum(0,1,True,True)),("r_t-1", Continuum(0,1,True,True))]
        targets = [("p_t", Continuum(0,1,True,True)),("q_t",  Continuum(0,1,True,True)),("r_t", Continuum(0,1,True,True))]
        feature_names = ["p_t-1","q_t-1","r_t-1"]
        target_names = ["p_t","q_t","r_t"]
        self.assertRaises(ValueError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, features, None, feature_names, None)
        self.assertRaises(ValueError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, None, targets, None, target_names)
        self.assertRaises(ValueError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, features, targets, feature_names, target_names)

        # target_names is not list of string
        data = [ \
        ([0,0,0],[0,0,1]), \
        ([0,0,0],[1,0,0])]
        feature_names = ["p_t-1","q_t-1","r_t-1"]
        target_names = ["p_t","q_t","r_t"]

        feature_names = ""
        self.assertRaises(TypeError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, None, None, feature_names, target_names)
        feature_names = [1,0.5,"lol"]
        self.assertRaises(ValueError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, None, None, feature_names, target_names)

        # target_names is not list of string
        data = [ \
        ([0,0,0],[0,0,1]), \
        ([0,0,0],[1,0,0])]
        feature_names = ["p_t-1","q_t-1","r_t-1"]
        target_names = ["p_t","q_t","r_t"]

        target_names = ""
        self.assertRaises(TypeError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, None, None, feature_names, target_names)
        target_names = [1,0.5,"lol"]
        self.assertRaises(ValueError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, None, None, feature_names, target_names)

        # Random tests
        for i in range(self._nb_random_tests):

            original_dataset = random_ContinuousStateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            min_value=self._min_value, \
            max_value=self._max_value, \
            min_continuum_size=self._min_continuum_size)


            data = original_dataset.data
            features = original_dataset.features
            targets = original_dataset.targets
            feature_names = [var for var, vals in features]
            target_names = [var for var, vals in targets]

            # empty dataset
            self.assertEqual(pylfit.preprocessing.continuous_state_transitions_dataset_from_array(data=[], feature_domains=features, target_domains=targets), ContinuousStateTransitionsDataset(data=[], features=features, targets=targets))

            # Only data given
            dataset = pylfit.preprocessing.continuous_state_transitions_dataset_from_array(data=data)

            # Only names given
            dataset = pylfit.preprocessing.continuous_state_transitions_dataset_from_array(data=data, feature_names=feature_names, target_names=target_names)

            self.assertEqual([var for var, vals in dataset.features], feature_names)
            self.assertEqual([var for var, vals in dataset.targets], target_names)

            # all domain value appear in data
            for var_id, (var, vals) in enumerate(dataset.features):
                for val in [vals.min_value, vals.max_value]:
                    appear = False
                    for s1,s2 in data:
                        if s1[var_id] == val:
                            appear = True
                    self.assertTrue(appear)

            for var_id, (var, vals) in enumerate(dataset.targets):
                for val in [vals.min_value, vals.max_value]:
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
            dataset = pylfit.preprocessing.continuous_state_transitions_dataset_from_array(data=data, feature_domains=features, target_domains=targets)

            self.assertEqual(dataset.features, features)
            self.assertEqual(dataset.targets, targets)
            self.assertEqual(len(dataset.data), len(data))
            for i in range(len(data)):
                self.assertTrue( (dataset.data[i][0]==data[i][0]).all() )
                self.assertTrue( (dataset.data[i][1]==data[i][1]).all() )

            # feature domains only
            dataset = pylfit.preprocessing.continuous_state_transitions_dataset_from_array(data=data, feature_domains=features, target_names=target_names)

            self.assertEqual(dataset.features, features)
            self.assertEqual(len(dataset.data), len(data))
            for i in range(len(data)):
                self.assertTrue( (dataset.data[i][0]==data[i][0]).all() )
                self.assertTrue( (dataset.data[i][1]==data[i][1]).all() )

            # target domains only
            dataset = pylfit.preprocessing.continuous_state_transitions_dataset_from_array(data=data, target_domains=targets, feature_names=feature_names)

            self.assertEqual(dataset.targets, targets)
            self.assertEqual(len(dataset.data), len(data))
            for i in range(len(data)):
                self.assertTrue( (dataset.data[i][0]==data[i][0]).all() )
                self.assertTrue( (dataset.data[i][1]==data[i][1]).all() )

            # Exceptions

            # empty dataset
            self.assertRaises(ValueError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, [], None, targets)
            self.assertRaises(ValueError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, [], features, None)

            # Wrong data format
            data = [(list(s1)+[0],list(s2)) for s1,s2 in original_dataset.data]
            self.assertRaises(ValueError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, features, targets)


            data = [(list(s1),list(s2)+[0]) for s1,s2 in original_dataset.data]
            self.assertRaises(ValueError, pylfit.preprocessing.continuous_state_transitions_dataset_from_array, data, features, targets)

'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
