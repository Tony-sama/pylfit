#-----------------------
# @author: Tony Ribeiro
# @created: 2020/12/23
# @updated: 2020/12/23
#
# @desc: dataset class unit test script
#
#-----------------------

import unittest
import random
import sys
import numpy as np

import pylfit

random.seed(0)

class tabular_dataset_tests(unittest.TestCase):
    """
        Unit test of module tabular_dataset.py
    """
    _nb_random_tests = 10

    _max_variables = 100

    _max_domain_size = 10

    def test_transitions_dataset_from_csv(self):
        print(">> pylfit.preprocessing.transitions_dataset_from_csv(path, feature_names, target_names)")

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
        print(">> pylfit.preprocessing.transitions_dataset_from_csv(path, feature_names, target_names)")

        for i in range(self._nb_random_tests):
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

            self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, feature_names, target_names)

            # Not same size for targets
            data = [ \
            ([0,0,0],[0,0,1]), \
            ([0,0,0],[1,0]), \
            ([1,0,0],[0,0,0])]

            self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, feature_names, target_names)

            # Not only int/string in features
            data = [ \
            ([0,0,0],[0,0,1]), \
            ([0,0.3,0],[1,0,0]), \
            ([1,0,0],[0,0,0])]

            self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, feature_names, target_names)

            # Not only int/string in targets
            data = [ \
            ([0,0,0],[0,0.11,1]), \
            ([0,0,0],[1,0,0]), \
            ([1,0,0],[0,0,0])]

            self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, feature_names, target_names)

            # target_names is not list of string
            data = [ \
            ([0,0,0],[0,0,1]), \
            ([0,0,0],[1,0,0])]
            feature_names = ["p_t-1","q_t-1","r_t-1"]
            target_names = ["p_t","q_t","r_t"]

            feature_names = ""
            self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, feature_names, target_names)
            feature_names = [1,0.5,"lol"]
            self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, feature_names, target_names)

            # target_names is not list of string
            data = [ \
            ([0,0,0],[0,0,1]), \
            ([0,0,0],[1,0,0])]
            feature_names = ["p_t-1","q_t-1","r_t-1"]
            target_names = ["p_t","q_t","r_t"]

            target_names = ""
            self.assertRaises(TypeError, pylfit.preprocessing.transitions_dataset_from_array, data, feature_names, target_names)
            target_names = [1,0.5,"lol"]
            self.assertRaises(ValueError, pylfit.preprocessing.transitions_dataset_from_array, data, feature_names, target_names)

'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
