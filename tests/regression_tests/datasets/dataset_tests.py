#-----------------------
# @author: Tony Ribeiro
# @created: 2020/12/23
# @updated: 2020/12/23
#
# @desc: dataset class unit test script
# done:
# - __init__
#
# Todo:
#-----------------------

import unittest
import random
import sys

import sys

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from pylfit.utils import eprint
from pylfit.datasets import Dataset

random.seed(0)

class Dataset_tests(unittest.TestCase):
    """
        Unit test of class Datatset from dataset.py
    """
    _nb_tests = 100

    _nb_features = 5

    _nb_targets = 5

    _nb_feature_values = 3

    _nb_target_values = 3

    #------------------
    # Constructors
    #------------------

    def test_empty_constructor(self):
        print(">> Dataset.__init__(self, data, feature_domains, targets_domains)")
        for i in range(self._nb_tests):
            features = [(var, ["x"+str(val) for val in range(0, random.randint(1,self._nb_feature_values))]) for var in range(0, random.randint(1,self._nb_features))]
            targets = [(var, ["y"+str(val) for val in range(0, random.randint(1,self._nb_target_values))]) for var in range(0, random.randint(1,self._nb_targets))]

            # Constructor data/features/targets
            self.assertRaises(NotImplementedError, Dataset, [], [], [])


'''
@desc: main
'''
if __name__ == '__main__':
    unittest.main()
