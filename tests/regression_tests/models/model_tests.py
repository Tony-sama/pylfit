#-----------------------
# @author: Tony Ribeiro
# @created: 2020/12/23
# @updated: 2020/12/23
#
# @desc: dataset class unit test script
# done:
# - __init__
# - interface
#   - compile
#   - fit
#   - summary
#   - to_string
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
from pylfit.models import Model
from pylfit.datasets import Dataset

random.seed(0)

class Model_tests(unittest.TestCase):
    """
        Unit test of class pylfit.models.Model from pylfit/models/model.py
    """
    _nb_tests = 100

    #------------------
    # Constructors
    #------------------

    def test_empty_constructor(self):
        print(">> pylfit.models.Model.__init__(self, data, feature_domains, targets_domains)")
        for i in range(self._nb_tests):
            class ModelSubClass(Model):
                _COMPATIBLE_DATASETS = [Dataset]
                _ALGORITHMS = ["algoname"]
                _OPTIMIZERS = []

            model = ModelSubClass()

            self.assertEqual(model.algorithm, None)

            # Exception:
            # ----------

            self.assertRaises(NotImplementedError, model.compile, "algoname")
            self.assertRaises(NotImplementedError, model.fit, [])
            self.assertRaises(NotImplementedError, model.summary)
            self.assertRaises(NotImplementedError, model.to_string)

            class ModelSubClass(Model):
                _COMPATIBLE_DATASETS = [int]
                _ALGORITHMS = ["algoname"]
                _OPTIMIZERS = []

            self.assertRaises(ValueError, ModelSubClass)

            class ModelSubClass(Model):
                _COMPATIBLE_DATASETS = [Dataset]
                _ALGORITHMS = [42]
                _OPTIMIZERS = []

            self.assertRaises(ValueError, ModelSubClass)

'''
@desc: main
'''
if __name__ == '__main__':
    unittest.main()
