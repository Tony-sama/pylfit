#-----------------------
# @author: Tony Ribeiro
# @created: 2021/02/17
# @updated: 2021/02/17
#
# @desc: semantics class unit test script
# done:
# - ,next
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
from pylfit.semantics import Semantics

random.seed(0)

class Semantics_tests(unittest.TestCase):
    """
        Unit test of class pylfit.semantics.Semantics from pylfit/semantics/semantics.py
    """
    _nb_tests = 100

    #------------------
    # Constructors
    #------------------

    def test_next(self):
        print(">> pylfit.semantics.Semantics.next(feature_state, targets, rules)")
        for i in range(self._nb_tests):
            class SemanticsSubClass(Semantics):
                _test = 0

            semantics = SemanticsSubClass()

            # Exception:
            # ----------

            self.assertRaises(NotImplementedError, semantics.next, "", "", "")

'''
@desc: main
'''
if __name__ == '__main__':
    unittest.main()
