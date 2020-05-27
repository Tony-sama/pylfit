#-----------------------
# @author: Tony Ribeiro
# @created: 2019/03/25
# @updated: 2019/05/02
#
# @desc: PyLFIT unit test script
#
#-----------------------

import unittest

import sys
sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')

from unit_tests_rule import RuleTest
from unit_tests_logicProgram import LogicProgramTest

from unit_tests_continuum import ContinuumTest
from unit_tests_continuumRule import ContinuumRuleTest
from unit_tests_continuumLogicProgram import ContinuumLogicProgramTest

from unit_tests_pride import PRIDETest
from unit_tests_lf1t import LF1TTest
from unit_tests_gula import GULATest
from unit_tests_lfkt import LFkTTest
from unit_tests_lust import LUSTTest
from unit_tests_acedia import ACEDIATest

#random.seed(0)

if __name__ == '__main__':
    """ Main """

    unittest.main()
