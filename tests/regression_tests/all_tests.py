#-----------------------
# @author: Tony Ribeiro
# @created: 2019/03/25
# @updated: 2019/05/02
#
# @desc: PyLFIT unit test script
#
#-----------------------

import unittest

# Core
#------

# Objects
from objects.rule_tests import Rule_tests
#from objects.logicProgram_tests import LogicProgramTest

#from objects.continuum_tests import ContinuumTest
#from objects.continuumRule_tests import ContinuumRuleTest
#from objects.continuumLogicProgram_tests import ContinuumLogicProgramTest

# Semantics
from semantics.semantics_tests import Semantics_tests
from semantics.synchronous_tests import Synchronous_tests
from semantics.asynchronous_tests import Asynchronous_tests
from semantics.general_tests import General_tests
from semantics.synchronousConstrained_tests import SynchronousConstrained_tests

# Algorithms
#from algorithms.lf1t_tests import LF1TTest
from algorithms.gula_tests import GULA_tests
from algorithms.pride_tests import PRIDE_tests
#from algorithms.lfkt_tests import LFkTTest
#from algorithms.lust_tests import LUSTTest
#from algorithms.acedia_tests import ACEDIATest
from algorithms.synchronizer_tests import Synchronizer_tests
#from algorithms.probabilizer_tests import ProbabilizerTest

# Benchmarks
from algorithms.gula_benchmark_tests import GULA_benchmark_tests
from algorithms.pride_benchmark_tests import PRIDE_benchmark_tests
from algorithms.synchronizer_benchmark_tests import Synchronizer_benchmark_tests

# Api
#-----

# Datasets
from datasets.dataset_tests import Dataset_tests
from datasets.stateTransitionsDataset_tests import StateTransitionsDataset_tests

# Preprocessing
from preprocessing.tabular_dataset_tests import tabular_dataset_tests

# Models
from models.model_tests import Model_tests
from models.dmvlp_tests import DMVLP_tests
from models.cdmvlp_tests import CDMVLP_tests
from models.wdmvlp_tests import WDMVLP_tests

# postprocessing

#random.seed(0)

if __name__ == '__main__':
    """ Main """

    unittest.main()
