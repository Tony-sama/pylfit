#-----------------------
# @author: Tony Ribeiro
# @created: 2019/03/25
# @updated: 2023/12/26
#
# @desc: PyLFIT unit test script
#
#-----------------------

import unittest
import os
import random
# Core
#------

# Objects
from .objects.rule_tests import Rule_tests

from .objects.continuum_tests import ContinuumTest
from .objects.continuumRule_tests import ContinuumRuleTest

# Semantics
from .semantics.semantics_tests import Semantics_tests
from .semantics.synchronous_tests import Synchronous_tests
from .semantics.asynchronous_tests import Asynchronous_tests
from .semantics.general_tests import General_tests
from .semantics.synchronousConstrained_tests import SynchronousConstrained_tests

# Algorithms
from .algorithms.gula_tests import GULA_tests
from .algorithms.pride_tests import PRIDE_tests
from .algorithms.bruteForce_tests import BruteForce_tests
from .algorithms.lfkt_tests import LFkTTest
from .algorithms.lust_tests import LUSTTest
from .algorithms.acedia_tests import ACEDIATest
from .algorithms.synchronizer_tests import Synchronizer_tests
from .algorithms.probalizer_tests import ProbalizerTest

# Benchmarks
from .algorithms.gula_benchmark_tests import GULA_benchmark_tests
from .algorithms.pride_benchmark_tests import PRIDE_benchmark_tests
#from algorithms.bruteForce_benchmark_tests import BruteForce_benchmark_tests # OK but too long
from .algorithms.synchronizer_benchmark_tests import Synchronizer_benchmark_tests
from .algorithms.acedia_benchmark_tests import ACEDIA_benchmark_tests

# Api
#-----

# Datasets
from .datasets.dataset_tests import Dataset_tests
from .datasets.discreteStateTransitionsDataset_tests import DiscreteStateTransitionsDataset_tests
from .datasets.continuousStateTransitionsDataset_tests import ContinuousStateTransitionsDataset_tests

# Preprocessing
from .preprocessing.tabular_dataset_tests import tabular_dataset_tests
from .preprocessing.boolean_network_tests import boolean_network_tests

# Models
from .models.model_tests import Model_tests
from .models.dmvlp_tests import DMVLP_tests
from .models.cdmvlp_tests import CDMVLP_tests
from .models.wdmvlp_tests import WDMVLP_tests
from .models.pdmvlp_tests import PDMVLP_tests
from .models.clp_tests import CLP_tests

# postprocessing
from .postprocessing.metrics_tests import metrics_tests

#random.seed(0)

if __name__ == '__main__':
    """ Main """

    random.seed(0)

    current_directory = os.getcwd()
    tmp_directory = os.path.join(current_directory, r'tmp')
    if not os.path.exists(tmp_directory):
       os.makedirs(tmp_directory)

    unittest.main()
