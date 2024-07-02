#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/23
# @updated: 2023/12/26
#
# @desc: PyLFIT unit test script
#
#-----------------------

import unittest
import random
import os
import sys
import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from pylfit.utils import eprint
from pylfit.algorithms.lust import LUST
from pylfit.objects.rule import Rule
from pylfit.semantics.synchronous import Synchronous
from pylfit.datasets.discreteStateTransitionsDataset import DiscreteStateTransitionsDataset
from pylfit.models.dmvlp import DMVLP

from tests_generator import random_DiscreteStateTransitionsDataset, random_DMVLP

#random.seed(0)

class LUSTTest(unittest.TestCase):
    """
        Unit test of class LUST from lust.py
    """

    _nb_unit_test = 100

    _nb_transitions = 100

    _nb_features = 4

    _nb_targets = 3

    _nb_feature_values = 3

    _nb_target_values = 3

    _max_programs = 5

    _body_size = 10

    _tmp_file_path = "tmp/unit_test_lust.tmp"

    #------------------
    # Test functions
    #------------------

    def test_fit(self):
        print(">> LUST.fit(data, features, targets)")

        # No transitions
        #p = self.random_program(self._nb_features, self._nb_targets, self._nb_values, self._body_size)
        dataset = random_DiscreteStateTransitionsDataset( \
                            nb_transitions=0, \
                            nb_features=random.randint(1,self._nb_features), \
                            nb_targets=random.randint(1,self._nb_targets), \
                            max_feature_values=self._nb_feature_values, max_target_values=self._nb_target_values)
        p = LUST.fit(dataset)
        self.assertEqual(len(p),1)

        for i in range(self._nb_unit_test):
            #eprint("test: ", i, "/", self._nb_unit_test)

            dataset = random_DiscreteStateTransitionsDataset( \
                            nb_transitions=self._nb_transitions, \
                            nb_features=random.randint(1,self._nb_features), \
                            nb_targets=random.randint(1,self._nb_targets), \
                            max_feature_values=self._nb_feature_values, max_target_values=self._nb_target_values)

            rules_set = LUST.fit(dataset)

            # Generate transitions
            predictions = []
            for rules in rules_set:
                p = DMVLP(dataset.features,dataset.targets,rules)
                predictions += [(tuple(s1),tuple(s2)) for (s1,S) in p.predict([s for s,_ in dataset.data]).items() for s2 in S]

            # Remove incomplete states
            #predictions = [ (s1,s2) for s1,s2 in predictions if -1 not in s2 ]

            #eprint("Expected: ", transitions)
            #eprint("Predicted: ", predictions)
            dataset_data = [(tuple(s1),tuple(s2)) for (s1,s2) in dataset.data]

            # All original transitions are predicted
            for s1, s2 in dataset_data:
                self.assertTrue((s1,s2) in predictions)

            # All predictions are in original transitions
            for s1, s2 in predictions:
                if (s1,s2) not in dataset_data:
                    eprint(s1,s2)
                    eprint(dataset_data)
                self.assertTrue((s1,s2) in dataset_data)

            #sys.exit()

    def test_interprete(self):
        print(">> LUST.interprete(dataset)")

        for i in range(self._nb_unit_test):
            #eprint("test: ", i, "/", self._nb_unit_test)

            # No transitions
            dataset = random_DiscreteStateTransitionsDataset( \
                            nb_transitions=0, \
                            nb_features=random.randint(1,self._nb_features), \
                            nb_targets=random.randint(1,self._nb_targets), \
                            max_feature_values=self._nb_feature_values, max_target_values=self._nb_target_values)
            features = dataset.features
            targets = dataset.targets

            p = random_DMVLP(self._nb_features, self._nb_targets, self._nb_feature_values, self._nb_target_values, "gula")

            DC, DS = LUST.interprete(dataset)
            self.assertEqual(DC.data,[])
            self.assertEqual(DS,[])

            # Regular case
            dataset = random_DiscreteStateTransitionsDataset( \
                            nb_transitions=self._nb_transitions, \
                            nb_features=random.randint(1,self._nb_features), \
                            nb_targets=random.randint(1,self._nb_targets), \
                            max_feature_values=self._nb_feature_values, max_target_values=self._nb_target_values)

            DC, DS = LUST.interprete(dataset)
            D = []
            ND = []

            DC_data = [(tuple(s1),tuple(s2)) for (s1,s2) in DC.data]
            DS_data = [[(tuple(s1),tuple(s2)) for (s1,s2) in s.data] for s in DS]

            for s1, s2 in dataset.data:
                deterministic = True
                for s3, s4 in dataset.data:
                    if tuple(s1) == tuple(s3) and tuple(s2) != tuple(s4):
                        ND.append( (tuple(s1),tuple(s2)) )
                        deterministic = False
                        break
                if deterministic:
                    D.append( (tuple(s1),tuple(s2)) )

            #eprint("DC: ",DC)
            #eprint("DS: ",DS)
            #eprint("D: ",D)
            #eprint("ND: ",ND)

            # All deterministic are only in DC
            for s1, s2 in D:
                self.assertTrue((s1,s2) in DC_data)
                for s in DS_data:
                    self.assertTrue((s1,s2) not in s)

            # All DC are deterministic
            for s1, s2 in DC_data:
                self.assertTrue((s1,s2) in D)

            # All non deterministic sets are set
            for s in DS_data:
                for s1, s2 in s:
                    occ = 0
                    for s3, s4 in s:
                        if s1 == s3 and s2 == s4:
                            occ += 1
                    self.assertEqual(occ,1)

            # All input origin state appears in each DS
            for s1, s2 in ND:
                for s in DS_data:
                    occurs = False
                    for s3, s4 in s:
                        if s1 == s3:
                            occurs = True
                    self.assertTrue(occurs)

            # All DS are deterministic
            for s in DS_data:
                for s1, s2 in s:
                    for s3, s4 in s:
                        if s1 == s3:
                            self.assertTrue(s2 == s4)

    #------------------
    # Tool functions
    #------------------


if __name__ == '__main__':
    """ Main """

    unittest.main()
