#-----------------------
# @author: Tony Ribeiro
# @created: 2021/02/17
# @updated: 2021/06/15
#
# @desc: SynchronousConstrained class unit test script
# done:
# - next
#
# Todo:
#-----------------------

import unittest
import random
import sys
import itertools

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

import pylfit
from pylfit.utils import eprint
from pylfit.semantics import SynchronousConstrained, Synchronous
from pylfit.algorithms import Algorithm
from pylfit.models import CDMVLP

from tests_generator import random_CDMVLP, random_StateTransitionsDataset

random.seed(0)

class SynchronousConstrained_tests(unittest.TestCase):
    """
        Unit test of class pylfit.semantics.SynchronousConstrained from pylfit/semantics/SynchronousConstrained.py
    """
    _nb_tests = 100

    _nb_transitions = 100

    _nb_features = 3

    _nb_targets = 3

    _nb_feature_values = 3

    _nb_target_values = 3

    #------------------
    # Constructors
    #------------------

    def test_next(self):
        print(">> pylfit.semantics.SynchronousConstrained.next(feature_state, targets, rules)")

        # Unit test
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
        feature_names=["p_t-1","q_t-1","r_t-1"]
        target_names=["p_t","q_t","r_t"]

        dataset = pylfit.preprocessing.transitions_dataset_from_array(data=data, feature_names=feature_names, target_names=target_names)

        model = CDMVLP(features=dataset.features, targets=dataset.targets)
        model.compile(algorithm="synchronizer")
        model.fit(dataset=dataset)

        feature_state = Algorithm.encode_state([0,0,0], model.features)
        self.assertEqual(set([tuple(s) for s in SynchronousConstrained.next(feature_state, model.targets, model.rules, model.constraints)]), set([(1,0,0), (0, 0, 1)]))
        feature_state = Algorithm.encode_state([1,1,1], model.features)
        self.assertEqual(set([tuple(s) for s in SynchronousConstrained.next(feature_state, model.targets, model.rules, model.constraints)]), set([(1,1,0)]))
        feature_state = Algorithm.encode_state([0,1,0], model.features)
        self.assertEqual(set([tuple(s) for s in SynchronousConstrained.next(feature_state, model.targets, model.rules, model.constraints)]), set([(1,0,1)]))

        # Random tests
        for i in range(self._nb_tests):

            # Apply CDMVLP correctly
            model = random_CDMVLP( \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values, \
            algorithm="synchronizer")

            feature_state = random.choice(model.feature_states())
            feature_state = Algorithm.encode_state(feature_state, model.features)

            target_states = SynchronousConstrained.next(feature_state, model.targets, model.rules, model.constraints)

            domains = [set() for var in model.targets]

            # Apply synchronous semantics
            candidates = Synchronous.next(feature_state, model.targets, model.rules)

            # Apply constraints
            expected = []
            for s in candidates:
                valid = True
                for c in model.constraints:
                    if c.matches(list(feature_state)+list(s)):
                        valid = False
                        #eprint(c, " matches ", feature_state, ", ", s)
                        break
                if valid:
                    # Decode state with domain values
                    expected.append(s)

            for s2 in target_states:
                self.assertTrue(s2 in expected)

            for s2 in expected:
                self.assertTrue(s2 in target_states)

            # Exception:
            # ----------

            #self.assertRaises(NotImplementedError, semantics.next, "", "", "")

'''
@desc: main
'''
if __name__ == '__main__':
    unittest.main()
