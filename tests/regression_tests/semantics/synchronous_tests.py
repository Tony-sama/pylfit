#-----------------------
# @author: Tony Ribeiro
# @created: 2021/02/17
# @updated: 2023/12/22
#
# @desc: synchronous class unit test script
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
from pylfit.semantics import Synchronous
from pylfit.algorithms import Algorithm
from pylfit.models import DMVLP
from pylfit.objects import Rule

from tests_generator import random_DMVLP

random.seed(0)

class Synchronous_tests(unittest.TestCase):
    """
        Unit test of class pylfit.semantics.Synchronous from pylfit/semantics/synchronous.py
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
        print(">> pylfit.semantics.Synchronous.next(feature_state, targets, rules)")

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

        dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=data, feature_names=feature_names, target_names=target_names)

        model = DMVLP(features=dataset.features, targets=dataset.targets)
        model.compile(algorithm="gula")
        model.fit(dataset=dataset)

        feature_state = ["0","0","0"]
        self.assertEqual(set([tuple(s) for s in Synchronous.next(feature_state, model.targets, model.rules)]), set([("1","0","0"), ("0","0","0"), ("0", "0", "1"), ("1","0","1")]))
        feature_state = ["1","1","1"]
        self.assertEqual(set([tuple(s) for s in Synchronous.next(feature_state, model.targets, model.rules)]), set([("1","1","0")]))
        feature_state = ["0","1","0"]
        self.assertEqual(set([tuple(s) for s in Synchronous.next(feature_state, model.targets, model.rules)]), set([("1","0","1")]))

        # test default
        rules = []
        for r in model.rules:
            if r.head.variable != "p_t":
                rules.append(r)

        model.rules = rules
        #print(model.rules)
        feature_state = ["0","0","0"]
        self.assertEqual(set([tuple(s) for s in Synchronous.next(feature_state, model.targets, model.rules)]), set([("?","0","0"), ("?", "0", "1"), ("?","0","1")]))

        default = [("p_t", ["1"]), ("q_t", ["0"]), ("r_t", ["0"])]
        feature_state = ["0","0","0"]
        self.assertEqual(set([tuple(s) for s in Synchronous.next(feature_state, model.targets, model.rules, default)]), set([("1","0","0"), ("1", "0", "1")]))

        # Random tests
        for i in range(self._nb_tests):
            model = random_DMVLP( \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values, \
            algorithm="pride")

            feature_state = random.choice(model.feature_states())
            feature_state = feature_state

            output = Synchronous.next(feature_state, model.targets, model.rules, default=None)

            domains = [set() for var in model.targets]

            # extract conclusion of all matching rules
            for r in model.rules:
                if(r.matches(feature_state)):
                    domains[r.head.state_position].add(r.head.value)

            # Check variables without next value
            for i,domain in enumerate(domains):
                if len(domain) == 0:
                    domains[i] = [-1]

            # generate all combination of domains
            expected = [list(i) for i in list(itertools.product(*domains))]

            target_states = output.keys()
            expected = [tuple(i) for i in expected]

            for s2 in target_states:
                self.assertTrue(s2 in expected)

            for s2 in expected:
                self.assertTrue(s2 in target_states)

            for state, rules in output.items():
                for r in rules:
                    self.assertTrue(r.matches(feature_state))
                    self.assertEqual(r.head.value,state[r.head.state_position])
                    self.assertEqual([],[r for r in model.rules if r not in rules and r.matches(feature_state) and r.head.value == state[r.head.state_position]])

            # Exception:
            # ----------

            #self.assertRaises(NotImplementedError, semantics.next, "", "", "")

'''
@desc: main
'''
if __name__ == '__main__':
    unittest.main()
