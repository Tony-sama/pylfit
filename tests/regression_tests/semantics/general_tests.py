#-----------------------
# @author: Tony Ribeiro
# @created: 2021/02/17
# @updated: 2021/06/15
#
# @desc: general class unit test script
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
from pylfit.semantics import General
from pylfit.algorithms import Algorithm
from pylfit.models import DMVLP
from pylfit.objects import Rule

from tests_generator import random_DMVLP, random_symmetric_StateTransitionsDataset

random.seed(0)

class General_tests(unittest.TestCase):
    """
        Unit test of class pylfit.semantics.General from pylfit/semantics/General.py
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
        print(">> pylfit.semantics.General.next(feature_state, targets, rules)")

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

        model = DMVLP(features=dataset.features, targets=dataset.targets)
        model.compile(algorithm="gula")
        model.fit(dataset=dataset)

        feature_state = Algorithm.encode_state(["0","0","0"], model.features)
        self.assertEqual(set([tuple(s) for s in General.next(feature_state, model.targets, model.rules)]), set([(1,0,0), (0,0,0), (0, 0, 1), (1,0,1)]))
        feature_state = Algorithm.encode_state(["1","1","1"], model.features)
        self.assertEqual(set([tuple(s) for s in General.next(feature_state, model.targets, model.rules)]), set([(1,1,0), (1,1,1)]))
        feature_state = Algorithm.encode_state(["0","1","0"], model.features)
        self.assertEqual(set([tuple(s) for s in General.next(feature_state, model.targets, model.rules)]), set([(1,0,1), (0,1,0), (1,0,0), (0,0,1), (1,1,1), (0,0,0), (1,1,0), (0,1,1)]))

        # incomplete program, semantics with default
        model = DMVLP(features=dataset.features, targets=dataset.targets)
        rules = [
        "p_t(1) :- q_t-1(1)",
        "q_t(1) :- p_t-1(1), r_t-1(1)",
        "r_t(1) :- p_t-1(0)"]
        model.rules = [Rule.from_string(s, model.features, model.targets) for s in rules]
        default = [("p_t", [0]), ("q_t", [0]), ("r_t", [0])]

        feature_state = Algorithm.encode_state([0,0,0], model.features)
        self.assertEqual(set([tuple(s) for s in General.next(feature_state, model.targets, model.rules, default)]), set([(0,0,0), (0,0,1)]))

        feature_state = Algorithm.encode_state([0,0,1], model.features)
        self.assertEqual(set([tuple(s) for s in General.next(feature_state, model.targets, model.rules, default)]), set([(0,0,1)]))

        feature_state = Algorithm.encode_state([0,1,0], model.features)
        self.assertEqual(set([tuple(s) for s in General.next(feature_state, model.targets, model.rules, default)]), set([(0,1,0), (0,0,0), (0,1,1), (0,0,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]))

        feature_state = Algorithm.encode_state([1,0,0], model.features)
        self.assertEqual(set([tuple(s) for s in General.next(feature_state, model.targets, model.rules, default)]), set([(1,0,0), (0,0,0)]))

        feature_state = Algorithm.encode_state([0,1,1], model.features)
        self.assertEqual(set([tuple(s) for s in General.next(feature_state, model.targets, model.rules, default)]), set([(0,1,1), (0,0,1), (1,1,1), (1,0,1)]))

        feature_state = Algorithm.encode_state([1,0,1], model.features)
        self.assertEqual(set([tuple(s) for s in General.next(feature_state, model.targets, model.rules, default)]), set([(1,0,1), (0,0,1), (1,0,0), (1,1,1), (0,1,1), (1,1,0), (0,1,0), (0,0,0)]))

        feature_state = Algorithm.encode_state([1,1,0], model.features)
        self.assertEqual(set([tuple(s) for s in General.next(feature_state, model.targets, model.rules, default)]), set([(1,1,0), (1,0,0)]))

        feature_state = Algorithm.encode_state([1,1,1], model.features)
        self.assertEqual(set([tuple(s) for s in General.next(feature_state, model.targets, model.rules, default)]), set([(1,1,1), (1,1,0)]))

        # Default to unknow
        feature_state = Algorithm.encode_state([1,1,1], model.features)
        self.assertEqual(set([tuple(s) for s in General.next(feature_state, model.targets, model.rules, None)]), set([(1,1,1), (1,1,-1)]))

        # Random tests
        for i in range(self._nb_tests):
            dataset = random_symmetric_StateTransitionsDataset(100, random.randint(1,self._nb_features), self._nb_feature_values)

            model = DMVLP(features=dataset.features, targets=dataset.targets)
            model.compile(algorithm="pride")
            model.fit(dataset=dataset)

            feature_state = random.choice(model.feature_states())
            feature_state = Algorithm.encode_state(feature_state, model.features)

            output = General.next(feature_state, model.targets, model.rules)

            domains = [set() for var in model.targets]

            # extract conclusion of all matching rules
            for r in model.rules:
                if(r.matches(feature_state)):
                    domains[r.head_variable].add(r.head_value)

            # Check variables without next value
            for i,domain in enumerate(domains):
                if i < len(model.features):
                    domains[i].add(feature_state[i])
                if len(domain) == 0:
                    domains[i] = set([-1])

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
                    self.assertEqual(r.head_value,state[r.head_variable])
                    self.assertEqual([],[r for r in model.rules if r not in rules and r.matches(feature_state) and r.head_value == state[r.head_variable]])

            # Exception:
            # ----------

            #self.assertRaises(NotImplementedError, semantics.next, "", "", "")

'''
@desc: main
'''
if __name__ == '__main__':
    unittest.main()
