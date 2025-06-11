#-----------------------
# @author: Tony Ribeiro
# @created: 2025/04/10
# @updated: 2025/04/10
#
# @desc: counterfactuals.py regression test script
#
#-----------------------

import unittest
import random
import sys
import numpy as np
import contextlib
import io

import pylfit
from pylfit.postprocessing import bruteforce_counterfactuals, compute_counterfactuals
from pylfit.models import DMVLP
from pylfit.datasets import DiscreteStateTransitionsDataset
from pylfit.utils import eprint

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from tests_generator import random_rule, random_WDMVLP, random_DiscreteStateTransitionsDataset

random.seed(0)

class metrics_tests(unittest.TestCase):
    """
        Unit test of module metrics.py
    """
    _nb_random_tests = 100

    _nb_features = 6

    _nb_feature_values = 4

    _nb_targets = 4

    _nb_target_values = 3

    _max_body_size = 10

    _nb_transitions = 100

    _max_rules_per_head = 4

    def test_bruteforce_counterfactuals(self):
        print(">> pylfit.postprocessing.bruteforce_counterfactuals(dmvlp, feature_state, target_variable, excluded_values, desired_values)")

        for test in range(self._nb_random_tests):
            dataset = random_DiscreteStateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            model = DMVLP(features=dataset.features, targets=dataset.targets)
            model.compile(algorithm="gula")
            model.fit(dataset=dataset)

            feature_state = dataset.data[random.randint(0,len(dataset.data)-1)][0]
            target_variable = random.randint(0, len(model.targets)-1)
            target_domain = model.targets[target_variable][1]
            target_variable = model.targets[target_variable][0]

            nb_excluded = random.randint(0, len(target_domain)-1)
            excluded_values = random.sample(target_domain, nb_excluded)

            desired_values = [x for x in target_domain if x not in excluded_values]
            nb_desired = random.randint(1, len(desired_values))
            desired_values = random.sample(desired_values, nb_desired)

            output = bruteforce_counterfactuals(model, feature_state, target_variable, excluded_values, desired_values)

            excluded_rules = [r for r in model.rules if r.head.variable == target_variable and r.head.value in excluded_values]

            # Rules that can produce desired values
            for value in desired_values:
                desired_rules = [r for r in model.rules if r.head.variable == target_variable and r.head.value == value]

                # None matches excluded rules
                for s in output[value]:
                    new_state = feature_state.copy()
                    for atom in s:
                        new_state[atom.state_position] = atom.value
                    # check matching
                    for r in excluded_rules:
                        self.assertFalse(r.matches(new_state))

                    # Matched by a desired rule
                    matched = False
                    for r in desired_rules:
                        if r.matches(new_state):
                            matched = True
                            break
                    self.assertTrue(matched)

    def test_compute_counterfactuals(self):
        print(">> pylfit.postprocessing.compute_counterfactuals(rules, feature_state, target_variable, excluded_values, desired_values, determinist, verbose)")

        for test in range(self._nb_random_tests):
            for verbose in [0,1]:
                dataset = random_DiscreteStateTransitionsDataset( \
                nb_transitions=random.randint(1, self._nb_transitions), \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values)

                model = DMVLP(features=dataset.features, targets=dataset.targets)
                model.compile(algorithm="gula")
                model.fit(dataset=dataset)

                feature_state = dataset.data[random.randint(0,len(dataset.data)-1)][0]
                target_variable = random.randint(0, len(model.targets)-1)
                target_domain = model.targets[target_variable][1]
                target_variable = model.targets[target_variable][0]

                nb_excluded = random.randint(0, len(target_domain)-1)
                excluded_values = random.sample(target_domain, nb_excluded)

                desired_values = [x for x in target_domain if x not in excluded_values]
                nb_desired = random.randint(1, len(desired_values))
                desired_values = random.sample(desired_values, nb_desired)

                # Limit the number of rules
                rules = []
                candidates = model.rules
                nb_rules = dict()
                for r in candidates:
                    if r.head not in nb_rules:
                        nb_rules[r.head] = 1
                    elif nb_rules[r.head] >= self._max_rules_per_head:
                        continue
                    else:
                        rules.append(r)
                        nb_rules[r.head] += 1
                
                model.rules = rules

                f = io.StringIO()
                with contextlib.redirect_stderr(f):
                    output = compute_counterfactuals(model, feature_state, target_variable, excluded_values, desired_values, verbose)
                expected = bruteforce_counterfactuals(model, feature_state, target_variable, excluded_values, desired_values)

                # Should be equal
                for value in output:
                    for c in output[value]:
                        if c not in expected[value]:
                            print("ERROR",c)
                        self.assertTrue(c in expected[value])
                
                for value in expected:
                    for c in expected[value]:
                        if c not in output[value]:
                            print("ERROR",c)
                        self.assertTrue(c in output[value])



'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
