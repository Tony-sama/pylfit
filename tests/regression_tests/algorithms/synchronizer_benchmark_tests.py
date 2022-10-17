#-----------------------
# @author: Tony Ribeiro
# @created: 2021/02/03
# @updated: 2021/06/15
#
# @desc: Synchronizer regression test script
# Tests algorithm methods on benchmark dataset
# Done:
#   - Repressilator
# Todo:
#   - Others
#
#-----------------------

import unittest
import random
import os

import sys

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

import itertools

from tests_generator import random_DiscreteStateTransitionsDataset

from pylfit.utils import eprint
from pylfit.algorithms import Synchronizer
from pylfit.objects import Rule

from pylfit.datasets import DiscreteStateTransitionsDataset
from pylfit.preprocessing import discrete_state_transitions_dataset_from_csv

from pylfit.models import CDMVLP

random.seed(0)

class Synchronizer_benchmark_tests(unittest.TestCase):
    """
        Regression tests of class Synchronizer from Synchronizer.py with benchmarks data
    """

    _nb_tests = 10

    #------------------
    # Test functions
    #------------------

    def test_repressilator(self):
        print(">> Synchronizer benchmark <repressilator>:")

        dataset_filepath = "datasets/repressilator.csv"
        features_col_header = ["p_t_1","q_t_1","r_t_1"]
        targets_col_header = ["p_t","q_t","r_t"]

        dataset = discrete_state_transitions_dataset_from_csv(path=dataset_filepath, feature_names=features_col_header, target_names=targets_col_header)

        # Expected rules
        expected_string_rules = """
        p_t(1) :- q_t_1(1).
        q_t(1) :- p_t_1(1), r_t_1(1).
        r_t(1) :- p_t_1(0).

        p_t(0) :- q_t_1(0).
        q_t(0) :- p_t_1(0).
        q_t(0) :- r_t_1(0).
        r_t(0) :- p_t_1(1)."""

        expected_string_constraints= ""

        self._check_rules_and_predictions(dataset, expected_string_rules, expected_string_constraints)

    def test_disjonctive_boolean_network(self):
        print(">> Synchronizer benchmark <disjonctive_boolean_network>:")

        dataset_filepath = "datasets/disjonctive_boolean_network.csv"
        features_col_header = ["a","b","c"]
        targets_col_header = ["a_t","b_t","c_t"]

        dataset = discrete_state_transitions_dataset_from_csv(path=dataset_filepath, feature_names=features_col_header, target_names=targets_col_header)

        # Expected rules
        expected_string_rules = """
        a_t(1) :- b(1).
        a_t(1) :- a(1), c(1).

        b_t(1) :- a(0).
        b_t(1) :- b(0), c(1).

        c_t(1) :- a(0).

        a_t(0) :- b(0).

        b_t(0) :- .

        c_t(0) :- ."""

        expected_string_constraints= """
        :- b_t(1), c_t(1).

        :- a(0), b_t(0), c_t(0).
        :- b(0), a_t(1), b_t(1).
        :- a(1), a_t(1), b_t(1).

        :- c(1), a_t(0), b_t(0), c_t(0).
        :- a(1), c(1), a_t(0), b_t(0).
        """

        self._check_rules_and_predictions(dataset, expected_string_rules, expected_string_constraints)

    def test_toy_all_or_nothing_change(self):
        print(">> Synchronizer benchmark <toy_all_or_nothing_change>:")

        dataset_filepath = "datasets/toy_all_or_nothing_change.csv"
        features_col_header = ["x0","x1"]
        targets_col_header = ["y0","y1"]

        dataset = discrete_state_transitions_dataset_from_csv(path=dataset_filepath, feature_names=features_col_header, target_names=targets_col_header)

        # Expected rules
        expected_string_rules = """
        y0(0) :- x0(0).
        y0(0) :- x1(1).
        y0(1) :- x0(1).
        y0(1) :- x1(0).
        y1(0) :- x0(1).
        y1(0) :- x1(0).
        y1(1) :- x0(0).
        y1(1) :- x1(1).
        """

        expected_string_constraints= """
        :- x1(0), y0(0), y1(1).
        :- x0(0), y0(1), y1(0).
        :- x1(1), y0(1), y1(0).
        :- x0(1), y0(0), y1(1).
        """

        self._check_rules_and_predictions(dataset, expected_string_rules, expected_string_constraints)

    def test_toy_asynchronous_example(self):
        print(">> Synchronizer benchmark <toy_asynchronous_example>:")

        dataset_filepath = "datasets/toy_asynchronous_example.csv"
        features_col_header = ["x0","x1","x2"]
        targets_col_header = ["y0","y1","y2"]

        dataset = discrete_state_transitions_dataset_from_csv(path=dataset_filepath, feature_names=features_col_header, target_names=targets_col_header)

        # Expected rules
        expected_string_rules = """
        y0(0) :- x0(0).
        y0(0) :- x1(0).
        y0(1) :- x1(1).
        y0(1) :- x0(1), x2(1).
        y1(0) :- x0(0).
        y1(0) :- x1(0).
        y1(0) :- x2(0).
        y1(1) :- x0(1), x2(1).
        y1(1) :- x0(0), x1(1).
        y1(1) :- x1(1), x2(1).
        y2(0) :- x0(1).
        y2(0) :- x1(1), x2(0).
        y2(1) :- x0(0).
        y2(1) :- x1(0), x2(1).
        """

        expected_string_constraints= """
        :- y0(0), y1(1), y2(0).
        :- x1(1), x2(0), y1(0), y2(1).
        :- x2(0), y0(1), y2(1).
        :- x2(1), y0(0), y1(1).
        :- x2(1), y0(0), y2(0).
        :- x0(0), y0(1), y1(0).
        :- y0(1), y1(0), y2(1).
        :- x1(0), y0(0), y1(1).
        :- x1(0), y1(1), y2(0).
        :- x0(1), y0(0), y1(1).
        :- x1(1), x2(1), y0(1), y1(0).
        """

        self._check_rules_and_predictions(dataset, expected_string_rules, expected_string_constraints)

    #------------------
    # Tool functions
    #------------------

    def _check_rules_and_predictions(self, dataset, expected_string_rules, expected_string_constraints):
        expected_string_rules = [s.strip() for s in expected_string_rules.strip().split("\n") if len(s) > 0 ]
        expected_string_constraints = [s.strip() for s in expected_string_constraints.strip().split("\n") if len(s) > 0 ]

        expected_rules = []
        for string_rule in expected_string_rules:
            expected_rules.append(Rule.from_string(string_rule, dataset.features, dataset.targets))

        expected_constraints = []
        for string_constraint in expected_string_constraints:
            expected_constraints.append(Rule.from_string(string_constraint, dataset.features, dataset.targets))

        #eprint(expected_rules)

        rules, constraints = Synchronizer.fit(dataset)

        #eprint(output)

        for r in expected_rules:
            if r not in rules:
                eprint("Missing rule: ", r)
            self.assertTrue(r in rules)

        for r in rules:
            if r not in expected_rules:
                eprint("Additional rule: ", r)
            self.assertTrue(r in expected_rules)

        for r in expected_constraints:
            if r not in constraints:
                eprint("Missing constraint: ", r)
            self.assertTrue(r in constraints)

        for r in constraints:
            if r not in expected_constraints:
                eprint("Additional constraint: ", r)
            self.assertTrue(r in constraints)

        model = CDMVLP(dataset.features, dataset.targets, rules, constraints)

        #model.compile("synchronizer")
        #model.summary()

        expected = set((tuple(s1),tuple(s2)) for s1,s2 in dataset.data)

        predicted = model.predict(model.feature_states())
        predicted = set((tuple(s1),tuple(s2)) for (s1, S2) in predicted.items() for s2, rules in S2.items())

        done = 0
        for s1,s2 in expected:
            done += 1
            eprint("\r.>>> Checking transitions ",done,"/",len(expected),end='')
            self.assertTrue((s1,s2) in predicted)
        eprint()

        done = 0
        for s1,s2 in predicted:
            done += 1
            eprint("\r.>>> Checking transitions ",done,"/",len(predicted),end='')
            self.assertTrue((s1,s2) in expected)
        eprint()


if __name__ == '__main__':
    """ Main """

    unittest.main()
