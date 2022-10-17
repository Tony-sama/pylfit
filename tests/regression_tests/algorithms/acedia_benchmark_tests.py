#-----------------------
# @author: Tony Ribeiro
# @created: 2022/08/31
# @updated: 2022/08/31
#
# @desc: ACEDIA regression test script
# Tests algorithm methods on random dataset
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

from pylfit.utils import eprint
from pylfit.algorithms import ACEDIA
from pylfit.objects import ContinuumRule

from pylfit.preprocessing import continuous_state_transitions_dataset_from_csv

from pylfit.models import CLP

random.seed(0)

class ACEDIA_benchmark_tests(unittest.TestCase):
    """
        Regression tests of class ACEDIA from acedia.py with benchmarks data
    """

    _nb_tests = 10

    #------------------
    # Test functions
    #------------------

    def test_repressilator(self):
        print(">> ACEDIA benchmark <repressilator>:")

        dataset_filepath = "datasets/repressilator.csv"
        features_col_header = ["p_t_1","q_t_1","r_t_1"]
        targets_col_header = ["p_t","q_t","r_t"]

        dataset = continuous_state_transitions_dataset_from_csv(path=dataset_filepath, feature_names=features_col_header, target_names=targets_col_header)

        # Expected rules
        expected_string_rules = """
        p_t([0,1]) :- .
        p_t([1,1]) :- q_t_1(]0,1]).
        p_t([0,0]) :- q_t_1([0,1[).

        q_t([0,1]) :- .
        q_t([1,1]) :- p_t_1(]0,1]), r_t_1(]0,1]).
        q_t([0,0]) :- p_t_1([0,1[).
        q_t([0,0]) :- r_t_1([0,1[).

        r_t([0,1]) :- .
        r_t([1,1]) :- p_t_1([0,1[).
        r_t([0,0]) :- p_t_1(]0,1])."""

        self._check_rules_and_predictions(dataset, expected_string_rules)


    def _test_mammalian(self):
        print(">> ACEDIA benchmark <mammalian>:")

        dataset_filepath = "datasets/mammalian.csv"
        features_col_header = ["CycD_t_1","CycE_t_1","Rb_t_1","E2F_t_1","CycA_t_1","p27_t_1","Cdc20_t_1","UbcH10_t_1","Cdh1_t_1","CycB_t_1"]
        targets_col_header = ["CycD_t","CycE_t","Rb_t","E2F_t","CycA_t","p27_t","Cdc20_t","UbcH10_t","Cdh1_t","CycB_t"]

        dataset = continuous_state_transitions_dataset_from_csv(path=dataset_filepath, feature_names=features_col_header, target_names=targets_col_header)

        # Expected rules
        expected_string_rules = """
        CycD_t([0,1]) :- .
        CycD_t([1,1]) :- CycD_t_1(]0,1]).

        CycE_t([0,1]) :- .
        CycE_t([1,1]) :- Rb_t_1([0,1[), E2F_t_1(]0,1]).

        Rb_t([0,1]) :- .
        Rb_t([1,1]) :- CycD_t_1([0,1[), CycE_t_1([0,1[), CycA_t_1([0,1[), CycB_t_1([0,1[).
        Rb_t([1,1]) :- CycD_t_1([0,1[), p27_t_1(]0,1]), CycB_t_1([0,1[).

        E2F_t([0,1]) :- .
        E2F_t([1,1]) :- Rb_t_1([0,1[), CycA_t_1([0,1[), CycB_t_1([0,1[).
        E2F_t([1,1]) :- Rb_t_1([0,1[), p27_t_1(]0,1]), CycB_t_1([0,1[).

        CycA_t([0,1]) :- .
        CycA_t([1,1]) :- Rb_t_1([0,1[), E2F_t_1(]0,1]), Cdc20_t_1([0,1[), UbcH10_t_1([0,1[).
        CycA_t([1,1]) :- Rb_t_1([0,1[), E2F_t_1(]0,1]), Cdc20_t_1([0,1[), Cdh1_t_1([0,1[).
        CycA_t([1,1]) :- Rb_t_1([0,1[), CycA_t_1(]0,1]), Cdc20_t_1([0,1[), UbcH10_t_1([0,1[).
        CycA_t([1,1]) :- Rb_t_1([0,1[), CycA_t_1(]0,1]), Cdc20_t_1([0,1[), Cdh1_t_1([0,1[).

        p27_t([0,1]) :- .
        p27_t([1,1]) :- CycD_t_1([0,1[), CycE_t_1([0,1[), CycA_t_1([0,1[), CycB_t_1([0,1[).
        p27_t([1,1]) :- CycD_t_1([0,1[), CycE_t_1([0,1[), p27_t_1(]0,1]), CycB_t_1([0,1[).
        p27_t([1,1]) :- CycD_t_1([0,1[), CycA_t_1([0,1[), p27_t_1(]0,1]), CycB_t_1([0,1[).

        Cdc20_t([0,1]) :- .
        Cdc20_t([1,1]) :- CycB_t_1(]0,1]).

        UbcH10_t([0,1]) :- .
        UbcH10_t([1,1]) :- Cdh1_t_1([0,1[).
        UbcH10_t([1,1]) :- CycA_t_1(]0,1]), UbcH10_t_1(]0,1]).
        UbcH10_t([1,1]) :- Cdc20_t_1(]0,1]), UbcH10_t_1(]0,1]).
        UbcH10_t([1,1]) :- UbcH10_t_1(]0,1]), CycB_t_1(]0,1]).

        Cdh1_t([0,1]) :- .
        Cdh1_t([1,1]) :- CycA_t_1([0,1[), CycB_t_1([0,1[).
        Cdh1_t([1,1]) :- Cdc20_t_1(]0,1]).
        Cdh1_t([1,1]) :- p27_t_1(]0,1]), CycB_t_1([0,1[).

        CycB_t([0,1]) :- .
        CycB_t([1,1]) :- Cdc20_t_1([0,1[), Cdh1_t_1([0,1[).

        CycD_t([0,0]) :- CycD_t_1([0,1[).
        CycE_t([0,0]) :- Rb_t_1(]0,1]).
        CycE_t([0,0]) :- E2F_t_1([0,1[).
        Rb_t([0,0]) :- CycD_t_1(]0,1]).
        Rb_t([0,0]) :- CycE_t_1(]0,1]), p27_t_1([0,1[).
        Rb_t([0,0]) :- CycA_t_1(]0,1]), p27_t_1([0,1[).
        Rb_t([0,0]) :- CycB_t_1(]0,1]).
        E2F_t([0,0]) :- Rb_t_1(]0,1]).
        E2F_t([0,0]) :- CycA_t_1(]0,1]), p27_t_1([0,1[).
        E2F_t([0,0]) :- CycB_t_1(]0,1]).
        CycA_t([0,0]) :- Rb_t_1(]0,1]).
        CycA_t([0,0]) :- E2F_t_1([0,1[), CycA_t_1([0,1[).
        CycA_t([0,0]) :- Cdc20_t_1(]0,1]).
        CycA_t([0,0]) :- UbcH10_t_1(]0,1]), Cdh1_t_1(]0,1]).
        p27_t([0,0]) :- CycD_t_1(]0,1]).
        p27_t([0,0]) :- CycE_t_1(]0,1]), CycA_t_1(]0,1]).
        p27_t([0,0]) :- CycE_t_1(]0,1]), p27_t_1([0,1[).
        p27_t([0,0]) :- CycA_t_1(]0,1]), p27_t_1([0,1[).
        p27_t([0,0]) :- CycB_t_1(]0,1]).
        Cdc20_t([0,0]) :- CycB_t_1([0,1[).
        UbcH10_t([0,0]) :- CycA_t_1([0,1[), Cdc20_t_1([0,1[), Cdh1_t_1(]0,1]), CycB_t_1([0,1[).
        UbcH10_t([0,0]) :- UbcH10_t_1([0,1[), Cdh1_t_1(]0,1]).
        Cdh1_t([0,0]) :- CycA_t_1(]0,1]), p27_t_1([0,1[), Cdc20_t_1([0,1[).
        Cdh1_t([0,0]) :- Cdc20_t_1([0,1[), CycB_t_1(]0,1]).
        CycB_t([0,0]) :- Cdc20_t_1(]0,1]).
        CycB_t([0,0]) :- Cdh1_t_1(]0,1])."""

        self._check_rules_and_predictions(dataset, expected_string_rules)

    def _test_fission_yeast(self):
        print(">> ACEDIA benchmark <fission_yeast>:")

        dataset_filepath = "datasets/fission_yeast.csv"
        features_col_header = ["Start_t_1","SK_t_1","Ste9_t_1","Cdc2/Cdc13_t_1","Rum1_t_1","PP_t_1","Cdc25_t_1","Slp1_t_1","Wee1/Mik1_t_1","Cdc2/Cdc13*_t_1"]
        targets_col_header = ["Start_t","SK_t","Ste9_t","Cdc2/Cdc13_t","Rum1_t","PP_t","Cdc25_t","Slp1_t","Wee1/Mik1_t","Cdc2/Cdc13*_t"]

        dataset = continuous_state_transitions_dataset_from_csv(path=dataset_filepath, feature_names=features_col_header, target_names=targets_col_header)

        # Expected rules
        expected_string_rules = """
        SK_t([0,1]) :- .
        SK_t([1,1]) :- Start_t_1(]0,1]).

        Ste9_t([0,1]) :- .
        Ste9_t([1,1]) :- SK_t_1([0,1[), Ste9_t_1(]0,1]), Cdc2/Cdc13_t_1([0,1[), Cdc2/Cdc13*_t_1([0,1[).
        Ste9_t([1,1]) :- SK_t_1([0,1[), Cdc2/Cdc13_t_1([0,1[), PP_t_1(]0,1]), Cdc2/Cdc13*_t_1([0,1[).
        Ste9_t([1,1]) :- SK_t_1([0,1[), Ste9_t_1(]0,1]), Cdc2/Cdc13_t_1([0,1[), PP_t_1(]0,1]).
        Ste9_t([1,1]) :- SK_t_1([0,1[), Ste9_t_1(]0,1]), PP_t_1(]0,1]), Cdc2/Cdc13*_t_1([0,1[).
        Ste9_t([1,1]) :- Ste9_t_1(]0,1]), Cdc2/Cdc13_t_1([0,1[), PP_t_1(]0,1]), Cdc2/Cdc13*_t_1([0,1[).

        Cdc2/Cdc13_t([0,1]) :- .
        Cdc2/Cdc13_t([1,1]) :- Ste9_t_1([0,1[), Rum1_t_1([0,1[), Slp1_t_1([0,1[).
        Cdc2/Cdc13_t([1,1]) :- Ste9_t_1([0,1[), Cdc2/Cdc13_t_1(]0,1]), Rum1_t_1([0,1[).
        Cdc2/Cdc13_t([1,1]) :- Ste9_t_1([0,1[), Cdc2/Cdc13_t_1(]0,1]), Slp1_t_1([0,1[).
        Cdc2/Cdc13_t([1,1]) :- Cdc2/Cdc13_t_1(]0,1]), Rum1_t_1([0,1[), Slp1_t_1([0,1[).

        Rum1_t([0,1]) :- .
        Rum1_t([1,1]) :- SK_t_1([0,1[), Cdc2/Cdc13_t_1([0,1[), Rum1_t_1(]0,1]), Cdc2/Cdc13*_t_1([0,1[).
        Rum1_t([1,1]) :- SK_t_1([0,1[), Cdc2/Cdc13_t_1([0,1[), PP_t_1(]0,1]), Cdc2/Cdc13*_t_1([0,1[).
        Rum1_t([1,1]) :- SK_t_1([0,1[), Cdc2/Cdc13_t_1([0,1[), Rum1_t_1(]0,1]), PP_t_1(]0,1]).
        Rum1_t([1,1]) :- SK_t_1([0,1[), Rum1_t_1(]0,1]), PP_t_1(]0,1]), Cdc2/Cdc13*_t_1([0,1[).
        Rum1_t([1,1]) :- Cdc2/Cdc13_t_1([0,1[), Rum1_t_1(]0,1]), PP_t_1(]0,1]), Cdc2/Cdc13*_t_1(]0,1]).

        PP_t([0,1]) :- .
        PP_t([1,1]) :- Slp1_t_1(]0,1]).

        Cdc25_t([0,1]) :- .
        Cdc25_t([1,1]) :- PP_t_1([0,1[), Cdc25_t_1(]0,1]).
        Cdc25_t([1,1]) :- Cdc2/Cdc13_t_1(]0,1]), PP_t_1([0,1[).
        Cdc25_t([1,1]) :- Cdc2/Cdc13_t_1(]0,1]), Cdc25_t_1(]0,1]).

        Slp1_t([0,1]) :- .
        Slp1_t([1,1]) :- Cdc2/Cdc13*_t_1(]0,1]).

        Wee1/Mik1_t([0,1]) :- .
        Wee1/Mik1_t([1,1]) :- Cdc2/Cdc13_t_1([0,1[), Wee1/Mik1_t_1(]0,1]).
        Wee1/Mik1_t([1,1]) :- Cdc2/Cdc13_t_1([0,1[), PP_t_1(]0,1]).
        Wee1/Mik1_t([1,1]) :- PP_t_1(]0,1]), Wee1/Mik1_t_1(]0,1]).

        Cdc2/Cdc13*_t([0,1]) :- .
        Cdc2/Cdc13*_t([1,1]) :- Ste9_t_1([0,1[), Rum1_t_1([0,1[), Cdc25_t_1(]0,1]), Slp1_t_1([0,1[), Wee1/Mik1_t_1([0,1[), Cdc2/Cdc13*_t_1(]0,1]).

        Start_t([0,0]) :- .
        SK_t([0,0]) :- Start_t_1([0,1[).
        Ste9_t([0,0]) :- SK_t_1(]0,1]), Ste9_t_1([0,1[).
        Ste9_t([0,0]) :- SK_t_1(]0,1]), Cdc2/Cdc13_t_1(]0,1]).
        Ste9_t([0,0]) :- SK_t_1(]0,1]), PP_t_1([0,1[).
        Ste9_t([0,0]) :- SK_t_1(]0,1]), Cdc2/Cdc13*_t_1(]0,1]).
        Ste9_t([0,0]) :- Ste9_t_1([0,1[), Cdc2/Cdc13_t_1(]0,1]).
        Ste9_t([0,0]) :- Ste9_t_1([0,1[), PP_t_1([0,1[).
        Ste9_t([0,0]) :- Ste9_t_1([0,1[), Cdc2/Cdc13*_t_1(]0,1]).
        Ste9_t([0,0]) :- Cdc2/Cdc13_t_1(]0,1]), PP_t_1([0,1[).
        Ste9_t([0,0]) :- Cdc2/Cdc13_t_1(]0,1]), Cdc2/Cdc13*_t_1(]0,1]).
        Ste9_t([0,0]) :- PP_t_1([0,1[), Cdc2/Cdc13*_t_1(]0,1]).
        Cdc2/Cdc13_t([0,0]) :- Ste9_t_1(]0,1]), Cdc2/Cdc13_t_1([0,1[).
        Cdc2/Cdc13_t([0,0]) :- Ste9_t_1(]0,1]), Rum1_t_1(]0,1]).
        Cdc2/Cdc13_t([0,0]) :- Ste9_t_1(]0,1]), Slp1_t_1(]0,1]).
        Cdc2/Cdc13_t([0,0]) :- Cdc2/Cdc13_t_1([0,1[), Rum1_t_1(]0,1]).
        Cdc2/Cdc13_t([0,0]) :- Rum1_t_1(]0,1]), Slp1_t_1(]0,1]).
        Cdc2/Cdc13_t([0,0]) :- Cdc2/Cdc13_t_1([0,1[), Slp1_t_1(]0,1]).
        Rum1_t([0,0]) :- SK_t_1(]0,1]), Cdc2/Cdc13_t_1(]0,1]).
        Rum1_t([0,0]) :- SK_t_1(]0,1]), Rum1_t_1([0,1[).
        Rum1_t([0,0]) :- SK_t_1(]0,1]), PP_t_1([0,1[).
        Rum1_t([0,0]) :- SK_t_1(]0,1]), Cdc2/Cdc13*_t_1([0,1[).
        Rum1_t([0,0]) :- Cdc2/Cdc13_t_1(]0,1]), Rum1_t_1([0,1[).
        Rum1_t([0,0]) :- Cdc2/Cdc13_t_1(]0,1]), PP_t_1([0,1[).
        Rum1_t([0,0]) :- Cdc2/Cdc13_t_1(]0,1]), Cdc2/Cdc13*_t_1(]0,1]).
        Rum1_t([0,0]) :- Rum1_t_1([0,1[), PP_t_1([0,1[).
        Rum1_t([0,0]) :- Rum1_t_1([0,1[), Cdc2/Cdc13*_t_1(]0,1]).
        Rum1_t([0,0]) :- PP_t_1([0,1[), Cdc2/Cdc13*_t_1(]0,1]).
        PP_t([0,0]) :- Slp1_t_1([0,1[).
        Cdc25_t([0,0]) :- Cdc2/Cdc13_t_1([0,1[), PP_t_1(]0,1]).
        Cdc25_t([0,0]) :- PP_t_1(]0,1]), Cdc25_t_1([0,1[).
        Cdc25_t([0,0]) :- Cdc2/Cdc13_t_1([0,1[), Cdc25_t_1([0,1[).
        Slp1_t([0,0]) :- Cdc2/Cdc13*_t_1([0,1[).
        Wee1/Mik1_t([0,0]) :- Cdc2/Cdc13_t_1(]0,1]), PP_t_1([0,1[).
        Wee1/Mik1_t([0,0]) :- Cdc2/Cdc13_t_1(]0,1]), Wee1/Mik1_t_1([0,1[).
        Wee1/Mik1_t([0,0]) :- PP_t_1([0,1[), Wee1/Mik1_t_1([0,1[).
        Cdc2/Cdc13*_t([0,0]) :- Ste9_t_1(]0,1]).
        Cdc2/Cdc13*_t([0,0]) :- Rum1_t_1(]0,1]).
        Cdc2/Cdc13*_t([0,0]) :- Cdc25_t_1([0,1[).
        Cdc2/Cdc13*_t([0,0]) :- Slp1_t_1(]0,1]).
        Cdc2/Cdc13*_t([0,0]) :- Wee1/Mik1_t_1(]0,1]).
        Cdc2/Cdc13*_t([0,0]) :- Cdc2/Cdc13*_t_1([0,1[).
        """

        self._check_rules_and_predictions(dataset, expected_string_rules)

    #------------------
    # Tool functions
    #------------------

    def _check_rules_and_predictions(self, dataset, expected_string_rules):
        expected_string_rules = [s.strip() for s in expected_string_rules.strip().split("\n") if len(s) > 0 ]

        expected_rules = []
        for string_rule in expected_string_rules:
            expected_rules.append(ContinuumRule.from_string(string_rule, dataset.features, dataset.targets))

        #eprint(expected_rules)

        output = ACEDIA.fit(dataset=dataset)

        #eprint(output)

        for r in expected_rules:
            if r not in output:
                eprint("Missing rule: ", r)
            self.assertTrue(r in output)

        for r in output:
            if r not in expected_rules:
                eprint("Additional rule: ", r)
            self.assertTrue(r in expected_rules)

        model = CLP(dataset.features, dataset.targets, output)

        expected = set((tuple(s1),tuple(s2)) for s1,s2 in dataset.data)
        predicted = set()

        for s1 in model.feature_states(1.0):
            prediction = model.predict([s1])
            for s2 in prediction[tuple(s1)]:
                predicted.add( (tuple(s1), tuple(s2)) )

        done = 0
        for s1,s2 in expected:
            done += 1
            eprint("\r.>>> Checking transitions ",done,"/",len(expected),end='')
            #self.assertTrue(s1 in [s for s,s_ in predicted])
            for s1_,s2_ in predicted:
                if s1 == s1_:
                    for var_id, var in enumerate(s2_):
                        self.assertTrue(s2_[var_id].includes(s2[var_id]))

        eprint()

        done = 0
        for s1,s2 in predicted:
            done += 1
            eprint("\r.>>> Checking transitions ",done,"/",len(predicted),end='')
            for s1_,s2_ in expected:
                if s1 == s1_:
                    for var_id, var in enumerate(s2_):
                        self.assertTrue(s2[var_id].includes(s2_[var_id]))
        eprint()


if __name__ == '__main__':
    """ Main """

    unittest.main()
