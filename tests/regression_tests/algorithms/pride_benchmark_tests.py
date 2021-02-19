#-----------------------
# @author: Tony Ribeiro
# @created: 2021/02/02
# @updated: 2021/02/02
#
# @desc: PRIDE regression test script
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

from tests_generator import random_StateTransitionsDataset

from pylfit.utils import eprint
from pylfit.algorithms import PRIDE
from pylfit.objects import Rule

from pylfit.datasets import StateTransitionsDataset
from pylfit.preprocessing import transitions_dataset_from_csv

from pylfit.models import DMVLP

random.seed(0)

class PRIDE_benchmark_tests(unittest.TestCase):
    """
        Regression tests of class PRIDE from pride.py with benchmarks data
    """

    _nb_tests = 10

    #------------------
    # Test functions
    #------------------

    def test_repressilator(self):
        print(">> PRIDE benchmark <repressilator>:")

        dataset_filepath = "datasets/repressilator.csv"
        features_col_header = ["p_t_1","q_t_1","r_t_1"]
        targets_col_header = ["p_t","q_t","r_t"]

        dataset = transitions_dataset_from_csv(path=dataset_filepath, feature_names=features_col_header, target_names=targets_col_header)

        # Expected rules
        expected_string_rules = """
        p_t(1) :- q_t_1(1).
        q_t(1) :- p_t_1(1), r_t_1(1).
        r_t(1) :- p_t_1(0).

        p_t(0) :- q_t_1(0).
        q_t(0) :- p_t_1(0).
        q_t(0) :- r_t_1(0).
        r_t(0) :- p_t_1(1)."""

        self._check_rules_and_predictions(dataset, expected_string_rules)


    def test_mammalian(self):
        print(">> PRIDE benchmark <mammalian>:")

        dataset_filepath = "datasets/mammalian.csv"
        features_col_header = ["CycD_t_1","CycE_t_1","Rb_t_1","E2F_t_1","CycA_t_1","p27_t_1","Cdc20_t_1","UbcH10_t_1","Cdh1_t_1","CycB_t_1"]
        targets_col_header = ["CycD_t","CycE_t","Rb_t","E2F_t","CycA_t","p27_t","Cdc20_t","UbcH10_t","Cdh1_t","CycB_t"]

        dataset = transitions_dataset_from_csv(path=dataset_filepath, feature_names=features_col_header, target_names=targets_col_header)

        # Expected rules
        expected_string_rules = """
        CycD_t(1) :- CycD_t_1(1).
        CycE_t(1) :- Rb_t_1(0), E2F_t_1(1).
        Rb_t(1) :- CycD_t_1(0), CycE_t_1(0), CycA_t_1(0), CycB_t_1(0).
        Rb_t(1) :- CycD_t_1(0), p27_t_1(1), CycB_t_1(0).
        E2F_t(1) :- Rb_t_1(0), CycA_t_1(0), CycB_t_1(0).
        E2F_t(1) :- Rb_t_1(0), p27_t_1(1), CycB_t_1(0).
        CycA_t(1) :- Rb_t_1(0), E2F_t_1(1), Cdc20_t_1(0), UbcH10_t_1(0).
        CycA_t(1) :- Rb_t_1(0), E2F_t_1(1), Cdc20_t_1(0), Cdh1_t_1(0).
        CycA_t(1) :- Rb_t_1(0), CycA_t_1(1), Cdc20_t_1(0), UbcH10_t_1(0).
        CycA_t(1) :- Rb_t_1(0), CycA_t_1(1), Cdc20_t_1(0), Cdh1_t_1(0).
        p27_t(1) :- CycD_t_1(0), CycE_t_1(0), CycA_t_1(0), CycB_t_1(0).
        p27_t(1) :- CycD_t_1(0), CycE_t_1(0), p27_t_1(1), CycB_t_1(0).
        p27_t(1) :- CycD_t_1(0), CycA_t_1(0), p27_t_1(1), CycB_t_1(0).
        Cdc20_t(1) :- CycB_t_1(1).
        UbcH10_t(1) :- Cdh1_t_1(0).
        UbcH10_t(1) :- CycA_t_1(1), UbcH10_t_1(1).
        UbcH10_t(1) :- Cdc20_t_1(1), UbcH10_t_1(1).
        UbcH10_t(1) :- UbcH10_t_1(1), CycB_t_1(1).
        Cdh1_t(1) :- CycA_t_1(0), CycB_t_1(0).
        Cdh1_t(1) :- Cdc20_t_1(1).
        Cdh1_t(1) :- p27_t_1(1), CycB_t_1(0).
        CycB_t(1) :- Cdc20_t_1(0), Cdh1_t_1(0).

        CycD_t(0) :- CycD_t_1(0).
        CycE_t(0) :- Rb_t_1(1).
        CycE_t(0) :- E2F_t_1(0).
        Rb_t(0) :- CycD_t_1(1).
        Rb_t(0) :- CycE_t_1(1), p27_t_1(0).
        Rb_t(0) :- CycA_t_1(1), p27_t_1(0).
        Rb_t(0) :- CycB_t_1(1).
        E2F_t(0) :- Rb_t_1(1).
        E2F_t(0) :- CycA_t_1(1), p27_t_1(0).
        E2F_t(0) :- CycB_t_1(1).
        CycA_t(0) :- Rb_t_1(1).
        CycA_t(0) :- E2F_t_1(0), CycA_t_1(0).
        CycA_t(0) :- Cdc20_t_1(1).
        CycA_t(0) :- UbcH10_t_1(1), Cdh1_t_1(1).
        p27_t(0) :- CycD_t_1(1).
        p27_t(0) :- CycE_t_1(1), CycA_t_1(1).
        p27_t(0) :- CycE_t_1(1), p27_t_1(0).
        p27_t(0) :- CycA_t_1(1), p27_t_1(0).
        p27_t(0) :- CycB_t_1(1).
        Cdc20_t(0) :- CycB_t_1(0).
        UbcH10_t(0) :- CycA_t_1(0), Cdc20_t_1(0), Cdh1_t_1(1), CycB_t_1(0).
        UbcH10_t(0) :- UbcH10_t_1(0), Cdh1_t_1(1).
        Cdh1_t(0) :- CycA_t_1(1), p27_t_1(0), Cdc20_t_1(0).
        Cdh1_t(0) :- Cdc20_t_1(0), CycB_t_1(1).
        CycB_t(0) :- Cdc20_t_1(1).
        CycB_t(0) :- Cdh1_t_1(1)."""

        self._check_rules_and_predictions(dataset, expected_string_rules)

    def test_fission_yeast(self):
        print(">> PRIDE benchmark <fission_yeast>:")

        dataset_filepath = "datasets/fission_yeast.csv"
        features_col_header = ["Start_t_1","SK_t_1","Ste9_t_1","Cdc2/Cdc13_t_1","Rum1_t_1","PP_t_1","Cdc25_t_1","Slp1_t_1","Wee1/Mik1_t_1","Cdc2/Cdc13*_t_1"]
        targets_col_header = ["Start_t","SK_t","Ste9_t","Cdc2/Cdc13_t","Rum1_t","PP_t","Cdc25_t","Slp1_t","Wee1/Mik1_t","Cdc2/Cdc13*_t"]

        dataset = transitions_dataset_from_csv(path=dataset_filepath, feature_names=features_col_header, target_names=targets_col_header)

        # Expected rules
        expected_string_rules = """
        SK_t(1) :- Start_t_1(1).
        Ste9_t(1) :- SK_t_1(0), Ste9_t_1(1), Cdc2/Cdc13_t_1(0), Cdc2/Cdc13*_t_1(0).
        Ste9_t(1) :- SK_t_1(0), Cdc2/Cdc13_t_1(0), PP_t_1(1), Cdc2/Cdc13*_t_1(0).
        Ste9_t(1) :- SK_t_1(0), Ste9_t_1(1), Cdc2/Cdc13_t_1(0), PP_t_1(1).
        Ste9_t(1) :- SK_t_1(0), Ste9_t_1(1), PP_t_1(1), Cdc2/Cdc13*_t_1(0).
        Ste9_t(1) :- Ste9_t_1(1), Cdc2/Cdc13_t_1(0), PP_t_1(1), Cdc2/Cdc13*_t_1(0).
        Cdc2/Cdc13_t(1) :- Ste9_t_1(0), Rum1_t_1(0), Slp1_t_1(0).
        Cdc2/Cdc13_t(1) :- Ste9_t_1(0), Cdc2/Cdc13_t_1(1), Rum1_t_1(0).
        Cdc2/Cdc13_t(1) :- Ste9_t_1(0), Cdc2/Cdc13_t_1(1), Slp1_t_1(0).
        Cdc2/Cdc13_t(1) :- Cdc2/Cdc13_t_1(1), Rum1_t_1(0), Slp1_t_1(0).
        Rum1_t(1) :- SK_t_1(0), Cdc2/Cdc13_t_1(0), Rum1_t_1(1), Cdc2/Cdc13*_t_1(0).
        Rum1_t(1) :- SK_t_1(0), Cdc2/Cdc13_t_1(0), PP_t_1(1), Cdc2/Cdc13*_t_1(0).
        Rum1_t(1) :- SK_t_1(0), Cdc2/Cdc13_t_1(0), Rum1_t_1(1), PP_t_1(1).
        Rum1_t(1) :- SK_t_1(0), Rum1_t_1(1), PP_t_1(1), Cdc2/Cdc13*_t_1(0).
        Rum1_t(1) :- Cdc2/Cdc13_t_1(0), Rum1_t_1(1), PP_t_1(1), Cdc2/Cdc13*_t_1(1).
        PP_t(1) :- Slp1_t_1(1).
        Cdc25_t(1) :- PP_t_1(0), Cdc25_t_1(1).
        Cdc25_t(1) :- Cdc2/Cdc13_t_1(1), PP_t_1(0).
        Cdc25_t(1) :- Cdc2/Cdc13_t_1(1), Cdc25_t_1(1).
        Slp1_t(1) :- Cdc2/Cdc13*_t_1(1).
        Wee1/Mik1_t(1) :- Cdc2/Cdc13_t_1(0), Wee1/Mik1_t_1(1).
        Wee1/Mik1_t(1) :- Cdc2/Cdc13_t_1(0), PP_t_1(1).
        Wee1/Mik1_t(1) :- PP_t_1(1), Wee1/Mik1_t_1(1).
        Cdc2/Cdc13*_t(1) :- Ste9_t_1(0), Rum1_t_1(0), Cdc25_t_1(1), Slp1_t_1(0), Wee1/Mik1_t_1(0), Cdc2/Cdc13*_t_1(1).

        Start_t(0) :- .
        SK_t(0) :- Start_t_1(0).
        Ste9_t(0) :- SK_t_1(1), Ste9_t_1(0).
        Ste9_t(0) :- SK_t_1(1), Cdc2/Cdc13_t_1(1).
        Ste9_t(0) :- SK_t_1(1), PP_t_1(0).
        Ste9_t(0) :- SK_t_1(1), Cdc2/Cdc13*_t_1(1).
        Ste9_t(0) :- Ste9_t_1(0), Cdc2/Cdc13_t_1(1).
        Ste9_t(0) :- Ste9_t_1(0), PP_t_1(0).
        Ste9_t(0) :- Ste9_t_1(0), Cdc2/Cdc13*_t_1(1).
        Ste9_t(0) :- Cdc2/Cdc13_t_1(1), PP_t_1(0).
        Ste9_t(0) :- Cdc2/Cdc13_t_1(1), Cdc2/Cdc13*_t_1(1).
        Ste9_t(0) :- PP_t_1(0), Cdc2/Cdc13*_t_1(1).
        Cdc2/Cdc13_t(0) :- Ste9_t_1(1), Cdc2/Cdc13_t_1(0).
        Cdc2/Cdc13_t(0) :- Ste9_t_1(1), Rum1_t_1(1).
        Cdc2/Cdc13_t(0) :- Ste9_t_1(1), Slp1_t_1(1).
        Cdc2/Cdc13_t(0) :- Cdc2/Cdc13_t_1(0), Rum1_t_1(1).
        Cdc2/Cdc13_t(0) :- Rum1_t_1(1), Slp1_t_1(1).
        Cdc2/Cdc13_t(0) :- Cdc2/Cdc13_t_1(0), Slp1_t_1(1).
        Rum1_t(0) :- SK_t_1(1), Cdc2/Cdc13_t_1(1).
        Rum1_t(0) :- SK_t_1(1), Rum1_t_1(0).
        Rum1_t(0) :- SK_t_1(1), PP_t_1(0).
        Rum1_t(0) :- SK_t_1(1), Cdc2/Cdc13*_t_1(0).
        Rum1_t(0) :- Cdc2/Cdc13_t_1(1), Rum1_t_1(0).
        Rum1_t(0) :- Cdc2/Cdc13_t_1(1), PP_t_1(0).
        Rum1_t(0) :- Cdc2/Cdc13_t_1(1), Cdc2/Cdc13*_t_1(1).
        Rum1_t(0) :- Rum1_t_1(0), PP_t_1(0).
        Rum1_t(0) :- Rum1_t_1(0), Cdc2/Cdc13*_t_1(1).
        Rum1_t(0) :- PP_t_1(0), Cdc2/Cdc13*_t_1(1).
        PP_t(0) :- Slp1_t_1(0).
        Cdc25_t(0) :- Cdc2/Cdc13_t_1(0), PP_t_1(1).
        Cdc25_t(0) :- PP_t_1(1), Cdc25_t_1(0).
        Cdc25_t(0) :- Cdc2/Cdc13_t_1(0), Cdc25_t_1(0).
        Slp1_t(0) :- Cdc2/Cdc13*_t_1(0).
        Wee1/Mik1_t(0) :- Cdc2/Cdc13_t_1(1), PP_t_1(0).
        Wee1/Mik1_t(0) :- Cdc2/Cdc13_t_1(1), Wee1/Mik1_t_1(0).
        Wee1/Mik1_t(0) :- PP_t_1(0), Wee1/Mik1_t_1(0).
        Cdc2/Cdc13*_t(0) :- Ste9_t_1(1).
        Cdc2/Cdc13*_t(0) :- Rum1_t_1(1).
        Cdc2/Cdc13*_t(0) :- Cdc25_t_1(0).
        Cdc2/Cdc13*_t(0) :- Slp1_t_1(1).
        Cdc2/Cdc13*_t(0) :- Wee1/Mik1_t_1(1).
        Cdc2/Cdc13*_t(0) :- Cdc2/Cdc13*_t_1(0).
        """

        self._check_rules_and_predictions(dataset, expected_string_rules)

    def test_budding_yeast(self):
        print(">> PRIDE benchmark <budding_yeast>:")

        dataset_filepath = "datasets/budding_yeast.csv"
        features_col_header = ["size_t_1","Cln3_t_1","SBF_t_1","MBF_t_1","Cln1_2_t_1","Sic1_t_1","Cln5_6_t_1","Cdh1_t_1","Clb1_2_t_1","Mcm1/SFF_t_1","Cdc20&Cdc14_t_1","Swi5_t_1"]
        targets_col_header = ["size_t","Cln3_t","SBF_t","MBF_t","Cln1_2_t","Sic1_t","Cln5_6_t","Cdh1_t","Clb1_2_t","Mcm1/SFF_t","Cdc20&Cdc14_t","Swi5_t"]

        dataset = transitions_dataset_from_csv(path=dataset_filepath, feature_names=features_col_header, target_names=targets_col_header)


        #Sic1_t(1) :- Cln1_2_t_1(0), Clb1_2_t_1(0), Cdc20&Cdc14_t_1(1), Swi5_t_1(1).
        #Sic1_t(1) :- Cln1_2_t_1(0), Sic1_t_1(1), Clb1_2_t_1(0), Cdc20&Cdc14_t_1(1).
        #Sic1_t(1) :- Cln1_2_t_1(0), Sic1_t_1(1), Cdc20&Cdc14_t_1(1), Swi5_t_1(1).
        #Sic1_t(1) :- Sic1_t_1(1), Clb1_2_t_1(0), Cdc20&Cdc14_t_1(1), Swi5_t_1(1).

        # Expected rules
        expected_string_rules = """
        Cln3_t(1) :- size_t_1(1).
        SBF_t(1) :- SBF_t_1(1), Clb1_2_t_1(0).
        SBF_t(1) :- Cln3_t_1(1), Clb1_2_t_1(0).
        SBF_t(1) :- Cln3_t_1(1), SBF_t_1(1).

        MBF_t(1) :- MBF_t_1(1), Clb1_2_t_1(0).
        MBF_t(1) :- Cln3_t_1(1), Clb1_2_t_1(0).
        MBF_t(1) :- Cln3_t_1(1), MBF_t_1(1).

        Cln1_2_t(1) :- SBF_t_1(1).

        Sic1_t(1) :- Cln1_2_t_1(0), Sic1_t_1(1), Cln5_6_t_1(0), Clb1_2_t_1(0).
        Sic1_t(1) :- Cln1_2_t_1(0), Cln5_6_t_1(0), Clb1_2_t_1(0), Swi5_t_1(1).
        Sic1_t(1) :- Cln1_2_t_1(0), Cln5_6_t_1(0), Clb1_2_t_1(0), Cdc20&Cdc14_t_1(1).
        Sic1_t(1) :- Cln1_2_t_1(0), Sic1_t_1(1), Cln5_6_t_1(0), Swi5_t_1(1).
        Sic1_t(1) :- Cln1_2_t_1(0), Cln5_6_t_1(0), Cdc20&Cdc14_t_1(1), Swi5_t_1(1).
        Sic1_t(1) :- Cln1_2_t_1(0), Sic1_t_1(1), Clb1_2_t_1(0), Swi5_t_1(1).
        Sic1_t(1) :- Cln1_2_t_1(0), Sic1_t_1(1), Clb1_2_t_1(0), Cdc20&Cdc14_t_1(1).
        Sic1_t(1) :- Cln1_2_t_1(0), Clb1_2_t_1(0), Cdc20&Cdc14_t_1(1), Swi5_t_1(1).
        Sic1_t(1) :- Sic1_t_1(1), Cln5_6_t_1(0), Clb1_2_t_1(0), Swi5_t_1(1).
        Sic1_t(1) :- Sic1_t_1(1), Cln5_6_t_1(0), Clb1_2_t_1(0), Cdc20&Cdc14_t_1(1).
        Sic1_t(1) :- Sic1_t_1(1), Cln5_6_t_1(0), Cdc20&Cdc14_t_1(1), Swi5_t_1(1).
        Sic1_t(1) :- Sic1_t_1(1), Clb1_2_t_1(0), Cdc20&Cdc14_t_1(1), Swi5_t_1(1).
        Sic1_t(1) :- Cln5_6_t_1(0), Clb1_2_t_1(0), Cdc20&Cdc14_t_1(1), Swi5_t_1(1).
        Sic1_t(1) :- Cln1_2_t_1(0), Sic1_t_1(1), Cln5_6_t_1(0), Cdc20&Cdc14_t_1(1).
        Sic1_t(1) :- Cln1_2_t_1(0), Sic1_t_1(1), Cdc20&Cdc14_t_1(1), Swi5_t_1(1).


        Cln5_6_t(1) :- Sic1_t_1(0), Cln5_6_t_1(1), Cdc20&Cdc14_t_1(0).
        Cln5_6_t(1) :- MBF_t_1(1), Sic1_t_1(0), Cdc20&Cdc14_t_1(0).
        Cln5_6_t(1) :- MBF_t_1(1), Sic1_t_1(0), Cln5_6_t_1(1).
        Cln5_6_t(1) :- MBF_t_1(1), Cln5_6_t_1(1), Cdc20&Cdc14_t_1(0).

        Cdh1_t(1) :- Cln1_2_t_1(0), Cln5_6_t_1(0), Cdh1_t_1(1), Clb1_2_t_1(0).
        Cdh1_t(1) :- Cln1_2_t_1(0), Cln5_6_t_1(0), Clb1_2_t_1(0), Cdc20&Cdc14_t_1(1).
        Cdh1_t(1) :- Cln1_2_t_1(0), Cln5_6_t_1(0), Cdh1_t_1(1), Cdc20&Cdc14_t_1(1).
        Cdh1_t(1) :- Cln1_2_t_1(0), Cdh1_t_1(1), Clb1_2_t_1(0), Cdc20&Cdc14_t_1(1).
        Cdh1_t(1) :- Cln5_6_t_1(0), Cdh1_t_1(1), Clb1_2_t_1(0), Cdc20&Cdc14_t_1(1).

        Clb1_2_t(1) :- Sic1_t_1(0), Cdh1_t_1(0), Clb1_2_t_1(1), Cdc20&Cdc14_t_1(0).
        Clb1_2_t(1) :- Sic1_t_1(0), Cdh1_t_1(0), Mcm1/SFF_t_1(1), Cdc20&Cdc14_t_1(0).
        Clb1_2_t(1) :- Sic1_t_1(0), Cdh1_t_1(0), Clb1_2_t_1(1), Mcm1/SFF_t_1(1).
        Clb1_2_t(1) :- Sic1_t_1(0), Clb1_2_t_1(1), Mcm1/SFF_t_1(1), Cdc20&Cdc14_t_1(0).
        Clb1_2_t(1) :- Sic1_t_1(0), Cln5_6_t_1(1), Cdh1_t_1(0), Cdc20&Cdc14_t_1(0).
        Clb1_2_t(1) :- Sic1_t_1(0), Cln5_6_t_1(1), Cdh1_t_1(0), Clb1_2_t_1(1).
        Clb1_2_t(1) :- Sic1_t_1(0), Cln5_6_t_1(1), Cdh1_t_1(0), Mcm1/SFF_t_1(1).
        Clb1_2_t(1) :- Sic1_t_1(0), Cln5_6_t_1(1), Clb1_2_t_1(1), Cdc20&Cdc14_t_1(0).
        Clb1_2_t(1) :- Sic1_t_1(0), Cln5_6_t_1(1), Mcm1/SFF_t_1(1), Cdc20&Cdc14_t_1(0).
        Clb1_2_t(1) :- Sic1_t_1(0), Cln5_6_t_1(1), Clb1_2_t_1(1), Mcm1/SFF_t_1(1).
        Clb1_2_t(1) :- Cdh1_t_1(0), Clb1_2_t_1(1), Mcm1/SFF_t_1(1), Cdc20&Cdc14_t_1(0).
        Clb1_2_t(1) :- Cln5_6_t_1(1), Cdh1_t_1(0), Clb1_2_t_1(1), Cdc20&Cdc14_t_1(0).
        Clb1_2_t(1) :- Cln5_6_t_1(1), Cdh1_t_1(0), Mcm1/SFF_t_1(1), Cdc20&Cdc14_t_1(0).
        Clb1_2_t(1) :- Cln5_6_t_1(1), Cdh1_t_1(0), Clb1_2_t_1(1), Mcm1/SFF_t_1(1).
        Clb1_2_t(1) :- Cln5_6_t_1(1), Clb1_2_t_1(1), Mcm1/SFF_t_1(1), Cdc20&Cdc14_t_1(0).

        Mcm1/SFF_t(1) :- Cln5_6_t_1(1).
        Mcm1/SFF_t(1) :- Clb1_2_t_1(1).

        Cdc20&Cdc14_t(1) :- Clb1_2_t_1(1).
        Cdc20&Cdc14_t(1) :- Mcm1/SFF_t_1(1).

        Swi5_t(1) :- Clb1_2_t_1(0), Cdc20&Cdc14_t_1(1).
        Swi5_t(1) :- Clb1_2_t_1(0), Mcm1/SFF_t_1(1).
        Swi5_t(1) :- Mcm1/SFF_t_1(1), Cdc20&Cdc14_t_1(1).

        size_t(0) :- .

        Cln3_t(0) :- size_t_1(0).

        SBF_t(0) :- Cln3_t_1(0), SBF_t_1(0).
        SBF_t(0) :- Cln3_t_1(0), Clb1_2_t_1(1).
        SBF_t(0) :- SBF_t_1(0), Clb1_2_t_1(1).

        MBF_t(0) :- Cln3_t_1(0), MBF_t_1(0).
        MBF_t(0) :- Cln3_t_1(0), Clb1_2_t_1(1).
        MBF_t(0) :- MBF_t_1(0), Clb1_2_t_1(1).

        Cln1_2_t(0) :- SBF_t_1(0).

        Sic1_t(0) :- Sic1_t_1(0), Cdc20&Cdc14_t_1(0), Swi5_t_1(0).
        Sic1_t(0) :- Sic1_t_1(0), Clb1_2_t_1(1), Cdc20&Cdc14_t_1(0).
        Sic1_t(0) :- Sic1_t_1(0), Clb1_2_t_1(1), Swi5_t_1(0).
        Sic1_t(0) :- Clb1_2_t_1(1), Cdc20&Cdc14_t_1(0), Swi5_t_1(0).
        Sic1_t(0) :- Sic1_t_1(0), Cln5_6_t_1(1), Cdc20&Cdc14_t_1(0).
        Sic1_t(0) :- Sic1_t_1(0), Cln5_6_t_1(1), Swi5_t_1(0).
        Sic1_t(0) :- Cln5_6_t_1(1), Cdc20&Cdc14_t_1(0), Swi5_t_1(0).
        Sic1_t(0) :- Sic1_t_1(0), Cln5_6_t_1(1), Clb1_2_t_1(1).
        Sic1_t(0) :- Cln5_6_t_1(1), Clb1_2_t_1(1), Cdc20&Cdc14_t_1(0).
        Sic1_t(0) :- Cln5_6_t_1(1), Clb1_2_t_1(1), Swi5_t_1(0).
        Sic1_t(0) :- Cln1_2_t_1(1), Sic1_t_1(0), Cdc20&Cdc14_t_1(0).
        Sic1_t(0) :- Cln1_2_t_1(1), Sic1_t_1(0), Swi5_t_1(0).
        Sic1_t(0) :- Cln1_2_t_1(1), Cdc20&Cdc14_t_1(0), Swi5_t_1(0).
        Sic1_t(0) :- Cln1_2_t_1(1), Sic1_t_1(0), Clb1_2_t_1(1).
        Sic1_t(0) :- Cln1_2_t_1(1), Clb1_2_t_1(1), Cdc20&Cdc14_t_1(0).
        Sic1_t(0) :- Cln1_2_t_1(1), Clb1_2_t_1(1), Swi5_t_1(0).
        Sic1_t(0) :- Cln1_2_t_1(1), Sic1_t_1(0), Cln5_6_t_1(1).
        Sic1_t(0) :- Cln1_2_t_1(1), Cln5_6_t_1(1), Clb1_2_t_1(1).
        Sic1_t(0) :- Cln1_2_t_1(1), Cln5_6_t_1(1), Cdc20&Cdc14_t_1(0).
        Sic1_t(0) :- Cln1_2_t_1(1), Cln5_6_t_1(1), Swi5_t_1(0).

        Cln5_6_t(0) :- MBF_t_1(0), Cln5_6_t_1(0).
        Cln5_6_t(0) :- MBF_t_1(0), Cdc20&Cdc14_t_1(1).
        Cln5_6_t(0) :- Cln5_6_t_1(0), Cdc20&Cdc14_t_1(1).
        Cln5_6_t(0) :- MBF_t_1(0), Sic1_t_1(1).
        Cln5_6_t(0) :- Sic1_t_1(1), Cln5_6_t_1(0).
        Cln5_6_t(0) :- Sic1_t_1(1), Cdc20&Cdc14_t_1(1).

        Cdh1_t(0) :- Cdh1_t_1(0), Cdc20&Cdc14_t_1(0).
        Cdh1_t(0) :- Cdh1_t_1(0), Clb1_2_t_1(1).
        Cdh1_t(0) :- Clb1_2_t_1(1), Cdc20&Cdc14_t_1(0).
        Cdh1_t(0) :- Cln5_6_t_1(1), Cdh1_t_1(0).
        Cdh1_t(0) :- Cln5_6_t_1(1), Clb1_2_t_1(1).
        Cdh1_t(0) :- Cln5_6_t_1(1), Cdc20&Cdc14_t_1(0).
        Cdh1_t(0) :- Cln1_2_t_1(1), Cln5_6_t_1(1).
        Cdh1_t(0) :- Cln1_2_t_1(1), Cdh1_t_1(0).
        Cdh1_t(0) :- Cln1_2_t_1(1), Clb1_2_t_1(1).
        Cdh1_t(0) :- Cln1_2_t_1(1), Cdc20&Cdc14_t_1(0).

        Clb1_2_t(0) :- Cln5_6_t_1(0), Clb1_2_t_1(0), Mcm1/SFF_t_1(0).
        Clb1_2_t(0) :- Cln5_6_t_1(0), Clb1_2_t_1(0), Cdc20&Cdc14_t_1(1).
        Clb1_2_t(0) :- Cln5_6_t_1(0), Mcm1/SFF_t_1(0), Cdc20&Cdc14_t_1(1).
        Clb1_2_t(0) :- Clb1_2_t_1(0), Mcm1/SFF_t_1(0), Cdc20&Cdc14_t_1(1).
        Clb1_2_t(0) :- Cln5_6_t_1(0), Cdh1_t_1(1), Clb1_2_t_1(0).
        Clb1_2_t(0) :- Cln5_6_t_1(0), Cdh1_t_1(1), Mcm1/SFF_t_1(0).
        Clb1_2_t(0) :- Cdh1_t_1(1), Clb1_2_t_1(0), Mcm1/SFF_t_1(0).
        Clb1_2_t(0) :- Cln5_6_t_1(0), Cdh1_t_1(1), Cdc20&Cdc14_t_1(1).
        Clb1_2_t(0) :- Cdh1_t_1(1), Clb1_2_t_1(0), Cdc20&Cdc14_t_1(1).
        Clb1_2_t(0) :- Cdh1_t_1(1), Mcm1/SFF_t_1(0), Cdc20&Cdc14_t_1(1).
        Clb1_2_t(0) :- Sic1_t_1(1), Cln5_6_t_1(0), Clb1_2_t_1(0).
        Clb1_2_t(0) :- Sic1_t_1(1), Cln5_6_t_1(0), Mcm1/SFF_t_1(0).
        Clb1_2_t(0) :- Sic1_t_1(1), Clb1_2_t_1(0), Mcm1/SFF_t_1(0).
        Clb1_2_t(0) :- Sic1_t_1(1), Cln5_6_t_1(0), Cdc20&Cdc14_t_1(1).
        Clb1_2_t(0) :- Sic1_t_1(1), Clb1_2_t_1(0), Cdc20&Cdc14_t_1(1).
        Clb1_2_t(0) :- Sic1_t_1(1), Mcm1/SFF_t_1(0), Cdc20&Cdc14_t_1(1).
        Clb1_2_t(0) :- Sic1_t_1(1), Cln5_6_t_1(0), Cdh1_t_1(1).
        Clb1_2_t(0) :- Sic1_t_1(1), Cdh1_t_1(1), Clb1_2_t_1(0).
        Clb1_2_t(0) :- Sic1_t_1(1), Cdh1_t_1(1), Mcm1/SFF_t_1(0).
        Clb1_2_t(0) :- Sic1_t_1(1), Cdh1_t_1(1), Cdc20&Cdc14_t_1(1).

        Mcm1/SFF_t(0) :- Cln5_6_t_1(0), Clb1_2_t_1(0).

        Cdc20&Cdc14_t(0) :- Clb1_2_t_1(0), Mcm1/SFF_t_1(0).

        Swi5_t(0) :- Mcm1/SFF_t_1(0), Cdc20&Cdc14_t_1(0).
        Swi5_t(0) :- Clb1_2_t_1(1), Mcm1/SFF_t_1(0).
        Swi5_t(0) :- Clb1_2_t_1(1), Cdc20&Cdc14_t_1(0).
        """

        self._check_rules_and_predictions(dataset, expected_string_rules)

    def _test_arabidopsis(self):
        print(">> PRIDE benchmark <arabidopsis>:")

        dataset_filepath = "datasets/arabidopsis.csv"
        features_col_header = ["AP3_t_1","UFO_t_1","FUL_t_1","FT_t_1","AP1_t_1","EMF1_t_1","LFY_t_1","AP2_t_1","WUS_t_1","AG_t_1","LUG_t_1","CLF_t_1","TFL1_t_1","PI_t_1","SEP_t_1"]
        targets_col_header = ["AP3_t","UFO_t","FUL_t","FT_t","AP1_t","EMF1_t","LFY_t","AP2_t","WUS_t","AG_t","LUG_t","CLF_t","TFL1_t","PI_t","SEP_t"]

        dataset = transitions_dataset_from_csv(path=dataset_filepath, feature_names=features_col_header, target_names=targets_col_header)

        # Expected rules
        expected_string_rules = """
        AP3_t(1) :- AP3_t_1(1), AP1_t_1(1), PI_t_1(1), SEP_t_1(1).
        AP3_t(1) :- AP3_t_1(1), AG_t_1(1), PI_t_1(1), SEP_t_1(1).
        AP3_t(1) :- UFO_t_1(1), LFY_t_1(1).
        UFO_t(1) :- UFO_t_1(1).
        FUL_t(1) :- AP1_t_1(0), TFL1_t_1(0).
        FT_t(1) :- EMF1_t_1(0).
        AP1_t(1) :- AG_t_1(0), TFL1_t_1(0).
        AP1_t(1) :- FT_t_1(1), AG_t_1(0).
        AP1_t(1) :- LFY_t_1(1), AG_t_1(0).
        EMF1_t(1) :- LFY_t_1(0).
        LFY_t(1) :- TFL1_t_1(0).
        LFY_t(1) :- EMF1_t_1(0).
        AP2_t(1) :- TFL1_t_1(0).
        WUS_t(1) :- WUS_t_1(1), SEP_t_1(0).
        WUS_t(1) :- WUS_t_1(1), AG_t_1(0).
        AG_t(1) :- AP2_t_1(0), TFL1_t_1(0).
        AG_t(1) :- LFY_t_1(1), AG_t_1(1), SEP_t_1(1).
        AG_t(1) :- LFY_t_1(1), CLF_t_1(0).
        AG_t(1) :- LFY_t_1(1), LUG_t_1(0).
        AG_t(1) :- AP1_t_1(0), LFY_t_1(1).
        AG_t(1) :- LFY_t_1(1), WUS_t_1(1).
        AG_t(1) :- LFY_t_1(1), AP2_t_1(0).
        LUG_t(1) :- .
        CLF_t(1) :- .
        TFL1_t(1) :- AP1_t_1(0), EMF1_t_1(1), LFY_t_1(0).
        PI_t(1) :- AP3_t_1(1), AP1_t_1(1), PI_t_1(1), SEP_t_1(1).
        PI_t(1) :- AP3_t_1(1), AG_t_1(1), PI_t_1(1), SEP_t_1(1).
        PI_t(1) :- LFY_t_1(1), AG_t_1(1).
        PI_t(1) :- AP3_t_1(1), LFY_t_1(1).
        SEP_t(1) :- LFY_t_1(1).

        AP3_t(0) :- AP3_t_1(0), UFO_t_1(0).
        AP3_t(0) :- AP3_t_1(0), LFY_t_1(0).
        AP3_t(0) :- UFO_t_1(0), AP1_t_1(0), AG_t_1(0).
        AP3_t(0) :- AP1_t_1(0), LFY_t_1(0), AG_t_1(0).
        AP3_t(0) :- UFO_t_1(0), PI_t_1(0).
        AP3_t(0) :- LFY_t_1(0), PI_t_1(0).
        AP3_t(0) :- UFO_t_1(0), SEP_t_1(0).
        AP3_t(0) :- LFY_t_1(0), SEP_t_1(0).
        UFO_t(0) :- UFO_t_1(0).
        FUL_t(0) :- AP1_t_1(1).
        FUL_t(0) :- TFL1_t_1(1).
        FT_t(0) :- EMF1_t_1(1).
        AP1_t(0) :- AG_t_1(1).
        AP1_t(0) :- FT_t_1(0), LFY_t_1(0), TFL1_t_1(1).
        EMF1_t(0) :- LFY_t_1(1).
        LFY_t(0) :- EMF1_t_1(1), TFL1_t_1(1).
        AP2_t(0) :- TFL1_t_1(1).
        WUS_t(0) :- WUS_t_1(0).
        WUS_t(0) :- AG_t_1(1), SEP_t_1(1).
        AG_t(0) :- LFY_t_1(0), AP2_t_1(1).
        AG_t(0) :- AP1_t_1(1), AP2_t_1(1), WUS_t_1(0), AG_t_1(0), LUG_t_1(1), CLF_t_1(1).
        AG_t(0) :- AP1_t_1(1), AP2_t_1(1), WUS_t_1(0), LUG_t_1(1), CLF_t_1(1), SEP_t_1(0).
        AG_t(0) :- LFY_t_1(0), TFL1_t_1(1).
        TFL1_t(0) :- AP1_t_1(1).
        TFL1_t(0) :- EMF1_t_1(0).
        TFL1_t(0) :- LFY_t_1(1).
        PI_t(0) :- AP3_t_1(0), LFY_t_1(0).
        PI_t(0) :- AP3_t_1(0), AG_t_1(0).
        PI_t(0) :- AP1_t_1(0), LFY_t_1(0), AG_t_1(0).
        PI_t(0) :- LFY_t_1(0), PI_t_1(0).
        PI_t(0) :- LFY_t_1(0), SEP_t_1(0).
        SEP_t(0) :- LFY_t_1(0).
        """

        self._check_rules_and_predictions(dataset, expected_string_rules)

    #------------------
    # Tool functions
    #------------------

    def _check_rules_and_predictions(self, dataset, expected_string_rules):
        expected_string_rules = [s.strip() for s in expected_string_rules.strip().split("\n") if len(s) > 0 ]

        expected_rules = []
        for string_rule in expected_string_rules:
            expected_rules.append(Rule.from_string(string_rule, dataset.features, dataset.targets))

        #eprint(expected_rules)

        output = PRIDE.fit(dataset)

        #eprint(output)

        #for r in expected_rules:
        #    if r not in output:
        #        eprint("Missing rule: ", r)
        #    self.assertTrue(r in output)

        for r in output:
            if r not in expected_rules:
                eprint("Additional rule: ", r)
            self.assertTrue(r in expected_rules)

        model = DMVLP(dataset.features, dataset.targets, output)

        expected = set((tuple(s1),tuple(s2)) for s1,s2 in dataset.data)
        predicted = set()

        for s1 in model.feature_states():
            prediction = model.predict(s1)
            for s2 in prediction:
                predicted.add( (tuple(s1), tuple(s2)) )

        eprint()
        done = 0
        for s1,s2 in expected:
            done += 1
            eprint("\rChecking transitions ",done,"/",len(expected),end='')
            self.assertTrue((s1,s2) in predicted)

        done = 0
        for s1,s2 in predicted:
            done += 1
            eprint("\rChecking transitions ",done,"/",len(predicted),end='')
            self.assertTrue((s1,s2) in expected)


if __name__ == '__main__':
    """ Main """

    unittest.main()
