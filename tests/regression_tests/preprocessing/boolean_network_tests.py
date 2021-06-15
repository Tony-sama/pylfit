#-----------------------
# @author: Tony Ribeiro
# @created: 2021/06/14
# @updated: 2021/06/15
#
# @desc: boolean network class unit test script
#
#-----------------------

import unittest
import random
import sys
import numpy as np
import io
import contextlib

import pylfit

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from pylfit.objects import Rule
from pylfit.preprocessing.boolean_network import dmvlp_from_boolean_network_file, dmvlp_from_bnet_file, dmvlp_from_net_file
from pylfit.utils import eprint

random.seed(0)

class boolean_network_tests(unittest.TestCase):
    """
        Unit test of module boolean_network.py
    """
    _nb_random_tests = 100

    _nb_transitions = 100

    _nb_features = 5

    _nb_targets = 5

    _nb_feature_values = 3

    _nb_target_values = 3

    def test_dmvlp_from_boolean_network_file(self):
        print(">> pylfit.preprocessing.boolean_network.dmvlp_from_boolean_network_file(file_path, compute_complementary_rules=False)")

        for compute_complementary_rules in [False,True]:
            # Unit tests

            # boolenet: fission_yeast.net
            file_path = "benchmarks/boolean_networks/boolenet/fission_yeast.net"
            f = io.StringIO()
            with contextlib.redirect_stderr(f):
                model = dmvlp_from_boolean_network_file(file_path=file_path, compute_complementary_rules=compute_complementary_rules)

            expected_string_rules = """
            SK_t(1) :- Start_t_1(1).
            Ste9_t(1) :- SK_t_1(0), Ste9_t_1(1), Cdc2/Cdc13_t_1(0), PP_t_1(0), Cdc2/Cdc13*_t_1(0).
            Ste9_t(1) :- SK_t_1(0), Cdc2/Cdc13_t_1(0), PP_t_1(1), Cdc2/Cdc13*_t_1(0).
            Ste9_t(1) :- SK_t_1(0), Ste9_t_1(1), Cdc2/Cdc13_t_1(0), PP_t_1(1), Cdc2/Cdc13*_t_1(1).
            Ste9_t(1) :- SK_t_1(0), Ste9_t_1(1), Cdc2/Cdc13_t_1(1), PP_t_1(1), Cdc2/Cdc13*_t_1(0).
            Ste9_t(1) :- SK_t_1(1), Ste9_t_1(1), Cdc2/Cdc13_t_1(0), PP_t_1(1), Cdc2/Cdc13*_t_1(0).
            Cdc2/Cdc13_t(1) :- Ste9_t_1(0), Rum1_t_1(0), Slp1_t_1(0).
            Cdc2/Cdc13_t(1) :- Ste9_t_1(0), Cdc2/Cdc13_t_1(1), Rum1_t_1(0), Slp1_t_1(1).
            Cdc2/Cdc13_t(1) :- Ste9_t_1(0), Cdc2/Cdc13_t_1(1), Rum1_t_1(1), Slp1_t_1(0).
            Cdc2/Cdc13_t(1) :-  Ste9_t_1(1), Cdc2/Cdc13_t_1(1), Rum1_t_1(0), Slp1_t_1(0).
            Rum1_t(1) :- SK_t_1(0), Cdc2/Cdc13_t_1(0), Rum1_t_1(1), PP_t_1(0), Cdc2/Cdc13*_t_1(0).
            Rum1_t(1) :- SK_t_1(0), Cdc2/Cdc13_t_1(0), PP_t_1(1), Cdc2/Cdc13*_t_1(0).
            Rum1_t(1) :- SK_t_1(0), Cdc2/Cdc13_t_1(0), Rum1_t_1(1), PP_t_1(1), Cdc2/Cdc13*_t_1(1).
            Rum1_t(1) :- SK_t_1(0), Cdc2/Cdc13_t_1(1), Rum1_t_1(1), PP_t_1(1), Cdc2/Cdc13*_t_1(0).
            Rum1_t(1) :- SK_t_1(1), Cdc2/Cdc13_t_1(0), Rum1_t_1(1), PP_t_1(1), Cdc2/Cdc13*_t_1(1).
            PP_t(1) :- Slp1_t_1(1).
            Cdc25_t(1) :- Cdc2/Cdc13_t_1(0), PP_t_1(0), Cdc25_t_1(1).
            Cdc25_t(1) :- Cdc2/Cdc13_t_1(1), PP_t_1(0).
            Cdc25_t(1) :- Cdc2/Cdc13_t_1(1), PP_t_1(1), Cdc25_t_1(1).
            Slp1_t(1) :- Cdc2/Cdc13*_t_1(1).
            Wee1/Mik1_t(1) :- Cdc2/Cdc13_t_1(0), PP_t_1(0), Wee1/Mik1_t_1(1).
            Wee1/Mik1_t(1) :- Cdc2/Cdc13_t_1(0), PP_t_1(1).
            Wee1/Mik1_t(1) :- Cdc2/Cdc13_t_1(1), PP_t_1(1), Wee1/Mik1_t_1(1).
            Cdc2/Cdc13*_t(1) :- Ste9_t_1(0), Rum1_t_1(0), Cdc25_t_1(1), Slp1_t_1(0), Wee1/Mik1_t_1(0), Cdc2/Cdc13*_t_1(1).
            """

            if compute_complementary_rules:
                expected_string_rules += """
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

            self._check_rules(model, expected_string_rules)

            # pyboolnet: xiao_wnt5a.bnet
            file_path = "benchmarks/boolean_networks/pyboolnet/bio/xiao_wnt5a.bnet"
            f = io.StringIO()
            with contextlib.redirect_stderr(f):
                model = dmvlp_from_boolean_network_file(file_path=file_path, compute_complementary_rules=compute_complementary_rules)

            expected_string_rules = """
            x1_t(1) :- x6_t_1(0).
            x2_t(1) :- x6_t_1(1), x4_t_1(1).
            x2_t(1) :- x2_t_1(1), x6_t_1(1).
            x2_t(1) :- x2_t_1(1), x4_t_1(1).
            x3_t(1) :- x7_t_1(0).
            x4_t(1) :- x4_t_1(1).
            x5_t(1) :- x7_t_1(0).
            x5_t(1) :- x2_t_1(1).
            x6_t(1) :- x3_t_1(1).
            x6_t(1) :- x4_t_1(1).
            x7_t(1) :- x7_t_1(1).
            x7_t(1) :- x2_t_1(0).
            """

            if compute_complementary_rules:
                expected_string_rules += """
                x1_t(0) :- x6_t_1(1).
                x2_t(0) :- x2_t_1(0), x6_t_1(0).
                x2_t(0) :- x4_t_1(0), x6_t_1(0).
                x2_t(0) :- x2_t_1(0), x4_t_1(0).
                x3_t(0) :- x7_t_1(1).
                x4_t(0) :- x4_t_1(0).
                x5_t(0) :- x2_t_1(0), x7_t_1(1).
                x6_t(0) :- x3_t_1(0), x4_t_1(0).
                x7_t(0) :- x2_t_1(1), x7_t_1(0).
                """

            self._check_rules(model, expected_string_rules)

            # exceptions

            # Bad file format
            self.assertRaises(ValueError,  dmvlp_from_boolean_network_file, "", compute_complementary_rules)
            self.assertRaises(ValueError,  dmvlp_from_boolean_network_file, "file.csv", compute_complementary_rules)

    def test_dmvlp_from_bnet_file(self):
        print(">> pylfit.preprocessing.boolean_network.dmvlp_from_bnet_file(file_path)")

        file_path = "tmp/special_bnet_file.bnet"
        content = """
        x4, 0

        x2,  1
        """
        f = open(file_path, "w")
        f.write(content)
        f.close()

        model = dmvlp_from_bnet_file(file_path=file_path)

    def test_dmvlp_from_net_file(self):
        print(">> pylfit.preprocessing.boolean_network.dmvlp_from_net_file(file_path)")

        # exceptions

        # Missing .v
        file_path = "tmp/bad_net_file.net"
        content = """
        # labels of nodes and names of corresponding components
        # 1 = Start
        # 2 = SK

        # 1 = Start
        .n 1 0

        # 2 = SK
        .n 2 1 1
        1 1
        """
        f = open(file_path, "w")
        f.write(content)
        f.close()

        self.assertRaises(ValueError, dmvlp_from_net_file, file_path)

        # Missng labels
        file_path = "tmp/bad_net_file.net"
        content = """
        .v 2

        # 1 = Start
        .n 1 0

        # 2 = SK
        .n 2 1 1
        1 1
        """
        f = open(file_path, "w")
        f.write(content)
        f.close()

        self.assertRaises(ValueError, dmvlp_from_net_file, file_path)

        # Missng .n
        file_path = "tmp/bad_net_file.net"
        content = """
        .v 2

        # labels of nodes and names of corresponding components
        # 1 = Start
        # 2 = SK

        # 1 = Start
        .n 1 0

        # 2 = SK
        n 2 1 1
        1 1
        """
        f = open(file_path, "w")
        f.write(content)
        f.close()

        self.assertRaises(ValueError, dmvlp_from_net_file, file_path)

        # warning , in label
        file_path = "tmp/warning_net_file.net"
        content = """
        #total number of nodes
        .v 2

        # labels of nodes and names of corresponding components
        # 1 = Start
        # 2 = SK,test

        # 1 = Start
        .n 1 0

        # 2 = SK,test
        .n 2 1 1
        1 1
        """
        f = open(file_path, "w")
        f.write(content)
        f.close()

        f = io.StringIO()
        with contextlib.redirect_stderr(f):
            model =  dmvlp_from_net_file(file_path)

        #self.assertRaises(ValueError, dmvlp_from_net_file, file_path)

        # id not in order
        file_path = "tmp/warning_net_file.net"
        content = """
        #total number of nodes
        .v 2

        # labels of nodes and names of corresponding components
        # 1 = Start
        # 2 = SK

        # 2 = SK,test
        .n 2 1 1
        1 1

        # 1 = Start
        .n 1 0
        """
        f = open(file_path, "w")
        f.write(content)
        f.close()

        self.assertRaises(ValueError, dmvlp_from_net_file, file_path)

        # Regulators not consistents
        file_path = "tmp/bad_net_file.net"
        content = """
        #total number of nodes
        .v 2

        # labels of nodes and names of corresponding components
        # 1 = Start
        # 2 = SK

        # 1 = Start
        .n 1 0

        # 2 = SK,test
        .n 2 1 1 2
        11 1
        10 1
        """
        f = open(file_path, "w")
        f.write(content)
        f.close()

        self.assertRaises(ValueError, dmvlp_from_net_file, file_path)

    def _check_rules(self, model, expected_string_rules):
        expected_string_rules = [s.strip() for s in expected_string_rules.strip().split("\n") if len(s.strip()) > 0 ]

        expected_rules = []
        for string_rule in expected_string_rules:
            expected_rules.append(Rule.from_string(string_rule, model.features, model.targets))

        for r in expected_rules:
            if r not in model.rules:
                eprint("Missing rule: ", r.logic_form(model.features, model.targets), " (", r.to_string(),")")
            self.assertTrue(r in model.rules)

        for r in model.rules:
            if r not in expected_rules:
                eprint("Additional rule: ", r.logic_form(model.features, model.targets), " (", r.to_string(),")")
            self.assertTrue(r in expected_rules)


'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
