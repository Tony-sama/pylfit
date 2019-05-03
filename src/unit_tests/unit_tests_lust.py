#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/23
# @updated: 2019/05/02
#
# @desc: PyLFIT unit test script
#
#-----------------------

import sys
import unittest
import random
import os

sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')

from utils import eprint
from lust import LUST
from rule import Rule
from logicProgram import LogicProgram

#random.seed(0)

class LUSTTest(unittest.TestCase):
    """
        Unit test of class LUST from lust.py
    """

    __nb_unit_test = 100

    __nb_variables = 6

    __nb_values = 2

    __max_programs = 5

    __body_size = 10

    __tmp_file_path = "tmp/unit_test_lust.tmp"

    #------------------
    # Test functions
    #------------------

    def test_fit(self):
        print(">> LUST.fit(variables, values, transitions)")

        # No transitions
        variables = ["x"+str(i) for i in range(random.randint(1,self.__nb_variables))]
        values = []

        for var in range(len(variables)):
            values.append([val for val in range(random.randint(2,self.__nb_values))])

        min_body_size = 0
        max_body_size = random.randint(min_body_size, len(variables))
        p = LogicProgram.random(variables, values, min_body_size, max_body_size)
        p_ = LUST.fit(p.get_variables(), p.get_values(), [])
        self.assertEqual(len(p_),1)
        p_ = p_[0]
        self.assertEqual(p_.get_variables(),p.get_variables())
        self.assertEqual(p_.get_values(),p.get_values())
        self.assertEqual(p_.get_rules(),[])

        for i in range(self.__nb_unit_test):
            #eprint("test: ", i, "/", self.__nb_unit_test)

            variables = ["x"+str(i) for i in range(random.randint(1,self.__nb_variables))]
            values = []

            for var in range(len(variables)):
                values.append([val for val in range(random.randint(2,self.__nb_values))])

            nb_programs = random.randint(1,self.__max_programs)
            transitions = []

            for j in range(nb_programs):
                # Generate transitions
                min_body_size = 0
                max_body_size = random.randint(min_body_size, len(variables))

                p = LogicProgram.random(variables, values, min_body_size, max_body_size)
                transitions += p.generate_all_transitions()

                #eprint(p.logic_form())
            #eprint(transitions)

            P = LUST.fit(p.get_variables(), p.get_values(), transitions)
            #rules = p_.get_rules()

            # Generate transitions
            predictions = []
            for p in P:
                #eprint(p.logic_form())
                predictions += p.generate_all_transitions()

            # Remove incomplete states
            #predictions = [ [s1,s2] for s1,s2 in predictions if -1 not in s2 ]

            #eprint("Expected: ", transitions)
            #eprint("Predicted: ", predictions)

            # All original transitions are predicted
            for s1, s2 in transitions:
                self.assertTrue([s1,s2] in predictions)

            # All predictions are in original transitions
            for s1, s2 in predictions:
                #eprint(s1,s2)
                self.assertTrue([s1,s2] in transitions)

            #sys.exit()

    def test_interprete(self):
        print(">> LUST.interprete(variables, values, transitions)")

        for i in range(self.__nb_unit_test):
            #eprint("test: ", i, "/", self.__nb_unit_test)

            # No transitions
            variables = ["x"+str(i) for i in range(random.randint(1,self.__nb_variables))]
            values = []

            for var in range(len(variables)):
                values.append([val for val in range(random.randint(2,self.__nb_values))])

            min_body_size = 0
            max_body_size = random.randint(min_body_size, len(variables))
            p = LogicProgram.random(variables, values, min_body_size, max_body_size)

            var = random.randint(0, len(p.get_variables())-1)
            val = random.randint(0, len(p.get_values()[var])-1)

            DC, DS = LUST.interprete(p.get_variables(), p.get_values(), [])
            self.assertEqual(DC,[])
            self.assertEqual(DS,[])

            # Regular case
            variables = ["x"+str(i) for i in range(random.randint(1,self.__nb_variables))]
            values = []

            for var in range(len(variables)):
                values.append([val for val in range(random.randint(2,self.__nb_values))])

            nb_programs = random.randint(1,self.__max_programs)
            transitions = []

            for j in range(nb_programs):
                # Generate transitions
                min_body_size = 0
                max_body_size = random.randint(min_body_size, len(variables))

                p = LogicProgram.random(variables, values, min_body_size, max_body_size)
                transitions += p.generate_all_transitions()

                #eprint(p.logic_form())
            #eprint(transitions)

            var = random.randint(0, len(p.get_variables())-1)
            val = random.randint(0, len(p.get_values()[var])-1)

            DC, DS = LUST.interprete(p.get_variables(), p.get_values(), transitions)
            D = []
            ND = []

            for s1, s2 in transitions:
                deterministic = True
                for s3, s4 in transitions:
                    if s1 == s3 and s2 != s4:
                        ND.append( [s1,s2] )
                        deterministic = False
                        break
                if deterministic:
                    D.append( [s1,s2] )

            #eprint("DC: ",DC)
            #eprint("DS: ",DS)
            #eprint("D: ",D)
            #eprint("ND: ",ND)

            # All deterministic are only in DC
            for s1, s2 in D:
                self.assertTrue([s1,s2] in DC)
                for s in DS:
                    self.assertTrue([s1,s2] not in s)

            # All DC are deterministic
            for s1, s2 in DC:
                self.assertTrue([s1,s2] in D)

            # All non deterministic sets are set
            for s in DS:
                for s1, s2 in s:
                    occ = 0
                    for s3, s4 in s:
                        if s1 == s3 and s2 == s4:
                            occ += 1
                    self.assertEqual(occ,1)

            # All input origin state appears in each DS TODO
            for s1, s2 in ND:
                for s in DS:
                    occurs = False
                    for s3, s4 in s:
                        if s1 == s3:
                            occurs = True
                    self.assertTrue(occurs)

            # All DS are deterministic
            for s in DS:
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
