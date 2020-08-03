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
sys.path.insert(0, 'src/semantics')

from utils import eprint
from lust import LUST
from rule import Rule
from logicProgram import LogicProgram
from synchronous import Synchronous

#random.seed(0)

class LUSTTest(unittest.TestCase):
    """
        Unit test of class LUST from lust.py
    """

    __nb_unit_test = 100

    __nb_features = 4

    __nb_targets = 3

    __nb_values = 2

    __max_programs = 5

    __body_size = 10

    __tmp_file_path = "tmp/unit_test_lust.tmp"

    #------------------
    # Test functions
    #------------------

    def test_fit(self):
        print(">> LUST.fit(data, features, targets)")

        # No transitions
        p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
        p_ = LUST.fit([], p.get_features(), p.get_targets())
        self.assertEqual(len(p_),1)
        p_ = p_[0]
        self.assertEqual(p_.get_features(),p.get_features())
        self.assertEqual(p_.get_targets(),p.get_targets())
        self.assertEqual(p_.get_rules(),[])

        for i in range(self.__nb_unit_test):
            #eprint("test: ", i, "/", self.__nb_unit_test)

            nb_programs = random.randint(1,self.__max_programs)
            transitions = []

            p_ = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            features = p_.get_features()
            targets = p.get_targets()

            for j in range(nb_programs):
                # Generate transitions
                min_body_size = 0
                max_body_size = random.randint(min_body_size, len(features))

                p = LogicProgram.random(features, targets, min_body_size, max_body_size)
                transitions += Synchronous.transitions(p)

                #eprint(p.logic_form())
            #eprint(transitions)

            P = LUST.fit(transitions, features, targets)
            #rules = p_.get_rules()

            # Generate transitions
            predictions = []
            for p in P:
                #eprint(p.logic_form())
                predictions += Synchronous.transitions(p)

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
            p_ = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            features = p_.get_features()
            targets = p_.get_targets()

            min_body_size = 0
            max_body_size = random.randint(min_body_size, len(features))
            p = LogicProgram.random(features, targets, min_body_size, max_body_size)

            DC, DS = LUST.interprete([])
            self.assertEqual(DC,[])
            self.assertEqual(DS,[])

            # Regular case
            nb_programs = random.randint(1,self.__max_programs)
            transitions = []

            for j in range(nb_programs):
                # Generate transitions
                min_body_size = 0
                max_body_size = random.randint(min_body_size, len(features))

                p = LogicProgram.random(features, targets, min_body_size, max_body_size)
                transitions += Synchronous.transitions(p)

                #eprint(p.logic_form())
            #eprint(transitions)

            DC, DS = LUST.interprete(transitions)
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

    def random_rule(self, features, targets, body_size):
        head_var = random.randint(0,len(targets)-1)
        head_val = random.randint(0,len(targets[head_var][1])-1)
        body = []
        conditions = []

        for j in range(0, random.randint(0,body_size)):
            var = random.randint(0,len(features)-1)
            val = random.randint(0,len(features[var][1])-1)
            if var not in conditions:
                body.append( (var, val) )
                conditions.append(var)

        return  Rule(head_var,head_val,len(features),body)


    def random_program(self, nb_features, nb_targets, nb_values, body_size):
        features = [("x"+str(i), [str(val) for val in range(0,random.randint(2,nb_values))]) for i in range(random.randint(1,nb_features))]
        targets = [("y"+str(i), [str(val) for val in range(0,random.randint(2,nb_values))]) for i in range(random.randint(1,nb_targets))]
        rules = []

        for j in range(random.randint(0,100)):
            r = self.random_rule(features, targets, body_size)
            rules.append(r)

        return LogicProgram(features, targets, rules)


if __name__ == '__main__':
    """ Main """

    unittest.main()
