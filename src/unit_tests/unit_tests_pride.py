#-----------------------
# @author: Tony Ribeiro
# @created: 2019/03/25
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
from pride import PRIDE
from rule import Rule
from logicProgram import LogicProgram
from semantics import Semantics
from synchronous import Synchronous

#random.seed(0)


class PRIDETest(unittest.TestCase):
    """
        Unit test of class PRIDE from pride.py
    """

    __nb_unit_test = 100

    __nb_features = 5

    __nb_targets = 3

    __nb_values = 3

    __body_size = 10

    __tmp_file_path = "tmp/unit_test_pride.tmp"

    #------------------
    # Test functions
    #------------------

    def test_load_input_from_csv(self):
        print(">> PRIDE.load_input_from_csv(filepath)")

        for i in range(self.__nb_unit_test):
            # Generate transitions
            p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            t1 = Synchronous.transitions(p)

            # Save to csv
            Semantics.transitions_to_csv(self.__tmp_file_path, t1, p.get_features(), p.get_targets())

            # Extract csv
            t2 = PRIDE.load_input_from_csv(self.__tmp_file_path)

            # Check if still same
            for i in range(len(t1)):
                self.assertEqual(t1[i],t2[i])


    def test_fit(self):
        print(">> PRIDE.fit(transitions)")

        # No transitions
        p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
        p_ = PRIDE.fit([], p.get_features(), p.get_targets())
        self.assertEqual(p_.get_features(),p.get_features())
        self.assertEqual(p_.get_targets(),p.get_targets())
        self.assertEqual(p_.get_rules(),[])

        for i in range(self.__nb_unit_test):
            # Generate transitions
            p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            t = Synchronous.transitions(p)

            p_ = PRIDE.fit(t, p.get_features(), p.get_targets())
            rules = p_.get_rules()

            for variable in range(len(p.get_targets())):
                for value in range(len(p.get_targets()[variable][1])):
                    #eprint("var="+str(variable)+", val="+str(value))
                    pos, neg = PRIDE.interprete(t, variable, value)

                    # Each positive is explained
                    for s in pos:
                        cover = False
                        for r in rules:
                            if r.get_head_variable() == variable \
                               and r.get_head_value() == value \
                               and r.matches(s):
                                cover = True
                        self.assertTrue(cover) # One rule cover the example

                    # No negative is covered
                    for s in neg:
                        cover = False
                        for r in rules:
                            if r.get_head_variable() == variable \
                               and r.get_head_value() == value \
                               and r.matches(s):
                                cover = True
                        self.assertFalse(cover) # no rule covers the example

                    # All rules are minimals
                    for r in rules:
                        if r.get_head_variable() == variable and r.get_head_value() == value:
                            for (var,val) in r.get_body():
                                r.remove_condition(var) # Try remove condition

                                conflict = False
                                for s in neg:
                                    if r.matches(s): # Cover a negative example
                                        conflict = True
                                        break

                                # # DEBUG:
                                if not conflict:
                                    eprint("not minimal "+r.to_string())
                                    eprint(neg)

                                self.assertTrue(conflict)
                                r.add_condition(var,val) # Cancel removal

    def test_interprete(self):
        print(">> PRIDE.interprete(transitions, variable, value)")

        for i in range(self.__nb_unit_test):
            # Generate transitions
            p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            t = Synchronous.transitions(p)

            var = random.randint(0, len(p.get_targets())-1)
            val = random.randint(0, len(p.get_targets()[var][1])-1)

            pos, neg = PRIDE.interprete(t, var, val)

            # All pos are valid
            for s in pos:
                occurs = False
                for s1, s2 in t:
                    if s1 == s and s2[var] == val :
                        occurs = True
                        break
                self.assertTrue(occurs)

            # All neg are valid
            for s in neg:
                for s1, s2 in t:
                    if s1 == s:
                        self.assertTrue(s2[var] != val)
                        break
            # All transitions are interpreted
            for s1, s2 in t:
                self.assertTrue(s1 in pos or s1 in neg)

    def test_fit_var_val(self):
        print(">> PRIDE.fit_var_val(variable, value, nb_variables, positives, negatives)")

        for i in range(self.__nb_unit_test):
            # Generate transitions
            p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            t = Synchronous.transitions(p)

            var = random.randint(0, len(p.get_targets())-1)
            val = random.randint(0, len(p.get_targets()[var][1])-1)

            pos, neg = PRIDE.interprete(t, var, val)

            rules = PRIDE.fit_var_val(var, val, len(p.get_features()), pos, neg)

            # Each positive is explained
            for s in pos:
                cover = False
                for r in rules:
                    if r.matches(s):
                        cover = True
                        self.assertEqual(r.get_head_variable(), var) # correct head var
                        self.assertEqual(r.get_head_value(), val) # Correct head val
                self.assertTrue(cover) # Alteast one rule covers the example

            # No negative is covered
            for s in neg:
                cover = False
                for r in rules:
                    if r.matches(s):
                        cover = True
                self.assertFalse(cover) # no rule covers the example

            # All rules are minimals
            for r in rules:
                for (var,val) in r.get_body():
                    r.remove_condition(var) # Try remove condition

                    conflict = False
                    for s in neg:
                        if r.matches(s): # Cover a negative example
                            conflict = True
                            break
                    self.assertTrue(conflict)
                    r.add_condition(var,val) # Cancel removal


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
