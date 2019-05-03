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

from utils import eprint
from lf1t import LF1T
from rule import Rule
from logicProgram import LogicProgram

#random.seed(0)


class LF1TTest(unittest.TestCase):
    """
        Unit test of class LF1T from lf1t.py
    """

    __nb_unit_test = 100

    __nb_variables = 3

    __nb_values = 2

    __body_size = 10

    __tmp_file_path = "tmp/unit_test_lf1t.tmp"

    #------------------
    # Test functions
    #------------------

    def test_load_input_from_csv(self):
        print(">> LF1T.load_input_from_csv(filepath)")

        for i in range(self.__nb_unit_test):
            # Generate transitions
            p = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)
            t1 = p.generate_all_transitions()

            # Save to csv
            p.transitions_to_csv(self.__tmp_file_path,t1)

            # Extract csv
            t2 = LF1T.load_input_from_csv(self.__tmp_file_path)

            # Check if still same
            for i in range(len(t1)):
                self.assertEqual(t1[i],t2[i])


    def test_fit(self):
        print(">> LF1T.fit(variables, values, transitions)")

        for i in range(self.__nb_unit_test):
            # Generate transitions
            p = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)
            t = p.generate_all_transitions()

            p_ = LF1T.fit(p.get_variables(), p.get_values(),t)
            rules = p_.get_rules()

            for head_var in range(len(p.get_variables())):
                for head_val in range(len(p.get_values()[head_var])):
                    for s1, s2 in t:
                        # Each positive is explained
                        if s2[head_var] == head_val:
                            cover = False
                            for r in rules:
                                if r.get_head_variable() == head_var and r.get_head_value() == head_val and r.matches(s1):
                                    cover = True
                                    self.assertEqual(r.get_head_variable(), head_var) # correct head var
                                    self.assertEqual(r.get_head_value(), head_val) # Correct head val
                            self.assertTrue(cover) # One rule cover the example

                        # No negative is covered
                        else:
                            cover = False
                            for r in rules:
                                if r.get_head_variable() == head_var and r.get_head_value() == head_val and r.matches(s1):
                                    cover = True
                            self.assertFalse(cover) # no rule covers the example

                    # All rules are minimals
                    for r in rules:
                        if r.get_head_variable() == head_var and r.get_head_value() == head_val:
                            for (var,val) in r.get_body():
                                r.remove_condition(var) # Try remove condition

                                conflict = False
                                for s1, s2 in t:
                                    if s2[head_var] != head_val and r.matches(s1): # Cover a negative example
                                        conflict = True
                                        break
                                self.assertTrue(conflict)
                                r.add_condition(var,val) # Cancel removal

    def test_fit_var_val(self):
        print(">> LF1T.fit_var_val(variable, value, positives, negatives)")

        for i in range(self.__nb_unit_test):
            # Generate transitions
            p = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)
            t = p.generate_all_transitions()

            head_var = random.randint(0, len(p.get_variables())-1)
            head_val = random.randint(0, len(p.get_values()[head_var]))

            rules = LF1T.fit_var_val(p.get_variables(), p.get_values(), t, head_var, head_val)

            for s1, s2 in t:
                # Each positive is explained
                if s2[head_var] == head_val:
                    cover = False
                    for r in rules:
                        if r.matches(s1):
                            cover = True
                            self.assertEqual(r.get_head_variable(), head_var) # correct head var
                            self.assertEqual(r.get_head_value(), head_val) # Correct head val
                    self.assertTrue(cover) # One rule cover the example

                # No negative is covered
                else:
                    cover = False
                    for r in rules:
                        if r.matches(s1):
                            cover = True
                    self.assertFalse(cover) # no rule covers the example

            # All rules are minimals
            for r in rules:
                for (var,val) in r.get_body():
                    r.remove_condition(var) # Try remove condition

                    conflict = False
                    for s1, s2 in t:
                        if s2[head_var] != head_val and r.matches(s1): # Cover a negative example
                            conflict = True
                            break
                    self.assertTrue(conflict)
                    r.add_condition(var,val) # Cancel removal


    #------------------
    # Tool functions
    #------------------

    def random_rule(self, variables, values, body_size):
        var = random.randint(0,len(variables)-1)
        val = random.choice(values[var])
        body = []
        conditions = []

        for j in range(0, random.randint(0,body_size)):
            var = random.randint(0,len(variables)-1)
            val = random.choice(values[var])
            if var not in conditions:
                body.append( (var, val) )
                conditions.append(var)

        return  Rule(var,val,body)


    def random_program(self, nb_variables, nb_values, body_size):
        variables = ["x"+str(i) for i in range(random.randint(1,nb_variables))]
        values = []
        rules = []

        for var in range(len(variables)):
            values.append([val for val in range(0,random.randint(2,nb_values))])

        for j in range(random.randint(0,100)):
            r = self.random_rule(variables, values, body_size)
            rules.append(r)

        return LogicProgram(variables, values, rules)


if __name__ == '__main__':
    """ Main """

    unittest.main()
