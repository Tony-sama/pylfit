#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/15
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
from gula import GULA
from rule import Rule
from logicProgram import LogicProgram

#random.seed(0)


class GULATest(unittest.TestCase):
    """
        Unit test of class GULA from gula.py
    """

    __nb_unit_test = 100

    __nb_variables = 5

    __nb_values = 2

    __body_size = 10

    __tmp_file_path = "tmp/unit_test_gula.tmp"

    #------------------
    # Test functions
    #------------------

    def test_load_input_from_csv(self):
        print(">> GULA.load_input_from_csv(filepath)")

        for i in range(self.__nb_unit_test):
            # Generate transitions
            p = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)
            t1 = p.generate_all_transitions()

            # Save to csv
            p.transitions_to_csv(self.__tmp_file_path,t1)

            # Extract csv
            t2 = GULA.load_input_from_csv(self.__tmp_file_path)

            # Check if still same
            for i in range(len(t1)):
                self.assertEqual(t1[i],t2[i])


    def test_fit(self):
        print(">> GULA.fit(transitions)")

        # No transitions
        p = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)
        p_ = GULA.fit(p.get_variables(), p.get_values(), [])
        self.assertEqual(p_.get_variables(),p.get_variables())
        self.assertEqual(p_.get_values(),p.get_values())
        rules = []
        for var in range(len(p.get_variables())):
            for val in p.get_values()[var]:
                rules.append(Rule(var,val))
        self.assertEqual(p_.get_rules(),rules)

        for i in range(self.__nb_unit_test):
            # Generate transitions
            p = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)
            t = p.generate_all_transitions()

            if random.choice([True,False]):
                p_ = GULA.fit(p.get_variables(), p.get_values(), t)
            else:
                rules = []
                for var in range(0,len(p.get_variables())):
                    for val in range(0,len(p.get_values()[var])):
                        rules.append(Rule(var,val))
                init = LogicProgram(p.get_variables(), p.get_values(), rules)
                p_ = GULA.fit(p.get_variables(), p.get_values(), t, init)
            rules = p_.get_rules()

            for variable in range(len(p.get_variables())):
                for value in range(len(p.get_values()[variable])):
                    #eprint("var="+str(variable)+", val="+str(value))
                    pos, neg = GULA.interprete(t, variable, value)

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
        print(">> GULA.interprete(transitions, variable, value)")

        for i in range(self.__nb_unit_test):
            # Generate transitions
            p = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)
            t = p.generate_all_transitions()

            var = random.randint(0, len(p.get_variables())-1)
            val = random.randint(0, len(p.get_values()[var])-1)

            pos, neg = GULA.interprete(t, var, val)

            # All pos are valid
            for s in pos:
                for s1, s2 in t:
                    if s1 == s:
                        self.assertEqual(s2[var], val)
                        break
            # All neg are valid
            for s in neg:
                for s1, s2 in t:
                    if s1 == s:
                        self.assertTrue(s2[var] != val)
                        break
            # All transitions are interpreted
            for s1, s2 in t:
                if s2[var] == val:
                    self.assertTrue(s1 in pos)
                else:
                    self.assertTrue(s1 in neg)


    def test_fit_var_val(self):
        print(">> GULA.fit_var_val(variables, values, variable, value, positives, negatives)")

        for i in range(self.__nb_unit_test):
            # Generate transitions
            p = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)
            t = p.generate_all_transitions()

            var = random.randint(0, len(p.get_variables())-1)
            val = random.randint(0, len(p.get_values()[var])-1)

            pos, neg = GULA.interprete(t, var, val)

            if random.choice([True,False]):
                rules = GULA.fit_var_val(p.get_variables(), p.get_values(), var, val, pos, neg)
            else:
                rules = GULA.fit_var_val(p.get_variables(), p.get_values(), var, val, pos, neg, LogicProgram(p.get_variables(), p.get_values(), [Rule(var, val)]))

            # Each positive is explained
            for s in pos:
                cover = False
                for r in rules:
                    if r.matches(s):
                        cover = True
                        self.assertEqual(r.get_head_variable(), var) # correct head var
                        self.assertEqual(r.get_head_value(), val) # Correct head val
                self.assertTrue(cover) # One rule cover the example

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
