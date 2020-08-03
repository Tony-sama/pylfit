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
sys.path.insert(0, 'src/semantics')

from utils import eprint
from gula import GULA
from rule import Rule
from logicProgram import LogicProgram
from synchronous import Synchronous

random.seed(0)


class GULATest(unittest.TestCase):
    """
        Unit test of class GULA from gula.py
    """

    __nb_unit_test = 100

    __nb_features = 3

    __nb_targets = 4

    __nb_values = 3

    __body_size = 10

    __tmp_file_path = "tmp/unit_test_gula.tmp"

    #------------------
    # Test functions
    #------------------

    def test_load_input_from_csv(self):
        print(">> GULA.load_input_from_csv(filepath)")

        for i in range(self.__nb_unit_test):
            # Generate transitions
            p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            t1 = Synchronous.transitions(p)

            # Save to csv
            Synchronous.transitions_to_csv(self.__tmp_file_path, t1, p.get_features(), p.get_targets())

            # Extract csv
            t2 = GULA.load_input_from_csv(self.__tmp_file_path, len(p.get_features()))

            # Check if still same
            for i in range(len(t1)):
                self.assertEqual(t1[i],t2[i])


    def test_fit(self):
        print(">> GULA.fit(data, features, targets)")

        # No transitions
        p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
        p_ = GULA.fit([],p.get_features(), p.get_targets())
        self.assertEqual(p_.get_features(),p.get_features())
        self.assertEqual(p_.get_targets(),p.get_targets())
        rules = []
        for var in range(len(p.get_targets())):
            for val in range(len(p.get_targets()[var][1])):
                rules.append(Rule(var,val,len(p.get_features())))
        self.assertEqual(p_.get_rules(),rules)

        for i in range(self.__nb_unit_test):
            # Generate transitions
            p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            t = Synchronous.transitions(p)
            random.shuffle(t)
            #t_ = {tuple(s1): [s2_ for (s1_, s2_) in t if s1 == s1_] for (s1, s2) in t}

            p_ = GULA.fit(t, p.get_features(), p.get_targets())
            rules = p_.get_rules()

            for variable in range(len(p.get_targets())):
                for value in range(len(p.get_targets()[variable][1])):
                    #eprint("var="+str(variable)+", val="+str(value))
                    #neg = GULA.interprete(t_, variable, value)

                    # Each positive is explained
                    pos = [s1 for s1,s2 in t if s2[variable] == value]
                    neg = [s1 for s1,s2 in t if s1 not in pos]
                    #eprint(variable, "=", value)
                    #eprint("t: ", t)
                    #eprint("neg: ", neg)
                    #eprint("pos: ", pos)
                    #eprint([r for r in rules if r.get_head_variable() == variable and r.get_head_value() == value])
                    for s in pos:
                        cover = False
                        for r in rules:
                            if r.get_head_variable() == variable \
                               and r.get_head_value() == value \
                               and r.matches(s):
                                cover = True
                        #eprint(variable,"=",value)
                        #eprint(rules)
                        #eprint(s)
                        self.assertTrue(cover) # One rule cover the example

                    # No negative is covered
                    neg = [s1 for s1,s2 in t if s1 not in pos]
                    for s in neg:
                        cover = False
                        for r in rules:
                            if r.get_head_variable() == variable \
                               and r.get_head_value() == value \
                               and r.matches(s):
                                cover = True
                                #eprint(r)
                                #eprint(s)
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
            p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            t = Synchronous.transitions(p)

            var = random.randint(0, len(p.get_targets())-1)
            val = random.randint(0, len(p.get_targets()[var][1])-1)

            t = sorted(t)
            t_ = [ (tuple(s1), [s2_ for (s1_, s2_) in t if s1 == s1_]) for (s1, s2) in t]
            #t_ = {tuple(s1): [s2_ for (s1_, s2_) in t if s1 == s1_] for (s1, s2) in t}
            neg = [tuple(s) for s in GULA.interprete(t_, var, val)]

            pos_ = [s1 for s1,s2 in t if s2[var] == val]
            neg_ = [tuple(s1) for s1,s2 in t if s1 not in pos_]

            if set(neg) != set(neg_):
                eprint(neg)
                eprint(neg_)
            self.assertEqual(set(neg),set(neg_))

            # All pos are valid
            for s in pos_:
                valid = False
                for s1, s2 in t:
                    if s1 == s:
                        valid = True
                        break
                self.assertTrue(valid)

            # All neg are valid
            for s in neg:
                for s1, s2 in t:
                    if s1 == s:
                        self.assertTrue(s2[var] != val)
                        break
            # All transitions are interpreted
            for s1,S2 in t_:
                if len([s2 for s2 in S2 if s2[var] == val]) == 0:
                    self.assertTrue(s1 in neg)


    def test_fit_var_val(self):
        print(">> GULA.fit_var_val(features, variable, value, negatives)")

        for i in range(self.__nb_unit_test):
            # Generate transitions
            p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            t = Synchronous.transitions(p)

            var = random.randint(0, len(p.get_targets())-1)
            val = random.randint(0, len(p.get_targets()[var])-1)

            t = sorted(t)
            t_ = [ (tuple(s1), [s2_ for (s1_, s2_) in t if s1 == s1_]) for (s1, s2) in t]
            #t_ = {tuple(s1): [s2_ for (s1_, s2_) in t if s1 == s1_] for (s1, s2) in t}
            neg = GULA.interprete(t_, var, val)

            rules = GULA.fit_var_val(p.get_features(), var, val, neg)

            # Each positive is explained
            pos = [s1 for s1,s2 in t if s2[var] == val]
            for s in pos:
                cover = False
                for r in rules:
                    self.assertEqual(r.get_head_variable(), var) # correct head var
                    self.assertEqual(r.get_head_value(), val) # Correct head val

                    if r.matches(s):
                        cover = True

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
