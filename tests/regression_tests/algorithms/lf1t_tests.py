#-----------------------
# @author: Tony Ribeiro
# @created: 2019/03/25
# @updated: 2019/05/02
#
# @desc: PyLFIT unit test script
#
#-----------------------

import unittest
import random
import os

from pylfit.utils import eprint, load_tabular_data_from_csv
from pylfit.algorithms.lf1t import LF1T
from pylfit.objects.rule import Rule
from pylfit.objects.logicProgram import LogicProgram
from pylfit.semantics.synchronous import Synchronous

random.seed(0)


class LF1TTest(unittest.TestCase):
    """
        Unit test of class LF1T from lf1t.py
    """

    __nb_unit_test = 100

    __nb_features = 4

    __nb_targets = 3

    __nb_values = 3

    __body_size = 10

    __tmp_file_path = "tmp/unit_test_lf1t.tmp"

    #------------------
    # Test functions
    #------------------

    def test_load_input_from_csv(self):
        print(">> LF1T.load_input_from_csv(filepath)")

        for i in range(self.__nb_unit_test):
            # Generate transitions
            p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            t1 = Synchronous.transitions(p)

            # Save to csv
            Synchronous.transitions_to_csv(self.__tmp_file_path, t1, p.get_features(), p.get_targets())

            # Extract csv
            t2, feature_domains, target_domains = load_tabular_data_from_csv(self.__tmp_file_path, [var for var,vals in p.get_features()], [var for var,vals in p.get_targets()])
            t2 = [ [[str(i) for i in s1], [str(i) for i in s2]] for (s1,s2) in t2 ]

            # Check if still same
            for i in range(len(t1)):
                self.assertEqual(t1[i],t2[i])


    def test_fit(self):
        print(">> LF1T.fit(variables, values, transitions)")

        for i in range(self.__nb_unit_test):
            # Generate transitions
            p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            t = Synchronous.transitions(p)

            # Make transitions deterministics
            locked = set()
            t_ = []
            for s1,s2 in t:
                if tuple(s1) not in locked:
                    t_.append((s1,s2))
                    locked.add(tuple(s1))
            t = t_
            t = LF1T.encode_transitions_set(t, p.get_features(), p.get_targets())

            p_ = LF1T.fit(t,p.get_features(),p.get_targets())
            rules = p_.get_rules()

            for head_var in range(len(p.get_targets())):
                for head_val in range(len(p.get_targets()[head_var][1])):
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
            p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            t = Synchronous.transitions(p)

            # Make transitions deterministics
            locked = set()
            t_ = []
            for s1,s2 in t:
                if tuple(s1) not in locked:
                    t_.append((s1,s2))
                    locked.add(tuple(s1))
            t = t_

            head_var = random.randint(0, len(p.get_targets())-1)
            head_val = random.randint(0, len(p.get_targets()[head_var][1]))

            rules = LF1T.fit_var_val(p.get_features(), head_var, head_val, t)

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
        features = [("x"+str(i), ["val_"+str(val) for val in range(0,random.randint(2,nb_values))]) for i in range(random.randint(1,nb_features))]
        targets = [(var+"_t", vals) for var,vals in features]
        rules = []

        for j in range(random.randint(0,100)):
            r = self.random_rule(features, targets, body_size)
            rules.append(r)

        return LogicProgram(features, targets, rules)


if __name__ == '__main__':
    """ Main """

    unittest.main()
