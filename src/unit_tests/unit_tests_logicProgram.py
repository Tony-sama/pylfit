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
import csv
from operator import mul
from functools import reduce

sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')
sys.path.insert(0, 'src/semantics')

from utils import eprint
from logicProgram import LogicProgram
from rule import Rule
from semantics import Semantics

seed = random.randint(0,1000000)
seed = 0
random.seed(seed)
#eprint("seed: ", seed)

class LogicProgramTest(unittest.TestCase):
    """p
        Unit test of class LogicProgram from logicProgram.py
    """

    __nb_unit_test = 100

    __nb_features = 3

    __nb_targets = 2

    __nb_values = 2

    __max_delay = 5

    __nb_rules = 100

    __body_size = 10

    __tmp_file_path = "tmp/unit_test_logicProgram.tmp"

    #------------------
    # Test functions
    #------------------

    def test___init__(self):
        print(">> LogicProgram.__init__(self, features, targets, rules, constraints=[])")

        for i in range(self.__nb_unit_test):
            features = [("x"+str(i), [val for val in range(0,random.randint(2,self.__nb_values))]) for i in range(random.randint(1,self.__nb_features))]
            targets = [("x"+str(i), [val for val in range(0,random.randint(2,self.__nb_values))]) for i in range(random.randint(1,self.__nb_targets))]
            rules = []

            for j in range(random.randint(0,self.__nb_rules)):
                r = self.random_rule(features, targets, self.__body_size)
                rules.append(r)

            p = LogicProgram(features, targets, rules)

            self.assertEqual(p.get_features(), features)
            self.assertEqual(p.get_targets(), targets)
            self.assertEqual(p.get_rules(), rules)

    def test_load_from_file(self):
        print(">> LogicProgram.load_from_file(self, file_path)")

        for i in range(self.__nb_unit_test):
            features = [("x"+str(i), [str(val) for val in range(0,random.randint(2,self.__nb_values))]) for i in range(random.randint(1,self.__nb_features))]
            targets = [("y"+str(i), [str(val) for val in range(0,random.randint(2,self.__nb_values))]) for i in range(random.randint(1,self.__nb_targets))]
            rules = []

            out = ""

            # Variables
            for var in range(len(features)):
                out += "FEATURE " + str(features[var][0]) + " "
                for val in features[var][1]:
                    out += str(val) + " "
                out = out[:-1] + "\n"

            for var in range(len(targets)):
                out += "TARGET " + str(targets[var][0]) + " "
                for val in targets[var][1]:
                    out += str(val) + " "
                out = out[:-1] + "\n"

            out += "\n"

           # eprint(out)
           # eprint(features)
           # eprint(targets)

            # Rules
            for j in range(random.randint(0,100)):
                r = self.random_rule(features, targets, self.__body_size)
                rules.append(r)
               # eprint(r)
                out += str(targets[r.get_head_variable()][0]) + "(" + str(targets[r.get_head_variable()][1][r.get_head_value()]) + ",T) :- "

                if len(r.get_body()) == 0:
                    out = out[:-4] + ".\n"
                else:
                    for var, val in r.get_body():
                        out += str(features[var][0]) + "(" + str(features[var][1][val]) + ",T-1), "
                    out = out[:-2] + ".\n"

                # Random empty line
                if random.randint(0,1):
                    out += "\n"

           # eprint(out)

            f = open(self.__tmp_file_path, "w")
            f.write(out)
            f.close()

            p = LogicProgram.load_from_file(self.__tmp_file_path)

            self.assertEqual(p.get_features(), features)
            self.assertEqual(p.get_targets(), targets)

            for r in rules:
                if r not in p.get_rules():
                    eprint(r.to_string())
                    eprint(p.to_string())
                self.assertTrue(r in p.get_rules())

            for r in p.get_rules():
                self.assertTrue(r in rules)

        if os.path.exists(self.__tmp_file_path):
            os.remove(self.__tmp_file_path)

    def test_random(self):
        print(">> LogicProgram.random(features, targets, rule_min_size, rule_max_size)")

        # No delay
        for i in range(self.__nb_unit_test):
            features = [("x"+str(i), [str(val) for val in range(0,random.randint(2,self.__nb_values))]) for i in range(random.randint(1,self.__nb_features))]
            targets = [("y"+str(i), [str(val) for val in range(0,random.randint(2,self.__nb_values))]) for i in range(random.randint(1,self.__nb_targets))]

            min_body_size = 0
            max_body_size = random.randint(min_body_size, len(features))

            p = LogicProgram.random(features, targets, min_body_size, max_body_size)
            #eprint(p.to_string())

            self.assertEqual(p.get_features(), features)
            self.assertEqual(p.get_targets(), targets)

            for r in p.get_rules():
                self.assertTrue(len(r.get_body()) >= min_body_size)
                self.assertTrue(len(r.get_body()) <= max_body_size)

            states = p.states()

            for s in states:
                for var in range(len(targets)):
                    matched = False
                    conclusion = -1
                    for r in p.get_rules():
                        if r.get_head_variable() == var and r.matches(s):
                            matched = True
                            if conclusion == -1: # stored first conclusion
                                conclusion = r.get_head_value()
                            else: # check conflict
                                self.assertEqual(conclusion, r.get_head_value())
                    self.assertTrue(matched)

            # No cross-matching
            for r1 in p.get_rules():
                for r2 in p.get_rules():
                    if r1 == r2 or r1.get_head_variable() != r2.get_head_variable():
                        continue
                    #eprint(r1)
                    #eprint(r2)
                    #eprint()
                    self.assertFalse(r1.cross_matches(r2))

    def test_to_string(self):
        print(">> LogicProgram.to_string(self)")

        #for i in range(self.__nb_unit_test):

        p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)

        result = p.to_string()
        #eprint(result)
        result = result.splitlines()
        #eprint(result)

        self.assertEqual(result[0], "{")
        self.assertEqual(result[1], "Features: " + str(p.get_features()))
        self.assertEqual(result[2], "Targets: " + str(p.get_targets()))
        self.assertEqual(result[3], "Rules:")

        for i in range(4,len(result)-1):
            rule_id = i - 4
            self.assertEqual(result[i], p.get_rules()[rule_id].to_string())

        self.assertEqual(result[-1], "}")
        self.assertEqual(p.to_string(), p.__str__())
        self.assertEqual(p.to_string(), p.__repr__())

    def test_logic_form(self):
        print(">> LogicProgram.logic_form(self)")

        #for i in range(self.__nb_unit_test):

        p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)

        result = p.logic_form()
        #eprint(result)
        result = result.splitlines()
        #eprint(result)

        for i in range(len(result)):
            #eprint(result[i])

            # Variable declaration
            if i < len(p.get_features()):
                variable = p.get_features()[i][0]
                values = ""
                for val in p.get_features()[i][1]:
                    values += " " + str(val)
                    #eprint(values)
                correct_string = "FEATURE " + variable + values
                #eprint("expected: " + correct_string)
                #eprint("got: " + result[i])
                self.assertEqual(result[i], correct_string)
                continue

            if i < len(p.get_features())+len(p.get_targets()):
                variable = p.get_targets()[i-len(p.get_features())][0]
                values = ""
                for val in p.get_targets()[i-len(p.get_features())][1]:
                    values += " " + str(val)
                    #eprint(values)
                correct_string = "TARGET " + variable + values
                #eprint("expected: " + correct_string)
                #eprint("got: " + result[i])
                self.assertEqual(result[i], correct_string)
                continue

            # Variable/rules empty line separator
            if i == len(p.get_features())+len(p.get_targets()):
                self.assertEqual(result[i],"")
                continue

            # Rule declaration
            rule_id = i - len(p.get_features())-len(p.get_targets()) - 1
            r = p.get_rules()[rule_id]

            #eprint(i)
            #eprint(result[i])
            #eprint(rule_id)
            #eprint(r.logic_form(p.get_features(), p.get_targets()))

            self.assertEqual(result[i], r.logic_form(p.get_features(), p.get_targets()))

    def test_compare(self):
        print(">> LogicProgram.compare(other)")

        # Equal programs
        for i in range(self.__nb_unit_test):
            p1 = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            p2 = LogicProgram(p1.get_features(), p1.get_targets(), p1.get_rules())
            common, missing, over = p1.compare(p2)

            self.assertEqual(len(common),len(p1.get_rules()))
            self.assertEqual(len(missing), 0)
            self.assertEqual(len(over), 0)

        # Equal programs reverse call
        for i in range(self.__nb_unit_test):
            p1 = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            p2 = LogicProgram(p1.get_features(), p1.get_targets(), p1.get_rules())
            common, missing, over = p2.compare(p1)

            self.assertEqual(len(common),len(p1.get_rules()))
            self.assertEqual(len(missing), 0)
            self.assertEqual(len(over), 0)

        # Random programs
        for i in range(self.__nb_unit_test):
            p1 = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            p2 = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            common, missing, over = p1.compare(p2)

            # All rules appear in a one of the set
            for r in p1.get_rules():
                self.assertTrue(r in common or r in missing)
            for r in p2.get_rules():
                self.assertTrue(r in common or r in over)

            # All rules are correctly placed
            for r in common:
                self.assertTrue(r in p1.get_rules() and r in p2.get_rules())
            for r in missing:
                self.assertTrue(r in p1.get_rules() and r not in p2.get_rules())
            for r in over:
                self.assertTrue(r not in p1.get_rules() and r in p2.get_rules())

    def test_get_rules_of(self):
        print(">> LogicProgram.get_rules_of(self, var)")

        for i in range(self.__nb_unit_test):
            p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            var = random.randint(0, len(p.get_targets())-1)
            val = random.randint(0, len(p.get_targets()[var][1])-1)

            rules = p.get_rules_of(var, val)

            for r in rules:
                self.assertEqual(r.get_head_variable(), var)
                self.assertTrue(r in p.get_rules())

            for r in p.get_rules():
                if r.get_head_variable() == var and r.get_head_value() == val:
                    self.assertTrue(r in rules)

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
