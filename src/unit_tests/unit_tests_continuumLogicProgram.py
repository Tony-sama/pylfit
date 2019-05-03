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

from utils import eprint
from continuumLogicProgram import ContinuumLogicProgram
from continuum import Continuum
from continuumRule import ContinuumRule

#seed = random.randint(0,1000000)
#seed = 438467
#random.seed(seed)
#eprint("seed: ", seed)

class ContinuumLogicProgramTest(unittest.TestCase):
    """
        Unit test of class ContinuumLogicProgram from continuumlogicProgram.py
    """

    __nb_unit_test = 100

    __max_variables = 2

    __min_epsilon = 0.3

    """ must be < __max_value"""
    __min_value = -100.0

    """ must be > __min_value"""
    __max_value = 100.0

    __min_domain_size = 1.0

    __min_continuum_size = 0.01

    __nb_rules = 100

    __body_size = 10

    __max_delay = 2

    __tmp_file_path = "tmp/unit_test_continuumlogicProgram.tmp"

    #------------------
    # Constructors
    #------------------

    def test___init__(self):
        print(">> ContinuumLogicProgram.__init__(self, variables, domains, rules)")

        for i in range(self.__nb_unit_test):
            variables, domains = self.random_system()
            rules = []

            for j in range(random.randint(0,self.__nb_rules)):
                r = self.random_rule(variables, domains)
                rules.append(r)

            p = ContinuumLogicProgram(variables, domains, rules)

            self.assertEqual(p.get_variables(), variables)
            self.assertEqual(p.get_domains(), domains)
            self.assertEqual(p.get_rules(), rules)

            modif = random.randint(1,len(variables))
            self.assertRaises(ValueError, ContinuumLogicProgram, variables, domains[:-modif], rules)
            for var in range(0, modif):
                domains.append(Continuum.random(self.__min_value, self.__max_value, self.__min_domain_size))
            self.assertRaises(ValueError, ContinuumLogicProgram, variables, domains, rules)

    def test_random(self):
        print(">> ContinuumLogicProgram.random(variables, values, rule_min_size, rule_max_size, epsilon, delay=1)")

        # No delay
        for i in range(self.__nb_unit_test):
            variables, domains = self.random_system()

            min_body_size = 0
            max_body_size = random.randint(min_body_size, len(variables))

            # Valid and realistic epsilon
            epsilon = round(random.uniform(0.1,1.0), 2)
            while epsilon == 1.0:
                epsilon = round(random.uniform(0.1,1.0), 2)

            p = ContinuumLogicProgram.random(variables, domains, min_body_size, max_body_size, epsilon)
            #eprint(p.to_string())

            self.assertEqual(p.get_variables(), variables)
            self.assertEqual(p.get_domains(), domains)

            for r in p.get_rules():
                self.assertTrue(len(r.get_body()) >= min_body_size)
                self.assertTrue(len(r.get_body()) <= max_body_size)

            states = ContinuumLogicProgram.states(domains, epsilon)

            for s in states:
                for var in range(len(s)):
                    matched = False
                    conclusion = -1
                    for r in p.get_rules():
                        if r.get_head_variable() == var and r.matches(s):
                            matched = True
                            #if conclusion == -1: # stored first conclusion
                            #    conclusion = r.get_head_value()
                            #else: # check conflict
                            #    self.assertEqual(conclusion, r.get_head_value())
                    self.assertTrue(matched)
        # Delay
        for i in range(self.__nb_unit_test):
            variables, domains = self.random_system()

            min_body_size = 0
            max_body_size = random.randint(min_body_size, len(variables))
            delay = random.randint(1, self.__max_delay)

            # Valid and realistic epsilon
            epsilon = round(random.uniform(0.1,1.0), 2)
            while epsilon == 1.0:
                epsilon = round(random.uniform(0.1,1.0), 2)

            p = ContinuumLogicProgram.random(variables, domains, min_body_size, max_body_size, epsilon, delay)
            #eprint(p.logic_form())

            extended_variables = variables.copy()
            extended_domains = domains.copy()
            for d in range(1,delay):
                extended_variables += [var+"_"+str(d) for var in variables]
                extended_domains += domains

            self.assertEqual(p.get_variables(), variables)
            self.assertEqual(p.get_domains(), domains)

            for r in p.get_rules():
                self.assertTrue(len(r.get_body()) >= min_body_size)
                self.assertTrue(len(r.get_body()) <= max_body_size)

            states = ContinuumLogicProgram.states(extended_domains, epsilon)

            for s in states:
                for var in range(len(variables)):
                    matched = False
                    conclusion = -1
                    for r in p.get_rules():
                        if r.get_head_variable() == var and r.matches(s):
                            matched = True
                            break
                            #if conclusion == -1: # stored first conclusion
                            #    conclusion = r.get_head_value()
                            #else: # check conflict
                            #    self.assertEqual(conclusion, r.get_head_value())
                    #if not matched:
                        #eprint(s)
                        #eprint(p)
                    self.assertTrue(matched)

    #--------------
    # Observers
    #--------------

    #--------------
    # Methods
    #--------------

    def test_to_string(self):
        print(">> ContinuumLogicProgram.to_string(self)")

        for i in range(self.__nb_unit_test):
            p = self.random_program()

            result = p.to_string()
            #eprint(result)
            result = result.splitlines()
            #eprint(result)

            self.assertEqual(result[0], "{")
            self.assertEqual(result[1], "Variables: " + str(p.get_variables()))
            self.assertEqual(result[2], "Domains: " + str(p.get_domains()))
            self.assertEqual(result[3], "Rules:")

            for i in range(4,len(result)-1):
                rule_id = i - 4
                self.assertEqual(result[i], p.get_rules()[rule_id].to_string())

            self.assertEqual(result[-1], "}")

            self.assertEqual(p.to_string(), p.__str__())
            self.assertEqual(p.to_string(), p.__repr__())

    def test_logic_form(self):
        print(">> ContinuumLogicProgram.logic_form(self)")

        for i in range(self.__nb_unit_test):

            p = self.random_program()

            result = p.logic_form()
            #eprint(result)
            result = result.splitlines()
            #eprint(result)

            for j in range(len(result)):
                #eprint(result[i])

                # Variable declaration
                if j < len(p.get_variables()):
                    variable = p.get_variables()[j]
                    domain = p.get_domains()[j]

                    correct_string = "VAR " + variable + " " + str(domain.get_min_value()) + " " + str(domain.get_max_value())
                    #eprint("expected: " + correct_string)
                    #eprint("got: " + result[i])
                    self.assertEqual(result[j], correct_string)
                    continue

                # Variable/rules empty line separator
                if j == len(p.get_variables()):
                    self.assertEqual(result[j],"")
                    continue

                # Rule declaration
                rule_id = j - len(p.get_variables()) - 1
                r = p.get_rules()[rule_id]

                self.assertEqual(result[j], r.logic_form(p.get_variables()))

    def test_next(self):
        print(">> ContinuumLogicProgram.next(self, state)")

        for i in range(self.__nb_unit_test):
            variables, domains = self.random_system()
            p = self.random_program(variables, domains)
            s1 = self.random_state(variables, domains)

            s2 = p.next(s1)

            for var in range(len(s2)):
                if s2[var] is not None:
                    exists = False
                    for r in p.get_rules():
                        if r.get_head_variable() == var \
                        and r.get_head_value() == s2[var] \
                        and r.matches(s1): # a corresponding rule holds
                            exists = True
                            break

                    self.assertTrue(exists)
                else:
                    exists = False
                    for r in p.get_rules():
                        if r.get_head_variable() == var \
                        and r.get_head_value() == s2[var] \
                        and r.matches(s1): # a corresponding rule holds
                            exists = True
                            break

                    self.assertFalse(exists)

    def test_states(self):
        print(">> ContinuumLogicProgram.states(self, epsilon)")

        for i in range(self.__nb_unit_test):
            variables, domains = self.random_system()
            p = self.random_program(variables, domains)

            # bad epsilon
            epsilon = random.uniform(-100, 0)

            self.assertRaises(ValueError, p.states, domains, epsilon)

            epsilon = random.uniform(0, 0.1)
            while epsilon == 0.0 or epsilon == 0.1:
                epsilon = random.uniform(0, 0.1)
            self.assertRaises(ValueError, p.states, domains, epsilon)

            epsilon = random.uniform(1.0, 100)
            while epsilon == 1.0:
                epsilon = random.uniform(1.0, 100)

            self.assertRaises(ValueError, p.states, domains, epsilon)

            # Valid and realistic epsilon
            epsilon = round(random.uniform(0.1,1.0), 2)
            while epsilon == 1.0:
                epsilon = round(random.uniform(0.1,1.0), 2)
            #eprint("epsilon: ", epsilon)

            # None value in domains
            var = random.randint(0, len(domains)-1)
            domains_none = domains.copy()
            domains_none[var] = None
            states = p.states(domains_none, epsilon)
            for s in states:
                self.assertTrue(s[var] is None)
                for v in len(s):
                    if s[v] != var:
                        self.assertFalse(s[v] is None)

            states = p.states(domains, epsilon)

            nb_total_state = 1
            for var in range(len(p.get_domains())):
                d = p.get_domains()[var]
                min = d.get_min_value()
                max = d.get_max_value()
                step = epsilon * (max - min)
                values = [min+(step*i) for i in range( int(1 / epsilon) )]
                if values[-1] != max:
                    values.append(max)

                nb_values = len(values)

                if not d.min_included:
                    nb_values -= 1
                if not d.max_included:
                    nb_values -= 1

                nb_total_state *= nb_values

            self.assertEqual(len(states),nb_total_state)
            #eprint("Nb states: ", nb_total_state)

            for s in states:
                self.assertEqual(states.count(s), 1)

                for var in range(len(s)):
                    d = p.get_domains()[var]
                    min = d.get_min_value()
                    max = d.get_max_value()
                    step = epsilon * (max - min)
                    values = [min+(step*i) for i in range( int(1.0 / epsilon) )]
                    if values[-1] != max:
                        values.append(max)
                    #eprint(s[var])
                    #eprint(values)
                    #eprint()
                    appears = False
                    for val in values:
                        if abs(s[var] - val) <= 0.0001:
                            appears = True
                            break
                    self.assertTrue(appears)

            for s1 in states:
                for s2 in states:
                    if s1 != s2:
                        for idx in range(len(s1)):
                            difference = abs(s1[idx] - s2[idx])
                            min = domains[idx].get_min_value()
                            max = domains[idx].get_max_value()
                            if difference != 0:
                                #eprint("min: ", min)
                                #eprint("max: ", max)
                                #eprint("s1: ", s1[idx])
                                #eprint("s2: ", s2[idx])
                                difference = abs(s1[idx] - s2[idx]) / (max - min)
                                #eprint("epsilon: ", round(epsilon,3))
                                #eprint("diff: ", round(difference,3))
                                #eprint()
                            self.assertTrue( difference <= 0.0001 or (round(difference,3) > 0 and round(difference,3) <= 1.0 and round(difference,3) >= round(epsilon,3)) )



    def test_generate_all_transitions(self):
        print(">> ContinuumLogicProgram.generate_all_transitions(epsilon)")

        for i in range(self.__nb_unit_test):
            variables, domains = self.random_system()
            p = self.random_program(variables, domains)

            # bad epsilon
            epsilon = random.uniform(-100, 0)

            self.assertRaises(ValueError, p.generate_all_transitions, epsilon)

            epsilon = random.uniform(0, 0.1)
            while epsilon == 0.0 or epsilon == 0.1:
                epsilon = random.uniform(0, 0.1)
            self.assertRaises(ValueError, p.generate_all_transitions, epsilon)

            epsilon = random.uniform(1.0, 100)
            while epsilon == 1.0:
                epsilon = random.uniform(1.0, 100)

            self.assertRaises(ValueError, p.generate_all_transitions, epsilon)

            # Valid and realistic epsilon
            epsilon = round(random.uniform(self.__min_epsilon,1.0), 2)
            while epsilon == 1.0:
                epsilon = round(random.uniform(self.__min_epsilon,1.0), 2)
            #eprint("epsilon: ", epsilon)

            transitions = p.generate_all_transitions(epsilon)

            #eprint(p.to_string())
            #eprint("transitions: ", transitions)
            #eprint("Nb transitions: ", len(transitions))
            #eprint(transitions)

            # All initial state appears
            S = ContinuumLogicProgram.states(p.get_domains(), epsilon)
            init = [s1 for s1, s2 in transitions]
            #eprint("init: ", init)
            #eprint("S: ", S)
            #eprint()
            for s in S:
                s2 = p.next(s)
                if None in s2: # uncomplete states ignored
                    continue
                appears = False
                for s1 in init:
                    different = False
                    for idx in range(len(s1)):
                        if abs(s1[idx]-s[idx]) > 0.0001:
                            different = True
                            break

                        if not different:
                            appears = True
                            break

                self.assertTrue(appears)

            # All transitions appears and are valid
            for s1, s2 in transitions:

                # all transitions from s1
                S = p.next(s1)
                S = ContinuumLogicProgram.states(S, epsilon)

                # s2 is a transion of s1
                self.assertTrue(s2 in S)

                # All transitions from s1 appears
                for s3 in S:
                    self.assertTrue( [s1,s3] in transitions)

    def test_precision(self):
        print(">> ContinuumLogicProgram.precision(expected, predicted)")

        self.assertEqual(ContinuumLogicProgram.precision([],[]), 1.0)

        # Equal programs
        for i in range(self.__nb_unit_test):
            variables, domains = self.random_system()
            nb_states = random.randint(1,100)

            expected = []
            predicted = []

            for j in range(nb_states):
                s1 = [ random.uniform(d.get_min_value(),d.get_max_value()) for d in domains ]
                s2 = [ random.uniform(d.get_min_value(),d.get_max_value()) for d in domains  ]
                s2_ = [ Continuum.random(d.get_min_value(), d.get_max_value()) for d in domains ]

                expected.append( (s1,s2) )
                predicted.append( (s1,s2_) )

            precision = ContinuumLogicProgram.precision(expected, predicted)

            error = 0
            for j in range(len(expected)):
                s1, s2 = expected[j]
                s1_, s2_ = predicted[j]

                for k in range(len(s2)):
                    if not s2_[k].includes(s2[k]):
                        error += 1

            total = nb_states * len(variables)

            self.assertEqual( precision, 1.0 - (error / total) )

            #eprint("precision: ", precision)

            # error of size
            state_id = random.randint(0, len(expected)-1)
            modif = random.randint(1,len(expected[state_id]))
            expected[state_id] = ( expected[state_id][0][:-modif], expected[state_id][1] )

            self.assertRaises(ValueError, ContinuumLogicProgram.precision, expected, predicted)


    def test_transitions_to_csv(self):
        print(">> ContinuumLogicProgram.transitions_to_csv(filepath, transitions)")

        for i in range(self.__nb_unit_test):
            p = self.random_program()

            #eprint("var "+str(len(p.get_values()))+", val "+str(len(p.get_variables())))
            epsilon = random.uniform(0.1,1.0)
            transitions = p.generate_all_transitions(epsilon)

            p.transitions_to_csv(self.__tmp_file_path, transitions)

            with open(self.__tmp_file_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                line_count = 0
                x_size = 0

                for row in csv_reader:
                    if line_count == 0:
                        x_size = row.index("y0")
                        self.assertEqual(x_size, len(p.get_variables()))
                        x = row[:x_size]
                        y = row[x_size:]
                        self.assertEqual(len(x), len(p.get_variables()))
                        self.assertEqual(len(y), len(p.get_variables()))
                    else:
                        row = [float(i) for i in row] # integer convertion
                        self.assertEqual(len(row),len(p.get_variables())*2)
                        self.assertEqual(row[:x_size], transitions[line_count-1][0])
                        self.assertEqual(row[x_size:], transitions[line_count-1][1])
                    line_count += 1

            if os.path.exists(self.__tmp_file_path):
                os.remove(self.__tmp_file_path)

    def test_get_rules_of(self):
        print(">> ContinuumLogicProgram.get_rules_of(self, var)")

        for i in range(self.__nb_unit_test):
            p = self.random_program()
            var = random.randint(0, len(p.get_variables()))

            rules = p.get_rules_of(var)

            for r in rules:
                self.assertEqual(r.get_head_variable(), var)
                self.assertTrue(r in p.get_rules())

            for r in p.get_rules():
                if r.get_head_variable() == var:
                    self.assertTrue(r in rules)
    #------------------
    # Tool functions
    #------------------

    def random_system(self):
        # generates variables/domains
        nb_variables = random.randint(1, self.__max_variables)
        variables = ["x"+str(var) for var in range(nb_variables)]
        domains = [ Continuum.random(self.__min_value, self.__max_value, self.__min_domain_size) for var in variables ]

        return variables, domains

    def random_rule(self, variables, domains):
        var = random.randint(0,len(variables)-1)
        var_domain = domains[var]
        val = Continuum.random(var_domain.get_min_value(), var_domain.get_max_value(), self.__min_continuum_size)
        min_size = random.randint(0, len(variables))
        max_size = random.randint(min_size, len(variables))

        return ContinuumRule.random(var, val, variables, domains, min_size, max_size)

    def random_state(self, variables, domains):
        return [random.uniform(domains[var].get_min_value(),domains[var].get_max_value()) for var in range(len(variables))]


    def random_program(self, variables=None, domains=None):
        if variables is None:
            variables, domains = self.random_system()
        rules = []

        for j in range(random.randint(0,self.__nb_rules)):
            r = self.random_rule(variables, domains)
            rules.append(r)

        return ContinuumLogicProgram(variables, domains, rules)


if __name__ == '__main__':
    """ Main """

    unittest.main()
