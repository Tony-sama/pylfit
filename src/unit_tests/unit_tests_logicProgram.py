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
from logicProgram import LogicProgram
from rule import Rule

#seed = random.randint(0,1000000)
#seed = 331710
#random.seed(seed)
#eprint("seed: ", seed)

class LogicProgramTest(unittest.TestCase):
    """
        Unit test of class LogicProgram from logicProgram.py
    """

    __nb_unit_test = 100

    __nb_variables = 3

    __nb_values = 2

    __max_delay = 5

    __nb_rules = 100

    __body_size = 10

    __tmp_file_path = "tmp/unit_test_logicProgram.tmp"

    #------------------
    # Test functions
    #------------------

    def test___init__(self):
        print(">> LogicProgram.__init__(self, variables, values, rules)")

        for i in range(self.__nb_unit_test):
            variables = ["x"+str(i) for i in range(random.randint(1,self.__nb_variables))]
            values = []
            rules = []

            for var in range(len(variables)):
                values.append([val for val in range(0,random.randint(2,self.__nb_values))])

            for j in range(random.randint(0,self.__nb_rules)):
                r = self.random_rule(variables, values, self.__body_size)
                rules.append(r)

            p = LogicProgram(variables, values, rules)

            self.assertEqual(p.get_variables(), variables)
            self.assertEqual(p.get_values(), values)
            self.assertEqual(p.get_rules(), rules)

    def test_load_from_file(self):
        print(">> LogicProgram.load_from_file(self, file_path)")

        for i in range(self.__nb_unit_test):
            variables = ["x"+str(i) for i in range(random.randint(1,self.__nb_variables))]
            values = []
            rules = []

            for var in range(len(variables)):
                values.append([str(val) for val in range(0,random.randint(2,self.__nb_values))])

            out = ""

            # Variables
            for var in range(len(variables)):
                out += "VAR x" + str(var) + " "
                for val in values[var]:
                    out += str(val) + " "
                out = out[:-1] + "\n"

            out += "\n"

            # Rules
            for j in range(random.randint(0,100)):
                r = self.random_rule(variables, values, self.__body_size)
                rules.append(r)
                out += "x"+str(r.get_head_variable()) + "(" + str(r.get_head_value()) + ",T) :- "

                if len(r.get_body()) == 0:
                    out = out[:-4] + ".\n"
                else:
                    for var, val in r.get_body():
                        out += "x" + str(var) + "(" + str(val) + ",T-1), "
                    out = out[:-2] + ".\n"

                # Random empty line
                if random.randint(0,1):
                    out += "\n"

            #eprint(out)

            f = open(self.__tmp_file_path, "w")
            f.write(out)
            f.close()

            p = LogicProgram.load_from_file(self.__tmp_file_path)

            self.assertEqual(p.get_variables(), variables)
            self.assertEqual(p.get_values(), values)

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
        print(">> LogicProgram.random(variables, values, rule_min_size, rule_max_size, delay=1)")

        # No delay
        for i in range(self.__nb_unit_test):
            variables = ["x"+str(i) for i in range(random.randint(1,self.__nb_variables))]
            values = []

            for var in range(len(variables)):
                values.append([val for val in range(random.randint(2,self.__nb_values))])

            min_body_size = 0
            max_body_size = random.randint(min_body_size, len(variables))

            p = LogicProgram.random(variables, values, min_body_size, max_body_size)
            #eprint(p.to_string())

            self.assertEqual(p.get_variables(), variables)
            self.assertEqual(p.get_values(), values)

            for r in p.get_rules():
                self.assertTrue(len(r.get_body()) >= min_body_size)
                self.assertTrue(len(r.get_body()) <= max_body_size)

            states = p.states()

            for s in states:
                for var in range(len(s)):
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

        # Delay
        for i in range(self.__nb_unit_test):
            variables = ["x"+str(i) for i in range(random.randint(1,self.__nb_variables))]
            values = []

            for var in range(len(variables)):
                values.append([val for val in range(random.randint(2,self.__nb_values))])

            min_body_size = 0
            max_body_size = random.randint(min_body_size, len(variables))
            delay = random.randint(1, self.__max_delay)

            p = LogicProgram.random(variables, values, min_body_size, max_body_size, delay)
            #eprint(p.logic_form())

            extended_variables = variables.copy()
            extended_values = values.copy()
            for d in range(1,delay):
                extended_variables += [var+"_"+str(d) for var in variables]
                extended_values += values

            self.assertEqual(p.get_variables(), variables)
            self.assertEqual(p.get_values(), values)

            for r in p.get_rules():
                self.assertTrue(len(r.get_body()) >= min_body_size)
                #self.assertTrue(len(r.get_body()) <= max_body_size)

            p_ = LogicProgram(extended_variables, extended_values,[])
            states = p_.states()

            for s in states:
                for var in range(len(variables)):
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

    def test_to_string(self):
        print(">> LogicProgram.to_string(self)")

        #for i in range(self.__nb_unit_test):

        p = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)

        result = p.to_string()
        #eprint(result)
        result = result.splitlines()
        #eprint(result)

        self.assertEqual(result[0], "{")
        self.assertEqual(result[1], "Variables: " + str(p.get_variables()))
        self.assertEqual(result[2], "Values: " + str(p.get_values()))
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

        p = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)

        result = p.logic_form()
        #eprint(result)
        result = result.splitlines()
        #eprint(result)

        for i in range(len(result)):
            #eprint(result[i])

            # Variable declaration
            if i < len(p.get_variables()):
                variable = p.get_variables()[i]
                values = ""
                for val in p.get_values()[i]:
                    values += " " + str(val)
                    #eprint(values)
                correct_string = "VAR " + variable + values
                #eprint("expected: " + correct_string)
                #eprint("got: " + result[i])
                self.assertEqual(result[i], correct_string)
                continue

            # Variable/rules empty line separator
            if i == len(p.get_variables()):
                self.assertEqual(result[i],"")
                continue

            # Rule declaration
            rule_id = i - len(p.get_variables()) - 1
            r = p.get_rules()[rule_id]

            self.assertEqual(result[i], r.logic_form(p.get_variables(), p.get_values()))


    def test_next(self):
        print(">> LogicProgram.next(self, state)")

        for i in range(self.__nb_unit_test):
            p = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)
            s1 = [random.randint(0,len(p.get_values()[var])-1) \
                  for var in range(len(p.get_variables()))]

            s2 = p.next(s1)

            for var in range(len(s2)):
                if s2[var] != -1:
                    exists = False
                    for r in p.get_rules():
                        if r.get_head_variable() == var \
                        and r.get_head_value() == s2[var] \
                        and r.matches(s1): # a corresponding rule holds
                            exists = True
                            break

                    self.assertTrue(exists)


    def test_generate_transitions(self):
        print(">> LogicProgram.generate_transitions(nb_states)")

        for i in range(self.__nb_unit_test):
            p = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)

            #eprint("var "+str(len(p.get_values()))+", val "+str(len(p.get_variables())))
            nb_states = random.randint(0, min(int(pow(len(p.get_values()), len(p.get_variables()))), 10))
            transitions = p.generate_transitions(nb_states)

            #transitions[1][1][0] = -1
            #eprint(transitions)

            self.assertEqual(len(transitions), nb_states)
            for s1, s2 in transitions:
                self.assertEqual(len(s1),len(s2))
                self.assertEqual(len(s2), len(p.get_variables()))
                for var in range(len(s1)):
                    self.assertTrue(s1[var] in p.get_values()[var])
                for var in range(len(s2)):
                    if s2[var] != -1:
                        exists = False
                        for r in p.get_rules():
                            if r.get_head_variable() == var \
                            and r.get_head_value() == s2[var] \
                            and r.matches(s1): # a corresponding rule holds
                                exists = True
                                break

                        self.assertTrue(exists)

    def test_generate_all_transitions(self):
        print(">> LogicProgram.generate_all_transitions(variable, state, transitions)")

        for i in range(self.__nb_unit_test):
            p = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)

            transitions = p.generate_all_transitions()

            #DBG
            #transitions[1][1][0] = -1
            #eprint(transitions)

            nb_total_state = reduce(mul, [len(val) for val in p.get_values()], 1)
            self.assertEqual(len(transitions), nb_total_state)
            for s1, s2 in transitions:
                self.assertEqual(len(s1),len(s2))
                self.assertEqual(len(s2), len(p.get_variables()))
                for var in range(len(s1)):
                    self.assertTrue(s1[var] in p.get_values()[var])
                for var in range(len(s2)):
                    if s2[var] != -1:
                        exists = False
                        for r in p.get_rules():
                            if r.get_head_variable() == var \
                            and r.get_head_value() == s2[var] \
                            and r.matches(s1): # a corresponding rule holds
                                exists = True
                                break

                        self.assertTrue(exists)

    def test_transitions_to_csv(self):
        print(">> LogicProgram.transitions_to_csv(filepath, transitions)")

        for i in range(self.__nb_unit_test):
            p = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)

            #eprint("var "+str(len(p.get_values()))+", val "+str(len(p.get_variables())))
            nb_states = random.randint(0, min(int(pow(len(p.get_values()), len(p.get_variables()))), 10))
            transitions = p.generate_transitions(nb_states)

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
                        row = [int(i) for i in row] # integer convertion
                        self.assertEqual(len(row),len(p.get_variables())*2)
                        self.assertEqual(row[:x_size], transitions[line_count-1][0])
                        self.assertEqual(row[x_size:], transitions[line_count-1][1])
                    line_count += 1

            if os.path.exists(self.__tmp_file_path):
                os.remove(self.__tmp_file_path)

    def test_next(self):
        print(">> LogicProgram.next(self, state)")

        for i in range(self.__nb_unit_test):
            p = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)
            size = random.randint(1, 10)
            time_serie = []
            for j in range(size):
                s = [random.randint(0,len(p.get_values()[var])-1) for var in range(len(p.get_variables()))]
                time_serie.append(s)

            time_serie_ = time_serie.copy()
            time_serie_.reverse()
            meta_state = [y for x in time_serie_ for y in x]

            s2 = p.next_state(time_serie)

            for var in range(len(s2)):
                if s2[var] != -1:
                    exists = False
                    for r in p.get_rules():
                        if r.get_head_variable() == var \
                        and r.get_head_value() == s2[var] \
                        and r.matches(meta_state): # a corresponding rule holds
                            exists = True
                            break

                    if not exists:
                        eprint(p)
                        eprint(time_serie)
                        eprint(meta_state)
                        eprint(s2)

                    self.assertTrue(exists)

    def test_generate_all_time_series(self):
        print(">> LogicProgram.generate_all_time_series(self, length)")

        for i in range(self.__nb_unit_test):
            p = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)

            size = random.randint(1, 5)
            time_serie = p.generate_all_time_series(size)

            #DBG
            #transitions[1][1][0] = -1
            #eprint(time_serie)

            nb_total_state = reduce(mul, [len(val) for val in p.get_values()], 1)
            self.assertEqual(len(time_serie), nb_total_state)
            for serie in time_serie:
                for j in range(1,len(serie)-1):
                    result = p.next_state(serie[:j])
                    #eprint(serie)
                    #eprint(serie[:j])
                    #eprint(serie[j])
                    #eprint(result)
                    for v in range(len(result)):
                        if result[v] == -1:
                            result[v] = serie[j-1][v]
                    self.assertEqual(result, serie[j])

    def test_compare(self):
        print(">> LogicProgram.compare(other)")

        # Equal programs
        for i in range(self.__nb_unit_test):
            p1 = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)
            p2 = LogicProgram(p1.get_variables(), p1.get_values(), p1.get_rules())
            common, missing, over = p1.compare(p2)

            self.assertEqual(len(common),len(p1.get_rules()))
            self.assertEqual(len(missing), 0)
            self.assertEqual(len(over), 0)

        # Equal programs reverse call
        for i in range(self.__nb_unit_test):
            p1 = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)
            p2 = LogicProgram(p1.get_variables(), p1.get_values(), p1.get_rules())
            common, missing, over = p2.compare(p1)

            self.assertEqual(len(common),len(p1.get_rules()))
            self.assertEqual(len(missing), 0)
            self.assertEqual(len(over), 0)

        # Random programs
        for i in range(self.__nb_unit_test):
            p1 = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)
            p2 = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)
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

    def test_states(self):
        print(">> LogicProgram.states(self)")

        for i in range(self.__nb_unit_test):
            p = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)

            states = p.states()

            nb_total_state = reduce(mul, [len(val) for val in p.get_values()], 1)

            self.assertEqual(len(states),nb_total_state)

            for s in states:
                self.assertEqual(states.count(s), 1)

    def test_precision(self):
        print(">> LogicProgram.precision(expected, predicted)")

        self.assertEqual(LogicProgram.precision([],[]), 1.0)

        # Equal programs
        for i in range(self.__nb_unit_test):
            nb_var = random.randint(1,self.__nb_variables)
            nb_values = random.randint(2,self.__nb_values)
            nb_states = random.randint(1,100)

            expected = []
            predicted = []

            for j in range(nb_states):
                s1 = [ random.randint(0,nb_values) for var in range(nb_var) ]
                s2 = [ random.randint(0,nb_values) for var in range(nb_var) ]
                s2_ = [ random.randint(0,nb_values) for var in range(nb_var) ]

                expected.append( (s1,s2) )
                predicted.append( (s1,s2_) )

            precision = LogicProgram.precision(expected, predicted)

            error = 0
            for i in range(len(expected)):
                s1, s2 = expected[i]

                for j in range(len(predicted)):
                    s1_, s2_ = predicted[j]

                    if s1 == s1_:
                        for var in range(len(s2)):
                            if s2_[var] != s2[var]:
                                error += 1
                        break
            
            #for i in range(len(expected)):
            #    s1, s2 = expected[i]
            #    s1_, s2_ = predicted[j]

            #    for k in range(len(s2)):
            #        if s2[k] != s2_[k]:
            #           error += 1

            total = nb_states * nb_var

            self.assertEqual( precision, 1.0 - (error / total) )

            # error of size
            state_id = random.randint(0, len(expected)-1)
            modif = random.randint(1,len(expected[state_id]))
            expected[state_id] = ( expected[state_id][0][:-modif], expected[state_id][1] )

            self.assertRaises(ValueError, LogicProgram.precision, expected, predicted)

    def test_get_rules_of(self):
        print(">> LogicProgram.get_rules_of(self, var)")

        for i in range(self.__nb_unit_test):
            p = self.random_program(self.__nb_variables, self.__nb_values, self.__body_size)
            var = random.randint(0, len(p.get_variables())-1)
            val = random.randint(0, len(p.get_values()[var])-1)

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


    def random_rule(self, variables, values, body_size):
        var = random.randint(0,len(variables)-1)
        val = random.randint(0,len(values[var])-1)
        body = []
        conditions = []

        for j in range(0, random.randint(0,body_size)):
            var = random.randint(0,len(variables)-1)
            val = random.randint(0,len(values[var])-1)
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
