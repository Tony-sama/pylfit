#-----------------------
# @author: Tony Ribeiro
# @created: 2019/11/25
# @updated: 2019/11/25
#
# @desc: PyLFIT unit test script
#
#-----------------------

import unittest
import random
import os

from pylfit.utils import eprint, load_tabular_data_from_csv
from pylfit.algorithms.probabilizer import Probabilizer
from pylfit.algorithms.gula import GULA
from pylfit.objects.rule import Rule
from pylfit.objects.logicProgram import LogicProgram
from pylfit.semantics.synchronous import Synchronous

import itertools
import math

random.seed(0)


class ProbabilizerTest(unittest.TestCase):
    """
        Unit test of class Probabilizer from probabilizer.py
    """

    __nb_unit_test = 100

    __nb_features = 5

    __nb_targets = 3

    __nb_values = 3

    __body_size = 10

    #------------------
    # Test functions
    #------------------

    def test_encode(self):
        print(">> Probabilizer.encode(transitions, synchronous_independant)")

        self.assertEqual(Probabilizer.encode([]),[])

        for i in range(self.__nb_unit_test):
            #eprint(i,"/",self.__nb_unit_test)

            p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            t = self.random_independant_transitions(p)

            #eprint(t)

            enco = Probabilizer.encode(t)

            # All initial state appear
            init_states = [s1 for s1,s2 in t]
            init_enco = [s1 for s1,s2 in enco]
            for s in init_states:
                self.assertTrue(s in init_enco)

            # Extract occurences of each transition
            next_states = dict()
            nb_transitions_from = dict()
            for (i,j) in t:
                s_i = tuple(i)
                s_j = tuple(j)
                # new init state
                if s_i not in next_states:
                    next_states[s_i] = dict()
                    nb_transitions_from[s_i] = 0
                # new next state
                if s_j not in next_states[s_i]:
                    next_states[s_i][s_j] = (s_i,s_j,0)

                (_, _, o) = next_states[s_i][s_j]
                next_states[s_i][s_j] = (s_i,s_j,o+1)
                nb_transitions_from[s_i] += 1

            #eprint(enco)

            # Encoded ratio correspond to observations
            for s1, s2 in enco:
                for var in range(0,len(s2)):
                    tokens = s2[var].split(",")
                    value = p.get_targets()[var][1].index(tokens[0])
                    ratio = tokens[1]
                    tokens = ratio.split("/")
                    top = int(tokens[0])
                    bot = int(tokens[1])

                    # count occurences of value after s1
                    occurences = 0
                    s_i = tuple(s1)
                    for s_j in next_states[s_i]: # For each transition from s1
                        (_, j, o) = next_states[s_i][s_j]
                        if j[var] == p.get_targets()[var][1][value]:
                            occurences += o

                    self.assertEqual(top/bot, occurences/nb_transitions_from[s_i])



    def test_fit(self):
        print(">> Probabilizer.fit(variables, values, transitions, complete, synchronous_independant)")

        # No transitions
        p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
        p_ = Probabilizer.fit([], p.get_features(), p.get_targets())
        self.assertEqual(p_.get_features(), p.get_features())
        self.assertEqual(p_.get_targets(), [(i[0],[]) for i in p.get_targets()])
        rules = []
        self.assertEqual(p_.get_rules(),rules)

        for i in range(self.__nb_unit_test):
            #eprint(i,"/",self.__nb_unit_test)

            p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            t = self.random_independant_transitions(p)

            eprint("Input: ",t)
            t = Probabilizer.encode_transitions_set(t, p.get_features(), p.get_targets())
            #eprint(t)

            p_ = Probabilizer.fit(t, p.get_features(), p.get_targets())

            #eprint(p_.logic_form())

            probability_encoded_input = Probabilizer.encode(t)
            probability_encoded_targets = Probabilizer.conclusion_values(p.get_targets(), probability_encoded_input)

            probability_encoded_input = [([p_.get_features()[var][1][i[var]] for var in range(len(i))],j) for (i,j) in probability_encoded_input]

            eprint(probability_encoded_input)

            # Only original transitions are produced from observed states
            for s1, s2 in probability_encoded_input:
                next = Synchronous.next(p_,s1)
                #eprint(s2)
                #eprint(next)
                #eprint(conclusion_values)
                next = [tuple(s) for s in next]
                eprint(s1)
                eprint(next)
                eprint(s2)
                self.assertTrue(s2 in next)
                for s in next:
                    self.assertTrue((s1,s) in probability_encoded_input)



            #exit()
    #------------------
    # Tool functions
    #------------------

    def random_independant_transitions(self, p):

        # Generate random transitions
        t = []
        states = p.feature_states()
        for s1 in states:

            ratio = [[0 for val in range(0,len(p.get_targets()[var]))] for var in range(0,len(p.get_targets()))]
            for var in range(0,len(p.get_targets())):
                can_appear = [random.choice([True,False]) for val in range(0,len(p.get_targets()[var]))]
                # at least one value possible
                if True not in can_appear:
                    can_appear[random.randint(0,len(can_appear)-1)] = True

                # Only x10% proba
                total = 10
                #eprint(total)
                while total > 0:
                    for val in range(0,len(p.get_targets()[var])):
                        if can_appear[val]:
                            added = random.randint(1,total)
                            ratio[var][val] += added # proba of appearance
                            total -= added
                            #eprint(total)
                        if total <= 0:
                            break

                # GCD ratio simplification
                rates = [i for i in ratio[var] if i != 0]
                if len(rates) < 2:
                    for val in range(0,len(p.get_targets()[var])):
                        if ratio[var][val] == 10:
                            ratio[var][val] = 1
                else:
                    gcd = int(self.find_gcd(rates[0],rates[1]))

                    for i in range(2,len(rates)):
                        gcd = int(self.find_gcd(gcd,rates[i]))

                    for val in range(0,len(p.get_targets()[var])):
                        ratio[var][val] = int(ratio[var][val] / gcd)

            # DBG
            #eprint(ratio)

            domains = [set() for var in p.get_targets()]
            for var in range(0,len(p.get_targets())):
                for val in range(0,len(p.get_targets()[var])):
                    if ratio[var][val] > 0:
                        domains[var].add(val)

            #eprint(domains)

            # generate all combination of conclusions
            possible = [i for i in list(itertools.product(*domains))]

            #eprint(possible)

            for s2 in possible:
                ratio_s2 = 1
                for var in range(0,len(s2)):
                    val = s2[var]
                    ratio_s2 *= ratio[var][val]

                t.extend([[s1, tuple([p.get_targets()[var][1][val] for val in s2])] for i in range(0,ratio_s2)])

        return t

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

    def random_constraint(self, variables, values, body_size):
        var = -1
        val = -1
        body = []
        conditions = []

        for j in range(0, random.randint(0,body_size)):
            var = random.randint(0,len(variables)*2-1)
            val = random.choice(values[var%len(variables)])
            if var not in conditions:
                body.append( (var, val) )
                conditions.append(var)

        return  Rule(var,val,len(variables)*2,body)


    def random_program(self, nb_features, nb_targets, nb_values, body_size):
        features = [("x"+str(i), ["val_"+str(val) for val in range(0,random.randint(2,nb_values))]) for i in range(random.randint(1,nb_features))]
        targets = [("y"+str(i), ["val_"+str(val) for val in range(0,random.randint(2,nb_values))]) for i in range(random.randint(1,nb_targets))]
        rules = []

        for j in range(random.randint(0,100)):
            r = self.random_rule(features, targets, body_size)
            rules.append(r)

        return LogicProgram(features, targets, rules)

    def find_gcd(self, x, y):
        while(y):
            x, y = y, x % y

        return x


if __name__ == '__main__':
    """ Main """

    unittest.main()
