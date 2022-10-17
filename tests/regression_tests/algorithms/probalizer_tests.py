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
import sys

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from pylfit.utils import eprint
from pylfit.algorithms.probalizer import Probalizer
from pylfit.algorithms.gula import GULA
from pylfit.objects.rule import Rule
from pylfit.semantics.synchronous import Synchronous
from pylfit.semantics.synchronousConstrained import SynchronousConstrained
from pylfit.datasets import DiscreteStateTransitionsDataset

from tests_generator import random_DMVLP, random_DiscreteStateTransitionsDataset

import itertools
import math

random.seed(0)


class ProbalizerTest(unittest.TestCase):
    """
        Unit test of class Probalizer from probabilizer.py
    """

    _nb_tests = 10

    _nb_transitions = 100

    _nb_features = 3

    _nb_targets = 3

    _nb_feature_values = 2

    _nb_target_values = 2

    #------------------
    # Test functions
    #------------------

    def test_encode(self):
        print(">> Probalizer.encode(transitions, synchronous_independant)")

        self.assertEqual(Probalizer.encode([]),[])

        for i in range(self._nb_tests):
            #eprint(i,"/",self.__nb_unit_test)

            p = random_DMVLP(self._nb_features, self._nb_targets, self._nb_feature_values, self._nb_target_values, "gula") #self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
            t = self.random_independant_transitions(p)

            #eprint(t)

            enco = Probalizer.encode(t)

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
                    value = p.targets[var][1].index(tokens[0])
                    ratio = tokens[1]
                    tokens = ratio.split("/")
                    top = int(tokens[0])
                    bot = int(tokens[1])

                    # count occurences of value after s1
                    occurences = 0
                    s_i = tuple(s1)
                    for s_j in next_states[s_i]: # For each transition from s1
                        (_, j, o) = next_states[s_i][s_j]
                        if j[var] == p.targets[var][1][value]:
                            occurences += o

                    self.assertEqual(top/bot, occurences/nb_transitions_from[s_i])



    def test_fit(self):
        print(">> Probalizer.fit(variables, values, transitions, complete, synchronous_independant)")

        # No transitions
        dataset = random_DiscreteStateTransitionsDataset( \
        nb_transitions=0, \
        nb_features=random.randint(1,self._nb_features), \
        nb_targets=random.randint(1,self._nb_targets), \
        max_feature_values=self._nb_feature_values, max_target_values=self._nb_target_values)

        proba_encoded_targets, rules, constraints = Probalizer.fit(dataset, dataset.features, dataset.targets)

        self.assertEqual(proba_encoded_targets, [(var, []) for (var, vals) in dataset.targets])
        self.assertEqual(rules,[])
        self.assertEqual(constraints,[])

        for i in range(self._nb_tests):
            #eprint(i,"/",self.__nb_unit_test)

            p = random_DMVLP(self._nb_features, self._nb_targets, self._nb_feature_values, self._nb_target_values, "gula")
            t = self.random_independant_transitions(p)

            #eprint("Input: ",t)
            #eprint(p.targets)
            dataset = DiscreteStateTransitionsDataset(t, p.features, p.targets)
            #eprint(t)

            for complete in [True,False]:
                for synchronous_independant in [True]:
                    for verbose in [0,1]:
                        for threads in [1,2]:
                            proba_encoded_targets, rules, constraints = Probalizer.fit(dataset, complete, synchronous_independant, threads)

                            #eprint(p_.logic_form())

                            probability_encoded_input = Probalizer.encode(t)
                            probability_encoded_targets = Probalizer.conclusion_values(p.targets, probability_encoded_input)

                            self.assertEqual(proba_encoded_targets, probability_encoded_targets)

                            final_encoded_input = DiscreteStateTransitionsDataset(probability_encoded_input, dataset.features, probability_encoded_targets)

                            #eprint(final_encoded_input)

                            # Only original transitions are produced from observed states
                            for s1, s2 in final_encoded_input.data:
                                s1_encoded = []
                                for var_id, val in enumerate(s1):
                                    val_id = final_encoded_input.features[var_id][1].index(str(val))
                                    s1_encoded.append(val_id)

                                if not synchronous_independant:
                                    encoded_target_states = SynchronousConstrained.next(s1_encoded, final_encoded_input.targets, rules, constraints).keys()
                                else:
                                    encoded_target_states = Synchronous.next(s1_encoded, final_encoded_input.targets, rules).keys()

                                target_states = []
                                for s in encoded_target_states:
                                    target_state = []
                                    for var_id, val_id in enumerate(s):
                                        #eprint(var_id, val_id)
                                        if val_id == -1:
                                            target_state.append("?")
                                        else:
                                            target_state.append(final_encoded_input.targets[var_id][1][val_id])

                                    target_states.append(target_state)

                                #eprint(s2)
                                #eprint(next)
                                #eprint(conclusion_values)
                                #eprint("------------------------------------")
                                #eprint(s1)
                                #eprint(target_states)
                                #eprint(s2)
                                self.assertTrue(list(s2) in target_states)
                                for s in target_states:
                                    self.assertTrue((list(s1),list(s)) in [(list(s1_),list(s2_)) for (s1_,s2_) in final_encoded_input.data])

                            # Ratio of learn rules fit to data



            #exit()
    #------------------
    # Tool functions
    #------------------

    def random_independant_transitions(self, p):

        # Generate random transitions
        t = []
        states = p.feature_states()
        for s1 in states:

            ratio = [[0 for val in range(0,len(p.targets[var][1]))] for var in range(0,len(p.targets))]
            for var in range(0,len(p.targets)):
                can_appear = [random.choice([True,False]) for val in range(0,len(p.targets[var][1]))]
                # at least one value possible
                if True not in can_appear:
                    can_appear[random.randint(0,len(can_appear)-1)] = True

                # Only x10% proba
                total = 10
                #eprint(total)
                while total > 0:
                    for val in range(0,len(p.targets[var][1])):
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
                    for val in range(0,len(p.targets[var][1])):
                        if ratio[var][val] == 10:
                            ratio[var][val] = 1
                else:
                    gcd = int(self.find_gcd(rates[0],rates[1]))

                    for i in range(2,len(rates)):
                        gcd = int(self.find_gcd(gcd,rates[i]))

                    for val in range(0,len(p.targets[var][1])):
                        ratio[var][val] = int(ratio[var][val] / gcd)

            # DBG
            #eprint(ratio)

            domains = [set() for var in p.targets]
            for var in range(0,len(p.targets)):
                for val in range(0,len(p.targets[var][1])):
                    if ratio[var][val] > 0:
                        domains[var].add(val)
            #eprint(p.targets)
            #eprint(domains)

            # generate all combination of conclusions
            possible = [i for i in list(itertools.product(*domains))]

            #eprint(possible)

            for s2 in possible:
                ratio_s2 = 1
                for var in range(0,len(s2)):
                    val = s2[var]
                    ratio_s2 *= ratio[var][val]

                decoded_s2 = [p.targets[var][1][s2[var]] for var in range(0,len(s2))]
                t.extend([(s1, decoded_s2) for i in range(0,ratio_s2)])

        return t

    def find_gcd(self, x, y):
        while(y):
            x, y = y, x % y

        return x


if __name__ == '__main__':
    """ Main """

    unittest.main()
