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
import itertools

sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')
sys.path.insert(0, 'src/semantics')

from utils import eprint
from lfkt import LFkT
from rule import Rule
from logicProgram import LogicProgram
from synchronous import Synchronous

random.seed(0)


class LFkTTest(unittest.TestCase):
    """
        Unit test of class LFkT from lfkt.py
    """

    __nb_unit_test = 10

    __nb_features = 3

    __nb_targets = 3

    __nb_values = 2

    __max_delay = 3

    __body_size = 10

    __tmp_file_path = "tmp/unit_test_lfkt.tmp"

    #------------------
    # Test functions
    #------------------

    def test_fit(self):
        print(">> LFkT.fit(variables, values, time_series)")

        # No transitions
        p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)

        min_body_size = 0
        max_body_size = random.randint(min_body_size, len(p.get_features()))
        delay_original = random.randint(2, self.__max_delay)

        features = []
        targets = p.get_targets()

        for d in range(1,delay_original+1):
            features += [(var+"_"+str(d), vals) for var,vals in p.get_features()]

        p = LogicProgram.random(features, targets, min_body_size, max_body_size)
        p_ = LFkT.fit([], p.get_features(), p.get_targets())
        self.assertEqual(p_.get_features(),p.get_features())
        self.assertEqual(p_.get_targets(),p.get_targets())
        self.assertEqual(p_.get_rules(),[])

        for i in range(self.__nb_unit_test):
            #eprint("\rTest ", i+1, "/", self.__nb_unit_test, end='')

            # Generate transitions
            p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)

            min_body_size = 0
            max_body_size = random.randint(min_body_size, len(p.get_features()))
            delay_original = random.randint(2, self.__max_delay)

            features = []
            targets = p.get_features()

            for d in range(0, delay_original):
                features += [(var+"_t-"+str(d+1), vals) for var,vals in p.get_features()]

            p = LogicProgram.random(features, targets, min_body_size, max_body_size)

            cut = len(targets)
            time_series = [[list(s[cut*(d-1):cut*d]) for d in range(1,delay_original+1)] for s in p.states()]

            #eprint(delay_original)
            #eprint(p)
            #eprint(p.states())
            #eprint(time_series)
            #exit()

            time_serie_size = delay_original + 2

            for serie in time_series:
                while len(serie) < time_serie_size:
                    serie_end = serie[-delay_original:]
                    #eprint(serie_end)
                    serie_end = list(itertools.chain.from_iterable(serie_end))
                    serie.append(Synchronous.next(p, serie_end)[0])

            #eprint(p.logic_form())
            #for s in time_series:
            #    eprint(s)

            p_ = LFkT.fit(time_series, targets, targets)
            rules = p_.get_rules()

            #eprint(p_.logic_form())

            for variable in range(len(targets)):
                for value in range(len(targets[variable][1])):
                    #eprint("var="+str(variable)+", val="+str(value))
                    pos, neg, delay = LFkT.interprete(time_series, targets, targets, variable, value)

                    #eprint("pos: ", pos)

                    # Each positive is explained
                    for s in pos:
                        cover = False
                        for r in rules:
                            if r.get_head_variable() == variable \
                               and r.get_head_value() == value \
                               and r.matches(s):
                                cover = True
                        #if not cover:
                        #    eprint(p_)
                        #    eprint(s)
                        self.assertTrue(cover) # One rule cover the example

                    #eprint("neg: ", neg)

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
        #eprint()

    def test_interprete(self):
        print(">> LFkT.interprete(transitions, variable, value)")

        for i in range(self.__nb_unit_test):
            #eprint("Start test ", i, "/", self.__nb_unit_test)
            # Generate transitions
            p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)

            min_body_size = 0
            max_body_size = random.randint(min_body_size, len(p.get_features()))
            delay_original = random.randint(1, self.__max_delay)

            features = []
            targets = p.get_features()

            for d in range(0, delay_original):
                features += [(var+"_t-"+str(d+1), vals) for var,vals in p.get_features()]

            p = LogicProgram.random(features, targets, min_body_size, max_body_size)
            #eprint("Generating series...")
            cut = len(targets)
            time_series = [[list(s[cut*(d-1):cut*d]) for d in range(1,delay_original+1)] for s in p.states()]

            #eprint(delay_original)
            #eprint(p)
            #eprint(p.states())
            #eprint(time_series)
            #exit()

            time_serie_size = delay_original + 2

            for serie in time_series:
                while len(serie) < time_serie_size:
                    serie_end = serie[-delay_original:]
                    #eprint(serie_end)
                    serie_end = list(itertools.chain.from_iterable(serie_end))
                    serie.append(Synchronous.next(p, serie_end)[0])

            var = random.randint(0, len(targets)-1)
            val = random.randint(0, len(targets[var])-1)

            #eprint("interpreting...")
            pos, neg, delay = LFkT.interprete(time_series, targets, targets, var, val)

            # DBG
            #eprint("variables: ", variables)
            #eprint("values", values)
            #eprint("delay: ", delay_original)
            #eprint(p.logic_form())
            #eprint(time_series)
            #eprint("var: ", var)
            #eprint("val: ", val)
            #eprint("pos: ", pos)
            #eprint("neg: ",neg)
            #eprint("delay detected: ", delay)

            # All pos are valid
            for s in pos:
                for serie in time_series:
                    for id in range(len(serie)-delay):
                        s1 = serie[id:id+delay].copy()
                        #s1.reverse()
                        s1 = [y for x in s1 for y in x]
                        #eprint(s1)
                        #eprint(s)
                        s2 = serie[id+delay]
                        if s1 == s:
                            self.assertEqual(s2[var], val)
                            break
            # All neg are valid
            for s in neg:
                for serie in time_series:
                    for id in range(len(serie)-delay):
                        s1 = serie[id:id+delay].copy()
                        #s1.reverse()
                        s1 = [y for x in s1 for y in x]
                        s2 = serie[id+delay]
                        if s1 == s:
                            self.assertTrue(s2[var] != val)
                            break

            # All transitions are interpreted
            #eprint("var/val: ", var, "/", val)
            #eprint("delay: ", delay)
            #eprint("Time serie: ", time_series)
            for serie in time_series:
                #eprint("checking: ", serie)
                for id in range(delay, len(serie)):
                    s1 = serie[id-delay:id].copy()
                    #s1.reverse()
                    s1 = [y for x in s1 for y in x]
                    s2 = serie[id]
                    #eprint("s1: ", s1, ", s2: ", s2)
                    #eprint("pos: ", pos)
                    #eprint("neg: ", neg)
                    if s2[var] == val:
                        self.assertTrue(s1 in pos)
                        self.assertFalse(s1 in neg)
                    else:
                        self.assertFalse(s1 in pos)
                        self.assertTrue(s1 in neg)

            # delay valid
            global_delay = 1
            for serie_1 in time_series:
                for id_state_1 in range(len(serie_1)-1):
                    state_1 = serie_1[id_state_1]
                    next_1 = serie_1[id_state_1+1]
                    # search duplicate with different future
                    for serie_2 in time_series:
                        for id_state_2 in range(len(serie_2)-1):
                            state_2 = serie_2[id_state_2]
                            next_2 = serie_2[id_state_2+1]

                            # Non-determinism detected
                            if state_1 == state_2 and next_1[var] != next_2[var]:
                                local_delay = 2
                                id_1 = id_state_1
                                id_2 = id_state_2
                                while id_1 > 0 and id_2 > 0:
                                    previous_1 = serie_1[id_1-1]
                                    previous_2 = serie_2[id_2-1]
                                    if previous_1 != previous_2:
                                        break
                                    local_delay += 1
                                    id_1 -= 1
                                    id_2 -= 1
                                global_delay = max(global_delay, local_delay)
                                self.assertTrue(local_delay <= delay)
            self.assertEqual(delay, global_delay)

            #eprint("FINISHED ", i, "/", self.__nb_unit_test)

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
        features = [("x"+str(i), [val for val in range(0,random.randint(2,nb_values))]) for i in range(random.randint(1,nb_features))]
        targets = [("y"+str(i), [val for val in range(0,random.randint(2,nb_values))]) for i in range(random.randint(1,nb_targets))]
        rules = []

        for j in range(random.randint(0,100)):
            r = self.random_rule(features, targets, body_size)
            rules.append(r)

        return LogicProgram(features, targets, rules)


if __name__ == '__main__':
    """ Main """

    unittest.main()
