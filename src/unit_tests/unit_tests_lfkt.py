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
from lfkt import LFkT
from rule import Rule
from logicProgram import LogicProgram

#random.seed(0)


class LFkTTest(unittest.TestCase):
    """
        Unit test of class LFkT from lfkt.py
    """

    __nb_unit_test = 10

    __nb_variables = 2

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
        variables = ["x"+str(i) for i in range(random.randint(1,self.__nb_variables))]
        values = []

        for var in range(len(variables)):
            values.append([val for val in range(random.randint(2,self.__nb_values))])

        min_body_size = 0
        max_body_size = random.randint(min_body_size, len(variables))
        delay_original = random.randint(2, self.__max_delay)
        p = LogicProgram.random(variables, values, min_body_size, max_body_size, delay_original)
        p_ = LFkT.fit(p.get_variables(), p.get_values(), [])
        self.assertEqual(p_.get_variables(),p.get_variables())
        self.assertEqual(p_.get_values(),p.get_values())
        self.assertEqual(p_.get_rules(),[])

        for i in range(self.__nb_unit_test):
            #eprint("\rTest ", i+1, "/", self.__nb_unit_test, end='')

            # Generate transitions
            variables = ["x"+str(i) for i in range(random.randint(1,self.__nb_variables))]
            values = []

            for var in range(len(variables)):
                values.append([val for val in range(random.randint(2,self.__nb_values))])

            min_body_size = 0
            max_body_size = random.randint(min_body_size, len(variables))
            delay_original = random.randint(2, self.__max_delay)

            p = LogicProgram.random(variables, values, min_body_size, max_body_size, delay_original)
            time_series = p.generate_all_time_series(delay_original*10)

            #eprint(p.logic_form())
            #eprint(time_series)

            p_ = LFkT.fit(p.get_variables(), p.get_values(), time_series)
            rules = p_.get_rules()

            #eprint(p_.logic_form())

            for variable in range(len(p.get_variables())):
                for value in range(len(p.get_values()[variable])):
                    #eprint("var="+str(variable)+", val="+str(value))
                    pos, neg, delay = LFkT.interprete(p.get_variables(), p.get_values(), time_series, variable, value)

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
            variables = ["x"+str(i) for i in range(random.randint(1,self.__nb_variables))]
            values = []

            for var in range(len(variables)):
                values.append([val for val in range(random.randint(2,self.__nb_values))])

            min_body_size = 0
            max_body_size = random.randint(min_body_size, len(variables))
            delay_original = random.randint(1, self.__max_delay)

            #eprint("Generating random program")
            #eprint("variables: ", variables)
            #eprint("Values: ", values)
            #eprint("delay: ", delay_original)
            p = LogicProgram.random(variables, values, min_body_size, max_body_size, delay_original)
            #eprint("Generating series...")
            time_series = p.generate_all_time_series(delay_original)

            var = random.randint(0, len(p.get_variables())-1)
            val = random.randint(0, len(p.get_values()[var])-1)

            #eprint("interpreting...")
            pos, neg, delay = LFkT.interprete(p.get_variables(), p.get_values(), time_series, var, val)

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
                        s1.reverse()
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
                        s1.reverse()
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
                    s1.reverse()
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


if __name__ == '__main__':
    """ Main """

    unittest.main()
