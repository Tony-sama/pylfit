#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/30
# @updated: 2019/04/30
#
# @desc: PyLFIT unit test script
#
#-----------------------

import sys
import unittest
import random
import os



from pylfit.utils import eprint
from pylfit.objects.continuum import Continuum
from pylfit.objects.continuumRule import ContinuumRule
from pylfit.objects.continuumLogicProgram import ContinuumLogicProgram
from pylfit.algorithms.acedia import ACEDIA

#seed = random.randint(0,1000000)
#seed = 381009
#random.seed(seed)
#eprint("seed: ", seed)

class ACEDIATest(unittest.TestCase):
    """
        Unit test of class ACEDIA from acedia.py
    """

    __nb_unit_test = 10

    __max_variables = 2

    __min_epsilon = 0.3

    """ must be < __max_value"""
    __min_value = 0.0

    """ must be > __min_value"""
    __max_value = 100.0

    __min_domain_size = 1.0

    __min_continuum_size = 0.01

    __nb_rules = 10

    __body_size = 10

    __tmp_file_path = "tmp/unit_test_acedia.tmp"

    #------------------
    # Test functions
    #------------------

    def test_load_input_from_csv(self):
        print(">> ACEDIA.load_input_from_csv(filepath)")

        for i in range(self.__nb_unit_test):
            # Generate transitions
            p = self.random_program()
            epsilon = random.uniform(self.__min_epsilon, 1.0)
            t1 = p.generate_all_transitions(epsilon)

            # Save to csv
            p.transitions_to_csv(self.__tmp_file_path,t1)

            # Extract csv
            t2 = ACEDIA.load_input_from_csv(self.__tmp_file_path)

            # Check if still same
            for i in range(len(t1)):
                self.assertEqual(t1[i],t2[i])

            #eprint(t1)
            #eprint(t2)

    def test_fit(self):
        eprint(">> ACEDIA.fit(variables, values, transitions)")

        for i in range(self.__nb_unit_test):

            eprint("\rTest ", i+1, "/", self.__nb_unit_test, end='')

            # Generate transitions
            epsilon = random.choice([0.1,0.25,0.3,0.5])
            variables, domains = self.random_system()
            p = ContinuumLogicProgram.random(variables, domains, 1, len(variables), epsilon)

            #eprint("Progam: ", p)

            # Valid and realistic epsilon
            #epsilon = round(random.uniform(0.1,1.0), 2)
            #while epsilon == 1.0:
            #    epsilon = round(random.uniform(0.1,1.0), 2)

            t = p.generate_all_transitions(epsilon)

            #sys.exit()

            #eprint("Transitions: ")
            #for s1, s2 in t:
            #    eprint(s1, s2)
            #eprint("Transitions: ", t)

            p_ = ACEDIA.fit(p.get_variables(), p.get_domains(),t)
            rules = p_.get_rules()

            #eprint("learned: ", p_)

            # All transitions are realized
            #------------------------------

            for head_var in range(len(p.get_variables())):
                    for s1, s2 in t:
                        for idx, val in enumerate(s2):
                            realized = 0
                            for r in rules:
                                if r.get_head_variable() == idx and r.get_head_value().includes(val) and r.matches(s1):
                                    realized += 1
                                    break
                            if realized <= 0:
                                eprint("head_var: ", head_var)
                                eprint("s1: ", s1)
                                eprint("s2: ", s2)
                                eprint("learned: ", p_)
                            self.assertTrue(realized >= 1) # One rule realize the example

            # All rules are minimals
            #------------------------
            for r in rules:

                #eprint("r: ", r)

                # Try reducing head min
                #-----------------------
                r_ = r.copy()
                h = r_.get_head_value()
                if h.get_min_value() + epsilon <= h.get_max_value():
                    r_.set_head_value( Continuum(h.get_min_value()+epsilon, h.get_max_value(), h.min_included(), h.max_included()) )

                    #eprint("spec: ", r_)

                    conflict = False
                    for s1, s2 in t:
                        if not r_.get_head_value().includes(s2[r_.get_head_variable()]) and r_.matches(s1): # Cover a negative example
                            conflict = True
                            #eprint("conflict")
                            break

                    if not conflict:
                        eprint("Non minimal rule: ", r)
                        eprint("head can be specialized into: ", r_.get_head_variable(), "=", r_.get_head_value())

                    self.assertTrue(conflict)

                # Try reducing head max
                #-----------------------
                r_ = r.copy()
                h = r_.get_head_value()
                if h.get_max_value() - epsilon >= h.get_min_value():
                    r_.set_head_value( Continuum(h.get_min_value(), h.get_max_value()-epsilon, h.min_included(), h.max_included()) )

                    #eprint("spec: ", r_)

                    conflict = False
                    for s1, s2 in t:
                        if not r_.get_head_value().includes(s2[r_.get_head_variable()]) and r_.matches(s1): # Cover a negative example
                            conflict = True
                            #eprint("conflict")
                            break

                    if not conflict:
                        eprint("Non minimal rule: ", r)
                        eprint("head can be generalized to: ", r_.get_head_variable(), "=", r_.get_head_value())

                    self.assertTrue(conflict)

                # Try extending condition
                #-------------------------
                for (var,val) in r.get_body():

                    # Try extend min
                    r_ = r.copy()
                    if val.get_min_value() - epsilon >= domains[var].get_min_value():
                        val_ = val.copy()
                        if not val_.min_included():
                            val_.set_lower_bound(val_.get_min_value(), True)
                        else:
                            val_.set_lower_bound(val_.get_min_value()-epsilon, False)
                        r_.set_condition(var, val_)

                        #eprint("gen: ", r_)

                        conflict = False
                        for s1, s2 in t:
                            if not r_.get_head_value().includes(s2[r_.get_head_variable()]) and r_.matches(s1): # Cover a negative example
                                conflict = True
                                #eprint("conflict")
                                break

                        if not conflict:
                            eprint("Non minimal rule: ", r)
                            eprint("condition can be generalized: ", var, "=", val_)

                        self.assertTrue(conflict)

                    # Try extend max
                    r_ = r.copy()
                    if val.get_max_value() + epsilon <= domains[var].get_max_value():
                        val_ = val.copy()
                        if not val_.max_included():
                            val_.set_upper_bound(val_.get_max_value(), True)
                        else:
                            val_.set_upper_bound(val_.get_max_value()+epsilon, False)
                        r_.set_condition(var, val_)

                        #eprint("gen: ", r_)

                        conflict = False
                        for s1, s2 in t:
                            if not r_.get_head_value().includes(s2[r_.get_head_variable()]) and r_.matches(s1): # Cover a negative example
                                conflict = True
                                #eprint("conflict")
                                break

                        if not conflict:
                            eprint("Non minimal rule: ", r)
                            eprint("condition can be generalized: ", var, "=", val_)

                        self.assertTrue(conflict)
        eprint()

    def test_fit_var(self):
        eprint(">> ACEDIA.fit_var(variables, domains, transitions, variable)")

        for i in range(self.__nb_unit_test):

            eprint("\rTest ", i+1, "/", self.__nb_unit_test, end='')

            # Generate transitions
            epsilon = random.choice([0.1,0.25,0.3,0.5])
            variables, domains = self.random_system()
            p = ContinuumLogicProgram.random(variables, domains, 1, len(variables), epsilon)

            #eprint("Progam: ", p)

            # Valid and realistic epsilon
            #epsilon = round(random.uniform(0.1,1.0), 2)
            #while epsilon == 1.0:
            #    epsilon = round(random.uniform(0.1,1.0), 2)

            t = p.generate_all_transitions(epsilon)

            #sys.exit()

            #eprint("Transitions: ")
            #for s1, s2 in t:
            #    eprint(s1, s2)
            #eprint("Transitions: ", t)
            head_var = random.randint(0, len(p.get_variables())-1)

            rules = ACEDIA.fit_var(p.get_variables(), p.get_domains(), t, head_var)

            #eprint("learned: ", p_)

            # All transitions are realized
            #------------------------------

            for s1, s2 in t:
                realized = 0
                for r in rules:
                    if r.get_head_variable() == head_var and r.get_head_value().includes(s2[head_var]) and r.matches(s1):
                        realized += 1
                        break
                if realized <= 0:
                    eprint("head_var: ", head_var)
                    eprint("s1: ", s1)
                    eprint("s2: ", s2)
                    eprint("learned: ", p_)
                self.assertTrue(realized >= 1) # One rule realize the example

            # All rules are minimals
            #------------------------
            for r in rules:

                #eprint("r: ", r)
                self.assertEqual(r.get_head_variable(), head_var)

                # Try reducing head min
                #-----------------------
                r_ = r.copy()
                h = r_.get_head_value()
                if h.get_min_value() + epsilon <= h.get_max_value():
                    r_.set_head_value( Continuum(h.get_min_value()+epsilon, h.get_max_value(), h.min_included(), h.max_included()) )

                    #eprint("spec: ", r_)

                    conflict = False
                    for s1, s2 in t:
                        if not r_.get_head_value().includes(s2[r_.get_head_variable()]) and r_.matches(s1): # Cover a negative example
                            conflict = True
                            #eprint("conflict")
                            break

                    if not conflict:
                        eprint("Non minimal rule: ", r)
                        eprint("head can be specialized into: ", r_.get_head_variable(), "=", r_.get_head_value())

                    self.assertTrue(conflict)

                # Try reducing head max
                #-----------------------
                r_ = r.copy()
                h = r_.get_head_value()
                if h.get_max_value() - epsilon >= h.get_min_value():
                    r_.set_head_value( Continuum(h.get_min_value(), h.get_max_value()-epsilon, h.min_included(), h.max_included()) )

                    #eprint("spec: ", r_)

                    conflict = False
                    for s1, s2 in t:
                        if not r_.get_head_value().includes(s2[r_.get_head_variable()]) and r_.matches(s1): # Cover a negative example
                            conflict = True
                            #eprint("conflict")
                            break

                    if not conflict:
                        eprint("Non minimal rule: ", r)
                        eprint("head can be generalized to: ", r_.get_head_variable(), "=", r_.get_head_value())

                    self.assertTrue(conflict)

                # Try extending condition
                #-------------------------
                for (var,val) in r.get_body():

                    # Try extend min
                    r_ = r.copy()
                    if val.get_min_value() - epsilon >= domains[var].get_min_value():
                        val_ = val.copy()
                        if not val_.min_included():
                            val_.set_lower_bound(val_.get_min_value(), True)
                        else:
                            val_.set_lower_bound(val_.get_min_value()-epsilon, False)
                        r_.set_condition(var, val_)

                        #eprint("gen: ", r_)

                        conflict = False
                        for s1, s2 in t:
                            if not r_.get_head_value().includes(s2[r_.get_head_variable()]) and r_.matches(s1): # Cover a negative example
                                conflict = True
                                #eprint("conflict")
                                break

                        if not conflict:
                            eprint("Non minimal rule: ", r)
                            eprint("condition can be generalized: ", var, "=", val_)

                        self.assertTrue(conflict)

                    # Try extend max
                    r_ = r.copy()
                    if val.get_max_value() + epsilon <= domains[var].get_max_value():
                        val_ = val.copy()
                        if not val_.max_included():
                            val_.set_upper_bound(val_.get_max_value(), True)
                        else:
                            val_.set_upper_bound(val_.get_max_value()+epsilon, False)
                        r_.set_condition(var, val_)

                        #eprint("gen: ", r_)

                        conflict = False
                        for s1, s2 in t:
                            if not r_.get_head_value().includes(s2[r_.get_head_variable()]) and r_.matches(s1): # Cover a negative example
                                conflict = True
                                #eprint("conflict")
                                break

                        if not conflict:
                            eprint("Non minimal rule: ", r)
                            eprint("condition can be generalized: ", var, "=", val_)

                        self.assertTrue(conflict)

        eprint()

    def test_least_revision(self):
        eprint(">> ACEDIA.least_revision(rule, state_1, state_2)")

        for i in range(self.__nb_unit_test):
            variables, domains = self.random_system()

            state_1 = self.random_state(variables, domains)
            state_2 = self.random_state(variables, domains)

            # not matching
            #--------------
            rule = self.random_rule(variables, domains)
            while rule.matches(state_1):
                rule = self.random_rule(variables, domains)

            self.assertRaises(ValueError, ACEDIA.least_revision, rule, state_1, state_2)

            # matching
            #--------------

            rule = self.random_rule(variables, domains)
            while not rule.matches(state_1):
                rule = self.random_rule(variables, domains)

            head_var = rule.get_head_variable()
            target_val = state_2[rule.get_head_variable()]

            # Consistent
            head_value = Continuum()
            while not head_value.includes(target_val):
                head_value = Continuum.random(domains[head_var].get_min_value(), domains[head_var].get_max_value())
            rule.set_head_value(head_value)
            self.assertRaises(ValueError, ACEDIA.least_revision, rule, state_1, state_2)

            # Empty set head
            rule.set_head_value(Continuum())

            LR = ACEDIA.least_revision(rule, state_1, state_2)
            lg = rule.copy()
            lg.set_head_value(Continuum(target_val,target_val,True,True))
            self.assertTrue(lg in LR)

            nb_valid_revision = 1

            for var, val in rule.get_body():
                state_value = state_1[var]

                # min rev
                ls = rule.copy()
                new_val = val.copy()
                new_val.set_lower_bound(state_value, False)
                if not new_val.is_empty():
                    ls.set_condition(var, new_val)
                    self.assertTrue(ls in LR)
                    nb_valid_revision += 1

                # max rev
                ls = rule.copy()
                new_val = val.copy()
                new_val.set_upper_bound(state_value, False)
                if not new_val.is_empty():
                    ls.set_condition(var, new_val)
                    self.assertTrue(ls in LR)
                    nb_valid_revision += 1

            self.assertEqual(len(LR), nb_valid_revision)

            #eprint(nb_valid_revision)

            # usual head
            head_value = Continuum.random(domains[head_var].get_min_value(), domains[head_var].get_max_value())
            while head_value.includes(target_val):
                head_value = Continuum.random(domains[head_var].get_min_value(), domains[head_var].get_max_value())
            rule.set_head_value(head_value)

            LR = ACEDIA.least_revision(rule, state_1, state_2)

            lg = rule.copy()
            head_value = lg.get_head_value()
            if target_val <= head_value.get_min_value():
                head_value.set_lower_bound(target_val, True)
            else:
                head_value.set_upper_bound(target_val, True)
            lg.set_head_value(head_value)
            self.assertTrue(lg in LR)

            nb_valid_revision = 1

            for var, val in rule.get_body():
                state_value = state_1[var]

                # min rev
                ls = rule.copy()
                new_val = val.copy()
                new_val.set_lower_bound(state_value, False)
                if not new_val.is_empty():
                    ls.set_condition(var, new_val)
                    self.assertTrue(ls in LR)
                    nb_valid_revision += 1

                # max rev
                ls = rule.copy()
                new_val = val.copy()
                new_val.set_upper_bound(state_value, False)
                if not new_val.is_empty():
                    ls.set_condition(var, new_val)
                    self.assertTrue(ls in LR)
                    nb_valid_revision += 1

            self.assertEqual(len(LR), nb_valid_revision)

    #------------------
    # Tool functions
    #------------------

    def random_system(self):
        # generates variables/domains
        nb_variables = random.randint(1, self.__max_variables)
        variables = ["x"+str(var) for var in range(nb_variables)]
        domains = [ Continuum.random(self.__min_value, self.__max_value, self.__min_domain_size) for var in variables ]
        # DBG
        #domains = [ Continuum(0, 1, True, True) for var in variables ]

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
