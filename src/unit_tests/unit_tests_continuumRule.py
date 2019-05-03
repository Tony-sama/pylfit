#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/26
# @updated: 2019/05/02
#
# @desc: PyLFIT unit test script
#
#-----------------------

import sys
import unittest
import random

sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')

from utils import eprint
from continuum import Continuum
from continuumRule import ContinuumRule

#random.seed(0)

class ContinuumRuleTest(unittest.TestCase):
    """
        Unit test of class ContinuumRule from continuumRule.py
    """

    __nb_unit_test = 10

    __max_variables = 5

    """ must be < __max_value"""
    __min_value = -100.0

    """ must be > __min_value"""
    __max_value = 100.0

    __min_domain_size = 0.01

    __min_continuum_size = 0.01

    __max_size = 5


    #------------------
    # Constructors
    #------------------

    def test_constructor_empty(self):
        eprint(">> ContinuumRule.__init__(self, head_variable, head_value)")

        for i in range(self.__nb_unit_test):
            var_id = random.randint(0, self.__max_variables)
            domain = Continuum.random(self.__min_value, self.__max_value)
            c = ContinuumRule(var_id, domain)

            self.assertEqual(c.get_head_variable(), var_id)
            self.assertEqual(c.get_head_value(), domain)
            self.assertEqual(c.get_body(), [])

    def test_constructor_full(self):
        eprint(">> ContinuumRule.__init__(self, head_variable, head_value, body)")

        for i in range(self.__nb_unit_test):
            head_var_id = random.randint(0, self.__max_variables)
            head_domain = Continuum.random(self.__min_value, self.__max_value)

            size = random.randint(1, self.__max_variables)
            body = []
            locked = []
            for j in range(size):
                var_id = random.randint(0, self.__max_variables)
                domain = Continuum.random(self.__min_value, self.__max_value)
                if var_id not in locked:
                    body.append( (var_id, domain) )
                    locked.append(var_id)

            r = ContinuumRule(head_var_id, head_domain, body)

            self.assertEqual(r.get_head_variable(), head_var_id)
            self.assertEqual(r.get_head_value(), head_domain)
            for e in body:
                self.assertTrue(e in r.get_body())
            for e in r.get_body():
                self.assertTrue(e in body)

    def test_copy(self):
        eprint(">> ContinuumRule.copy(self)")

        for i in range(self.__nb_unit_test):
            head_var_id = random.randint(0, self.__max_variables)
            head_domain = Continuum.random(self.__min_value, self.__max_value)

            size = random.randint(1, self.__max_variables)
            body = []
            locked = []
            for j in range(size):
                var_id = random.randint(0, self.__max_variables)
                domain = Continuum.random(self.__min_value, self.__max_value)
                if var_id not in locked:
                    body.append( (var_id, domain) )
                    locked.append(var_id)

            r_ = ContinuumRule(head_var_id, head_domain, body)
            r = r_.copy()

            self.assertEqual(r.get_head_variable(), head_var_id)
            self.assertEqual(r.get_head_value(), head_domain)
            for e in body:
                self.assertTrue(e in r.get_body())
            for e in r.get_body():
                self.assertTrue(e in body)

            self.assertEqual(r.get_head_variable(), r_.get_head_variable())
            self.assertEqual(r.get_head_value(), r_.get_head_value())
            for e in r_.get_body():
                self.assertTrue(e in r.get_body())
            for e in r.get_body():
                self.assertTrue(e in r_.get_body())

    def test_static_random(self):
        eprint(">> ContinuumRule.random(min_value, max_value)")

        for i in range(self.__nb_unit_test):

            variables, domains = self.random_system()

            # rule characteristics
            var = random.randint(0,len(variables)-1)
            var_domain = domains[var]
            val = Continuum.random(var_domain.get_min_value(), var_domain.get_max_value())
            min_size = random.randint(0, len(variables))
            max_size = random.randint(min_size, len(variables))
            r = ContinuumRule.random(var, val, variables, domains, min_size, max_size)

            # Check head
            self.assertEqual(r.get_head_variable(), var)
            self.assertEqual(r.get_head_value(), val)

            # Check body
            self.assertTrue(r.size() >= min_size)
            self.assertTrue(r.size() <= max_size)

            appears = []
            for var, val in r.get_body():
                self.assertTrue(var >= 0 and var < len(variables))
                self.assertTrue(domains[var].includes(val))

                self.assertFalse(var in appears)
                appears.append(var)

            # min > max
            min_size = random.randint(0, len(variables))
            max_size = random.randint(-100, min_size-1)
            self.assertRaises(ValueError, ContinuumRule.random, var, val, variables, domains, min_size, max_size)

            # min > nb variables
            min_size = random.randint(len(variables)+1, len(variables)+100)
            max_size = random.randint(min_size, len(variables)+100)
            self.assertRaises(ValueError, ContinuumRule.random, var, val, variables, domains, min_size, max_size)

            # max > nb variables
            min_size = random.randint(0, len(variables))
            max_size = random.randint(len(variables)+1, len(variables)+100)
            self.assertRaises(ValueError, ContinuumRule.random, var, val, variables, domains, min_size, max_size)

    #--------------
    # Observers
    #--------------

    def test_size(self):
        print(">> ContinuumRule.size(self)")
        for i in range(self.__nb_unit_test):
            variables, domains = self.random_system()
            r = self.random_rule(variables, domains)

            self.assertEqual(r.size(), len(r.get_body()))

    def test_get_condition(self):
        print(">> ContinuumRule.get_condition(self, variable)")

        for i in range(self.__nb_unit_test):
            # Empty rule
            variables, domains = self.random_system()
            var = random.randint(0,len(variables))
            val = Continuum.random(self.__min_value, self.__max_value)
            r = ContinuumRule(var,val)

            for var in range(len(variables)):
                self.assertEqual(r.get_condition(var),None)

            # Regular rule
            variables, domains = self.random_system()
            while r.size() == 0:
                r = self.random_rule(variables, domains)
            body = r.get_body()

            appears = []
            for var, val in body:
                appears.append(var)
                self.assertEqual(r.get_condition(var),val)

            for var in range(len(variables)):
                if var not in appears:
                    self.assertEqual(r.get_condition(var),None)

    def test_has_condition(self):
        print(">> ContinuumRule.has_condition(self, variable)")

        for i in range(self.__nb_unit_test):
            # Empty rule
            variables, domains = self.random_system()
            var = random.randint(0,len(variables))
            val = Continuum.random(self.__min_value, self.__max_value)
            r = ContinuumRule(var,val)

            for var in range(len(variables)):
                self.assertEqual(r.has_condition(var),False)

            # Regular rule
            variables, domains = self.random_system()
            while r.size() == 0:
                r = self.random_rule(variables, domains)
            body = r.get_body()

            appears = []
            for var, val in body:
                appears.append(var)
                self.assertEqual(r.has_condition(var),True)

            for var in range(len(variables)):
                if var not in appears:
                    self.assertEqual(r.has_condition(var),False)

    #--------------
    # Operators
    #--------------

    #--------------
    # Methods
    #--------------

    def test_to_string(self):
        print(">> ContinuumRule.to_string(self)")
        for i in range(self.__nb_unit_test):

            variables, domains = self.random_system()
            r = self.random_rule(variables, domains)

            string = str(r.get_head_variable()) + "=" + r.get_head_value().to_string() + " :- "
            for a, b in r.get_body():
                string += str(a) + "=" + b.to_string() + ", "

            if len(r.get_body()) > 0:
                string = string[:-2]
            string += "."

            self.assertEqual(r.to_string(), string)
            self.assertEqual(r.to_string(), str(r))
            self.assertEqual(r.to_string(), repr(r))
            #eprint(r)

    def test_logic_form(self):
        print(">> ContinuumRule.logic_form(self, variables)")

        # One step rules
        for i in range(self.__nb_unit_test):
            variables, domains = self.random_system()
            r = self.random_rule(variables, domains)

            max_var_id = r.get_head_variable()
            max_val_id = r.get_head_value()

            string = variables[r.get_head_variable()] + "(" + r.get_head_value().to_string() + ",T) :- "
            for var, val in r.get_body():
                string += variables[var] + "(" + val.to_string() + ",T-1), "

            if len(r.get_body()) > 0:
                string = string[:-2]
            string += "."

            self.assertEqual(r.logic_form(variables),string)

        # Delayed rules
        for i in range(self.__nb_unit_test):
            variables, domains = self.random_system()
            r = self.random_rule(variables, domains)

            variables = ["x"+str(var) for var in range(int(len(variables)/3)+1)]

            string = variables[r.get_head_variable() % len(variables)] + "(" + r.get_head_value().to_string() + ",T) :- "
            for var, val in r.get_body():
                delay = int(var / len(variables)) + 1
                string += variables[var % len(variables)] + "(" + val.to_string() + ",T-" + str(delay) + "), "

            if len(r.get_body()) > 0:
                string = string[:-2]
            string += "."

            self.assertEqual(r.logic_form(variables),string)
            #eprint(string)

    def test_matches(self):
        print(">> ContinuumRule.matches(self, state)")

        for i in range(self.__nb_unit_test):
            variables, domains = self.random_system()
            r = self.random_rule(variables, domains)
            state = self.random_state(variables, domains)

            body = r.get_body()

            for var, val in body:
                min = val.get_min_value()
                max = val.get_max_value()
                if not val.min_included:
                    min += self.__min_continuum_size * 0.1
                if not val.max_included:
                    max -= self.__min_continuum_size * 0.1
                state[var] = random.uniform(min, max)

            self.assertTrue(r.matches(state))

            while r.size() == 0:
                r = self.random_rule(variables, domains)
            state = self.random_state(variables, domains)

            var, val = random.choice(r.get_body())
            val_ = val.get_max_value() - ( (val.get_min_value() - val.get_max_value()) / 2.0 )

            while val.includes(val_):
                val_ = random.uniform(self.__min_value-100.0, self.__max_value+100.0)

            state[var] = val_

            self.assertFalse(r.matches(state))

            # no out of bound (delay enconding)
            self.assertFalse(r.matches([]))

    def test_dominates(self):
        print(">> ContinuumRule.dominates(self, rule)")

        for i in range(self.__nb_unit_test):
            variables, domains = self.random_system()
            var = random.randint(0, len(variables)-1)
            r = self.random_rule(variables, domains)

            # Undominated rule: var = emptyset if anything
            r0 = ContinuumRule(r.get_head_variable(),Continuum())

            self.assertTrue(r0.dominates(r0))

            # r1 is a regular rule
            r1 = self.random_rule(variables, domains)
            while r1.size() == 0:
                r1 = self.random_rule(variables, domains)
            var = r.get_head_variable()
            val = Continuum.random(domains[var].get_min_value(),domains[var].get_max_value())
            r1 = ContinuumRule(var, val, r1.get_body()) # set head

            self.assertTrue(r0.dominates(r1))
            self.assertFalse(r1.dominates(r0))

            # r2 is precision of r1 (head specification)
            r2 = r1.copy()
            var = r2.get_head_variable()
            val = r2.get_head_value()
            while r2.get_head_value() == val or not r2.get_head_value().includes(val):
                val = Continuum.random(val.get_min_value(), val.get_max_value())
            r2.set_head_value(val)

            self.assertEqual(r0.dominates(r2), True)
            self.assertEqual(r2.dominates(r0), False)
            self.assertEqual(r2.dominates(r1), True)
            self.assertEqual(r1.dominates(r2), False)

            # r3 is a generalization of r1 (body generalization)
            r3 = r1.copy()
            var, val = random.choice(r3.get_body())
            val_ = val
            modif_min = random.uniform(0, self.__max_value)
            modif_max = random.uniform(0, self.__max_value)
            while val_ == val or not val_.includes(val):
                val_ = Continuum.random(val.get_min_value()-modif_min, val.get_max_value()+modif_max)
            r3.set_condition(var, val_)

            self.assertEqual(r0.dominates(r3), True)
            self.assertEqual(r3.dominates(r0), False)
            self.assertEqual(r3.dominates(r1), True)
            self.assertEqual(r1.dominates(r3), False)

            # r4 is unprecision of r1 (head generalization)
            r4 = r1.copy()
            var = r4.get_head_variable()
            val = r4.get_head_value()
            modif_min = random.uniform(0, self.__max_value)
            modif_max = random.uniform(0, self.__max_value)
            while r4.get_head_value() == val or not val.includes(r4.get_head_value()):
                val = Continuum(val.get_min_value()-modif_min, val.get_max_value()+modif_max,random.choice([True,False]),random.choice([True,False]))
            r4.set_head_value(val)

            self.assertEqual(r0.dominates(r4), True)
            self.assertEqual(r4.dominates(r0), False)
            self.assertEqual(r4.dominates(r1), False)
            self.assertEqual(r1.dominates(r4), True)

            # r5 is specialization of r1 (body specialization)
            r5 = r1.copy()
            var, val = random.choice(r5.get_body())
            val_ = val
            while val_ == val or not val.includes(val_):
                val_ = Continuum.random(val.get_min_value(), val.get_max_value())
            r5.set_condition(var,val_)

            self.assertEqual(r0.dominates(r5), True)
            self.assertEqual(r5.dominates(r0), False)
            self.assertEqual(r5.dominates(r1), False)
            self.assertEqual(r1.dominates(r5), True)

            # r6 is a random rule
            r6 = self.random_rule(variables, domains)

            # head var difference
            if r6.get_head_variable() != r1.get_head_variable():
                self.assertFalse(r6.dominates(r1))
                self.assertFalse(r1.dominates(r6))

            r6 = ContinuumRule(r1.get_head_variable(), r6.get_head_value(), r6.get_body()) # same head var
            #eprint("r1: ", r1)
            #eprint("r6: ", r6)

            # head val inclusion
            if not r1.get_head_value().includes(r6.get_head_value()):
                self.assertFalse(r6.dominates(r1))

            r6 = ContinuumRule(r1.get_head_variable(), r1.get_head_value(), r6.get_body()) # same head var, same head val

            # body subsumption
            if r1.dominates(r6):
                for var,val in r1.get_body():
                    self.assertTrue(val.includes(r6.get_condition(var)))

            # body subsumption
            if r6.dominates(r1):
                for var,val in r6.get_body():
                    self.assertTrue(val.includes(r1.get_condition(var)))

            # incomparable
            if not r1.dominates(r6) and not r6.dominates(r1):
                #eprint("r1: ", r1)
                #eprint("r6: ", r6)

                conflicts = False
                dominant_r1 = False
                dominant_r6 = False
                for var, val in r1.get_body():
                    # condition not appearing
                    if not r6.has_condition(var):
                        conflicts = True
                        break

                    # value not included
                    if not r6.get_condition(var).includes(val) and not val.includes(r6.get_condition(var)):
                        conflicts = True
                        break

                    # local dominates
                    if val.includes(r6.get_condition(var)):
                        dominant_r1 = True

                    # local dominated
                    if r6.get_condition(var).includes(val):
                        if dominant_r1:
                            conflicts = True
                            break

                for var, val in r6.get_body():
                    # condition not appearing
                    if not r1.has_condition(var):
                        conflicts = True
                        break

                    # value not included
                    if not r1.get_condition(var).includes(val) and not val.includes(r1.get_condition(var)):
                        conflicts = True
                        break

                    # local dominates
                    if val.includes(r1.get_condition(var)):
                        dominant_r6 = True
                        if dominant_r1:
                            conflicts = True
                            break

                    # local dominated
                    if r1.get_condition(var).includes(val):
                        if dominant_r6:
                            conflicts = True
                            break

                self.assertTrue(conflicts)

            #eprint("r0:", r0)
            #eprint("r1:", r1)
            #eprint("r2:", r2)
            #eprint("r3:", r3)
            #eprint("r4:", r4)
            #eprint("r5:", r5)
            #eprint()

    def test_set_condition(self):
        print(">> ContinuumRule.set_condition(self, variable, value)")

        for i in range(self.__nb_unit_test):
            variables, domains = self.random_system()
            var = random.randint(0, len(variables)-1)

            # empty rule
            r = self.random_rule(variables, domains)
            r = ContinuumRule(r.get_head_variable(), r.get_head_value(), [])

            for j in range(len(variables)):

                var = random.randint(0, len(variables)-1)
                val = Continuum.random(domains[var].get_min_value(), domains[var].get_max_value())
                size = r.size()
                exist = r.has_condition(var)
                r.set_condition(var, val)

                # condition updated
                self.assertEqual(r.get_condition(var), val)

                # change rather than add when variable exist in a condition
                if exist:
                    self.assertEqual(r.size(), size)
                else:
                    self.assertEqual(r.size(), size+1)

                # Ordering ensured
                prev = -1
                for v, val in r.get_body():
                    v > prev
                    prev = v

            # regular rule
            r = self.random_rule(variables, domains)

            for j in range(len(variables)):
                var = random.randint(0, len(variables)-1)
                val = Continuum.random(domains[var].get_min_value(), domains[var].get_max_value())
                size = r.size()
                exist = r.has_condition(var)


                #eprint("r: ", r)
                #eprint("var: ", var)
                #eprint("val: ", val)

                r.set_condition(var, val)

                # condition updated
                self.assertEqual(r.get_condition(var), val)

                # change rather than add when variable exist in a condition
                if exist:
                    self.assertEqual(r.size(), size)
                else:
                    self.assertEqual(r.size(), size+1)

                # Ordering ensured
                prev = -1
                for v, val in r.get_body():
                    v > prev
                    prev = v

    def test___eq__(self):
        print(">> ContinuumRule.__eq__(self, other)")

        for i in range(self.__nb_unit_test):
            variables, domains = self.random_system()
            r = self.random_rule(variables, domains)

            self.assertTrue(r == r)
            self.assertFalse(r != r)

            r_ = self.random_rule(variables, domains)

            if r.get_head_variable() != r_.get_head_variable():
                self.assertFalse(r == r_)

            if r.get_head_value() != r_.get_head_value():
                self.assertFalse(r == r_)

            if r.get_body() != r_.get_body():
                self.assertFalse(r == r_)

            # different type
            self.assertFalse(r == "")
            self.assertFalse(r == 0)
            self.assertFalse(r == [])
            self.assertFalse(r == [1,2,3])

            # different size
            r_ = self.random_rule(variables, domains)
            while r_.size() == r.size():
                r_ = self.random_rule(variables, domains)

            r_.set_head_variable(r.get_head_variable())
            r_.set_head_value(r.get_head_value())
            self.assertFalse(r == r_)

            # Same size, same head
            while r.size() == 0:
                r = self.random_rule(variables, domains)
            r_ = r.copy()
            var, val = random.choice(r.get_body())

            new_val = val
            while new_val == val:
                new_val = Continuum.random(domains[var].get_min_value(), domains[var].get_max_value())
                r_.set_condition(var, new_val)

            self.assertFalse(r == r_)

    def test_remove_condition(self):
        print(">> ContinuumRule.remove_condition(self, variable)")

        # Empty rule
        for i in range(self.__nb_unit_test):
            variables, domains = self.random_system()
            var = random.randint(0,len(variables)-1)
            val = Continuum.random(domains[var].get_min_value(), domains[var].get_max_value())
            r = ContinuumRule(var,val)

            var = random.randint(0,len(variables)-1)

            self.assertFalse(r.has_condition(var))
            r.remove_condition(var)
            self.assertFalse(r.has_condition(var))

            while r.size() == 0:
                r = self.random_rule(variables, domains)

            var, val = random.choice(r.get_body())

            self.assertTrue(r.has_condition(var))
            r.remove_condition(var)
            self.assertFalse(r.has_condition(var))

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
'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
