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

sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')

from utils import eprint
from rule import Rule

#random.seed(0)

class RuleTest(unittest.TestCase):
    """
        Unit test of class Rule from rule.py
    """

    __nb_unit_test = 100

    __var_size = 100

    __val_size = 10

    __max_delay = 5

    __body_size = 10


    #------------------
    # Constructors
    #------------------

    def test_empty_constructor(self):
        print(">> Rule.__init__(self, head_variable, head_value)")
        for i in range(self.__nb_unit_test):
            var = random.randint(0,self.__var_size)
            val = random.randint(0,self.__val_size)
            r = Rule(var,val)
            self.assertEqual(r.get_head_variable(),var)
            self.assertEqual(r.get_head_value(),val)
            self.assertEqual(r.get_body(),[])

    def test_body_constructor(self):
        print(">> Rule.__init__(self, head_variable, head_value, body)")
        for i in range(self.__nb_unit_test):
            var = random.randint(0,self.__var_size)
            val = random.randint(0,self.__val_size)
            body = []
            for j in range(0, random.randint(0,self.__body_size)):
                body += (random.randint(0,self.__var_size), random.randint(0,self.__val_size))
            r = Rule(var,val,body)
            self.assertEqual(r.get_head_variable(),var)
            self.assertEqual(r.get_head_value(),val)
            self.assertEqual(r.get_body(),body)

    def test_copy(self):
        print(">> Rule.copy(self)")
        for i in range(self.__nb_unit_test):
            r1 = self.random_rule()
            r2 = r1.copy()

            self.assertEqual(r1.get_head_variable(), r2.get_head_variable())
            self.assertEqual(r1.get_head_value(), r2.get_head_value())
            self.assertEqual(r1.get_body(), r2.get_body())

    def test_static_random(self):
        print(">> Rule.static random(head_variable, head_value, variables, values, min_body_size, max_body_size)")
        for i in range(self.__nb_unit_test):
            var = random.randint(0, self.__var_size)
            val = random.randint(0, self.__val_size)
            variables = ["x"+str(var) for var in range(self.__var_size)]
            values = [ ["v"+str(val) for val in range(self.__val_size)] for var in variables ]
            min_size = random.randint(0, self.__body_size)
            max_size = random.randint(min_size, self.__body_size)
            r = Rule.random(var, val, variables, values, min_size, max_size)

            # Check head
            self.assertEqual(r.get_head_variable(), var)
            self.assertEqual(r.get_head_value(), val)

            # Check body
            self.assertTrue(r.size() >= min_size)
            self.assertTrue(r.size() <= max_size)
            for var, val in r.get_body():
                self.assertTrue(var >= 0 and var < len(variables))
                self.assertTrue(val >= 0 and val < len(values[var]))

            # bad argument
            while max_size >= min_size:
                max_size = random.randint(-100, min_size)

            self.assertRaises(ValueError, Rule.random, var, val, variables, values, min_size, max_size)

            # bad argument
            while max_size <= len(variables):
                max_size = random.randint(len(variables), len(variables) + 10)

            self.assertRaises(ValueError, Rule.random, var, val, variables, values, min_size, max_size)

            # bad argument
            while min_size <= len(variables):
                min_size = random.randint(len(variables), self.__var_size+10)

            max_size = min_size + 10

            self.assertRaises(ValueError, Rule.random, var, val, variables, values, min_size, max_size)


    #--------------
    # Observers
    #--------------

    def test_size(self):
        print(">> Rule.size(self)")
        for i in range(self.__nb_unit_test):
            r = self.random_rule()

            self.assertEqual(r.size(), len(r.get_body()))

    def test_to_string(self):
        print(">> Rule.to_string(self)")
        for i in range(self.__nb_unit_test):
            r = self.random_rule()

            string = str(r.get_head_variable()) + "=" + str(r.get_head_value()) + " :- "
            for a, b in r.get_body():
                string += str(a) + "=" + str(b) + ", "

            if len(r.get_body()) > 0:
                string = string[:-2]
            string += "."

            self.assertEqual(r.to_string(),string)
            self.assertEqual(r.to_string(), r.__str__())
            self.assertEqual(r.to_string(), r.__repr__())

    def test_logic_form(self):
        print(">> Rule.logic_form(self, variables, values)")

        # One step rules
        for i in range(self.__nb_unit_test):
            r = self.random_rule()

            max_var_id = r.get_head_variable()
            max_val_id = r.get_head_value()
            for var, val in r.get_body():
                if var > max_var_id:
                    max_var_id = var
                if val > max_val_id:
                    max_val_id = val

            variables = ["x"+str(var) for var in range(max_var_id+1)]
            values = [ ["v"+str(val) for val in range(max_val_id+1)] for var in variables ]

            string = variables[r.get_head_variable()] + "(" + values[r.get_head_variable()][r.get_head_value()] + ",T) :- "
            for var, val in r.get_body():
                string += variables[var] + "(" + values[var][val] + ",T-1), "

            if len(r.get_body()) > 0:
                string = string[:-2]
            string += "."

            self.assertEqual(r.logic_form(variables, values),string)

        # Delayed rules
        for i in range(self.__nb_unit_test):
            r = self.random_rule()

            max_var_id = r.get_head_variable()
            max_val_id = r.get_head_value()
            for var, val in r.get_body():
                if var > max_var_id:
                    max_var_id = var
                if val > max_val_id:
                    max_val_id = val

            variables = ["x"+str(var) for var in range(int(max_var_id/3)+1)]
            values = [ ["v"+str(val) for val in range(max_val_id+1)] for var in variables ]

            string = variables[r.get_head_variable() % len(variables)] + "(" + values[r.get_head_variable() % len(variables)][r.get_head_value()] + ",T) :- "
            for var, val in r.get_body():
                delay = int(var / len(variables)) + 1
                string += variables[var % len(variables)] + "(" + values[var % len(variables)][val] + ",T-" + str(delay) + "), "

            if len(r.get_body()) > 0:
                string = string[:-2]
            string += "."

            self.assertEqual(r.logic_form(variables, values),string)

    def test_get_condition(self):
        print(">> Rule.get_condition(self, variable)")

        # Empty rule
        for i in range(self.__nb_unit_test):
            var = random.randint(0,self.__var_size)
            val = random.randint(0,self.__val_size)
            r = Rule(var,val)

            self.assertEqual(r.get_condition(random.randint(0,self.__var_size)),None)

        for i in range(self.__nb_unit_test):
            r = self.random_rule()
            body = r.get_body()

            if (len(body) == 0):
                i -= 1
                continue

            cond = random.randint(0,len(body)-1)
            var, val = body[cond]

            self.assertEqual(r.get_condition(var),val)

    def test_has_condition(self):
        print(">> Rule.has_condition(self, variable)")

        for i in range(self.__nb_unit_test):
            r = self.random_rule()

            body = r.get_body()

            for var, val in body:
                self.assertTrue(r.has_condition(var))

            for j in range(10):
                conditions = [var for var, val in body]

                var = random.randint(0,self.__var_size)
                if var in conditions:
                    self.assertTrue(r.has_condition(var))
                else:
                    self.assertFalse(r.has_condition(var))

    def test_matches(self):
        print(">> Rule.matches(self, state)")

        for i in range(self.__nb_unit_test):
            r = self.random_rule()

            state = [random.randint(0,self.__val_size) for var in range(self.__var_size+1)]

            body = r.get_body()

            for var, val in body:
                state[var] = val

            self.assertTrue(r.matches(state))

            if len(body) == 0:
                i -= 1
                continue

            state = [random.randint(0,self.__val_size) for var in range(self.__var_size)]

            var, val = random.choice(body)
            val_ = val

            while val_ == val:
                val_ = random.randint(0,self.__val_size)

            state[var] = val_

            self.assertFalse(r.matches(state))

            # delay greater than state
            greater_var_id = 0
            for var, val in r.get_body():
                greater_var_id = max(greater_var_id, var)
            state = state[:greater_var_id]

            self.assertFalse(r.matches(state))

    def test_cross_matches(self):
        print(">> Rule.cross_matches(self, other)")

        for i in range(self.__nb_unit_test):
            r = self.random_rule()
            r_ = self.random_rule()

            #eprint(r.to_string())
            #eprint(r_.to_string())

            if r.cross_matches(r_):

                for var, val in r.get_body():
                    if r_.has_condition(var):
                        self.assertEqual(r_.get_condition(var), val)

                for var, val in r_.get_body():
                    if r.has_condition(var):
                        self.assertEqual(r.get_condition(var), val)

            else:
                exist_difference = False
                for var, val in r.get_body():
                    if r_.has_condition(var) and r_.get_condition(var) != val:
                        exist_difference = True
                        break
                self.assertTrue(exist_difference)

    def test___eq__(self):
        print(">> Rule.__eq__(self, other)")

        for i in range(self.__nb_unit_test):
            r = self.random_rule()

            self.assertTrue(r == r)
            self.assertFalse(r != r)

            r_ = self.random_rule()

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
            r_ = self.random_rule()
            while r_.size() == r.size():
                r_ = self.random_rule()

            r_.set_head_variable(r.get_head_variable())
            r_.set_head_value(r.get_head_value())
            self.assertFalse(r == r_)

            # Same size, same head
            while r.size() == 0:
                r = self.random_rule()
            r_ = r.copy()
            var, val = random.choice(r.get_body())

            new_val = val
            while new_val == val:
                new_val = random.randint(0, self.__val_size)
                r_.remove_condition(var)
                r_.add_condition(var, new_val)

            self.assertFalse(r == r_)

    def test_subsumes(self):
        print(">> Rule.__eq__(self, other)")

        for i in range(self.__nb_unit_test):
            r0 = Rule(0,0)

            r1 = self.random_rule()
            while r1.size() == 0:
                r1 = self.random_rule()
            r2 = r1.copy()
            r3 = r1.copy()
            r4 = self.random_rule()

            # r2 is a simplification of r1
            if r2.size() > 0:
                removed = random.randint(1,len(r2.get_body()))
                for j in range(0,removed):
                    var = random.choice(r2.get_body())[0]
                    r2.remove_condition(var)

            # r3 is a complication of r1
            to_add = random.randint(1,self.__var_size - len(r3.get_body()))
            while to_add > 0:
                var = random.randint(0, self.__var_size)
                val = random.randint(0, self.__val_size)
                if r3.has_condition(var):
                    continue
                r3.add_condition(var,val)
                to_add -= 1

            # Self subsumption
            self.assertTrue(r0.subsumes(r0))
            self.assertTrue(r1.subsumes(r1))
            self.assertTrue(r2.subsumes(r2))
            self.assertTrue(r3.subsumes(r3))

            # Empty rule
            self.assertTrue(r0.subsumes(r1))
            self.assertTrue(r0.subsumes(r2))
            self.assertEqual(r1.subsumes(r0), len(r1.get_body()) == 0)
            self.assertEqual(r2.subsumes(r0), len(r2.get_body()) == 0)

            # Reduction
            self.assertTrue(r2.subsumes(r1))
            self.assertFalse(r1.subsumes(r2))

            # Extension
            self.assertTrue(r1.subsumes(r3))
            self.assertFalse(r3.subsumes(r1))

            # Random
            if r1.subsumes(r4):
                for var, val in r1.get_body():
                    self.assertEqual(val, r4.get_condition(var))
            else:
                conflicts = 0
                for var, val in r1.get_body():
                    if r4.get_condition(var) != val:
                        conflicts += 1
                self.assertTrue(conflicts > 0)

    def test_add_condition(self):
        print(">> Rule.add_condition(self, variable, value)")

        # Empty rule
        for i in range(self.__nb_unit_test):
            var = random.randint(0,self.__var_size)
            val = random.randint(0,self.__val_size)
            r = Rule(var,val)

            var = random.randint(0,self.__var_size)
            val = random.randint(0,self.__val_size)

            self.assertFalse(r.has_condition(var))
            r.add_condition(var,val)
            self.assertEqual(r.get_condition(var), val)

        for i in range(self.__nb_unit_test):
            r = self.random_rule()

            if len(r.get_body()) == 0:
                i -= 1
                continue

            conditions = []
            for var, val in r.get_body():
                conditions.append(var)

            var = conditions[0]
            while var in conditions:
                var = random.randint(0,self.__var_size)

            val = random.randint(0,self.__val_size)

            self.assertFalse(r.has_condition(var))
            r.add_condition(var,val)
            self.assertEqual(r.get_condition(var), val)

    def test_remove_condition(self):
        print(">> Rule.remove_condition(self, variable)")

        # Empty rule
        for i in range(self.__nb_unit_test):
            var = random.randint(0,self.__var_size)
            val = random.randint(0,self.__val_size)
            r = Rule(var,val)

            var = random.randint(0,self.__var_size)

            self.assertFalse(r.has_condition(var))
            r.remove_condition(var)
            self.assertFalse(r.has_condition(var))

        for i in range(self.__nb_unit_test):
            r = self.random_rule()

            if len(r.get_body()) == 0:
                i -= 1
                continue

            var, val = random.choice(r.get_body())

            self.assertTrue(r.has_condition(var))
            r.remove_condition(var)
            self.assertFalse(r.has_condition(var))

    #------------------
    # Tool functions
    #------------------

    def random_rule(self):
        var = random.randint(0,self.__var_size-1)
        val = random.randint(0,self.__val_size-1)
        body = []
        conditions = []
        nb_conditions = random.randint(0,self.__body_size)
        while len(body) < nb_conditions:
            var = random.randint(0,self.__var_size-1)
            val = random.randint(0,self.__val_size-1)
            if var not in conditions:
                body.append( (var, val) )
                conditions.append(var)
        r = Rule(var,val,body)

        return r

'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
