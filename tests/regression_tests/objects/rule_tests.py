#-----------------------
# @author: Tony Ribeiro
# @created: 2019/03/25
# @updated: 2021/06/15
#
# @desc: PyLFIT unit test script
#
# done:
# - __init__
# - copy
# - size
# - to_string
# - logic_form
# - get_condition
# - has_condition
# - matches
# - cross_matches
# - __eq__
# - subsumes
# - add_condition
# - remove_condition
#
# Todo:
# - from_string
# - pop_condition
#
#-----------------------

import unittest
import random


import sys

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

import itertools

from tests_generator import random_rule, random_features, random_targets, random_constraint

from pylfit.utils import eprint
from pylfit.objects.rule import Rule


#random.seed(0)

class Rule_tests(unittest.TestCase):
    """
        Unit test of class Rule from rule.py
    """

    _nb_tests = 100

    _nb_features = 20

    _nb_targets = 15

    _nb_values = 10

    _max_delay = 5

    _max_body_size = 10


    #------------------
    # Constructors
    #------------------

    def test_empty_constructor(self):
        print(">> Rule.__init__(self, head_variable, head_value, nb_body_variables)")
        for i in range(self._nb_tests):
            var = random.randint(0,self._nb_targets-1)
            val = random.randint(0,self._nb_values-1)
            nb_var = random.randint(var,self._nb_features)
            r = Rule(var,val,nb_var)
            self.assertEqual(r.head_variable,var)
            self.assertEqual(r.head_value,val)
            self.assertEqual(r.body, [])

    def test_body_constructor(self):
        print(">> Rule.__init__(self, head_variable, head_value, nb_body_variables, body)")
        for i in range(self._nb_tests):
            var = random.randint(0,self._nb_targets-1)
            val = random.randint(0,self._nb_values-1)
            nb_var = random.randint(1,self._nb_features)
            body = []
            cond_var = set()
            for j in range(0, random.randint(0,nb_var)):
                var = random.randint(0,nb_var-1)
                if var not in cond_var:
                    body.append((var, random.randint(0,self._nb_values-1)))
                    cond_var.add(var)
                else:
                    j-=1
            r = Rule(var,val,nb_var,body)
            self.assertEqual(r.head_variable,var)
            self.assertEqual(r.head_value,val)
            self.assertEqual(sorted(body),sorted(r.body))

    def test_copy(self):
        print(">> Rule.copy(self)")
        for i in range(self._nb_tests):
            r1 = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))
            r2 = r1.copy()

            self.assertEqual(r1.head_variable, r2.head_variable)
            self.assertEqual(r1.head_value, r2.head_value)
            self.assertEqual(sorted(r1.body), sorted(r2.body))

    def test_from_string(self):
        print(">> Rule.from_string(string_format, features, targets)")
        for i in range(self._nb_tests):
            features = random_features(self._nb_features, self._nb_values)
            targets = random_targets(self._nb_targets, self._nb_values)

            head_var = random.randint(0,len(targets)-1)
            head_val = random.randint(0,len(targets[head_var][1])-1)
            body = []
            conditions = []
            nb_conditions = random.randint(0,min(len(features),self._max_body_size))
            while len(body) < nb_conditions:
                var = random.randint(0,len(features)-1)
                val = random.randint(0,len(features[var][1])-1)
                if var not in conditions:
                    body.append( (var, val) )
                    conditions.append(var)
            r = Rule(head_var,head_val,len(features),body)

            # generating string
            head = targets[r.head_variable][0] + "(" + targets[r.head_variable][1][r.head_value] + ")"
            body = ""
            for var, val in r.body:
                body += features[var][0] + "(" + features[var][1][val] + "),"

            body = body[:-1]

            r2 = Rule.from_string(head + " :- " + body, features, targets)

            self.assertEqual(r,r2)

            # Constraint
            head_var = -1
            head_val = -1
            body = []
            conditions = []
            nb_conditions = random.randint(0,min(len(features)+len(targets),self._max_body_size))
            while len(body) < nb_conditions:
                var = random.randint(0,len(features)+len(targets)-1)
                if var < len(features):
                    val = random.randint(0,len(features[var][1])-1)
                else:
                    val = random.randint(0,len(targets[var-len(features)][1])-1)
                if var not in conditions:
                    body.append( (var, val) )
                    conditions.append(var)
            r = Rule(head_var,head_val,len(features)+len(targets),body)

            # generating string
            body = ""
            for var, val in r.body:
                if var < len(features):
                    body += features[var][0] + "(" + features[var][1][val] + "),"
                else:
                    body += targets[var-len(features)][0] + "(" + targets[var-len(features)][1][val] + "),"


            body = body[:-1]

            r2 = Rule.from_string(":- " + body, features+targets, [])

            self.assertEqual(r,r2)
    #--------------
    # Observers
    #--------------

    def test_size(self):
        print(">> Rule.size(self)")
        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            self.assertEqual(r.size(), len(r.body))

    def test_to_string(self):
        print(">> Rule.to_string(self)")
        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            string = str(r.head_variable) + "=" + str(r.head_value) + " :- "
            for a, b in r.body:
                string += str(a) + "=" + str(b) + ", "

            if len(r.body) > 0:
                string = string[:-2]
            string += "."

            self.assertEqual(r.to_string(),string)
            self.assertEqual(r.to_string(), r.__str__())
            self.assertEqual(r.to_string(), r.__repr__())
            self.assertEqual(r.__hash__(),hash(r.to_string()))

    def test_logic_form(self):
        print(">> Rule.logic_form(self, features, targets)")

        # One step rules
        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            max_var_id = r.head_variable
            max_val_id = r.head_value
            for var, val in r.body:
                if var > max_var_id:
                    max_var_id = var
                if val > max_val_id:
                    max_val_id = val

            features = [("x"+str(var), ["v"+str(val) for val in range(max_val_id+1)]) for var in range(max_var_id+1)]
            targets = [("y"+str(var), ["v"+str(val) for val in range(max_val_id+1)]) for var in range(max_var_id+1)]

            string = targets[r.head_variable][0] + "(" + targets[r.head_variable][1][r.head_value] + ") :- "
            for var, val in r.body:
                string += features[var][0] + "(" + features[var][1][val] + "), "

            if len(r.body) > 0:
                string = string[:-2]
            string += "."

            self.assertEqual(r.logic_form(features, targets),string)

            # cosntraints
            c = random_constraint(len(features), len(targets), max_val_id+1, min(len(features)+len(targets), self._max_body_size))
            #eprint(c.logic_form(features,targets))
            string = ":- "
            for var, val in c.body:
                if var >= len(features):
                    string += targets[var-len(features)][0] + "(" + targets[var-len(features)][1][val] + "), "
                else:
                    string += features[var][0] + "(" + features[var][1][val] + "), "
            if len(c.body) > 0:
                string = string[:-2]
            string += "."
            self.assertEqual(c.logic_form(features,targets), string)

            # exceptions
            self.assertRaises(ValueError, r.logic_form, features, [])
            self.assertRaises(ValueError, r.logic_form, features, [(i,[]) for i,j in targets])

            r1 = r.copy()
            r1.add_condition(-1,0)
            self.assertRaises(ValueError, r1.logic_form, features, targets)

            self.assertRaises(ValueError, r.logic_form, features, [])

            if r.size() > 0:
                self.assertRaises(ValueError, r.logic_form, [], targets)

            if c.size() > 0:
                self.assertRaises(ValueError, c.logic_form, [], [])



    def test_get_condition(self):
        print(">> Rule.get_condition(self, variable)")

        # Empty rule
        for i in range(self._nb_tests):
            var = random.randint(0,self._nb_targets-1)
            val = random.randint(0,self._nb_values-1)
            r = Rule(var,val,self._nb_features)

            self.assertEqual(r.get_condition(random.randint(0,self._nb_features-1)),-1)

        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))
            body = r.body

            if (len(body) == 0):
                i -= 1
                continue

            cond = random.randint(0,len(body)-1)
            var, val = body[cond]

            self.assertEqual(r.get_condition(var),val)

    def test_has_condition(self):
        print(">> Rule.has_condition(self, variable)")

        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            body = r.body

            for var, val in body:
                self.assertTrue(r.has_condition(var))

            for j in range(10):
                conditions = [var for var, val in body]

                var = random.randint(0,self._nb_features-1)
                if var in conditions:
                    self.assertTrue(r.has_condition(var))
                else:
                    self.assertFalse(r.has_condition(var))

    def test_matches(self):
        print(">> Rule.matches(self, state)")

        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            state = [random.randint(0,self._nb_values) for var in range(self._nb_features+1)]

            body = r.body

            for var, val in body:
                state[var] = val

            self.assertTrue(r.matches(state))

            if len(body) == 0:
                i -= 1
                continue

            state = [random.randint(0,self._nb_values) for var in range(self._nb_features)]

            var, val = random.choice(body)
            val_ = val

            while val_ == val:
                val_ = random.randint(0,self._nb_values)

            state[var] = val_

            self.assertFalse(r.matches(state))

            # delay greater than state
            greater_var_id = 0
            for var, val in r.body:
                greater_var_id = max(greater_var_id, var)
            state = state[:greater_var_id]

            self.assertFalse(r.matches(state))

    def test_cross_matches(self):
        print(">> Rule.cross_matches(self, other)")

        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))
            r_ = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            #eprint(r.to_string())
            #eprint(r_.to_string())

            if r.cross_matches(r_):

                for var, val in r.body:
                    if r_.has_condition(var):
                        self.assertEqual(r_.get_condition(var), val)

                for var, val in r_.body:
                    if r.has_condition(var):
                        self.assertEqual(r.get_condition(var), val)

            else:
                exist_difference = False
                for var, val in r.body:
                    if r_.has_condition(var) and r_.get_condition(var) != val:
                        exist_difference = True
                        break
                self.assertTrue(exist_difference)

#--------------
# Operators
#--------------

    def test___eq__(self):
        print(">> Rule.__eq__(self, other)")

        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            self.assertTrue(r == r)
            self.assertFalse(r != r)

            r_ = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            if r.head_variable != r_.head_variable:
                self.assertFalse(r == r_)

            if r.head_value != r_.head_value:
                self.assertFalse(r == r_)

            if r.body != r_.body:
                self.assertFalse(r == r_)

            # different type
            self.assertFalse(r == "")
            self.assertFalse(r == 0)
            self.assertFalse(r == [])
            self.assertFalse(r == [1,2,3])

            # different size
            r_ = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))
            while r_.size() == r.size():
                r_ = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            r_.head_variable = r.head_variable
            r_.head_value = r.head_value
            self.assertFalse(r == r_)

            # Same size, same head
            while r.size() == 0:
                r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))
            r_ = r.copy()
            var, val = random.choice(r.body)

            new_val = val
            while new_val == val:
                new_val = random.randint(0, self._nb_values)
                r_.remove_condition(var)
                r_.add_condition(var, new_val)

            self.assertFalse(r == r_)

            r1 = Rule(r.head_variable, r.head_value, len(r._body_values)+10, [(len(r._body_values)+2,0) for i,j in r.body])
            self.assertFalse(r1 == r)
            self.assertFalse(r == r1)

    def test_subsumes(self):
        print(">> Rule.__eq__(self, other)")

        for i in range(self._nb_tests):
            r0 = Rule(0,0,self._nb_features)

            r1 = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))
            while r1.size() == 0:
                r1 = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))
            r2 = r1.copy()
            r3 = r1.copy()
            r4 = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            # r2 is a simplification of r1
            if r2.size() > 0:
                removed = random.randint(1,len(r2.body))
                for j in range(0,removed):
                    var = random.choice(r2.body)[0]
                    r2.remove_condition(var)

            # r3 is a complication of r1
            to_add = random.randint(1,self._nb_features - len(r3.body))
            while to_add > 0:
                var = random.randint(0, self._nb_features-1)
                val = random.randint(0, self._nb_values)
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
            self.assertEqual(r1.subsumes(r0), len(r1.body) == 0)
            self.assertEqual(r2.subsumes(r0), len(r2.body) == 0)

            # Reduction
            self.assertTrue(r2.subsumes(r1))
            self.assertFalse(r1.subsumes(r2))

            # Extension
            self.assertTrue(r1.subsumes(r3))
            self.assertFalse(r3.subsumes(r1))

            # Random
            if r1.subsumes(r4):
                for var, val in r1.body:
                    self.assertEqual(val, r4.get_condition(var))
            else:
                conflicts = 0
                for var, val in r1.body:
                    if r4.get_condition(var) != val:
                        conflicts += 1
                self.assertTrue(conflicts > 0)

    def test_add_condition(self):
        print(">> Rule.add_condition(self, variable, value)")

        # Empty rule
        for i in range(self._nb_tests):
            var = random.randint(0,self._nb_targets-1)
            val = random.randint(0,self._nb_values)
            r = Rule(var,val,self._nb_features)

            var = random.randint(0,self._nb_features-1)
            val = random.randint(0,self._nb_values)

            self.assertFalse(r.has_condition(var))
            r.add_condition(var,val)
            self.assertEqual(r.get_condition(var), val)

        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            if len(r.body) == 0:
                i -= 1
                continue

            conditions = []
            for var, val in r.body:
                conditions.append(var)

            var = conditions[0]
            while var in conditions:
                var = random.randint(0,self._nb_features-1)

            val = random.randint(0,self._nb_values)

            self.assertFalse(r.has_condition(var))
            r.add_condition(var,val)
            self.assertEqual(r.get_condition(var), val)

    def test_remove_condition(self):
        print(">> Rule.remove_condition(self, variable)")

        # Empty rule
        for i in range(self._nb_tests):
            var = random.randint(0,self._nb_targets-1)
            val = random.randint(0,self._nb_values)
            r = Rule(var,val,self._nb_features)

            var = random.randint(0,self._nb_features-1)

            self.assertFalse(r.has_condition(var))
            r.remove_condition(var)
            self.assertFalse(r.has_condition(var))

        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            if len(r.body) == 0:
                i -= 1
                continue

            var, val = random.choice(r.body)

            self.assertTrue(r.has_condition(var))
            r.remove_condition(var)
            self.assertFalse(r.has_condition(var))

    def test_pop_condition(self):
        print(">> Rule.pop_condition(self)")

        # Empty rule
        for i in range(self._nb_tests):
            var = random.randint(0,self._nb_targets-1)
            val = random.randint(0,self._nb_values)
            r = Rule(var,val,self._nb_features)

            var = random.randint(0,self._nb_features-1)

            r1 = r.copy()
            r1.pop_condition()
            self.assertEqual(r,r1)

        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            if len(r.body) == 0:
                i -= 1
                continue

            old_size = r.size()
            r1 = r.copy()
            r.pop_condition()

            self.assertTrue(r.size() == old_size-1)
            for var, val in r.body:
                self.assertTrue(r1.get_condition(var) == val)

    #------------------
    # Tool functions
    #------------------

'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
