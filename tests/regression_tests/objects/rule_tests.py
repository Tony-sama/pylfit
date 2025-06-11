#-----------------------
# @author: Tony Ribeiro
# @created: 2019/03/25
# @updated: 2023/12/07
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
import numpy

from tests_generator import random_legacy_atom, random_rule, random_features, random_targets, random_constraint

from pylfit.utils import eprint
from pylfit.objects.rule import Rule
from pylfit.objects.legacyAtom import LegacyAtom

random.seed(0)

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
        print(">> Rule.__init__(self, variable, value, body={})")
        for i in range(self._nb_tests):
            head = random_legacy_atom(self._nb_targets, self._nb_values)
            r = Rule(head)
            self.assertEqual(r.head,head)
            self.assertEqual(r.body, {})

    def test_body_constructor(self):
        print(">> Rule.__init__(self, variable, value, body={})")
        for i in range(self._nb_tests):
            head = random_legacy_atom(self._nb_targets, self._nb_values)
            nb_var = random.randint(1,self._nb_features)
            body = {}
            for j in range(0, random.randint(0,nb_var)):
                atom = random_legacy_atom(nb_var, self._nb_values)
                body[atom.variable] = atom

            r = Rule(head,body)
            self.assertEqual(r.head,head)
            self.assertEqual(r.body,body)

    def test_copy(self):
        print(">> Rule.copy(self)")
        for i in range(self._nb_tests):
            r1 = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))
            r2 = r1.copy()

            self.assertEqual(r1.head, r2.head)
            self.assertEqual(r1.body, r2.body)

    def test_from_string(self):
        print(">> Rule.from_string(string_format, features, targets)")
        for i in range(self._nb_tests):
            features = random_features(self._nb_features, self._nb_values)
            targets = random_targets(self._nb_targets, self._nb_values)
            r1 = random_rule(len(features), len(targets), self._nb_values, min(self._max_body_size,self._nb_features-1))
            r2 = Rule.from_string(r1.to_string(), features, targets)

            # Equality beside domain/state_position
            self.assertEqual(r1.head.variable,r2.head.variable)
            self.assertEqual(r1.head.value,r2.head.value)
            for var in r1.body:
                self.assertEqual(r1.body[var].value,r2.body[var].value)
            for var in r2.body:
                self.assertEqual(r2.body[var].value,r1.body[var].value)

            # Constraint case
            r1.head = None
            r2 = Rule.from_string(r1.to_string(), features, targets)
            self.assertEqual(r1.head,r2.head)
            for var in r1.body:
                self.assertEqual(r1.body[var].value,r2.body[var].value)
            for var in r2.body:
                self.assertEqual(r2.body[var].value,r1.body[var].value)
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

            string = str(r.head) + " :- "
            variables = list(r.body.keys())
            variables.sort()
            for var in variables:
                string += str(r.body[var]) + ", "

            if len(r.body) > 0:
                string = string[:-2]
            string += "."

            self.assertEqual(r.to_string(),string)
            self.assertEqual(r.to_string(), r.__str__())
            self.assertEqual(r.to_string(), r.__repr__())
            self.assertEqual(r.__hash__(),hash(r.to_string()))

    def test_get_condition(self):
        print(">> Rule.get_condition(self, variable)")

        # Empty rule
        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))
            r.body = {}

            self.assertEqual(r.get_condition(random.randint(0,self._nb_features-1)),None)

        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))
            if (len(r.body) == 0):
                i -= 1
                continue

            for var in r.body:
                self.assertEqual(r.get_condition(var),r.body[var])

    def test_has_condition(self):
        print(">> Rule.has_condition(self, variable)")

        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            for var in r.body:
                self.assertTrue(r.has_condition(var))
                self.assertFalse(r.has_condition(var+"_"))

    def test_matches(self):
        print(">> Rule.matches(self, state)")

        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            state = [str(random.randint(0,self._nb_values)) for var in range(self._nb_features+1)]

            for var in r.body:
                state[r.body[var].state_position] = r.body[var].value

            self.assertTrue(r.matches(state))

            if len(r.body) == 0:
                i -= 1
                continue

            state = [random.randint(0,self._nb_values) for var in range(self._nb_features)]

            var = random.choice(list(r.body.keys()))
            val_ = r.body[var].value

            while val_ == r.body[var].value:
                val_ = str(random.randint(0,self._nb_values))

            state[r.body[var].state_position] = val_

            self.assertFalse(r.matches(state))

    def test_partial_matches(self):
        print(">> Rule.partial_matches(self, state)")

        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            state = [str(random.randint(0,self._nb_values)) for var in range(self._nb_features+1)]

            for var in r.body:
                state[r.body[var].state_position] = r.body[var].value

            self.assertEqual(r.partial_matches(state,["?"]), Rule._FULL_MATCH)

            if len(r.body) == 0:
                i -= 1
                continue

            for var in r.body:
                state[r.body[var].state_position] = "?"
                if random.choice([True,False]):
                    break

            self.assertEqual(r.partial_matches(state,["?"]), Rule._PARTIAL_MATCH)

            var = random.choice(list(r.body.keys()))
            val = random.choice(list(r.body[var].domain))

            while val == r.body[var].value:
                val = random.choice(list(r.body[var].domain))

            state[r.body[var].state_position] = val

            self.assertEqual(r.partial_matches(state,["?"]), Rule._NO_MATCH)


    def test_cross_matches(self):
        print(">> Rule.cross_matches(self, other)")

        for i in range(self._nb_tests):
            continue # DBG
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

    def test_subsumes(self):
        print(">> Rule.subsumes(self, rule)")

        for i in range(self._nb_tests):
            r0 = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))
            r0.body = {}

            r1 = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))
            while r1.size() == 0:
                r1 = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))
            r2 = r1.copy()
            r3 = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            # r2 is a simplification of r1
            if r2.size() > 0:
                removed = random.randint(1,len(r2.body))
                for j in range(0,removed):
                    var = random.choice(list(r2.body.keys()))
                    r2.remove_condition(var)

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

            # Random
            if r1.subsumes(r3):
                for var in r1.body:
                    self.assertEqual(r1.get_condition(var), r3.get_condition(var))
            else:
                conflicts = 0
                for var in r1.body:
                    if r3.get_condition(var) != r1.body[var].value:
                        conflicts += 1
                self.assertTrue(conflicts > 0)

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

            if r.head != r_.head:
                self.assertFalse(r == r_)

            if r.body != r_.body:
                self.assertFalse(r == r_)

            # different type
            self.assertFalse(r == "")
            self.assertFalse(r == 0)
            self.assertFalse(r == [])
            self.assertFalse(r == [1,2,3])

    

    def test_add_condition(self):
        print(">> Rule.add_condition(self, atom)")

        # Empty rule
        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))
            r.body = {}

            atom = random_legacy_atom(self._nb_features, self._nb_values)

            self.assertFalse(r.has_condition(atom.variable))
            r.add_condition(atom)
            self.assertEqual(r.get_condition(atom.variable), atom)

        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            if len(r.body) == 0:
                i -= 1
                continue
            
            atom =  random_legacy_atom(self._nb_features, self._nb_values)
            r.add_condition(atom)
            self.assertEqual(r.get_condition(atom.variable), atom)

    def test_remove_condition(self):
        print(">> Rule.remove_condition(self, variable)")

        # Empty rule
        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))
            r.body = {}

            atom = random_legacy_atom(self._nb_features, self._nb_values)

            self.assertFalse(r.has_condition(atom.variable))
            r.remove_condition(atom.variable)
            self.assertFalse(r.has_condition(atom.variable))

        for i in range(self._nb_tests):
            r = random_rule(self._nb_features, self._nb_targets, self._nb_values, min(self._max_body_size,self._nb_features-1))

            if len(r.body) == 0:
                i -= 1
                continue

            var = random.choice(list(r.body.keys()))
            size = r.size()
            self.assertTrue(r.has_condition(var))
            r.remove_condition(var)
            self.assertFalse(r.has_condition(var))
            self.assertEqual(r.size(),size-1)


    def test_least_specialization(self):
        print(">> Rule.least_specialization(self, state, features)")

        
        for i in range(self._nb_tests):
            features = dict()
            state = [str(random.randint(0,self._nb_values)) for var in range(self._nb_features+1)]
            for idx, val in enumerate(state):
                dom = set(random.choices(state,k=random.randint(1,len(state))))
                dom.add(state[idx])
                atom = LegacyAtom(str(idx), dom, val, idx)
                features[atom.variable] = atom.void_atom()

            # Empty rule
            r = random_rule(len(state), self._nb_targets, self._nb_values, min(self._max_body_size,len(state)-1))
            r.body = {}

            least_specs = r.least_specialization(state, features)

            for ls in least_specs:
                self.assertEqual(ls.size(),r.size()+1)
                self.assertTrue(r.subsumes(ls))

            # Regular rule
            r = random_rule(len(state), self._nb_targets, self._nb_values, min(self._max_body_size,len(state)-1))

            least_specs = r.least_specialization(state, features)

            for ls in least_specs:
                changes = 0
                new_cond = 0
                for var in r.body:
                    if ls.get_condition(var) != r.get_condition(var):
                        changes += 1
                for var in ls.body:
                    if not r.has_condition(var):
                        new_cond += 1
                self.assertEqual(changes+new_cond,1)
                self.assertFalse(ls.matches(state))

    #------------------
    # Tool functions
    #------------------

'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
