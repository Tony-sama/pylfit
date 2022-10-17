#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/26
# @updated: 2019/05/02
#
# @desc: PyLFIT unit test script
#
#-----------------------

import unittest
import random
import sys

from pylfit.utils import eprint
from pylfit.objects.continuum import Continuum
from pylfit.objects.continuumRule import ContinuumRule

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from tests_generator import random_Continuum, random_ContinuumRule, random_CLP, random_continuous_state

random.seed(0)

class ContinuumRuleTest(unittest.TestCase):
    """
        Unit test of class ContinuumRule from continuumRule.py
    """

    _nb_tests = 10

    _nb_transitions = 10

    _nb_features = 3

    _nb_targets = 3

    _min_epsilon = 0.3

    """ must be < _max_value"""
    _min_value = -100.0

    """ must be > _min_value"""
    _max_value = 100.0

    _min_domain_size = 1.0

    _min_continuum_size = 1

    _nb_rules = 10

    _body_size = 10


    #------------------
    # Constructors
    #------------------

    def test_constructor_empty(self):
        eprint(">> ContinuumRule.__init__(self, head_variable, head_value)")

        for i in range(self._nb_tests):
            var_id = random.randint(0, self._nb_features)
            domain = random_Continuum(self._min_value, self._max_value)
            c = ContinuumRule(var_id, domain)

            self.assertEqual(c.head_variable, var_id)
            self.assertEqual(c.head_value, domain)
            self.assertEqual(c.body, [])

    def test_constructor_full(self):
        eprint(">> ContinuumRule.__init__(self, head_variable, head_value, body)")

        for i in range(self._nb_tests):
            head_var_id = random.randint(0, self._nb_features)
            head_domain = random_Continuum(self._min_value, self._max_value)

            size = random.randint(1, self._nb_features)
            body = []
            locked = []
            for j in range(size):
                var_id = random.randint(0, self._nb_features)
                domain = random_Continuum(self._min_value, self._max_value)
                if var_id not in locked:
                    body.append( (var_id, domain) )
                    locked.append(var_id)

            r = ContinuumRule(head_var_id, head_domain, body)

            self.assertEqual(r.head_variable, head_var_id)
            self.assertEqual(r.head_value, head_domain)
            for e in body:
                self.assertTrue(e in r.body)
            for e in r.body:
                self.assertTrue(e in body)

    def test_copy(self):
        eprint(">> ContinuumRule.copy(self)")

        for i in range(self._nb_tests):
            head_var_id = random.randint(0, self._nb_features)
            head_domain = random_Continuum(self._min_value, self._max_value)

            size = random.randint(1, self._nb_features)
            body = []
            locked = []
            for j in range(size):
                var_id = random.randint(0, self._nb_features)
                domain = random_Continuum(self._min_value, self._max_value)
                if var_id not in locked:
                    body.append( (var_id, domain) )
                    locked.append(var_id)

            r_ = ContinuumRule(head_var_id, head_domain, body)
            r = r_.copy()

            self.assertEqual(r.head_variable, head_var_id)
            self.assertEqual(r.head_value, head_domain)
            for e in body:
                self.assertTrue(e in r.body)
            for e in r.body:
                self.assertTrue(e in body)

            self.assertEqual(r.head_variable, r_.head_variable)
            self.assertEqual(r.head_value, r_.head_value)
            for e in r_.body:
                self.assertTrue(e in r.body)
            for e in r.body:
                self.assertTrue(e in r_.body)

    #--------------
    # Observers
    #--------------

    def test_size(self):
        print(">> ContinuumRule.size(self)")
        for i in range(self._nb_tests):
            model = random_CLP( \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            algorithm="acedia")

            r = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)

            self.assertEqual(r.size(), len(r.body))

    def test_get_condition(self):
        print(">> ContinuumRule.get_condition(self, variable)")

        for i in range(self._nb_tests):
            model = random_CLP( \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            algorithm="acedia")

            # Empty rule
            var = random.randint(0,len(model.features))
            val = random_Continuum(self._min_value, self._max_value)
            r = ContinuumRule(var,val)

            for var in range(len(model.features)):
                self.assertEqual(r.get_condition(var),None)

            # Regular rule
            while r.size() == 0:
                r = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)
            body = r.body

            appears = []
            for var, val in body:
                appears.append(var)
                self.assertEqual(r.get_condition(var),val)

            for var in range(len(model.features)):
                if var not in appears:
                    self.assertEqual(r.get_condition(var),None)

    def test_has_condition(self):
        print(">> ContinuumRule.has_condition(self, variable)")

        for i in range(self._nb_tests):
            model = random_CLP( \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            algorithm="acedia")

            # Empty rule
            var = random.randint(0,len(model.features))
            val = random_Continuum(self._min_value, self._max_value)
            r = ContinuumRule(var,val)

            for var in range(len(model.features)):
                self.assertEqual(r.has_condition(var),False)

            # Regular rule
            while r.size() == 0:
                r = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)
            body = r.body

            appears = []
            for var, val in body:
                appears.append(var)
                self.assertEqual(r.has_condition(var),True)

            for var in range(len(model.features)):
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
        for i in range(self._nb_tests):

            model = random_CLP( \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            algorithm="acedia")

            r = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)

            string = str(r.head_variable) + "=" + r.head_value.to_string() + " :- "
            for a, b in r.body:
                string += str(a) + "=" + b.to_string() + ", "

            if len(r.body) > 0:
                string = string[:-2]
            string += "."

            self.assertEqual(r.to_string(), string)
            self.assertEqual(r.to_string(), str(r))
            self.assertEqual(r.to_string(), repr(r))
            #eprint(r)

    def test_logic_form(self):
        print(">> ContinuumRule.logic_form(self, features)")

        # One step rules
        for i in range(self._nb_tests):
            model = random_CLP( \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            algorithm="acedia")

            r = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)

            max_var_id = r.head_variable
            max_val_id = r.head_value

            string = model.targets[r.head_variable][0] + "(" + r.head_value.to_string() + ") :- "
            for var, val in r.body:
                string += model.features[var][0] + "(" + val.to_string() + "), "

            if len(r.body) > 0:
                string = string[:-2]
            string += "."
            self.assertEqual(r.logic_form(model.features, model.targets),string)

    def test_matches(self):
        print(">> ContinuumRule.matches(self, state)")

        for i in range(self._nb_tests):
            model = random_CLP( \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            algorithm="acedia")

            r = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)
            state = random_continuous_state(model.features)

            body = r.body

            for var, val in body:
                min = val.min_value
                max = val.max_value
                if not val.min_included:
                    min += self._min_continuum_size * 0.1
                if not val.max_included:
                    max -= self._min_continuum_size * 0.1
                state[var] = random.uniform(min, max)

            self.assertTrue(r.matches(state))

            while r.size() == 0:
                r = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)
            state = random_continuous_state(model.features)

            var, val = random.choice(r.body)
            val_ = val.max_value - ( (val.min_value - val.max_value) / 2.0 )

            while val.includes(val_):
                val_ = random.uniform(self._min_value-100.0, self._max_value+100.0)

            state[var] = val_

            self.assertFalse(r.matches(state))

            # no out of bound (delay enconding)
            self.assertFalse(r.matches([]))

    def test_dominates(self):
        print(">> ContinuumRule.dominates(self, rule)")

        for i in range(self._nb_tests):
            model = random_CLP( \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            algorithm="acedia")

            var = random.randint(0, len(model.features)-1)
            r = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)

            # Undominated rule: var = emptyset if anything
            r0 = ContinuumRule(r.head_variable,Continuum())

            self.assertTrue(r0.dominates(r0))

            # r1 is a regular rule
            r1 = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)
            while r1.size() == 0:
                r1 = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)
            var = r.head_variable
            val = random_Continuum(model.targets[var][1].min_value, model.targets[var][1].max_value)
            r1 = ContinuumRule(var, val, r1.body) # set head

            self.assertTrue(r0.dominates(r1))
            self.assertFalse(r1.dominates(r0))

            # r2 is precision of r1 (head specification)
            r2 = r1.copy()
            var = r2.head_variable
            val = r2.head_value
            while r2.head_value == val or not r2.head_value.includes(val):
                val = random_Continuum(val.min_value, val.max_value)
            r2.head_value = val

            self.assertEqual(r0.dominates(r2), True)
            self.assertEqual(r2.dominates(r0), False)
            self.assertEqual(r2.dominates(r1), True)
            self.assertEqual(r1.dominates(r2), False)

            # r3 is a generalization of r1 (body generalization)
            r3 = r1.copy()
            var, val = random.choice(r3.body)
            val_ = val
            modif_min = random.uniform(0, self._max_value)
            modif_max = random.uniform(0, self._max_value)
            while val_ == val or not val_.includes(val):
                val_ = random_Continuum(val.min_value-modif_min, val.max_value+modif_max)
            r3.set_condition(var, val_)

            self.assertEqual(r0.dominates(r3), True)
            self.assertEqual(r3.dominates(r0), False)
            self.assertEqual(r3.dominates(r1), True)
            self.assertEqual(r1.dominates(r3), False)

            # r4 is unprecision of r1 (head generalization)
            r4 = r1.copy()
            var = r4.head_variable
            val = r4.head_value
            modif_min = random.uniform(0, self._max_value)
            modif_max = random.uniform(0, self._max_value)
            while r4.head_value == val or not val.includes(r4.head_value):
                val = Continuum(val.min_value-modif_min, val.max_value+modif_max,random.choice([True,False]),random.choice([True,False]))
            r4.head_value = val

            self.assertEqual(r0.dominates(r4), True)
            self.assertEqual(r4.dominates(r0), False)
            self.assertEqual(r4.dominates(r1), False)
            self.assertEqual(r1.dominates(r4), True)

            # r5 is specialization of r1 (body specialization)
            r5 = r1.copy()
            var, val = random.choice(r5.body)
            val_ = val
            while val_ == val or not val.includes(val_):
                val_ = random_Continuum(val.min_value, val.max_value)
            r5.set_condition(var,val_)

            self.assertEqual(r0.dominates(r5), True)
            self.assertEqual(r5.dominates(r0), False)
            self.assertEqual(r5.dominates(r1), False)
            self.assertEqual(r1.dominates(r5), True)

            # r6 is a random rule
            r6 = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)

            # head var difference
            if r6.head_variable != r1.head_variable:
                self.assertFalse(r6.dominates(r1))
                self.assertFalse(r1.dominates(r6))

            r6 = ContinuumRule(r1.head_variable, r6.head_value, r6.body) # same head var
            #eprint("r1: ", r1)
            #eprint("r6: ", r6)

            # head val inclusion
            if not r1.head_value.includes(r6.head_value):
                self.assertFalse(r6.dominates(r1))

            r6 = ContinuumRule(r1.head_variable, r1.head_value, r6.body) # same head var, same head val

            # body subsumption
            if r1.dominates(r6):
                for var,val in r1.body:
                    self.assertTrue(val.includes(r6.get_condition(var)))

            # body subsumption
            if r6.dominates(r1):
                for var,val in r6.body:
                    self.assertTrue(val.includes(r1.get_condition(var)))

            # incomparable
            if not r1.dominates(r6) and not r6.dominates(r1):
                #eprint("r1: ", r1)
                #eprint("r6: ", r6)

                conflicts = False
                dominant_r1 = False
                dominant_r6 = False
                for var, val in r1.body:
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

                for var, val in r6.body:
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

        for i in range(self._nb_tests):
            model = random_CLP( \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            algorithm="acedia")

            var = random.randint(0, len(model.features)-1)

            # empty rule
            r = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)
            r = ContinuumRule(r.head_variable, r.head_value, [])

            for j in range(len(model.features)):

                var = random.randint(0, len(model.features)-1)
                val = random_Continuum(model.features[var][1].min_value, model.features[var][1].max_value)
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
                for v, val in r.body:
                    v > prev
                    prev = v

            # regular rule
            r = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)

            for j in range(len(model.features)):
                var = random.randint(0, len(model.features)-1)
                val = random_Continuum(model.features[var][1].min_value, model.features[var][1].max_value)
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
                for v, val in r.body:
                    v > prev
                    prev = v

    def test___eq__(self):
        print(">> ContinuumRule.__eq__(self, other)")

        for i in range(self._nb_tests):
            model = random_CLP( \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            algorithm="acedia")

            r = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)

            self.assertTrue(r == r)
            self.assertFalse(r != r)

            r_ = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)

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
            r_ = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)
            while r_.size() == r.size():
                r_ = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)

            r_.head_variable = r.head_variable
            r_.head_value = r.head_value
            self.assertFalse(r == r_)

            # Same size, same head
            while r.size() == 0:
                r = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)
            r_ = r.copy()
            var, val = random.choice(r.body)

            new_val = val
            while new_val == val:
                new_val = random_Continuum(model.features[var][1].min_value, model.features[var][1].max_value)
                r_.set_condition(var, new_val)

            self.assertFalse(r == r_)

    def test_remove_condition(self):
        print(">> ContinuumRule.remove_condition(self, variable)")

        # Empty rule
        for i in range(self._nb_tests):
            model = random_CLP( \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            algorithm="acedia")

            var = random.randint(0,len(model.features)-1)
            val = random_Continuum(model.features[var][1].min_value, model.features[var][1].max_value)
            r = ContinuumRule(var,val)

            var = random.randint(0,len(model.features)-1)

            self.assertFalse(r.has_condition(var))
            r.remove_condition(var)
            self.assertFalse(r.has_condition(var))

            while r.size() == 0:
                r = random_ContinuumRule(model.features, model.targets, self._min_continuum_size)

            var, val = random.choice(r.body)

            self.assertTrue(r.has_condition(var))
            r.remove_condition(var)
            self.assertFalse(r.has_condition(var))

    #------------------
    # Tool functions
    #------------------
'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
