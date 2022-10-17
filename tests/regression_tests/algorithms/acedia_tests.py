#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/30
# @updated: 2022/08/31
#
# @desc: PyLFIT unit test script
#
#-----------------------

import sys
import unittest
import random
import os
import contextlib
import io

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from pylfit.utils import eprint
from pylfit.objects.continuum import Continuum
from pylfit.objects.continuumRule import ContinuumRule
from pylfit.algorithms.acedia import ACEDIA

from tests_generator import random_ContinuousStateTransitionsDataset, random_ContinuumRule, random_Continuum

#seed = random.randint(0,1000000)
#seed = 381009
#random.seed(seed)
#eprint("seed: ", seed)

random.seed(0)

class ACEDIATest(unittest.TestCase):
    """
        Unit test of class ACEDIA from acedia.py
    """

    _nb_tests = 10

    _nb_transitions = 10

    _nb_features = 2

    _nb_targets = 2

    _min_epsilon = 0.3

    """ must be < _max_value"""
    _min_value = 0.0

    """ must be > _min_value"""
    _max_value = 100.0

    _min_domain_size = 1.0

    _min_continuum_size = 1

    _nb_rules = 10

    _body_size = 10

    _tmp_file_path = "tmp/unit_test_acedia.tmp"

    #------------------
    # Test functions
    #------------------

    def test_fit(self):
        eprint(">> ACEDIA.fit(dataset, targets_to_learn, verbose, threads)")

        for i in range(self._nb_tests):
            for verbose in [0,1]:
                for threads in [1,2]:

                    # Exceptions
                    dataset = "" # not a StateTransitionsDataset
                    f = io.StringIO()
                    with contextlib.redirect_stderr(f):
                        self.assertRaises(ValueError, ACEDIA.fit, dataset, None, verbose, threads)

                    dataset = random_ContinuousStateTransitionsDataset( \
                    nb_transitions=random.randint(1, self._nb_transitions), \
                    nb_features=random.randint(1, self._nb_features), \
                    nb_targets=random.randint(1, self._nb_targets), \
                    min_value=self._min_value, max_value=self._max_value, min_continuum_size=self._min_continuum_size)

                    # Exceptions
                    self.assertRaises(ValueError, ACEDIA.fit, dataset, "")
                    self.assertRaises(ValueError, ACEDIA.fit, dataset, [""])

                    f = io.StringIO()
                    with contextlib.redirect_stderr(f):
                        rules = ACEDIA.fit(dataset=dataset, targets_to_learn=None, verbose=verbose, threads=threads)

                    #eprint("learned: ", p_)

                    # All transitions are realized
                    #------------------------------

                    for head_var in range(len(dataset.targets)):
                        for s1, s2 in dataset.data:
                            for idx, val in enumerate(s2):
                                realized = 0
                                for r in rules:
                                    if r.head_variable == idx and r.head_value.includes(val) and r.matches(s1):
                                        realized += 1
                                        break
                                if realized <= 0:
                                    eprint("head_var: ", head_var)
                                    eprint("s1: ", s1)
                                    eprint("s2: ", s2)
                                    eprint("learned: ", rules)
                                self.assertTrue(realized >= 1) # One rule realize the example

                    # All rules are minimals
                    #------------------------
                    epsilon = 0.0001
                    for r in rules:

                        #eprint("r: ", r)

                        # Try reducing head min
                        #-----------------------
                        r_ = r.copy()
                        h = r_.head_value
                        if h.min_value + epsilon <= h.max_value:
                            r_.head_value = Continuum(h.min_value+epsilon, h.max_value, h.min_included, h.max_included)

                            #eprint("spec: ", r_)

                            conflict = False
                            for s1, s2 in dataset.data:
                                if not r_.head_value.includes(s2[r_.head_variable]) and r_.matches(s1): # Miss a positive example
                                    conflict = True
                                    #eprint("conflict")
                                    break

                            if not conflict:
                                eprint("Non minimal rule: ", r)
                                eprint("head can be specialized into: ", r_.head_variable, "=", r_.head_value)

                            self.assertTrue(conflict)

                        # Try reducing head max
                        #-----------------------
                        r_ = r.copy()
                        h = r_.head_value
                        if h.max_value - epsilon >= h.min_value:
                            r_.head_value = Continuum(h.min_value, h.max_value-epsilon, h.min_included, h.max_included)

                            #eprint("spec: ", r_)

                            conflict = False
                            for s1, s2 in dataset.data:
                                if not r_.head_value.includes(s2[r_.head_variable]) and r_.matches(s1): # Miss a positive example
                                    conflict = True
                                    #eprint("conflict")
                                    break

                            if not conflict:
                                eprint("Non minimal rule: ", r)
                                eprint("head can be generalized to: ", r_.head_variable, "=", r_.head_value)

                            self.assertTrue(conflict)

                        # Try extending condition
                        #-------------------------
                        for (var,val) in r.body:

                            # Try extend min
                            r_ = r.copy()
                            if val.min_value - epsilon >= dataset.features[var][1].min_value:
                                val_ = val.copy()
                                if not val_.min_included:
                                    val_.set_lower_bound(val_.min_value, True)
                                else:
                                    val_.set_lower_bound(val_.min_value-epsilon, False)
                                r_.set_condition(var, val_)

                                #eprint("gen: ", r_)

                                conflict = False
                                for s1, s2 in dataset.data:
                                    if not r_.head_value.includes(s2[r_.head_variable]) and r_.matches(s1): # Cover a negative example
                                        conflict = True
                                        #eprint("conflict")
                                        break

                                if not conflict:
                                    eprint("Non minimal rule: ", r)
                                    eprint("condition can be generalized: ", var, "=", val_)

                                self.assertTrue(conflict)

                            # Try extend max
                            r_ = r.copy()
                            if val.max_value + epsilon <= dataset.features[var][1].min_value:
                                val_ = val.copy()
                                if not val_.max_included:
                                    val_.set_upper_bound(val_.max_value, True)
                                else:
                                    val_.set_upper_bound(val_.max_value+epsilon, False)
                                r_.set_condition(var, val_)

                                #eprint("gen: ", r_)

                                conflict = False
                                for s1, s2 in datset.data:
                                    if not r_.head_value.includes(s2[r_.head_variable]) and r_.matches(s1): # Cover a negative example
                                        conflict = True
                                        #eprint("conflict")
                                        break

                                if not conflict:
                                    eprint("Non minimal rule: ", r)
                                    eprint("condition can be generalized: ", var, "=", val_)

                                self.assertTrue(conflict)


    def test_fit_var(self):
        eprint(">> ACEDIA.fit_var(variables, domains, transitions, variable)")

        for i in range(self._nb_tests):

            #eprint("\rTest ", i+1, "/", self._nb_tests, end='')

            dataset = random_ContinuousStateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1, self._nb_features), \
            nb_targets=random.randint(1, self._nb_targets), \
            min_value=self._min_value, max_value=self._max_value, min_continuum_size=self._min_continuum_size)


            #sys.exit()

            #eprint("Transitions: ")
            #for s1, s2 in t:
            #    eprint(s1, s2)
            #eprint("Transitions: ", t)
            head_var = random.randint(0, len(dataset.targets)-1)


            rules = ACEDIA.fit_var(dataset.features, dataset.data, head_var)

            #eprint("learned: ", p_)

            # All transitions are realized
            #------------------------------

            for s1, s2 in dataset.data:
                realized = 0
                for r in rules:
                    if r.head_variable == head_var and r.head_value.includes(s2[head_var]) and r.matches(s1):
                        realized += 1
                        break
                if realized <= 0:
                    eprint("head_var: ", head_var)
                    eprint("s1: ", s1)
                    eprint("s2: ", s2)
                    eprint("learned: ", rules)
                self.assertTrue(realized >= 1) # One rule realize the example

            # All rules are minimals
            #------------------------
            epsilon = 0.001
            for r in rules:

                #eprint("r: ", r)
                self.assertEqual(r.head_variable, head_var)

                # Try reducing head min
                #-----------------------
                r_ = r.copy()
                h = r_.head_value
                if h.min_value + epsilon <= h.max_value:
                    r_.head_value = Continuum(h.min_value+epsilon, h.max_value, h.min_included, h.max_included)

                    #eprint("spec: ", r_)

                    conflict = False
                    for s1, s2 in dataset.data:
                        if not r_.head_value.includes(s2[r_.head_variable]) and r_.matches(s1): # Cover a negative example
                            conflict = True
                            #eprint("conflict")
                            break

                    if not conflict:
                        eprint("Non minimal rule: ", r)
                        eprint("head can be specialized into: ", r_.head_variable, "=", r_.head_value)

                    self.assertTrue(conflict)

                # Try reducing head max
                #-----------------------
                r_ = r.copy()
                h = r_.head_value
                if h.max_value - epsilon >= h.min_value:
                    r_.head_value = Continuum(h.min_value, h.max_value-epsilon, h.min_included, h.max_included)

                    #eprint("spec: ", r_)

                    conflict = False
                    for s1, s2 in dataset.data:
                        if not r_.head_value.includes(s2[r_.head_variable]) and r_.matches(s1): # Cover a negative example
                            conflict = True
                            #eprint("conflict")
                            break

                    if not conflict:
                        eprint("Non minimal rule: ", r)
                        eprint("head can be generalized to: ", r_.head_variable, "=", r_.head_value)

                    self.assertTrue(conflict)

                # Try extending condition
                #-------------------------
                for (var,val) in r.body:

                    # Try extend min
                    r_ = r.copy()
                    if val.min_value - epsilon >= dataset.features[var][1].min_value:
                        val_ = val.copy()
                        if not val_.min_included:
                            val_.set_lower_bound(val_.min_value, True)
                        else:
                            val_.set_lower_bound(val_.min_value-epsilon, False)
                        r_.set_condition(var, val_)

                        #eprint("gen: ", r_)

                        conflict = False
                        for s1, s2 in dataset.data:
                            if not r_.head_value.includes(s2[r_.head_variable]) and r_.matches(s1): # Cover a negative example
                                conflict = True
                                #eprint("conflict")
                                break

                        if not conflict:
                            eprint("Non minimal rule: ", r)
                            eprint("condition can be generalized: ", var, "=", val_)

                        self.assertTrue(conflict)

                    # Try extend max
                    r_ = r.copy()
                    if val.max_value + epsilon <= dataset.features[var][1].max_value:
                        val_ = val.copy()
                        if not val_.max_included:
                            val_.set_upper_bound(val_.max_value, True)
                        else:
                            val_.set_upper_bound(val_.max_value+epsilon, False)
                        r_.set_condition(var, val_)

                        #eprint("gen: ", r_)

                        conflict = False
                        for s1, s2 in dataset.data:
                            if not r_.head_value.includes(s2[r_.head_variable]) and r_.matches(s1): # Cover a negative example
                                conflict = True
                                #eprint("conflict")
                                break

                        if not conflict:
                            eprint("Non minimal rule: ", r)
                            eprint("condition can be generalized: ", var, "=", val_)

                        self.assertTrue(conflict)

    def test_least_revision(self):
        eprint(">> ACEDIA.least_revision(rule, state_1, state_2)")

        for i in range(self._nb_tests):
            dataset = random_ContinuousStateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1, self._nb_features), \
            nb_targets=random.randint(1, self._nb_targets), \
            min_value=self._min_value, max_value=self._max_value, min_continuum_size=self._min_continuum_size)

            state_1, state_2 = random.choice(dataset.data)

            # not matching
            #--------------
            rule = random_ContinuumRule(dataset.features, dataset.targets, self._min_continuum_size)
            while rule.matches(state_1):
                rule = random_ContinuumRule(dataset.features, dataset.targets, self._min_continuum_size)

            self.assertRaises(ValueError, ACEDIA.least_revision, rule, state_1, state_2)

            # matching
            #--------------

            rule = random_ContinuumRule(dataset.features, dataset.targets, self._min_continuum_size)
            while not rule.matches(state_1):
                rule = random_ContinuumRule(dataset.features, dataset.targets, self._min_continuum_size)

            head_var = rule.head_variable
            target_val = state_2[rule.head_variable]

            # Consistent
            head_value = Continuum()
            while not head_value.includes(target_val):
                head_value = random_Continuum(dataset.targets[head_var][1].min_value, dataset.targets[head_var][1].max_value)
            rule.head_value = head_value
            self.assertRaises(ValueError, ACEDIA.least_revision, rule, state_1, state_2)

            # Empty set head
            rule.head_value = Continuum()

            LR = ACEDIA.least_revision(rule, state_1, state_2)
            lg = rule.copy()
            lg.head_value = Continuum(target_val,target_val,True,True)
            self.assertTrue(lg in LR)

            nb_valid_revision = 1

            for var, val in rule.body:
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
            head_value = random_Continuum(dataset.targets[head_var][1].min_value, dataset.targets[head_var][1].max_value)
            while head_value.includes(target_val):
                head_value = random_Continuum(dataset.targets[head_var][1].min_value, dataset.targets[head_var][1].max_value)
            rule.head_value = head_value

            LR = ACEDIA.least_revision(rule, state_1, state_2)

            lg = rule.copy()
            head_value = lg.head_value
            if target_val <= head_value.min_value:
                head_value.set_lower_bound(target_val, True)
            else:
                head_value.set_upper_bound(target_val, True)
            lg.head_value = head_value
            self.assertTrue(lg in LR)

            nb_valid_revision = 1

            for var, val in rule.body:
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



if __name__ == '__main__':
    """ Main """

    unittest.main()
