#-----------------------
# @author: Tony Ribeiro
# @created: 2021/05/10
# @updated: 2023/12/20
#
# @desc: BruteForce regression test script
# Tests algorithm methods on random dataset
# Done:
#   - BruteForce.fit(dataset)
#   - BruteForce.interprete(transitions, variable, value)
#   - BruteForce.fit_var_val(feature_domains, variable, value, negatives)
# Todo:
#   - check all minimal rules are learned
#   - check exceptions
#
#-----------------------

import unittest
import random
import os
import io
import contextlib

import sys
import numpy as np

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

import itertools

from tests_generator import random_DiscreteStateTransitionsDataset

from pylfit.utils import eprint
from pylfit.algorithms.bruteForce import BruteForce
from pylfit.objects.rule import Rule
from pylfit.objects.legacyAtom import LegacyAtom

from pylfit.datasets import DiscreteStateTransitionsDataset

random.seed(0)
np.random.seed(0)

class BruteForce_tests(unittest.TestCase):
    """
        Regression tests of class BruteForce from bruteForce.py
    """

    _nb_tests = 100

    _nb_transitions = 100

    _nb_features = 3

    _nb_targets = 3

    _nb_feature_values = 3

    _nb_target_values = 3

    #------------------
    # Test functions
    #------------------

    def test_fit(self):
        print(">> BruteForce.fit(dataset, verbose):")

        for test_id in range(self._nb_tests):

            # 0) exceptions
            #---------------

            # Datatset type
            dataset = "" # not a DiscreteStateTransitionsDataset
            self.assertRaises(ValueError, BruteForce.fit, dataset)

            # 1) No transitions
            #--------------------
            dataset = random_DiscreteStateTransitionsDataset( \
            nb_transitions=0, \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, max_target_values=self._nb_target_values)

            output = BruteForce.fit(dataset=dataset)

            # Output must be one empty rule for each target value
            self.assertEqual(len(output), len([val for (var,vals) in dataset.targets for val in vals]))

            expected = [Rule(LegacyAtom(var,set(vals),val,var_id)) for var_id, (var,vals) in enumerate(dataset.targets) for val_id, val in enumerate(vals)]
            #eprint(expected)
            #eprint(output)

            for r in expected:
                self.assertTrue(r in output)

            # 2) Random observations
            # ------------------------

            for impossibility_mode in [False,True]:
                for verbose in [0,1]:

                    # Generate transitions
                    dataset = random_DiscreteStateTransitionsDataset( \
                    nb_transitions=random.randint(1, self._nb_transitions), \
                    nb_features=random.randint(1,self._nb_features), \
                    nb_targets=random.randint(1,self._nb_targets), \
                    max_feature_values=self._nb_feature_values, \
                    max_target_values=self._nb_target_values)

                    #dataset.summary()

                    f = io.StringIO()
                    with contextlib.redirect_stderr(f):
                        output = BruteForce.fit(dataset=dataset, impossibility_mode=impossibility_mode, verbose=verbose)

                    # Encode data to check BruteForce output rules
                    data_encoded = dataset.data

                    # 2.1.1) Correctness (explain all)
                    # -----------------
                    # all transitions are fully explained, i.e. each target value is explained by atleast one rule
                    if impossibility_mode == False:
                        for (s1,s2) in dataset.data:
                            for var_id,(var,vals) in enumerate(dataset.targets):
                                realises_target = False

                                for r in output:
                                    if r.head.variable == var and r.head.matches(s2):
                                        realises_target = True
                                        #eprint(s1_encoded, " => ", target_id,"=",expected_value, " by ", r)
                                        break
                                if not realises_target:
                                    print(s1,s2)
                                    print(var_id,(var,vals))
                                self.assertTrue(realises_target)

                    #eprint("-------------------")
                    #eprint(data_encoded)
                    #eprint(output)

                    # 2.1.2) Correctness (no spurious observation)
                    # -----------------
                    # No rules generate a unobserved target value from an observed state
                    for r in output:
                        for (s1,s2) in data_encoded:
                            if r.matches(s1):
                                observed = False
                                for (s1_,s2_) in data_encoded: # Must be in a target state after s1
                                    if tuple(s1_) == tuple(s1) and r.head.matches(s2_):
                                        observed = True
                                        #eprint(r, " => ", s1_, s2_)
                                        break
                                if impossibility_mode:
                                    self.assertFalse(observed)
                                else:
                                    if not observed:
                                        eprint(r)
                                        eprint(s1)
                                        for (s1_,s2_) in data_encoded: # Must be in a target state after s1
                                            if tuple(s1_) == tuple(s1):
                                                eprint(s2_)
                                    self.assertTrue(observed)

                    # 2.2) Completness
                    # -----------------
                    # all possible initial state is matched by a rule of each target

                    # generate all combination of domains
                    if impossibility_mode == False:
                        encoded_domains = [set(domain) for (var, domain) in dataset.features]
                        init_states_encoded = set([i for i in list(itertools.product(*encoded_domains))])

                        for s in init_states_encoded:
                            for var, vals in dataset.targets:
                                realises_target = False
                                for r in output:
                                    if r.head.variable == var and r.matches(s):
                                        realises_target = True
                                        #eprint(s, " => ", target_id,"=",expected_value, " by ", r)
                                        break

                                self.assertTrue(realises_target)

                    # 2.3) minimality
                    # -----------------
                    # All rules conditions are necessary, i.e. removing a condition makes realizes unobserved target value from observation
                    # Encode data with DiscreteStateTransitionsDataset
                    data_grouped_by_init_state = []
                    for (s1,s2) in data_encoded:
                        added = False
                        for (s1_,S) in data_grouped_by_init_state:
                            if tuple(s1_) == tuple(s1):
                                if tuple(s2) not in S:
                                    S.append(tuple(s2))
                                added = True
                                break

                        if not added:
                            data_grouped_by_init_state.append((s1,[tuple(s2)])) # new init state

                    for r in output:
                        pos, neg = BruteForce.interprete(data_grouped_by_init_state, r.head)
                        if impossibility_mode:
                            pos_ = pos
                            pos = neg
                            neg = pos_
                        for var in r.body:
                            r_ = r.copy()
                            r_.remove_condition(var) # Try remove condition

                            minimal = False
                            for s in neg:
                                if r_.matches(s):
                                    minimal = True
                                    break

                            # # DEBUG:
                            if not minimal:
                                eprint("Simplification exists: "+r_.to_string())

                            # # DEBUG:
                            if not minimal:
                                eprint("not minimal "+r.to_string())
                                eprint("pos: ",pos)
                                eprint()
                                eprint("neg: ",neg)
                                eprint(output)

                            self.assertTrue(minimal)

    def test_interprete(self):
        print(">> BruteForce.interprete(transitions, head)")

        for i in range(self._nb_tests):
            # Generate transitions
            dataset = random_DiscreteStateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            # Encode data with DiscreteStateTransitionsDataset
            data_encoded = []
            for (s1,s2) in dataset.data:
                s1_encoded = [domain.index(s1[var_id]) for var_id, (var,domain) in enumerate(dataset.features)]
                s2_encoded = [domain.index(s2[var_id]) for var_id, (var,domain) in enumerate(dataset.targets)]
                data_encoded.append((s1_encoded,s2_encoded))

            #dataset.summary()

            # Group transitions by initial state
            data_grouped_by_init_state = []
            for (s1,s2) in data_encoded:
                added = False
                for (s1_,S) in data_grouped_by_init_state:
                    if s1_ == s1:
                        if s2 not in S:
                            S.append(s2)
                        added = True
                        break

                if not added:
                    data_grouped_by_init_state.append((s1,[s2])) # new init state

            #eprint(data_encoded)
            #eprint()
            #eprint(data_grouped_by_init_state)

            # each pos/neg interpretation
            for var_id, (var,vals) in enumerate(dataset.targets):
                for val_id, val in enumerate(vals):
                    #eprint("var_id: ", var_id)
                    #eprint("val_id: ", val_id)
                    head = LegacyAtom(var, set(vals), val, var_id)
                    pos, neg = BruteForce.interprete(data_grouped_by_init_state, head)

                    # All pos are valid
                    for s in pos:
                        found = False
                        for s1, s2 in data_encoded:
                            if s1 == s:
                                if head.matches(s2):
                                    found = True
                                    break
                        self.assertTrue(found)
                        self.assertFalse(s in neg)

                    # All neg are valid
                    for s in neg:
                        for s1, s2 in data_encoded:
                            if s1 == s:
                                self.assertFalse(head.matches(s2))
                        self.assertFalse(s in pos)

                    # All transitions are interpreted
                    for s1, S2 in data_grouped_by_init_state:
                        if len([s2 for s2 in S2 if head.matches(s2)]) == 0:
                            self.assertTrue(s1 in neg)
                        else:
                            self.assertTrue(s1 in pos)


    def test_fit_var_val(self):
        print(">> BruteForce.fit_var_val(features, variable, value, negatives)")

        for i in range(self._nb_tests):

            # Generate transitions
            dataset = random_DiscreteStateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            #dataset.summary()

            # Encode data with DiscreteStateTransitionsDataset
            data_encoded = []
            for (s1,s2) in dataset.data:
                s1_encoded = tuple(s1) #[domain.index(s1[var_id]) for var_id, (var,domain) in enumerate(dataset.features)]
                s2_encoded = tuple(s2) #[domain.index(s2[var_id]) for var_id, (var,domain) in enumerate(dataset.targets)]
                data_encoded.append((s1_encoded,s2_encoded))

            # Group transitions by initial state
            data_grouped_by_init_state = []
            for (s1,s2) in data_encoded:
                added = False
                for (s1_,S) in data_grouped_by_init_state:
                    if s1_ == s1:
                        if s2 not in S:
                            S.append(s2)
                        added = True
                        break

                if not added:
                    data_grouped_by_init_state.append((s1,[s2])) # new init state

            #eprint(data_grouped_by_init_state)

            # each target value
            for var_id, (var,vals) in enumerate(dataset.targets):
                for val_id, val in enumerate(vals):
                    #eprint("var: ", var_id)
                    #eprint("val: ", val_id)
                    head = LegacyAtom(var, set(vals), val, var_id)
                    pos, neg = BruteForce.interprete(data_grouped_by_init_state, head)
                    #eprint("neg: ", neg)
                    f = io.StringIO()
                    with contextlib.redirect_stderr(f):
                        output = BruteForce.fit_var_val(head, dataset.features_void_atoms, neg)
                    #eprint()
                    #eprint("rules: ", output)

                    # Check head
                    for r in output:
                        self.assertEqual(r.head, head)

                    # Each positive is explained
                    for s in pos:
                        cover = False
                        for r in output:
                            if r.matches(s):
                                cover = True

                        #if not cover:
                        #    print(s)
                        #    print(output)
                        self.assertTrue(cover) # One rule cover the example

                    # No negative is covered
                    for s in neg:
                        cover = False
                        for r in output:
                            if r.matches(s):
                                cover = True
                        self.assertFalse(cover) # no rule covers the example

                    # All rules are minimals
                    for r in output:
                        for var in r.body:
                            r_ = r.copy()
                            r_.remove_condition(var) # Try remove condition

                            conflict = False
                            for s in neg:
                                if r_.matches(s): # Cover a negative example
                                    conflict = True
                                    break
                            self.assertTrue(conflict)

    #------------------
    # Tool functions
    #------------------


if __name__ == '__main__':
    """ Main """

    unittest.main()
