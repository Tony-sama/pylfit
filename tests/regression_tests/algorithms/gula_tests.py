#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/15
# @updated: 2021/06/15
#
# @desc: GULA regression test script
# Tests algorithm methods on random dataset
# Done:
#   - GULA.fit(dataset, targets_to_learn)
#   - GULA.interprete(transitions, variable, value, supported_only=False)
#   - GULA.fit_var_val(feature_domains, variable, value, negatives, positives=None)
# Todo:
#   - check all minimal rules are learned
#   - check exceptions
#
#-----------------------

import unittest
import random
import os
import contextlib
import io

import sys

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

import itertools
import numpy as np

from tests_generator import random_DiscreteStateTransitionsDataset, random_unknown_values_dataset

from pylfit.utils import eprint
from pylfit.algorithms.gula import GULA
from pylfit.objects.rule import Rule
from pylfit.objects.legacyAtom import LegacyAtom
from pylfit.algorithms.algorithm import Algorithm

from pylfit.datasets import DiscreteStateTransitionsDataset

random.seed(0)
np.random.seed(0)

class GULA_tests(unittest.TestCase):
    """
        Regression tests of class GULA from gula.py
    """

    _nb_tests = 10

    _nb_transitions = 100

    _nb_features = 6

    _nb_targets = 3

    _nb_feature_values = 3

    _nb_target_values = 3

    #------------------
    # Test functions
    #------------------

    def test_fit(self):
        print(">> GULA.fit(dataset, targets_to_learn, impossibility_mode, verbose, threads):")

        for test_id in range(self._nb_tests):
            for partial_dataset in [False,True]:
                for impossibility_mode in [False,True]:
                    for verbose in [0,1]:
                        for threads in [1,2]:
                            if threads > 1 and verbose > 0: # Avoid flooding
                                continue
                            # 0) exceptions
                            #---------------

                            # Dataset type
                            dataset = "" # not a DiscreteStateTransitionsDataset
                            self.assertRaises(ValueError, GULA.fit, dataset)

                            # 1) No transitions
                            #--------------------
                            dataset = random_DiscreteStateTransitionsDataset( \
                            nb_transitions=0, \
                            nb_features=random.randint(1,self._nb_features), \
                            nb_targets=random.randint(1,self._nb_targets), \
                            max_feature_values=self._nb_feature_values, max_target_values=self._nb_target_values)

                            if partial_dataset:
                                data = random_unknown_values_dataset(dataset.data)
                                dataset = DiscreteStateTransitionsDataset(data, dataset.features, dataset.targets)

                            f = io.StringIO()
                            with contextlib.redirect_stderr(f):
                                output = GULA.fit(dataset=dataset, impossibility_mode=impossibility_mode, verbose=verbose, threads=threads)

                            # Output must be one empty rule for each target value
                            self.assertEqual(len(output), len([val for (var,vals) in dataset.targets for val in vals]))

                            expected = [Rule(LegacyAtom(var,set(vals),val,var_id)) for var_id, (var,vals) in enumerate(dataset.targets) for val_id, val in enumerate(vals)]
                            #eprint(expected)
                            #eprint(output)

                            for r in expected:
                                self.assertTrue(r in output)

                            # 2) Random observations
                            # ------------------------

                            # Generate transitions
                            dataset = random_DiscreteStateTransitionsDataset( \
                            nb_transitions=random.randint(1, self._nb_transitions), \
                            nb_features=random.randint(1,self._nb_features), \
                            nb_targets=random.randint(1,self._nb_targets), \
                            max_feature_values=self._nb_feature_values, \
                            max_target_values=self._nb_target_values)

                            if partial_dataset:
                                data = random_unknown_values_dataset(dataset.data)
                                dataset = DiscreteStateTransitionsDataset(data, dataset.features, dataset.targets)

                            #dataset.summary()

                            # Empty target list
                            f = io.StringIO()
                            with contextlib.redirect_stderr(f):
                                self.assertEqual(GULA.fit(dataset=dataset, targets_to_learn=dict()), [])

                            #dataset.summary()

                            f = io.StringIO()
                            with contextlib.redirect_stderr(f):
                                output = GULA.fit(dataset=dataset, impossibility_mode=impossibility_mode, verbose=verbose, threads=threads)

                            # Encode data to check GULA output rules
                            #data_encoded = []
                            #for (s1,s2) in dataset.data:
                            #    s1_encoded = [domain.index(s1[var_id]) for var_id, (var,domain) in enumerate(dataset.features)]
                            #    s2_encoded = [domain.index(s2[var_id]) for var_id, (var,domain) in enumerate(dataset.targets)]
                            #    data_encoded.append((s1_encoded,s2_encoded))

                            #if partial_dataset:
                            #    data_encoded = Algorithm.encode_transitions_set(dataset.data, dataset.features, dataset.targets, dataset._UNKNOWN_VALUE)
                            #else:
                            #    data_encoded = Algorithm.encode_transitions_set(dataset.data, dataset.features, dataset.targets)

                            data_encoded = dataset.data

                            # 2.1.1) Correctness (explain all)
                            # -----------------
                            # all transitions are fully explained, i.e. each target value is explained by atleast one rule
                            if impossibility_mode == False:

                                for (s1,s2) in dataset.data:
                                    # Partial s1 no guaranty
                                    s1_partial = False
                                    for i in s1:
                                        if i in [dataset._UNKNOWN_VALUE]:
                                            s1_partial = True
                                            break
                                    if s1_partial:
                                        continue
                                    for var_id,(var,vals) in enumerate(dataset.targets):
                                        realises_target = False

                                        # Can't explain unknown
                                        if s2[var_id] in [dataset._UNKNOWN_VALUE]:
                                            continue

                                        for r in output:
                                            if r.head.variable == var and r.head.matches(s2) and \
                                            ( (not partial_dataset and r.matches(s1)) or (partial_dataset and r.partial_matches(s1,[dataset._UNKNOWN_VALUE]) != Rule._NO_MATCH)):
                                                realises_target = True
                                                #eprint(s1_encoded, " => ", target_id,"=",expected_value, " by ", r)
                                                break
                                        if not realises_target:
                                            print(s1,s2)
                                            print(var_id,(var,vals))
                                        self.assertTrue(realises_target)

                            #eprint("-------------------")
                            #eprint(data_encoded)

                            # 2.1.2) Correctness (no spurious observation)
                            # -----------------
                            # No rules generate a unobserved target value from an observed state
                            for r in output:
                                for (s1,s2) in data_encoded:
                                    if r.matches(s1):
                                        observed = False
                                        for (s1_,s2_) in data_encoded: # Must be in a target state after s1
                                            if tuple(s1_) == tuple(s1) and (r.head.matches(s2_) or s2_[r.head.state_position] == LegacyAtom._UNKNOWN_VALUE):
                                                observed = True
                                                #eprint(r, " => ", s1_, s2_)
                                                break
                                            
                                            # Check if state is a potential false negative
                                            if partial_dataset and (r.head.matches(s2_) or s2_[r.head.state_position] == LegacyAtom._UNKNOWN_VALUE):# and not impossibility_mode:
                                                compatible_state = True
                                                for i in range(len(s1_)):
                                                    if s1[i] != s1_[i] and s1[i] not in [dataset._UNKNOWN_VALUE] and s1_[i] not in [dataset._UNKNOWN_VALUE]:
                                                        compatible_state = False
                                                if compatible_state:
                                                    observed = True
                                                    break

                                        if impossibility_mode:
                                            if not partial_dataset:
                                                # DBG
                                                if observed:
                                                    print(r)
                                                    print(s1,s2)
                                                    print(s1_,s2_)
                                                    print(dataset)
                                                self.assertFalse(observed)
                                        else:
                                            # DBG
                                            if not observed:
                                                print(r)
                                                print(s1,s2)
                                                print(s1_,s2_)
                                                print(dataset)
                                            self.assertTrue(observed)

                            # 2.2) Completness
                            # -----------------
                            # all possible initial state is matched by a rule of each target

                            # generate all combination of domains
                            if impossibility_mode == False and not partial_dataset:
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

                            # Group transitions by initial state
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
                                pos, neg = GULA.interprete(data_grouped_by_init_state, r.head)
                                if impossibility_mode:
                                    pos_ = pos
                                    pos = neg
                                    neg = pos_
                                for var in r.body:
                                        r_ = r.copy()
                                        r_.remove_condition(var) # Try remove condition

                                        minimal = True
                                        neg_full_match = False
                                        neg_partial_match = False
                                        for s in neg:
                                            if r_.matches(s):
                                                neg_full_match = True
                                                break

                                            if partial_dataset and r_.partial_matches(s,[dataset._UNKNOWN_VALUE]) == Rule._PARTIAL_MATCH:
                                                neg_partial_match = True

                                        if not neg_full_match:
                                            if neg_partial_match:
                                                for s_ in pos:
                                                    if r_.matches(s):
                                                        minimal = False
                                                        break
                                            else:
                                                minimal = False

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

                            # TODO: check that all minimal rules are learned
                            # - generate all rules and delete non minimals
                            # - check output of gula is this set

    def test_fit__targets_to_learn(self):
        print(">> GULA.fit(dataset, targets_to_learn):")

        for test_id in range(self._nb_tests):

            # 0) exceptions
            #---------------

            # Datatset type
            dataset = "" # not a DiscreteStateTransitionsDataset
            self.assertRaises(ValueError, GULA.fit, dataset, dict())

            # targets_to_learn type
            dataset = random_DiscreteStateTransitionsDataset( \
            nb_transitions=0, \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, max_target_values=self._nb_target_values)

            targets_to_learn = "" # not a dict
            self.assertRaises(ValueError, GULA.fit, dataset, targets_to_learn)

            targets_to_learn = {"1":["1","2"], 2:["1","2"]} # bad key
            self.assertRaises(ValueError, GULA.fit, dataset, targets_to_learn)

            targets_to_learn = {"1":"1,2", "2":["1","2"]} # bad values (not list)
            self.assertRaises(ValueError, GULA.fit, dataset, targets_to_learn)

            targets_to_learn = {"1":["1",2], "2":[1,"2"]} # bad values (not string)
            self.assertRaises(ValueError, GULA.fit, dataset, targets_to_learn)

            targets_to_learn = {"y0":["val_0","val_2"], "lool":["val_0","val_1"]} # bad values (not in targets)
            self.assertRaises(ValueError, GULA.fit, dataset, targets_to_learn)

            targets_to_learn = {"y0":["lool","val_2"]} # bad values (not domain)
            self.assertRaises(ValueError, GULA.fit, dataset, targets_to_learn)

            # 1) No transitions
            #--------------------
            dataset = random_DiscreteStateTransitionsDataset( \
            nb_transitions=0, \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, max_target_values=self._nb_target_values)

            f = io.StringIO()
            with contextlib.redirect_stderr(f):
                output = GULA.fit(dataset=dataset)

            # Output must be one empty rule for each target value
            self.assertEqual(len(output), len([val for (var,vals) in dataset.targets for val in vals]))

            expected = [Rule(LegacyAtom(var,set(vals),val,var_id)) for var_id, (var,vals) in enumerate(dataset.targets) for val_id, val in enumerate(vals)]
            #print(dataset.features)
            #print(dataset.targets)
            #eprint(expected)
            #eprint(output)

            for r in expected:
                self.assertTrue(r in output)

            # 2) Random observations
            # ------------------------

            # Generate transitions
            dataset = random_DiscreteStateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            # Empty target list
            self.assertEqual(GULA.fit(dataset=dataset, targets_to_learn=dict()), [])

            #dataset.summary()

            targets_to_learn = dict()
            for a, b in dataset.targets:
                if random.choice([True,False]):
                    b_ = random.sample(b, random.randint(0,len(b)))
                    targets_to_learn[a] = b_

            #eprint(targets_to_learn)

            f = io.StringIO()
            with contextlib.redirect_stderr(f):
                output = GULA.fit(dataset=dataset, targets_to_learn=targets_to_learn)

            # 2.1.1) Correctness (explain all)
            # -----------------
            # all transitions are fully explained, i.e. each target value is explained by atleast one rule
            for (s1,s2) in dataset.data:
                for var_id,(var,vals) in enumerate(dataset.targets):
                    realises_target = False

                    # In partial mode only requested target values are expected
                    if var not in targets_to_learn:
                        continue
                    if s2[var_id] not in targets_to_learn[var]:
                        continue

                    for r in output:
                        if r.head.matches(s2) and r.matches(s1):
                            realises_target = True
                            #eprint(s1_encoded, " => ", target_id,"=",expected_value, " by ", r)
                            break
                    self.assertTrue(realises_target)

            #eprint("-------------------")
            #eprint(data_encoded)

            # 2.1.2) Correctness (no spurious observation)
            # -----------------
            # No rules generate a unobserved target value from an observed state
            for r in output:
                for (s1,s2) in dataset.data:
                    if r.matches(s1):
                        observed = False
                        for (s1_,s2_) in dataset.data: # Must be in a target state after s1
                            if tuple(s1_) == tuple(s1) and r.head.matches(s2_):
                                observed = True
                                #eprint(r, " => ", s1_, s2_)
                                break
                        self.assertTrue(observed)

            # 2.2) Completness
            # -----------------
            # all possible initial state is matched by a rule of each target

            # generate all combination of domains
            encoded_domains = [set(domain) for (var, domain) in dataset.features]
            init_states_encoded = set([i for i in list(itertools.product(*encoded_domains))])

            for s in init_states_encoded:
                for var, vals in dataset.targets:
                    realises_target = False

                    # In partial mode only requested target values are expected
                    if var not in targets_to_learn:
                        continue
                    if len(targets_to_learn[var]) == 0:
                        continue
                    if len(targets_to_learn[var]) != len(vals): # completude cannot be garanty
                        continue

                    for r in output:
                        if r.head.variable == var and r.matches(s):
                            realises_target = True
                            #eprint(s, " => ", target_id,"=",expected_value, " by ", r)
                            break

                    #eprint("ttl: ", targets_to_learn)
                    #eprint("t: ", target_name)

                    self.assertTrue(realises_target)

            # 2.3) minimality
            # -----------------
            # All rules conditions are necessary, i.e. removing a condition makes realizes unobserved target value from observation
            for r in output:
                for var in r.body:
                    r_ = r.copy()
                    r_.remove_condition(var) # Try remove condition

                    conflict = False
                    for (s1,s2) in dataset.data:
                        if r_.matches(s1):
                            observed = False
                            for (s1_,s2_) in dataset.data: # Must be in a target state after s1
                                if tuple(s1_) == tuple(s1) and r_.head.matches(s2_):
                                    observed = True
                                    #eprint(r, " => ", s1_, s2_)
                                    break
                            if not observed:
                                conflict = True
                                break

                    # # DEBUG:
                    if not conflict:
                        eprint("not minimal "+r)

                    self.assertTrue(conflict)

            # 2.4) only requested targets value appear in rule head
            # ------------

            for r in output:
                self.assertTrue(r.head.variable in targets_to_learn)
                self.assertTrue(r.head.value in targets_to_learn[r.head.variable])


            # TODO: check that all minimal rules are learned
            # - generate all rules and delete non minimals
            # - check output of gula is this set

    def test_interprete(self):
        print(">> GULA.interprete(transitions, variable, value)")

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
                    pos, neg = GULA.interprete(data_grouped_by_init_state, head)

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
        print(">> GULA.fit_var_val(feature_domains, variable, value, negatives, positives, verbose)")

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
                    pos, neg = GULA.interprete(data_grouped_by_init_state, head)
                    #eprint("neg: ", neg)
                    f = io.StringIO()
                    with contextlib.redirect_stderr(f):
                        output = GULA.fit_var_val(head, dataset.features_void_atoms, neg)
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

                    # TODO: check all minimal rules are output (see above)
                    # - Check equality with brute force output

    #------------------
    # Tool functions
    #------------------


if __name__ == '__main__':
    """ Main """

    unittest.main()
