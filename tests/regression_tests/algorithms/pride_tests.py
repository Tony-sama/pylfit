#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/15
# @updated: 2023/12/13
#
# @desc: PRIDE regression test script
# Tests algorithm methods on random dataset
# Done:
#   - PRIDE.fit(dataset, targets_to_learn)
#   - PRIDE.interprete(transitions, variable, value)
#   - PRIDE.fit_var_val(feature_domains, variable, value, negatives, positives)
# Todo:
#   - check exceptions
#
#-----------------------

import unittest
import random
import os
import io
import contextlib
from itertools import chain, combinations

import sys

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

import itertools
import numpy as np

from tests_generator import random_DiscreteStateTransitionsDataset, random_unknown_values_dataset

from pylfit.utils import eprint
from pylfit.algorithms.pride import PRIDE
from pylfit.objects.rule import Rule
from pylfit.objects.legacyAtom import LegacyAtom
from pylfit.algorithms.algorithm import Algorithm

from pylfit.datasets import DiscreteStateTransitionsDataset

#random.seed(0)
#np.random.seed(0)

class PRIDE_tests(unittest.TestCase):
    """
        Regression tests of class PRIDE from pride.py
    """

    _nb_tests = 10

    _nb_transitions = 10

    _nb_features = 6

    _nb_targets = 3

    _nb_feature_values = 3

    _nb_target_values = 3

    #------------------
    # Test functions
    #------------------

    def test_fit(self):
        print(">> PRIDE.fit(dataset, targets_to_learn, verbose):")

        for i in range(self._nb_tests):
            #print("\r>>> "+ str(i+1)+"/"+str(self._nb_tests))
            # Datatset type
            dataset = "" # not a DiscreteStateTransitionsDataset
            self.assertRaises(ValueError, PRIDE.fit, dataset)

            #heuristics_list = ["try_all_atoms", "max_coverage_dynamic", "max_coverage_static", "max_diversity", "multi_thread_at_rule_level"]
            heuristics_list = PRIDE._HEURISTICS # TODO
            
            for partial_dataset in [False,True]:
                for impossibility_mode in [False,True]:
                    for verbose in [0,1]:
                        for heuristics in [None] + [heuristics_list] + [j for j in list(PRIDE_tests.powerset(heuristics_list))[1:] if len(j) == 1]: #list(PRIDE_tests.powerset(heuristics_list))[1:]:
                            for threads in [1,2]:
                                if threads > 1 and verbose > 0: # Avoid flooding
                                    continue
                                if heuristics is not None:
                                    heuristics = list(heuristics)

                                #eprint(">>> Parameters: partial_dataset=",partial_dataset,", impossibility_mode=",impossibility_mode, ", verbose=", verbose, ", heuristics=", heuristics, ", threads=", threads)

                                # 1) No transitions
                                #--------------------
                                dataset = random_DiscreteStateTransitionsDataset( \
                                nb_transitions=0, \
                                nb_features=random.randint(1,self._nb_features), \
                                nb_targets=random.randint(1,self._nb_targets), \
                                max_feature_values=self._nb_feature_values, max_target_values=self._nb_target_values)

                                f = io.StringIO()
                                with contextlib.redirect_stderr(f):
                                    output = PRIDE.fit(dataset=dataset, options={"impossibility_mode":impossibility_mode, "verbose":verbose, "heuristics":heuristics, "threads":threads})

                                # Output must be empty
                                self.assertTrue(output == [])

                                # 2) Random observations
                                # ------------------------

                                # Generate transitions
                                dataset = random_DiscreteStateTransitionsDataset( \
                                nb_transitions=random.randint(1, self._nb_transitions), \
                                nb_features=random.randint(1,self._nb_features), \
                                nb_targets=random.randint(1,self._nb_targets), \
                                max_feature_values=self._nb_feature_values, \
                                max_target_values=self._nb_target_values)

                                #dataset.summary()

                                if partial_dataset:
                                    data = random_unknown_values_dataset(dataset.data)
                                    dataset = DiscreteStateTransitionsDataset(data, dataset.features, dataset.targets)

                                f = io.StringIO()
                                with contextlib.redirect_stderr(f):
                                    output = PRIDE.fit(dataset=dataset, options={"impossibility_mode":impossibility_mode, "verbose":verbose, "heuristics":heuristics, "threads":threads})

                                # Encode data to check PRIDE output rules
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
                                            if i == dataset._UNKNOWN_VALUE:
                                                s1_partial = True
                                                break
                                        if s1_partial:
                                            continue
                                        for var_id,(var,vals) in enumerate(dataset.targets):
                                            realises_target = False

                                            # Can't explain unknown
                                            if s2[var_id] == dataset._UNKNOWN_VALUE:
                                                continue

                                            for r in output:
                                                if r.head.variable == var and r.head.matches(s2) and \
                                                ( (not partial_dataset and r.matches(s1)) or (partial_dataset and r.partial_matches(s1,dataset._UNKNOWN_VALUE) != Rule._NO_MATCH)):
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
                                                if not observed:
                                                        print("impossibility:", impossibility_mode)
                                                        print("heuristics:", heuristics)
                                                        print(r)
                                                        print([(s1,s2) for s1,s2 in data_encoded if r.matches(s1)])
                                                self.assertTrue(observed)

                                # 2.2) minimality
                                # -----------------
                                # All rules conditions are necessary, i.e. removing a condition makes realizes unobserved target value from observation

                                # Group transitions by initial state

                                for r in output:
                                    pos, neg = PRIDE.interprete(dataset, r.head)
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

                                            if partial_dataset and r_.partial_matches(s,dataset._UNKNOWN_VALUE) == Rule._PARTIAL_MATCH:
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

                                # TODO: check exceptions and targets to learn mode
                                if heuristics is not None:
                                    self.assertRaises(ValueError, PRIDE.fit, dataset, {"impossibility_mode":impossibility_mode, "verbose":verbose, "heuristics":heuristics+["bad_heuristic_name"], "threads":threads})

    def test_interprete(self):
        print(">> PRIDE.interprete(transitions, variable, value)")

        for i in range(self._nb_tests):

            # Generate transitions
            dataset = random_DiscreteStateTransitionsDataset( \
            nb_transitions=random.randint(1, self._nb_transitions), \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, \
            max_target_values=self._nb_target_values)

            # Encode data with DiscreteStateTransitionsDataset
            data_encoded = [(tuple(s1),tuple(s2)) for (s1,s2) in dataset.data]

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
                    pos, neg = PRIDE.interprete(dataset, head)

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

    def test_fit_targets_to_learn(self):
        print(">> PRIDE.fit(dataset, targets_to_learn):")

        for test_id in range(self._nb_tests):

            # 0) exceptions
            #---------------

            # Dataset type
            dataset = "" # not a DiscreteStateTransitionsDataset
            self.assertRaises(ValueError, PRIDE.fit, dataset, dict())

            # targets_to_learn type
            dataset = random_DiscreteStateTransitionsDataset( \
            nb_transitions=0, \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, max_target_values=self._nb_target_values)

            targets_to_learn = "" # not a dict
            self.assertRaises(ValueError, PRIDE.fit, dataset, {"targets_to_learn":targets_to_learn})

            targets_to_learn = {"1":["1","2"], 2:["1","2"]} # bad key
            self.assertRaises(ValueError, PRIDE.fit, dataset, {"targets_to_learn":targets_to_learn})

            targets_to_learn = {"1":"1,2", "2":["1","2"]} # bad values (not list)
            self.assertRaises(ValueError, PRIDE.fit, dataset, {"targets_to_learn":targets_to_learn})

            targets_to_learn = {"1":["1",2], "2":[1,"2"]} # bad values (not string)
            self.assertRaises(ValueError, PRIDE.fit, dataset, {"targets_to_learn":targets_to_learn})

            targets_to_learn = {"y0":["val_0","val_2"], "lool":["val_0","val_1"]} # bad values (not in targets)
            self.assertRaises(ValueError, PRIDE.fit, dataset, {"targets_to_learn":targets_to_learn})

            targets_to_learn = {"y0":["lool","val_2"]} # bad values (not domain)
            self.assertRaises(ValueError, PRIDE.fit, dataset, {"targets_to_learn":targets_to_learn})

            threads = random.randint(-10,0)
            self.assertRaises(ValueError, PRIDE.fit, dataset, {"threads":threads})

            # 1) No transitions
            #--------------------
            dataset = random_DiscreteStateTransitionsDataset( \
            nb_transitions=0, \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, max_target_values=self._nb_target_values)

            f = io.StringIO()
            with contextlib.redirect_stderr(f):
                output = PRIDE.fit(dataset=dataset)

            # Output must be empty
            self.assertEqual(output, [])

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
            self.assertEqual(PRIDE.fit(dataset=dataset, options={"targets_to_learn":dict()}), [])

            #dataset.summary()

            targets_to_learn = dict()
            for a, b in dataset.targets:
                if random.choice([True,False]):
                    b_ = random.sample(b, random.randint(0,len(b)))
                    targets_to_learn[a] = b_

            #eprint(targets_to_learn)

            f = io.StringIO()
            with contextlib.redirect_stderr(f):
                output = PRIDE.fit(dataset=dataset, options={"targets_to_learn":targets_to_learn})

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

            # 2.2) minimality
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

            # 2.3) only requested targets value appear in rule head
            # ------------

            for r in output:
                self.assertTrue(r.head.variable in targets_to_learn)
                self.assertTrue(r.head.value in targets_to_learn[r.head.variable])

    def test_fit_var_val(self):
        print(">> PRIDE.fit_var_val(variable, value, nb_features, positives, negatives)")

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

            # each target value
            for state_position, (var,vals) in enumerate(dataset.targets):
                for val_id, val in enumerate(vals):
                    #eprint("var: ", var_id)
                    #eprint("val: ", val_id)
                    head = LegacyAtom(var, set(vals), val, state_position)
                    pos, neg = PRIDE.interprete(dataset, head)
                    #eprint("neg: ", neg)
                    f = io.StringIO()
                    with contextlib.redirect_stderr(f):
                        threads = random.randint(-10,0)
                        self.assertRaises(ValueError, PRIDE.fit_var_val, head, dataset, pos, neg, 0, None, threads)
                        output = PRIDE.fit_var_val(head, dataset, pos, neg)
                    #eprint()
                    #eprint("rules: ", output)

                    # Check head
                    for r in output:
                        self.assertEqual(r.head, head)

                    # Each positive is explained
                    for s in pos:
                        values = [(var,val) for var, val in enumerate(s) if val != dataset._UNKNOWN_VALUE]
                        s_rule = Rule(head)
                        for var_id,val in values:
                            var_name = dataset.features[var_id][0]
                            domain = set(dataset.features[var_id][1])
                            s_rule.add_condition(LegacyAtom(var_name,domain,val,var_id))

                        #print(s, s_rule)
                        cover = False

                        # Cannot be explained
                        for s_ in neg:
                            if s_rule.matches(s_):
                                cover = True
                                break
                        if cover:
                            continue

                        for r in output:
                            if r.partial_matches(s, dataset._UNKNOWN_VALUE) != Rule._NO_MATCH:
                                cover = True
                                break
                            
                            if s_rule.subsumes(r):
                                cover = True
                                break

                        if not cover:
                            print(s,"not cover")
                            for r in output:
                                print(r)
                            print("pos:", pos)
                            print("neg:", neg)
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
                        for cond_var in r.body:
                            r_ = r.copy()
                            r_.remove_condition(cond_var) # Try remove condition

                            minimal = True
                            neg_full_match = False
                            neg_partial_match = False
                            for s in neg:
                                if r_.matches(s):
                                    neg_full_match = True
                                    break

                                if r_.partial_matches(s,dataset._UNKNOWN_VALUE) == Rule._PARTIAL_MATCH:
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

    #------------------
    # Tool functions
    #------------------

    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


if __name__ == '__main__':
    """ Main """

    unittest.main()
