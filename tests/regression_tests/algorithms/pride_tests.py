#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/15
# @updated: 2022/08/16
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

from tests_generator import random_DiscreteStateTransitionsDataset

from pylfit.utils import eprint
from pylfit.algorithms.pride import PRIDE
from pylfit.objects.rule import Rule

from pylfit.datasets import DiscreteStateTransitionsDataset

random.seed(0)

class PRIDE_tests(unittest.TestCase):
    """
        Regression tests of class PRIDE from pride.py
    """

    _nb_tests = 10

    _nb_transitions = 10

    _nb_features = 3

    _nb_targets = 3

    _nb_feature_values = 3

    _nb_target_values = 3

    #------------------
    # Test functions
    #------------------

    def test_fit(self):
        print(">> PRIDE.fit(dataset, targets_to_learn, verbose):")

        for i in range(self._nb_tests):

            # Datatset type
            dataset = "" # not a DiscreteStateTransitionsDataset
            self.assertRaises(ValueError, PRIDE.fit, dataset)

            #heuristics_list = ["try_all_atoms", "max_coverage_dynamic", "max_coverage_static", "max_diversity", "multi_thread_at_rule_level"]
            heuristics_list = PRIDE._HEURISTICS

            for impossibility_mode in [False,True]:
                for verbose in [0,1]:
                    for heuristics in [None] + list(PRIDE_tests.powerset(heuristics_list))[1:]:
                        for threads in [1,2]:
                            if heuristics is not None:
                                heuristics = list(heuristics)

                            #eprint(">>> Parameters: impossibility_mode=",impossibility_mode, ", verbose=", verbose, ", heuristics=", heuristics, ", threads=", threads)

                            # 1) No transitions
                            #--------------------
                            dataset = random_DiscreteStateTransitionsDataset( \
                            nb_transitions=0, \
                            nb_features=random.randint(1,self._nb_features), \
                            nb_targets=random.randint(1,self._nb_targets), \
                            max_feature_values=self._nb_feature_values, max_target_values=self._nb_target_values)

                            f = io.StringIO()
                            with contextlib.redirect_stderr(f):
                                output = PRIDE.fit(dataset=dataset, impossibility_mode=impossibility_mode, verbose=verbose, heuristics=heuristics, threads=threads)

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

                            f = io.StringIO()
                            with contextlib.redirect_stderr(f):
                                output = PRIDE.fit(dataset=dataset, impossibility_mode=impossibility_mode, verbose=verbose, heuristics=heuristics, threads=threads)

                            # Encode data to check PRIDE output rules
                            data_encoded = []
                            for (s1,s2) in dataset.data:
                                s1_encoded = [domain.index(s1[var_id]) for var_id, (var,domain) in enumerate(dataset.features)]
                                s2_encoded = [domain.index(s2[var_id]) for var_id, (var,domain) in enumerate(dataset.targets)]
                                data_encoded.append((s1_encoded,s2_encoded))

                            # 2.1.1) Correctness (explain all)
                            # -----------------
                            # all transitions are fully explained, i.e. each target value is explained by atleast one rule
                            for (s1,s2) in data_encoded:
                                for target_id in range(len(dataset.targets)):
                                    expected_value = s2_encoded[target_id]
                                    realizes_target = False
                                    for r in output:
                                        if r.head_variable == target_id and r.head_value == expected_value and r.matches(s1_encoded):
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
                                for (s1,s2) in data_encoded:
                                    if r.matches(s1):
                                        observed = False
                                        for (s1_,s2_) in data_encoded: # Must be in a target state after s1
                                            if s1_ == s1 and s2_[r.head_variable] == r.head_value:
                                                observed = True
                                                #eprint(r, " => ", s1_, s2_)
                                                break
                                        if impossibility_mode:
                                            self.assertFalse(observed)
                                        else:
                                            self.assertTrue(observed)

                            # 2.2) minimality
                            # -----------------
                            # All rules conditions are necessary, i.e. removing a condition makes realizes unobserved target value from observation

                            for r in output:
                                pos, neg = PRIDE.interprete(data_encoded, r.head_variable, r.head_value)
                                if impossibility_mode:
                                    pos_ = pos
                                    pos = neg
                                    neg = pos_
                                for (var_id, val_id) in r.body:
                                        r.remove_condition(var_id) # Try remove condition

                                        conflict = False
                                        for s in neg:
                                            if r.matches(s):
                                                conflict = True
                                                break

                                        r.add_condition(var_id,val_id) # Cancel removal

                                        # # DEBUG:
                                        if not conflict:
                                            eprint("not minimal "+r.to_string())

                                        self.assertTrue(conflict)

                            # TODO: check exceptions and targets to learn mode
                            if heuristics is not None:
                                self.assertRaises(ValueError, PRIDE.fit, dataset, None, impossibility_mode, verbose, heuristics+["bad_heuristic_name"])

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
                    pos, neg = PRIDE.interprete(data_encoded, var_id, val_id)

                    # All neg are valid
                    for s in neg:
                        for s1, s2 in data_encoded:
                            if s1 == s:
                                self.assertTrue(s2[var_id] != val_id)

                    # All transitions are interpreted
                    for s1, S2 in data_grouped_by_init_state:
                        if len([s2 for s2 in S2 if s2[var_id] == val_id]) == 0:
                            self.assertTrue(tuple(s1) in neg)

    def test_fit_targets_to_learn(self):
        print(">> PRIDE.fit(dataset, targets_to_learn):")

        for test_id in range(self._nb_tests):

            # 0) exceptions
            #---------------

            # Datatset type
            dataset = "" # not a DiscreteStateTransitionsDataset
            self.assertRaises(ValueError, PRIDE.fit, dataset, dict())

            # targets_to_learn type
            dataset = random_DiscreteStateTransitionsDataset( \
            nb_transitions=0, \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, max_target_values=self._nb_target_values)

            targets_to_learn = "" # not a dict
            self.assertRaises(ValueError, PRIDE.fit, dataset, targets_to_learn)

            targets_to_learn = {"1":["1","2"], 2:["1","2"]} # bad key
            self.assertRaises(ValueError, PRIDE.fit, dataset, targets_to_learn)

            targets_to_learn = {"1":"1,2", "2":["1","2"]} # bad values (not list)
            self.assertRaises(ValueError, PRIDE.fit, dataset, targets_to_learn)

            targets_to_learn = {"1":["1",2], "2":[1,"2"]} # bad values (not string)
            self.assertRaises(ValueError, PRIDE.fit, dataset, targets_to_learn)

            targets_to_learn = {"y0":["val_0","val_2"], "lool":["val_0","val_1"]} # bad values (not in targets)
            self.assertRaises(ValueError, PRIDE.fit, dataset, targets_to_learn)

            targets_to_learn = {"y0":["lool","val_2"]} # bad values (not domain)
            self.assertRaises(ValueError, PRIDE.fit, dataset, targets_to_learn)

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
            self.assertEqual(PRIDE.fit(dataset=dataset, targets_to_learn=dict()), [])

            #dataset.summary()

            targets_to_learn = dict()
            for a, b in dataset.targets:
                if random.choice([True,False]):
                    b_ = random.sample(b, random.randint(0,len(b)))
                    targets_to_learn[a] = b_

            #eprint(targets_to_learn)

            f = io.StringIO()
            with contextlib.redirect_stderr(f):
                output = PRIDE.fit(dataset=dataset, targets_to_learn=targets_to_learn)

            # Encode data to check PRIDE output rules
            data_encoded = []
            for (s1,s2) in dataset.data:
                s1_encoded = [domain.index(s1[var_id]) for var_id, (var,domain) in enumerate(dataset.features)]
                s2_encoded = [domain.index(s2[var_id]) for var_id, (var,domain) in enumerate(dataset.targets)]
                data_encoded.append((s1_encoded,s2_encoded))

            # 2.1.1) Correctness (explain all)
            # -----------------
            # all transitions are fully explained, i.e. each target value is explained by atleast one rule
            for (s1,s2) in data_encoded:
                for target_id in range(len(dataset.targets)):
                    expected_value = s2_encoded[target_id]
                    realizes_target = False

                    # In partial mode only requested target values are expected
                    target_name = dataset.targets[target_id][0]
                    target_value_name = dataset.targets[target_id][1][expected_value]
                    if target_name not in targets_to_learn:
                        continue
                    if target_value_name not in targets_to_learn[target_name]:
                        continue

                    for r in output:
                        if r.head_variable == target_id and r.head_value == expected_value and r.matches(s1_encoded):
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
                for (s1,s2) in data_encoded:
                    if r.matches(s1):
                        observed = False
                        for (s1_,s2_) in data_encoded: # Must be in a target state after s1
                            if s1_ == s1 and s2_[r.head_variable] == r.head_value:
                                observed = True
                                #eprint(r, " => ", s1_, s2_)
                                break
                        self.assertTrue(observed)

            # 2.2) minimality
            # -----------------
            # All rules conditions are necessary, i.e. removing a condition makes realizes unobserved target value from observation
            for r in output:
                for (var_id, val_id) in r.body:
                        r.remove_condition(var_id) # Try remove condition

                        conflict = False
                        for (s1,s2) in data_encoded:
                            if r.matches(s1):
                                observed = False
                                for (s1_,s2_) in data_encoded: # Must be in a target state after s1
                                    if s1_ == s1 and s2_[r.head_variable] == r.head_value:
                                        observed = True
                                        #eprint(r, " => ", s1_, s2_)
                                        break
                                if not observed:
                                    conflict = True
                                    break

                        r.add_condition(var_id,val_id) # Cancel removal

                        # # DEBUG:
                        if not conflict:
                            eprint("not minimal "+r)

                        self.assertTrue(conflict)

            # 2.3) only requested targets value appear in rule head
            # ------------

            for r in output:
                target_name = dataset.targets[r.head_variable][0]
                target_value = dataset.targets[r.head_variable][1][r.head_value]

                self.assertTrue(target_name in targets_to_learn)
                self.assertTrue(target_value in targets_to_learn[target_name])

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
            data_encoded = []
            for (s1,s2) in dataset.data:
                s1_encoded = [domain.index(s1[var_id]) for var_id, (var,domain) in enumerate(dataset.features)]
                s2_encoded = [domain.index(s2[var_id]) for var_id, (var,domain) in enumerate(dataset.targets)]
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
                    pos, neg = PRIDE.interprete(data_encoded, var_id, val_id)
                    #eprint("neg: ", neg)
                    f = io.StringIO()
                    with contextlib.redirect_stderr(f):
                        output = PRIDE.fit_var_val(var_id, val_id, len(dataset.features), pos, neg)
                    #eprint()
                    #eprint("rules: ", output)

                    # Check head
                    for r in output:
                        self.assertEqual(r.head_variable, var_id)
                        self.assertEqual(r.head_value, val_id)

                    # Each positive is explained
                    pos = [s1 for s1,s2 in data_encoded if s2[var_id] == val_id]
                    for s in pos:
                        cover = False
                        for r in output:
                            if r.matches(s):
                                cover = True

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
                        for (var_id_, val_id_) in r.body:
                            r.remove_condition(var_id_) # Try remove condition

                            conflict = False
                            for s in neg:
                                if r.matches(s): # Cover a negative example
                                    conflict = True
                                    break
                            self.assertTrue(conflict)
                            r.add_condition(var_id_,val_id_) # Cancel removal

    def test_find_one_optimal_rule_of(self):
        print(">> PRIDE.find_one_optimal_rule_of(variable, value, nb_features, positives, negatives, feature_state_to_match, verbose=0)")

        for i in range(self._nb_tests):

            for verbose in [0,1]:
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
                    s1_encoded = [domain.index(s1[var_id]) for var_id, (var,domain) in enumerate(dataset.features)]
                    s2_encoded = [domain.index(s2[var_id]) for var_id, (var,domain) in enumerate(dataset.targets)]
                    data_encoded.append((s1_encoded,s2_encoded))

                values_ids = [[j for j in range(0,len(dataset.features[i][1]))] for i in range(0,len(dataset.features))]
                feature_states = [list(i) for i in list(itertools.product(*values_ids))]

                # each target value
                for var_id, (var,vals) in enumerate(dataset.targets):
                    for val_id, val in enumerate(vals):
                        #eprint("var: ", var_id)
                        #eprint("val: ", val_id)
                        pos, neg = PRIDE.interprete(data_encoded, var_id, val_id)

                        if len(pos) == 0:
                            continue

                        for feature_states in [pos,neg]:
                            if(len(feature_states) == 0):
                                continue
                            feature_state_to_match = random.choice([s for s in feature_states])
                            #eprint("neg: ", neg)
                            f = io.StringIO()

                            # No pos case
                            with contextlib.redirect_stderr(f):
                                output = PRIDE.find_one_optimal_rule_of(var_id, val_id, len(dataset.features), [], neg, feature_state_to_match, verbose)

                            with contextlib.redirect_stderr(f):
                                output = PRIDE.find_one_optimal_rule_of(var_id, val_id, len(dataset.features), pos, neg, feature_state_to_match, verbose)
                            #eprint()
                            #eprint("rules: ", output)

                            # Check no consistent rule exists
                            if output is None:
                                for s in pos:
                                    # Most specific rule that match both the pos and request feature state
                                    r = Rule(var_id, val_id, len(dataset.features))
                                    for var in range(len(dataset.features)):
                                        if feature_state_to_match[var] == s[var]:
                                            r.add_condition(var,s[var])
                                    # Must match atleast a neg
                                    if len(neg) > 0:
                                        cover = False
                                        for s in neg:
                                            if r.matches(s):
                                                cover = True
                                                break

                                        if not cover:
                                            eprint(feature_state_to_match)
                                            eprint(s)
                                            eprint(r.to_string())

                                        self.assertTrue(cover)
                                continue

                            # Check head
                            self.assertEqual(output.head_variable, var_id)
                            self.assertEqual(output.head_value, val_id)

                            # Cover at least a positive
                            cover = False
                            for s in pos:
                                if output.matches(s):
                                    cover = True
                                    break

                            self.assertTrue(cover)

                            # No negative is covered
                            cover = False
                            for s in neg:
                                if output.matches(s):
                                    cover = True
                                    break
                            self.assertFalse(cover)

                            # Rules is minimal
                            for (var_id_, val_id_) in output.body:
                                output.remove_condition(var_id_) # Try remove condition

                                conflict = False
                                for s in neg:
                                    if output.matches(s): # Cover a negative example
                                        conflict = True
                                        break
                                self.assertTrue(conflict)
                                output.add_condition(var_id_,val_id_) # Cancel removal

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
