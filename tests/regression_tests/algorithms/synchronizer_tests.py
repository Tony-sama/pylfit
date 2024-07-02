#-----------------------
# @author: Tony Ribeiro
# @created: 2019/11/25
# @updated: 2023/12/22
#
# @desc: PyLFIT unit test script
#
#-----------------------

import unittest
import random
import os
import io
import contextlib

import sys

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from tests_generator import random_DiscreteStateTransitionsDataset

from pylfit.utils import eprint
from pylfit.algorithms.synchronizer import Synchronizer
from pylfit.objects.legacyAtom import LegacyAtom
from pylfit.objects.rule import Rule
from pylfit.semantics.synchronousConstrained import SynchronousConstrained

from pylfit.datasets import DiscreteStateTransitionsDataset

import itertools
import numpy as np

random.seed(0)

class Synchronizer_tests(unittest.TestCase):
    """
        Unit test of class Synchronizer from synchronizer.py
    """

    _nb_tests = 10

    _nb_transitions = 100

    _nb_features = 3

    _nb_targets = 3

    _nb_feature_values = 3

    _nb_target_values = 3

    #------------------
    # Test functions
    #------------------

    def test_fit(self):
        print(">> Synchronizer.fit(dataset, complete, verbose)")

        for test_id in range(self._nb_tests):
            for complete in [True,False]:
                for verbose in [0,1]:
                    for threads in [1,2]:
                        if threads > 1 and verbose > 0: # Avoid flooding
                            continue

                        # 0) exceptions
                        #---------------

                        # Datatset type
                        dataset = "" # not a DiscreteStateTransitionsDataset
                        self.assertRaises(ValueError, Synchronizer.fit, dataset)

                        # 1) No transitions
                        #--------------------
                        dataset = random_DiscreteStateTransitionsDataset( \
                        nb_transitions=0, \
                        nb_features=random.randint(1,self._nb_features), \
                        nb_targets=random.randint(1,self._nb_targets), \
                        max_feature_values=self._nb_feature_values, max_target_values=self._nb_target_values)

                        f = io.StringIO()
                        with contextlib.redirect_stderr(f):
                            rules, constraints = Synchronizer.fit(dataset=dataset, complete=True, verbose=verbose, threads=threads)

                        # Output must be one empty rule for each target value and the empty constraint
                        self.assertEqual(len(rules), len([val for (var,vals) in dataset.targets for val in vals]))
                        self.assertEqual(len(constraints), 1)

                        expected = [Rule(LegacyAtom(var, set(vals), val, var_id)) for var_id, (var,vals) in enumerate(dataset.targets) for val_id, val in enumerate(vals)]
                        #eprint(expected)
                        #eprint(output)

                        for r in expected:
                            self.assertTrue(r in rules)

                        # 2) Random observations
                        # ------------------------

                        for heuristic_partial in [True,False]:
                            Synchronizer.HEURISTIC_PARTIAL_IMPOSSIBLE_STATE = heuristic_partial

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
                                rules, constraints = Synchronizer.fit(dataset=dataset, complete=complete, verbose=verbose, threads=threads)

                            # Encode data to check Synchronizer output rules
                            #data_encoded = []
                            #for (s1,s2) in dataset.data:
                            #    s1_encoded = [domain.index(s1[var_id]) for var_id, (var,domain) in enumerate(dataset.features)]
                            #    s2_encoded = [domain.index(s2[var_id]) for var_id, (var,domain) in enumerate(dataset.targets)]
                            #    data_encoded.append((s1_encoded,s2_encoded))

                            data_encoded = [(tuple(s1),tuple(s2)) for (s1,s2) in dataset.data]

                            # 2.1) Correctness (explain all and no spurious observation)
                            # -----------------
                            # all transitions are fully explained, i.e. each target state are reproduce
                            for (s1,s2) in data_encoded:
                                next_states = SynchronousConstrained.next(s1, dataset.targets, rules, constraints)
                                #eprint()
                                #eprint(complete)
                                #eprint("rules: ", rules)
                                #eprint("constraints: ", constraints)
                                #eprint("s1: ", s1)
                                #eprint("s2: ", s2)
                                #eprint("next: ", next_states)
                                self.assertTrue(tuple(s2) in next_states)
                                for s3 in next_states:
                                    self.assertTrue((tuple(s1),tuple(s3)) in data_encoded)

                            #eprint("-------------------")
                            #eprint(data_encoded)

                            # 2.2) Completness
                            # -----------------
                            # all non observed initial state has no next state under synchronous constrainted semantics

                            # generate all combination of domains
                            encoded_domains = [set([i for i in domain]) for (var, domain) in dataset.features]
                            init_states_encoded = set([tuple(i) for i in list(itertools.product(*encoded_domains))])
                            observed_init_states = set([s1 for (s1,s2) in data_encoded])

                            #print(init_states_encoded)
                            #print(observed_init_states)
                            for s in init_states_encoded:
                                next_states = SynchronousConstrained.next(s, dataset.targets, rules, constraints)
                                if s not in observed_init_states:
                                    #eprint(s)
                                    if complete == True:
                                        self.assertEqual(len(next_states), 0)


                            # 2.3) minimality
                            # -----------------
                            # All rules conditions are necessary, i.e. removing a condition makes realizes unobserved target value from observation
                            for r in rules:
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

                            # 2.4) Constraints are minimals
                            #--------------------------------
                            # All constraints conditions are necessary, i.e. removing a condition makes some observed transitions impossible
                            for r in constraints:
                                for var in r.body:
                                        r_ = r.copy()
                                        r_.remove_condition(var) # Try remove condition

                                        conflict = False
                                        for (s1,s2) in data_encoded:
                                            if r_.matches(s1+s2):
                                                conflict = True
                                                break

                                        # # DEBUG:
                                        if not conflict:
                                            eprint("not minimal "+r)

                                        self.assertTrue(conflict)

                            # 2.5) Constraints are all applicable
                            #-------------------------------------
                            for constraint in constraints:
                                applicable = True
                                for (var,atom) in constraint.body.items():
                                    # Each condition on targets must be achievable by a rule head
                                    if atom.state_position >= len(dataset.features):
                                        #head_var = var-len(dataset.features)
                                        matching_rule = False
                                        # The conditions of the rule must be in the constraint
                                        for rule in rules:
                                            #eprint(rule)
                                            if rule.head.variable == atom.variable and rule.head.value == atom.value:
                                                matching_conditions = True
                                                for (cond_var,cond_val) in rule.body.items():
                                                    if constraint.has_condition(cond_var) and constraint.get_condition(cond_var) != cond_val:
                                                        matching_conditions = False
                                                        break
                                                if matching_conditions:
                                                    matching_rule = True
                                                    break
                                        if not matching_rule:
                                            applicable = False
                                            break
                                self.assertTrue(applicable)

                                # Get applicables rules
                                compatible_rules = []
                                for (var,atom) in constraint.body.items():
                                    #eprint(var)
                                    # Each condition on targets must be achievable by a rule head
                                    if atom.state_position >= len(dataset.features):
                                        compatible_rules.append([])
                                        #head_var = var-len(dataset.features)
                                        #eprint(var," ",val)
                                        # The conditions of the rule must be in the constraint
                                        for rule in rules:
                                            #eprint(rule)
                                            if rule.head.variable == atom.variable and rule.head.value == atom.value:
                                                matching_conditions = True
                                                for (cond_var,cond_val) in rule.body.items():
                                                    if constraint.has_condition(cond_var) and constraint.get_condition(cond_var) != cond_val:
                                                        matching_conditions = False
                                                        #eprint("conflict on: ",cond_var,"=",cond_val)
                                                        break
                                                if matching_conditions:
                                                    compatible_rules[-1].append(rule)

                                nb_combinations = np.prod([len(l) for l in compatible_rules])
                                done = 0

                                applicable = False
                                for combination in itertools.product(*compatible_rules):
                                    done += 1
                                    #eprint(done,"/",nb_combinations)

                                    condition_variables = set()
                                    conditions = set()
                                    valid_combo = True
                                    for r in combination:
                                        for (var,val) in r.body.items():
                                            if var not in condition_variables:
                                                condition_variables.add(var)
                                                conditions.add((var,val))
                                            elif (var,val) not in conditions:
                                                valid_combo = False
                                                break
                                        if not valid_combo:
                                            break

                                    if valid_combo:
                                        #eprint("valid combo: ", combination)
                                        applicable = True
                                        break

                                self.assertTrue(applicable)



    #------------------
    # Tool functions
    #------------------

if __name__ == '__main__':
    """ Main """

    unittest.main()
