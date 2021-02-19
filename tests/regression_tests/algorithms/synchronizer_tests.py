#-----------------------
# @author: Tony Ribeiro
# @created: 2019/11/25
# @updated: 2021/02/03
#
# @desc: PyLFIT unit test script
#
#-----------------------

import unittest
import random
import os

import sys

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from tests_generator import random_StateTransitionsDataset

from pylfit.utils import eprint, load_tabular_data_from_csv
from pylfit.algorithms.synchronizer import Synchronizer
from pylfit.objects.rule import Rule
from pylfit.semantics.synchronousConstrained import SynchronousConstrained

from pylfit.datasets import StateTransitionsDataset

import itertools
import numpy as np

random.seed(0)

class Synchronizer_tests(unittest.TestCase):
    """
        Unit test of class Synchronizer from synchronizer.py
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
        print(">> Synchronizer.fit(dataset, complete, verbose)")

        for test_id in range(self._nb_tests):

            # 0) exceptions
            #---------------

            # Datatset type
            dataset = "" # not a StateTransitionsDataset
            self.assertRaises(ValueError, Synchronizer.fit, dataset)

            # 1) No transitions
            #--------------------
            dataset = random_StateTransitionsDataset( \
            nb_transitions=0, \
            nb_features=random.randint(1,self._nb_features), \
            nb_targets=random.randint(1,self._nb_targets), \
            max_feature_values=self._nb_feature_values, max_target_values=self._nb_target_values)

            rules, constraints = Synchronizer.fit(dataset=dataset)

            # Output must be one empty rule for each target value and the empty constraint
            self.assertEqual(len(rules), len([val for (var,vals) in dataset.targets for val in vals]))
            self.assertEqual(len(constraints), 1)

            expected = [Rule(var_id,val_id,len(dataset.features)) for var_id, (var,vals) in enumerate(dataset.targets) for val_id, val in enumerate(vals)]
            #eprint(expected)
            #eprint(output)

            for r in expected:
                self.assertTrue(r in rules)

            # 2) Random observations
            # ------------------------

            for verbose in [0,1]:

                # Generate transitions
                dataset = random_StateTransitionsDataset( \
                nb_transitions=random.randint(1, self._nb_transitions), \
                nb_features=random.randint(1,self._nb_features), \
                nb_targets=random.randint(1,self._nb_targets), \
                max_feature_values=self._nb_feature_values, \
                max_target_values=self._nb_target_values)

                dataset.summary()

                rules, constraints = Synchronizer.fit(dataset=dataset,verbose=verbose)

                # Encode data to check Synchronizer output rules
                data_encoded = []
                for (s1,s2) in dataset.data:
                    s1_encoded = [domain.index(s1[var_id]) for var_id, (var,domain) in enumerate(dataset.features)]
                    s2_encoded = [domain.index(s2[var_id]) for var_id, (var,domain) in enumerate(dataset.targets)]
                    data_encoded.append((s1_encoded,s2_encoded))

                # 2.1) Correctness (explain all and no spurious observation)
                # -----------------
                # all transitions are fully explained, i.e. each target state are reproduce
                for (s1,s2) in data_encoded:
                    next_states = SynchronousConstrained.next(s1, dataset.targets, rules, constraints)
                    #eprint("rules: ", rules)
                    #eprint("constraints: ", constraints)
                    #eprint("s1: ", s1)
                    #eprint("s2: ", s2)
                    #eprint("next: ", next_states)
                    self.assertTrue(s2 in next_states)
                    for s3 in next_states:
                        self.assertTrue((s1,s3) in data_encoded)

                #eprint("-------------------")
                #eprint(data_encoded)

                # 2.2) Completness
                # -----------------
                # all non observed initial state has no next state under synchronous constrainted semantics

                # generate all combination of domains
                encoded_domains = [set([i for i in range(len(domain))]) for (var, domain) in dataset.features]
                init_states_encoded = [list(i) for i in list(itertools.product(*encoded_domains))]
                observed_init_states = [s1 for (s1,s2) in data_encoded]

                for s in init_states_encoded:
                    next_states = SynchronousConstrained.next(s, dataset.targets, rules, constraints)
                    if s not in observed_init_states:
                        #eprint(s)
                        self.assertEqual(len(next_states), 0)


                # 2.3) minimality
                # -----------------
                # All rules conditions are necessary, i.e. removing a condition makes realizes unobserved target value from observation
                for r in rules:
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

                # 2.4) Constraints are minimals
                #--------------------------------
                # All constraints conditions are necessary, i.e. removing a condition makes some observed transitions impossible
                for r in constraints:
                    for (var_id, val_id) in r.body:
                            r.remove_condition(var_id) # Try remove condition

                            conflict = False
                            for (s1,s2) in data_encoded:
                                if r.matches(s1+s2):
                                    conflict = True
                                    break

                            r.add_condition(var_id,val_id) # Cancel removal

                            # # DEBUG:
                            if not conflict:
                                eprint("not minimal "+r)

                            self.assertTrue(conflict)

                # 2.5) Constraints are all applicable
                #-------------------------------------
                for constraint in constraints:
                    applicable = True
                    for (var,val) in constraint.body:
                        # Each condition on targets must be achievable by a rule head
                        if var >= len(dataset.features):
                            head_var = var-len(dataset.features)
                            matching_rule = False
                            # The conditions of the rule must be in the constraint
                            for rule in rules:
                                #eprint(rule)
                                if rule.head_variable == head_var and rule.head_value == val:
                                    matching_conditions = True
                                    for (cond_var,cond_val) in rule.body:
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
                    for (var,val) in constraint.body:
                        #eprint(var)
                        # Each condition on targets must be achievable by a rule head
                        if var >= len(dataset.features):
                            compatible_rules.append([])
                            head_var = var-len(dataset.features)
                            #eprint(var," ",val)
                            # The conditions of the rule must be in the constraint
                            for rule in rules:
                                #eprint(rule)
                                if rule.head_variable == head_var and rule.head_value == val:
                                    matching_conditions = True
                                    for (cond_var,cond_val) in rule.body:
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
                            for var,val in r.body:
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

                # TODO: check PRIDE mode

    def __test_fit(self):
        # No transitions
        p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
        p_ = Synchronizer.fit([], p.get_features(), p.get_targets())
        self.assertEqual(p_.get_features(),p.get_features())
        self.assertEqual(p_.get_targets(),p.get_targets())
        rules = []
        for var in range(len(p.get_targets())):
            for val in range(len(p.get_targets()[var][1])):
                rules.append(Rule(var,val,len(p.get_features())))
        self.assertEqual(p_.get_rules(),rules)

        for unit_test in range(self.__nb_unit_test):
            # DBG
            eprint()
            eprint(unit_test,"/",self.__nb_unit_test)

            for arg in [True]:#, False]:
                if arg:
                    # DBG
                    #continue
                    eprint(">>> GULA version")
                else:
                    eprint(">>> PRIDE version")

                # Generate random program
                p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)

                for full_random in [True,False]:

                    if full_random:
                        # Generate random transitions with program size
                        t = []
                        states = p.feature_states()
                        states = random.sample(states, random.randint(1,len(states)))
                        for s1 in states:
                            nb_next = random.randint(1,5)
                            for i in range(0,nb_next):
                                s2 = [p.get_targets()[var][1][random.randint(0,len(p.get_targets()[var][1])-1)] for var in range(0,len(p.get_targets()))]
                                if [s1,s2] not in t:
                                    t.append((s1,s2))
                    else:
                        # Generate transitions from the random program
                        t = Synchronous.transitions(p)

                    t = [(tuple(s1),tuple(s2)) for s1,s2 in t]


                    # DBG
                    #eprint("Transitions: ", len(t))
                    #eprint("Transitions: ", t)

                    t_ = Synchronizer.encode_transitions_set(t, p.get_features(), p.get_targets())

                    p_ = Synchronizer.fit(t_, p.get_features(), p.get_targets(), arg)
                    rules = p_.get_rules()

                    #eprint("Model: ", p_)

                    # Check prediction correspond to input
                    predictions = Synchronous.transitions(p_)
                    predictions = set([(tuple(s1),tuple(s2)) for s1,s2 in predictions])
                    t_set = set([(tuple(s1),tuple(s2)) for s1,s2 in t])
                    #eprint("t: ", t_set)
                    #eprint("pred: ", predictions)
                    self.assertEqual(t_set, predictions)

                    # Check correctness/completness/minimality of rules
                    for variable in range(len(p.get_targets())):
                        for value in range(len(p.get_targets()[variable][1])):
                            #eprint("var="+str(variable)+", val="+str(value))
                            #neg = GULA.interprete(t, variable, value)
                            pos = [s1 for s1,s2 in t_ if s2[variable] == value]
                            neg = [s1 for s1,s2 in t_ if s1 not in pos]

                            # Each positive is explained
                            for s in pos:
                                cover = False
                                for r in rules:
                                    if r.get_head_variable() == variable \
                                       and r.get_head_value() == value \
                                       and r.matches(s):
                                        cover = True
                                self.assertTrue(cover) # Atleast one rule cover the example

                            # No negative is covered
                            for s in neg:
                                cover = False
                                for r in rules:
                                    if r.get_head_variable() == variable \
                                       and r.get_head_value() == value \
                                       and r.matches(s):
                                        cover = True
                                self.assertFalse(cover) # no rule covers the example

                            # All rules are minimals
                            for r in rules:
                                if r.get_head_variable() == variable and r.get_head_value() == value:
                                    for (var,val) in r.get_body():
                                        r.remove_condition(var) # Try remove condition

                                        conflict = False
                                        for s in neg:
                                            if r.matches(s): # Cover a negative example
                                                conflict = True
                                                break

                                        # # DEBUG:
                                        if not conflict:
                                            eprint("not minimal "+r.to_string())
                                            eprint(neg)

                                        self.assertTrue(conflict)
                                        r.add_condition(var,val) # Cancel removal

                    # Check correctness/completness/minimality of constraints
                    constraints = p_.get_constraints()
                    #eprint()
                    #eprint("transitions:", t)
                    #eprint("P", p_.to_string())


                    # No observations are match by a constraint
                    for s1, s2 in t_:
                        for c in constraints:
                            self.assertFalse(c.matches(list(s1)+list(s2)))

                    # All constraints are minimals
                    for c in constraints:
                            for (var,val) in c.get_body():
                                c.remove_condition(var) # Try remove condition

                                conflict = False
                                for s1, s2 in t_:
                                    if c.matches(list(s1)+list(s2)): # Cover a negative example
                                        conflict = True
                                        break

                                # # DEBUG:
                                if not conflict:
                                    eprint("not minimal "+c.to_string())
                                    eprint(s1, s2)

                                self.assertTrue(conflict)
                                c.add_condition(var,val) # Cancel removal

                    # Only original transitions are produced from observed states
                    #eprint("OOOOO: ", t)
                    for s1, s2 in t:
                        next = Synchronous.next(p_,s1)
                        self.assertTrue(list(s2) in next)
                        for s in next:
                            if((tuple(s1),tuple(s)) not in t):
                                eprint("HERE: ", (tuple(s1),tuple(s)))
                            self.assertTrue((tuple(s1),tuple(s)) in t)




    #------------------
    # Tool functions
    #------------------
    def __random_rule(self, features, targets, body_size):
        head_var = random.randint(0,len(targets)-1)
        head_val = random.randint(0,len(targets[head_var][1])-1)
        body = []
        conditions = []

        for j in range(0, random.randint(0,body_size)):
            var = random.randint(0,len(features)-1)
            val = random.randint(0,len(features[var][1])-1)
            if var not in conditions:
                body.append( (var, val) )
                conditions.append(var)

        return  Rule(head_var,head_val,len(features),body)


    def __random_constraint(self, features, targets, body_size):
        head_var = -1
        head_val = -1
        body = []
        conditions = []

        for j in range(0, random.randint(0,body_size)):
            var = random.randint(0,len(features)+len(targets)-1)
            if var < len(features):
                val = random.choice(range(0,len(features[var][1])))
            else:
                val = random.choice(range(0,len(targets[var-len(features)][1])))
            if var not in conditions:
                body.append( (var, val) )
                conditions.append(var)

        return  Rule(head_var,head_val,len(features)+len(targets),body)


    def __random_program(self, nb_features, nb_targets, nb_values, body_size):
        features = [("x"+str(i), ["val_"+str(val) for val in range(0,random.randint(2,nb_values))]) for i in range(random.randint(1,nb_features))]
        targets = [("y"+str(i), ["val_"+str(val) for val in range(0,random.randint(2,nb_values))]) for i in range(random.randint(1,nb_targets))]
        rules = []
        constraints = []

        for j in range(random.randint(0,100)):
            r = self.random_rule(features, targets, body_size)
            rules.append(r)

        for j in range(random.randint(0,100)):
            r = self.random_constraint(features, targets, body_size)
            # Force constraint to have condition at t
            var = random.randint(len(features), len(features)+len(targets)-1)
            val = random.randint(0, len(targets[var-len(features)])-1)
            r.add_condition(var, val)
            constraints.append(r)

        return LogicProgram(features, targets, rules, constraints)


if __name__ == '__main__':
    """ Main """

    unittest.main()
