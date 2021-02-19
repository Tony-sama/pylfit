#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/15
# @updated: 2021/01/26
#
# @desc: simple GULA implementation, the General Usage LFIT Algorithm.
#   - INPUT: a set of pairs of discrete multi-valued states
#   - OUTPUT: the optimal logic program that realizes the input
#   - THEORY:
#       - ILP 2018: Learning Dynamics with Synchronous, Asynchronous and General Semantics
#           https://hal.archives-ouvertes.fr/hal-01826564
#   - COMPLEXITY:
#       - Variables: exponential
#       - Values: exponential
#       - Observations: polynomial
#       - about O( |observations| * |values| ^ (2 * |variables|) )
#-----------------------

from ..utils import eprint
from ..objects.rule import Rule
from ..algorithms.algorithm import Algorithm
from ..datasets import StateTransitionsDataset

import csv
import numpy as np

class GULA (Algorithm):
    """
    Define a simple complete version of the GULA algorithm.
    Learn logic rules that explain state transitions
    of a dynamic system, whatever its semantic:
        - discrete
        - synchronous/asynchronous/general/other semantic
    INPUT: a set of pairs of discrete states
    OUTPUT: a logic program
    """

    @staticmethod
    def fit(dataset, targets_to_learn=None, impossibility_mode=False, supported_only=False, verbose=0): #variables, values, transitions, conclusion_values=None, program=None): #, partial_heuristic=False):
        """
        Preprocess transitions and learn rules for all given features/targets variables/values.

        Args:
            dataset: pylfit.datasets.StateTransitionsDataset
                state transitions of a the system
            targets: dict of {String: list of String}
                target variables values of the dataset for wich we want to learn rules.
                If not given, all targets values will be learned.

        Returns:
            list of pylfit.objects.Rule
                A set of DMVLP rules that is:
                    - correct: explain/reproduce all the transitions of the dataset.
                    - complete: matches all possible feature states (even not in the dataset).
                    - optimal: all rules are minimals
        """
        #eprint("Start GULA learning...")

        # Parameters checking
        if not isinstance(dataset, StateTransitionsDataset):
            raise ValueError('Dataset type not supported, GULA expect ' + str(StateTransitionsDataset.__name__))

        if targets_to_learn is None:
            targets_to_learn = dict()
            for a, b in dataset.targets:
                targets_to_learn[a] = b
        elif not isinstance(targets_to_learn, dict) \
            or not all(isinstance(key, str) and isinstance(value, list) for key, value in targets_to_learn.items()) \
            or not all(isinstance(v, str) for key, value in targets_to_learn.items() for v in value):
            raise ValueError('targets_to_learn must be a dict of format {String: list of String}')
        else:
            for key, values in targets_to_learn.items():
                targets_names = [var for var, vals in dataset.targets]
                if key not in targets_names:
                    raise ValueError('targets_to_learn keys must be dataset target variables')
                var_id = targets_names.index(key)
                for val in values:
                    if val not in dataset.targets[var_id][1]:
                        raise ValueError('targets_to_learn values must be in target variable domain')

        feature_domains = dataset.features
        target_domains = dataset.targets
        rules = []

        #if conclusion_values == None:
        #    conclusion_values = values

        # Replace state variable value (string) by their domain id (int)
        encoded_data = Algorithm.encode_transitions_set(dataset.data, dataset.features, dataset.targets)
        #eprint(encoded_data)

        #eprint(transitions)
        if verbose > 0:
            eprint("\nConverting transitions to nparray...")
        #processed_transitions = np.array([tuple(s1)+tuple(s2) for s1,s2 in encoded_data])
        processed_transitions = np.array([np.concatenate( (s1, s2), axis=None) for s1,s2 in encoded_data])

        #exit()

        if len(processed_transitions) > 0:
            #eprint("flattened: ", processed_transitions)
            if verbose > 0:
                eprint("Sorting transitions...")
            processed_transitions = processed_transitions[np.lexsort(tuple([processed_transitions[:,col] for col in reversed(range(0,len(feature_domains)))]))]
            #for i in range(0,len(variables)):
            #processed_transitions = processed_transitions[np.argsort(processed_transitions[:,i])]
            #eprint("sorted: ", processed_transitions)

            if verbose > 0:
                eprint("Grouping transitions by initial state...")
            #processed_transitions = np.array([ (row[:len(variables)], row[len(variables):]) for row in processed_transitions])

            processed_transitions_ = []
            s1 = processed_transitions[0][:len(feature_domains)]
            S2 = []
            for row in processed_transitions:
                if not np.array_equal(row[:len(feature_domains)], s1): # New initial state
                    #eprint("new state: ", s1)
                    processed_transitions_.append((s1,S2))
                    s1 = row[:len(feature_domains)]
                    S2 = []

                #eprint("adding ", row[len(feature_domains):], " to ", s1)
                S2.append(row[len(feature_domains):]) # Add new next state

            # Last state
            processed_transitions_.append((s1,S2))

            processed_transitions = processed_transitions_

        #eprint(processed_transitions)

        # Learn rules for each observed variable/value
        #for var in range(0, len(target_domains)):
        #    for val in range(0, len(target_domains[var][1])):
        #eprint(targets_to_learn)
        for var_id, (var_name, var_domain) in enumerate(dataset.targets):
            #eprint(var_id, (var_name, var_domain))
            for val_id, val_name in enumerate(var_domain):
                #eprint(val_id, val_name)
                if var_name not in targets_to_learn:
                    continue
                if val_name not in targets_to_learn[var_name]:
                    continue

                if impossibility_mode:
                    negatives, positives = GULA.interprete(processed_transitions, var_id, val_id, True)#, partial_heuristic)
                    rules += GULA.fit_var_val(feature_domains, var_id, val_id, positives, negatives) #variables, values, var, val, negatives, program)#, partial_heuristic)
                else:
                    negatives, positives = GULA.interprete(processed_transitions, var_id, val_id, supported_only)#, partial_heuristic)
                    rules += GULA.fit_var_val(feature_domains, var_id, val_id, negatives, positives, verbose) #variables, values, var, val, negatives, program)#, partial_heuristic)
                # DBG
                #eprint(negatives)
                if verbose > 0:
                    eprint("\nStart learning of var=", var_id+1,"/", len(target_domains), ", val=", val_id+1, "/", len(target_domains[var_id][1]))

        return rules


    @staticmethod
    def interprete(transitions, variable, value, supported_only=False): #, partial_heuristic=False):
        """
        Split transition into positive/negatives states for the given variable/value

        Args:
            transitions: list of tuple (tuple of int, list of tuple of int)
                state transitions grouped by intiial state
            variable: int
                variable id
            value: int
                variable value id
        """
        # DBG
        #eprint("Interpreting transitions to:",variable,"=",value)
        #positives = [t1 for t1,t2 in transitions if t2[variable] == value]
        #negatives = [t1 for t1,t2 in transitions if t1 not in positives]

        positives = None
        if supported_only:
            positives = []

        negatives = []
        for s1, S2 in transitions:
            negative = True
            for s2 in S2:
                if s2[variable] == value:
                    negative = False
                    break
            if negative:
                negatives.append(s1)
            elif supported_only:
                positives.append(s1)

        return negatives, positives


    @staticmethod
    def fit_var_val(feature_domains, variable, value, negatives, positives=None, verbose=0): #variables, values, variable, value, negatives, program=None):#, partial_heuristic=False):
        """
        Learn minimal rules that explain positive examples while consistent with negatives examples

        Args:
            feature_domains: list of (name, list of int)
                Features variables
            variable: int
                variable id
            value: int
                variable value id
            negatives: list of (list of int)
                States of the system where the variable cannot take this value in the next state
            positives: list of (list of int)
                States of the system where the variable can take this value in the next state
                Optional, if given rule will be enforced to matches alteast one of those states
        """

        # 0) Initialize program as most the general rule
        #------------------------------------------------
        minimal_rules = [Rule(variable, value, len(feature_domains), [])]

        #if program is not None:
        #    minimal_rules = program.get_rules_of(variable, value)

        # DBG
        neg_count = 0

        # Revise learned rules against each negative example
        for neg in negatives:

            neg_count += 1
            if verbose > 0:
                eprint("\rNegative examples satisfied: ",neg_count,"/",len(negatives), ", rules: ", len(minimal_rules), "               ", end='')

            # 1) Extract unconsistents rules
            #--------------------------------

            # Simple way
            #unconsistents = [ rule for rule in minimal_rules if rule.matches(neg) ]
            #minimal_rules = [ rule for rule in minimal_rules if rule not in unconsistents ]

            # Efficient way
            unconsistents = []
            index=0
            while index < len(minimal_rules):
                if minimal_rules[index].matches(neg):
                    #print "length of %s is: %s" %(x[index], len(x[index]))
                    unconsistents.append(minimal_rules[index])
                    del minimal_rules[index]
                    continue
                index+=1

            # 2) Revise unconsistents rules
            #--------------------------------

            new_rules = []

            for unconsistent in unconsistents:

                # Generates all least specialisation of the rule
                ls = []
                for var in range(len(feature_domains)):
                    for val in range(len(feature_domains[var][1])):

                        # Variable availability
                        if unconsistent.has_condition(var):
                            continue

                        # Value validity
                        if val == neg[var]:
                            continue

                        # Create least specialization of r on var/val
                        #least_specialization = unconsistent#.copy()
                        unconsistent.add_condition(var,val)

                        # Heuristic: discard rule that cover no positives example (partial input only)
                        #if partial_heuristic:
                        #    supported = False
                        #    for s in positives:
                        #        if unconsistent.matches(s):
                        #            supported = True
                        #            break
                        #    if not supported:
                        #        unconsistent.pop_condition()
                        #        continue

                        # Discard if subsumed by a consistent minimal rule
                        subsumed = False
                        for minimal_rule in minimal_rules:
                            if minimal_rule.subsumes(unconsistent):
                                subsumed = True
                                break

                        if subsumed:
                            unconsistent.pop_condition()
                            continue

                        # Discard if subsumed by another least specialization
                        subsumed = False
                        for new_rule in new_rules:
                            if new_rule.subsumes(unconsistent):
                                subsumed = True
                                break

                        if subsumed:
                            unconsistent.pop_condition()
                            continue

                        # Heuristic 1: check if the rule matches atleast one positive example
                        if positives is not None:
                            supported = False
                            for s in positives:
                                if unconsistent.matches(s):
                                    supported = True
                                    break
                            if not supported:
                                unconsistent.pop_condition()
                                continue

                        # Discard other least specialization subsumed by this least specialization
                        #new_rules = [new_rule for new_rule in new_rules if not least_specialization.subsumes(new_rule)]
                        index=0
                        while index < len(new_rules):
                            if unconsistent.subsumes(new_rules[index]):
                                del new_rules[index]
                                continue
                            index+=1
                        least_specialization = unconsistent.copy()
                        new_rules.append(least_specialization)
                        unconsistent.pop_condition()

            # Add new minimal rules
            for new_rule in new_rules:
                minimal_rules.append(new_rule)

        #DBG
        #eprint("\r",end='')

        return minimal_rules
