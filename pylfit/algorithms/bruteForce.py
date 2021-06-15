#-----------------------
# @author: Tony Ribeiro
# @created: 2021/05/10
# @updated: 2021/06/15
#
# @desc: simple brute force implementation, enumerate of all rules and keep the non-dominated consistent ones
#   - INPUT: a set of pairs of discrete multi-valued states
#   - OUTPUT: the optimal logic program that realizes the input
#   - THEORY:
#       - MLJ 2020: TODO
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

import itertools

class BruteForce (Algorithm):
    """
    Define a simple brute force enumeration algorithm.
    Generate all logic rules that explain state transitions
    of a dynamic system, whatever its semantic:
        - discrete
        - synchronous/asynchronous/general/other semantic
    INPUT: a set of pairs of discrete states
    OUTPUT: a logic program
    """

    @staticmethod
    def fit(dataset, impossibility_mode=False, verbose=0):
        """
        Preprocess transitions and learn rules for all given features/targets variables/values.

        Args:
            dataset: pylfit.datasets.StateTransitionsDataset
                state transitions of a the system

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
            raise ValueError('Dataset type not supported, BruteForce expect ' + str(StateTransitionsDataset.__name__))

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

                if impossibility_mode:
                    negatives, positives = BruteForce.interprete(processed_transitions, var_id, val_id)#, partial_heuristic)
                    rules += BruteForce.fit_var_val(feature_domains, var_id, val_id, positives, negatives, verbose) #variables, values, var, val, negatives, program)#, partial_heuristic)
                else:
                    negatives, positives = BruteForce.interprete(processed_transitions, var_id, val_id)#, partial_heuristic)
                    rules += BruteForce.fit_var_val(feature_domains, var_id, val_id, negatives, positives, verbose) #variables, values, var, val, negatives, program)#, partial_heuristic)
                # DBG
                #eprint(negatives)
                if verbose > 0:
                    eprint("\nStart learning of var=", var_id+1,"/", len(target_domains), ", val=", val_id+1, "/", len(target_domains[var_id][1]))

        return rules


    @staticmethod
    def interprete(transitions, variable, value): #, partial_heuristic=False):
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
            else:
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

        # 0) Generate all possible rules
        #---------------------------------
        if verbose > 0:
            eprint("Generating all rules...")
        domains = [[(var_id,-1)]+[(var_id,val_id) for val_id, val in enumerate(vals)] for var_id,(var,vals) in enumerate(feature_domains)]
        all_rules = [Rule(variable, value, len(feature_domains), [(var,val) for var,val in combination if val > -1]) for combination in list(itertools.product(*domains))]

        #eprint(all_rules)

        # 1) Remove unconsistent rules
        #---------------------------------
        if verbose > 0:
            eprint("Removing unconsistent rules...")
        consistent_rules = []
        for rule in all_rules:
            consistent = True
            for neg in negatives:
                if rule.matches(neg):
                    consistent = False
                    break
            if consistent:
                consistent_rules.append(rule)

        #eprint(consistent_rules)

        # 2) Remove dominated rules
        #---------------------------
        if verbose > 0:
            eprint("Removing dominated rules...")
        minimal_rules = []
        for rule in consistent_rules:
            dominated = False
            for other_rule in consistent_rules:
                if other_rule.subsumes(rule) and other_rule != rule:
                    dominated = True
                    break
            if not dominated:
                minimal_rules.append(rule)

        #eprint(minimal_rules)

        return minimal_rules
