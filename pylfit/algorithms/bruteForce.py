#-----------------------
# @author: Tony Ribeiro
# @created: 2021/05/10
# @updated: 2023/12/27
#
# @desc: simple brute force implementation, enumerates all rules and keep the non-dominated consistent ones
#   - INPUT: a set of pairs of discrete multi-valued states
#   - OUTPUT: the optimal logic program that realizes the input
#   - THEORY:
#       - MLJ 2021: Learning any memory-less discrete semantics for dynamical systems represented by logic programs
#           https://hal.archives-ouvertes.fr/hal-02925942/
#   - COMPLEXITY:
#       - Variables: exponential
#       - Values: exponential
#       - Observations: polynomial
#       - about O( |observations| * |values| ^ (2 * |variables|) )
#-----------------------

from ..utils import eprint
from ..objects.legacyAtom import LegacyAtom
from ..objects.rule import Rule
from ..algorithms.algorithm import Algorithm
from ..datasets import DiscreteStateTransitionsDataset

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
    OUTPUT: a list of discrete rules
    """

    @staticmethod
    def fit(dataset, impossibility_mode=False, verbose=0):
        """
        Preprocess transitions and learn rules for all given features/targets variables/values.

        Args:
            dataset: pylfit.datasets.DiscreteStateTransitionsDataset
                state transitions of a the system
            impossibility_mode: Boolean
            verbose: int

        Returns:
            list of pylfit.objects.Rule
                A set of DMVLP rules that is:
                    - correct: explain/reproduce all the transitions of the dataset.
                    - complete: matches all possible feature states (even not in the dataset).
                    - optimal: all rules are minimals
        """
        #eprint("Start GULA learning...")

        # Parameters checking
        if not isinstance(dataset, DiscreteStateTransitionsDataset):
            raise ValueError('Dataset type not supported, BruteForce expect ' + str(DiscreteStateTransitionsDataset.__name__))

        feature_domains = dataset.features
        target_domains = dataset.targets
        rules = []

        if verbose > 0:
            eprint("\nConverting transitions to nparray...")
        processed_transitions = np.array([np.concatenate( (s1, s2), axis=None) for s1,s2 in dataset.data])

        #exit()

        if len(processed_transitions) > 0:
            if verbose > 0:
                eprint("Sorting transitions...")
            processed_transitions = processed_transitions[np.lexsort(tuple([processed_transitions[:,col] for col in reversed(range(0,len(feature_domains)))]))]

            if verbose > 0:
                eprint("Grouping transitions by initial state...")

            processed_transitions_ = []
            s1 = processed_transitions[0][:len(feature_domains)]
            S2 = []
            for row in processed_transitions:
                if not np.array_equal(row[:len(feature_domains)], s1): # New initial state
                    processed_transitions_.append((s1,S2))
                    s1 = row[:len(feature_domains)]
                    S2 = []
                S2.append(row[len(feature_domains):]) # Add new next state

            # Last state
            processed_transitions_.append((s1,S2))
            processed_transitions = processed_transitions_

        for var_id, (var_name, var_domain) in enumerate(dataset.targets):
            for val_id, val_name in enumerate(var_domain):
                head = LegacyAtom(var_name, set(var_domain), val_name, var_id)
                positives, negatives = BruteForce.interprete(processed_transitions, head)

                # Remove potential false negatives
                if dataset.has_unknown_values():
                    certain_negatives = []
                    for neg in negatives:
                        uncertain_negative = False
                        for pos in positives:
                            possible_same_state = True
                            for i in range(len(pos)):
                                if neg[i] != pos[i] and pos[i] != dataset._UNKNOWN_VALUE and neg[i] != dataset._UNKNOWN_VALUE: # explicit difference
                                    possible_same_state = False
                                    break
                            if possible_same_state:
                                uncertain_negative = True
                                break
                        if not uncertain_negative:
                            certain_negatives.append(neg)

                    negatives = certain_negatives

                if impossibility_mode:
                    negatives = positives.copy()
                
                rules += BruteForce.fit_var_val(head, dataset.features_void_atoms, negatives, verbose)
                
                if verbose > 0:
                    eprint("\nStart learning of var=", var_id+1,"/", len(target_domains), ", val=", val_id+1, "/", len(target_domains[var_id][1]))

        return rules


    @staticmethod
    def interprete(transitions, head):
        """
        Split transition into positive/negatives states for the given variable/value

        Args:
            transitions: list of tuple (tuple of string, list of tuple of string)
                state transitions grouped by initial state
            head: pylfit.objects.LegacyAtom
                target atom
        Returns:
            positives: list of tuple of string
            negatives: list of tuple of string
        """

        positives = []
        negatives = []

        for s1, S2 in transitions:
            negative = True
            for s2 in S2:
                if head.matches(s2) or s2[head.state_position] == LegacyAtom._UNKNOWN_VALUE:
                    positives.append(s1)
                    negative = False
                    break
            if negative:
                negatives.append(s1)
                
        return positives, negatives


    @staticmethod
    def fit_var_val(head, features_void_atoms, negatives, verbose=0):
        """
        Learn minimal rules that explain positive examples while consistent with negatives examples

        Args:
            head: pylfit.objects.LegacyAtom
                target atom
            features_void_atoms: dict of (string, LegacyAtom)
                Features variables void atoms
            negatives: list of (list of int)
                States of the system where the variable cannot take this value in the next state
            verbose: int
        """

        # 0) Generate all possible rules
        #---------------------------------
        if verbose > 0:
            eprint("Generating all rules...")
        conditions = []
        for (key,atom) in features_void_atoms.items():
            condition = [None]
            for val in atom.domain:
                a = atom.copy()
                a.value = val
                condition.append(a)
            conditions.append(condition)

        all_rules = []
        for combination in list(itertools.product(*conditions)):
            rule = Rule(head)
            for atom in combination:
                if atom is not None:
                    rule.add_condition(atom)
            all_rules.append(rule)

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

        return minimal_rules
