#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/15
# @updated: 2023/12/27
#
# @desc: simple GULA implementation, the General Usage LFIT Algorithm.
#   - INPUT: a set of pairs of discrete multi-valued states
#   - OUTPUT: the optimal logic program that realizes the input
#   - THEORY:
#       - ILP 2018: Learning Dynamics with Synchronous, Asynchronous and General Semantics
#           https://hal.archives-ouvertes.fr/hal-01826564
#       - MLJ 2021: Learning any memory-less discrete semantics for dynamical systems represented by logic programs
#           https://hal.archives-ouvertes.fr/hal-02925942/
#   - COMPLEXITY:
#       - Variables: exponential
#       - Values: exponential
#       - Observations: polynomial
#       - about O( |observations| * |values| ^ (2 * |variables|) )
#-----------------------

from ..utils import eprint
from ..objects.rule import Rule
from ..algorithms.algorithm import Algorithm
from ..datasets import DiscreteStateTransitionsDataset
from ..objects.legacyAtom import LegacyAtom

import numpy as np
import multiprocessing
import itertools

class GULA (Algorithm):
    """
    Define a simple complete version of the GULA algorithm.
    Learn logic rules that explain state transitions
    of a dynamic system, whatever its semantic:
        - discrete
        - synchronous/asynchronous/general/other semantic
    INPUT: a set of pairs of discrete states
    OUTPUT: a list of discrete rules
    """

    @staticmethod
    def fit(dataset, targets_to_learn=None, impossibility_mode=False, verbose=0, threads=1): #variables, values, transitions, conclusion_values=None, program=None): #, partial_heuristic=False):
        """
        Preprocess transitions and learn rules for all given features/targets variables/values.

        Args:
            dataset: pylfit.datasets.DiscreteStateTransitionsDataset
                state transitions of a the system
            targets_to_learn: dict of {String: list of String}
                target variables values of the dataset for wich we want to learn rules.
                If not given, all targets values will be learned.
            impossibility_mode: Boolean
            verbose: int (0 or 1)
            threads: int (>=1)

        Returns:
            list of pylfit.objects.Rule
                A set of DMVLP rules that is:
                    - correct: explain/reproduce all the transitions of the dataset.
                    - complete: matches all possible feature states (even not in the dataset).
                    - optimal: all rules are minimals
        """
        # Parameters checking
        if not isinstance(dataset, DiscreteStateTransitionsDataset):
            raise ValueError('Dataset type not supported, GULA expect ' + str(DiscreteStateTransitionsDataset.__name__))

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
        features_void_atoms = dataset.features_void_atoms
        rules = []

        if verbose > 0:
            eprint("\nConverting transitions to nparray...")
        processed_transitions = np.array([np.concatenate( (s1, s2), axis=None) for s1,s2 in dataset.data])

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

        thread_parameters = []
        for var_id, (var_name, var_domain) in enumerate(dataset.targets):
            for val_id, val_name in enumerate(var_domain):
                if var_name not in targets_to_learn:
                    continue
                if val_name not in targets_to_learn[var_name]:
                    continue

                head = LegacyAtom(var_name, set(var_domain), val_name, var_id)

                if(threads == 1):
                    rules += GULA.fit_thread([processed_transitions, features_void_atoms, head, [dataset._UNKNOWN_VALUE], dataset.has_unknown_values(), impossibility_mode, verbose])
                else:
                    thread_parameters.append([processed_transitions, features_void_atoms, head, [dataset._UNKNOWN_VALUE], dataset.has_unknown_values(), impossibility_mode, verbose])

        if(threads > 1):
            if(verbose):
                eprint("Start learning over "+str(threads)+" threads")
            with multiprocessing.Pool(processes=threads) as pool:
                rules = pool.map(GULA.fit_thread, thread_parameters)
            rules = list(itertools.chain.from_iterable(rules))

        return rules

    @staticmethod
    def fit_thread(args):
        """
        Thread wrapper for fit_var/fit_var_val_with_unknown_values functions (see below)
        """
        processed_transitions, features_void_atoms, head, unknown_values, has_unknown_values, impossibility_mode, verbose = args
        if verbose > 0:
            eprint("\nStart learning of ", head)
        
        positives, negatives = GULA.interprete(processed_transitions, head)

        # Remove potential false negatives
        if has_unknown_values:
            certain_negatives = []
            for neg in negatives:
                uncertain_negative = False
                for pos in positives:
                    possible_same_state = True
                    for i in range(len(pos)):
                        if neg[i] != pos[i] and pos[i] not in unknown_values and neg[i] not in unknown_values: # explicit difference
                            possible_same_state = False
                            break
                    if possible_same_state:
                        uncertain_negative = True
                        break
                if not uncertain_negative:
                    certain_negatives.append(neg)

            negatives = certain_negatives

        if impossibility_mode:
            positives, negatives = negatives.copy(), positives.copy()

        #if has_unknown_values:
        #    rules = GULA.fit_var_val_with_unknown_values(head, features_void_atoms, negatives, positives, unknown_values, verbose)
        #else:
        #    rules = GULA.fit_var_val(head, features_void_atoms, negatives, verbose)
        rules = GULA.fit_var_val(head, features_void_atoms, negatives, verbose)

        if verbose > 0:
            eprint("\nFinished learning of ", head)
        return rules


    @staticmethod
    def interprete(transitions, head):
        """
        Split transition into positive/negatives states for the given head atom

        Args:
            transitions: list of tuple (tuple of any, list of tuple of any)
                state transitions grouped by initial state
            head: Atom
                target atom
        Returns:
            positives: list of tuple of any
            negatives: list of tuple of any
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
            head: LegacyAtom
                head of the rules.
            features_void_atoms: dictionary of string:Atom
                Features variables void atoms.
            negatives: list of (list of any)
                States of the system where the variable cannot take this value in the next state.
            verbose: int (0 or 1)
        Returns:
            list of pylfit.objects.Rule
                minimals consistent rules
        """

        # 0) Initialize program as most the general rule
        #------------------------------------------------
        minimal_rules = [Rule(head)] # HACK legacy atom target

        # DBG
        neg_count = 0

        # Revise learned rules against each negative example
        for neg in negatives:

            neg_count += 1
            if verbose > 0:
                eprint("\rNegative examples satisfied: ",neg_count,"/",len(negatives), ", rules: ", len(minimal_rules), "               ", end='')

            # 1) Extract unconsistents rules
            #--------------------------------

            unconsistents = []
            index=0
            while index < len(minimal_rules):
                if minimal_rules[index].matches(neg):
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
                ls = unconsistent.least_specialization(neg, features_void_atoms)

                for candidate in ls:
                    # Discard if subsumed by a consistent minimal rule
                    subsumed = False
                    for minimal_rule in minimal_rules:
                        if minimal_rule.subsumes(candidate):
                            subsumed = True
                            break

                    if subsumed:
                        continue

                    new_rules.append(candidate)

            # Add new minimal rules
            for new_rule in new_rules:
                minimal_rules.append(new_rule)

        return minimal_rules
