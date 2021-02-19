#-----------------------
# @author: Tony Ribeiro
# @created: 2019/11/06
# @updated: 2019/11/06
#
# @desc: simple Probabilizer implementation, for Learning from probabilistic states transitions.
#   - INPUT: a set of pairs of discrete multi-valued states
#   - OUTPUT: multiple set of minimal rules which together realize only the input transitions
#   - THEORY:
#       - NEW
#   - COMPLEXITY:
#       - Variables: exponential or polynomial
#       - Values: exponential or polynomial
#       - Observations: polynomial
#-----------------------

from ..utils import eprint
from ..objects.rule import Rule
from ..objects.logicProgram import LogicProgram
from ..algorithms.gula import GULA
from ..algorithms.pride import PRIDE
from ..algorithms.synchronizer import Synchronizer
from ..algorithms.algorithm import Algorithm

import csv
import itertools
import math

class Probabilizer (Algorithm):
    """
    Define a simple complete version of the Probabilizer algorithm.
    Learn logic rules that explain sequences of state transitions
    of a probabilistic dynamic system:
        - discrete
        - non-deterministic
        - probabilistic rules (head atom encoded)
    INPUT: a multiset of pairs of states
    OUTPUT: a logic programs
    """

    @staticmethod
    def fit(data, feature_domains, target_domains, complete=True, synchronous_independant=True): # variables, values, transitions, complete=True, synchronous_independant=True):
        """
        Preprocess transitions and learn rules for all variables/values.

        Args:
            data: list of tuple (list of int, list of int)
                state transitions of a the system
            feature_domains: list of (String, list of int)
                feature variables of the system and their values id that can appear in rules
            target_domains: list of (String, list of int)
                target variables of the system and their values id that can appear in rules

        Returns:
            LogicProgram
                - each rules are minimals
                - the output set explains/reproduces the input transitions
        """
        #eprint("Start LUST learning...")

        # Nothing to learn
        #if len(transitions) == 0:
        #    return [LogicProgram(variables, values, [])]

        #eprint("Raw transitions:", input)

        # Replace target values by their probability
        probability_encoded_input = Probabilizer.encode(data, synchronous_independant)
        probability_encoded_targets = Probabilizer.conclusion_values(target_domains, probability_encoded_input)

        # DBG
        eprint("total encoded transitions: ",len(probability_encoded_input))

        #for i in encoded_input:
        #    eprint(i)
        #exit()

        #eprint("Variables:", variables)
        #eprint("Condition values:", values)
        #eprint("Conclusion values:", conclusion_values)

        # Encode transition with id of values
        #for (i,j) in occurence_ratio_encoded_input:
        #    for var in range(0,len(j)):
        #        for val_id in range(0,len(occurence_ratio_encoded_targets[var][1])):
        #            if occurence_ratio_encoded_targets[var][1][val_id] == j[var]:
        #                j[var] = occurence_ratio_encoded_targets[var][1][val_id].index(j[var])
        #                break

        final_encoded_input = [(i, tuple([probability_encoded_targets[var][1].index(j[var]) for var in range(len(j))])) for (i,j) in probability_encoded_input]

        eprint("Probabilistic target value domain id encoded transitions:", final_encoded_input)

        #domain_encoded_input = Probabilizer.encode_transitions_set(encoded_input, feature_domains, conclusion_values)

        #eprint("Conclusion value domain id encoded transitions:", domain_encoded_input)

        if synchronous_independant:
            if complete:
                model = GULA.fit(final_encoded_input, feature_domains, probability_encoded_targets) #variables, values, encoded_input, conclusion_values)
            else:
                model = PRIDE.fit(final_encoded_input, feature_domains, probability_encoded_targets) #variables, values, encoded_input)
        else:
            model = Synchronizer.fit(final_encoded_input, feature_domains, probability_encoded_targets, complete) # variables, values, encoded_input, conclusion_values, complete)

        output = model #LogicProgram(feature_domains, conclusion_values, model.get_rules(), model.get_constraints()) #variables, values, model.get_rules(), model.get_constraints(), conclusion_values)

        eprint("Probabilizer output raw: \n", output.to_string())
        #eprint("Probabilizer output logic form: \n", output.logic_form())

        return output #, conclusion_values

    def encode(transitions, synchronous_independant=True):
        # Extract occurences of each transition
        next_states = dict()
        nb_transitions_from = dict()
        for (i,j) in transitions:
            s_i = tuple(i)
            s_j = tuple(j)
            # new init state
            if s_i not in next_states:
                next_states[s_i] = dict()
                nb_transitions_from[s_i] = 0
            # new next state
            if s_j not in next_states[s_i]:
                next_states[s_i][s_j] = (s_i,s_j,0)

            (_, _, p) = next_states[s_i][s_j]
            next_states[s_i][s_j] = (s_i,s_j,p+1)
            nb_transitions_from[s_i] += 1

        #eprint("Transitions counts:", next_states)

        # Extract probability of each transition
        #for s_i in next_states:
        #    for s_j in next_states[s_i]:
        #        (i, j, p) = next_states[s_i][s_j]
                #next_states[s_i][s_j] = (i, j, p / nb_transitions_from[s_i])
        #        next_states[s_i][s_j] = (i, j, p)

        #eprint("Transitions ratio:", next_states)

        # Encode probability locally
        encoded_input = []
        for s_i in next_states:
            if synchronous_independant:
                local_proba = dict()
                for s_j in next_states[s_i]: # For each transition
                    (_, j, p) = next_states[s_i][s_j]
                    for var in range(0,len(j)): # for each variable
                        if var not in local_proba:
                            local_proba[var] = dict()
                        if j[var] not in local_proba[var]: # for each value
                            #local_proba[var][j[var]] = 0.0
                            local_proba[var][j[var]] = 0

                        local_proba[var][j[var]] += p # accumulate probability

            # DBG
            #print("local proba:",local_proba)
            for s_j in next_states[s_i]: # For each transition
                (i, j, p) = next_states[s_i][s_j]
                # DBG
                #print(j)
                if synchronous_independant: # state proba is the product of local proba
                    encoded_j = [
                    str(j[var])+
                    ","+
                    str(int(local_proba[var][j[var]]/math.gcd(local_proba[var][j[var]],nb_transitions_from[s_i])))+
                    "/"+str(int(nb_transitions_from[s_i]/math.gcd(local_proba[var][j[var]],nb_transitions_from[s_i])))
                    for var in range(0,len(j))]
                else: # state probability is independant
                    encoded_j = [
                    str(j[var])+
                    ","+
                    str(int(p/math.gcd(p,nb_transitions_from[s_i])))+
                    "/"+str(int(nb_transitions_from[s_i]/math.gcd(p,nb_transitions_from[s_i])))
                    for var in range(0,len(j))]
                encoded_input.append([i,tuple(encoded_j)])

        #eprint("String encoded transitions:", encoded_input)

        return encoded_input

    def conclusion_values(target_domains, transitions):
        conclusion_values = []

        # Extract each variable possible value
        for var in range(0,len(target_domains)):
            domain = []
            for (i,j) in transitions:
                if j[var] not in domain:
                    domain.append(j[var])
            conclusion_values.append((target_domains[var][0],domain))

        return conclusion_values
