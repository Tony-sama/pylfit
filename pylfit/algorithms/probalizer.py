#-----------------------
# @author: Tony Ribeiro
# @created: 2019/11/06
# @updated: 2023/12/27
#
# @desc: simple Probalizer implementation, for Learning from probabilistic states transitions.
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
from ..algorithms.gula import GULA
from ..algorithms.pride import PRIDE
from ..algorithms.synchronizer import Synchronizer
from ..algorithms.algorithm import Algorithm
from ..datasets import DiscreteStateTransitionsDataset

import math

class Probalizer (Algorithm):
    """
    Define a simple complete version of the Probalizer algorithm.
    Learn logic rules that explain sequences of state transitions
    of a probabilistic dynamic system:
        - discrete
        - non-deterministic
        - probabilistic rules (head atom encoded)
    INPUT: a multiset of pairs of states
    OUTPUT: a logic programs
    """

    @staticmethod
    def fit(dataset, complete=True, synchronous_independant=True, verbose=0, threads=1): # variables, values, transitions, complete=True, synchronous_independant=True):
        """
        Preprocess transitions and learn rules for all variables/values.

        Args:
            dataset: pylfit.datasets.DiscreteStateTransitionsDataset
                state transitions of a the system
            complete: Boolean
                if completeness is required or not
            synchronous_independant: Boolean
                if the transitions are synchronous independant or not
            verbose: int
                When greater than 0 progress of learning will be print in stderr
            threads: int (>=1)
                Number of CPU threads to be used

        Returns:
            list of pylfit.objects.Rule
                A set of DMVLP rules that is:
                    - correct: explain/reproduce all the transitions of the dataset.
                    - complete: (if complete=True) matches all possible feature states (even not in the dataset).
                    - optimal: all rules are minimals
        """
        # Replace target values by their probability
        probability_encoded_input = Probalizer.encode(dataset.data, synchronous_independant)
        probability_encoded_targets = Probalizer.conclusion_values(dataset.targets, probability_encoded_input)

        final_encoded_input = DiscreteStateTransitionsDataset(probability_encoded_input, dataset.features, probability_encoded_targets)
        rules = []
        constraints = []
        if synchronous_independant:
            if complete:
                rules = GULA.fit(dataset=final_encoded_input, verbose=0, threads=threads)
            else:
                rules = PRIDE.fit(dataset=final_encoded_input, options={"verbose":0, "threads":threads})
        else:
            rules, constraints = Synchronizer.fit(dataset=final_encoded_input, complete=complete, verbose=0, threads=threads)

        return probability_encoded_targets, rules, constraints

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

            for s_j in next_states[s_i]: # For each transition
                (i, j, p) = next_states[s_i][s_j]
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
                encoded_input.append((list(i),list(encoded_j)))

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
