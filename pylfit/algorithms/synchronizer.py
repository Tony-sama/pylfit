#-----------------------
# @author: Tony Ribeiro
# @created: 2019/11/05
# @updated: 2023/12/27
#
# @desc: simple Synchronizer implementation, for Learning from states transitions from ANY semantic.
#   - INPUT: a set of pairs of discrete multi-valued states
#   - OUTPUT: a logic program with constraints that realize only the input transitions
#   - THEORY:
#       - MLJ 2021: Learning any memory-less discrete semantics for dynamical systems represented by logic programs
#           https://hal.archives-ouvertes.fr/hal-02925942/
#   - COMPLEXITY:
#       - Variables: exponential
#       - Values: exponential
#       - Observations: polynomial
#-----------------------

from ..utils import eprint

from ..algorithms.algorithm import Algorithm
from ..algorithms.gula import GULA
from ..algorithms.pride import PRIDE

import numpy as np
import itertools

class Synchronizer (Algorithm):
    """
    Define a simple complete version of the Synchronizer algorithm.
    Learn logic rules that explain sequences of state transitions
    of a dynamic system:
        - discrete
        - non-deterministic
        - semantic constraint rules
    INPUT: a set of pairs of states
    OUTPUT: a logic programs
    """

    # Enable heuristic: only partial state will be generated for impossible state generation for constraints learning
    HEURISTIC_PARTIAL_IMPOSSIBLE_STATE = True

    @staticmethod
    def fit(dataset, complete=True, verbose=0, threads=1):
        """
        Preprocess state transitions and learn rules for all variables/values.

        Args:
            dataset: pylfit.datasets.DiscreteStateTransitionsDataset
                state transitions of a the system
            complete: Boolean
                if completeness is required or not
            verbose: int
                When greater than 0 progress of learning will be print in stderr
            threads: int (>=1)
                Number of CPU threads to be used

        Returns:
            CDMVLP
                - each rules/constraints are minimals
                - the output set explains/reproduces the input transitions
        """

        # 1) Use GULA to learn local possibilities
        #------------------------------------------
        if complete:
            rules = GULA.fit(dataset=dataset, threads=threads)
        else:
            rules = PRIDE.fit(dataset=dataset, options={"threads":threads})

        # 2) Learn constraints
        #------------------------------------------

        encoded_data = dataset.data

        negatives = np.array([tuple(s1)+tuple(s2) for s1,s2 in encoded_data])
        if len(negatives) > 0:
            negatives = negatives[np.lexsort(tuple([negatives[:,col] for col in reversed(range(0,len(dataset.features)))]))]

        # Encode target void atoms positions
        void_atoms = dataset.features_void_atoms.copy()
        for key in dataset.targets_void_atoms:
            atom = dataset.targets_void_atoms[key].copy()
            atom.state_position += len(dataset.features)
            void_atoms[atom.variable] = atom

        if complete:
            constraints = GULA.fit_var_val(None, void_atoms, negatives)
        else:
            # Extract occurences of each transition
            next_states = dict()
            for (i,j) in encoded_data:
                s_i = tuple(i)
                s_j = tuple(j)
                # new init state
                if s_i not in next_states:
                    next_states[s_i] = (s_i,[])
                # new next state
                next_states[s_i][1].append(s_j)

            impossible = set()
            for i in next_states:
                # Extract all possible value of each variable in next state
                domains = [set() for var in dataset.targets]

                for s in next_states[i][1]:
                    for var in range(0,len(s)):
                        domains[var].add(s[var])
                combinations = Synchronizer.partial_combinations(next_states[i][1], domains)

                # Extract unobserved combinations
                if Synchronizer.HEURISTIC_PARTIAL_IMPOSSIBLE_STATE:
                    missings = [(next_states[i][0], tuple(j)) for j in combinations]
                else:
                    missings = [(next_states[i][0], j) for j in list(itertools.product(*domains)) if j not in next_states[i][1]]

                if missings != []:
                    impossible.update(missings)

            # convert impossible transition for PRIDE input
            positives = [list(i)+list(j) for i,j in impossible]

            # Hack the dataset for PRIDE
            dataset_copy = dataset.copy()
            dataset_copy.features_void_atoms = void_atoms

            constraints = PRIDE.fit_var_val(None, dataset_copy, positives, negatives)

        # 3) Discard non-applicable constraints
        #---------------------------------------
        necessary_constraints = []
        if not complete:
            necessary_constraints = constraints
        else:
            # Heuristic: clean constraint with not even a rule for each target condition
            for constraint in constraints:
                applicable = True
                for (var,atom) in constraint.body.items():
                    # Each condition on targets must be achievable by a rule head
                    if atom.state_position >= len(dataset.features):
                        matching_rule = False
                        # The conditions of the rule must be in the constraint
                        for rule in rules:
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
                if applicable:
                    necessary_constraints.append(constraint)

        constraints = necessary_constraints

        # Clean remaining constraints
        necessary_constraints = []
        for constraint in constraints:
            # Get applicables rules
            compatible_rules = []
            for (var,atom) in constraint.body.items():
                # Each condition on targets must be achievable by a rule head
                if atom.state_position >= len(dataset.features):
                    compatible_rules.append([])
                    # The conditions of the rule must be in the constraint
                    for rule in rules:
                        if rule.head.variable == atom.variable and rule.head.value == atom.value:
                            matching_conditions = True
                            for (cond_var,cond_val) in rule.body.items():
                                if constraint.has_condition(cond_var) and constraint.get_condition(cond_var) != cond_val:
                                    matching_conditions = False
                                    break
                            if matching_conditions:
                                compatible_rules[-1].append(rule)

            nb_combinations = np.prod([len(l) for l in compatible_rules])
            done = 0

            applicable = False
            for combination in itertools.product(*compatible_rules):
                done += 1

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
                    applicable = True
                    break

            if applicable:
                necessary_constraints.append(constraint)


        return rules, necessary_constraints

    @staticmethod
    def partial_combinations(states, domains):
        """
        Returns: list of list of any
            partial states
        """
        output = []
        Synchronizer._partial_combinations(states, domains, [], output)
        return output

    @staticmethod
    def _partial_combinations(states, domains, current, output):
        # full state constructed
        if len(current) >= len(domains):
            # Check if cover observation
            for s in states:
                if Synchronizer.cover(current, s):
                    return

            output.append(current.copy())
            return

        for val in domains[len(current)]:
            current.append(val)

            # heuristic: check if cover some states
            enough = True
            for s in states:
                if Synchronizer.cover(current, s):
                    enough = False
                    break

            if enough: # partial state already enough to encode not observed state
                # complete state with -1 encoding all value
                completed_current = current+[-1 for i in range(0, len(domains)-len(current))]
                output.append(completed_current)
            else: # complete current state
                Synchronizer._partial_combinations(states, domains, current, output)

            current.pop()

    
    @staticmethod
    def cover(partial_state, complete_state):
        """
        Check covering of a state by a partial state
        """
        for i in range(0, len(partial_state)):
            if partial_state[i] == -1 or partial_state[i] != complete_state[i]:
                return False
        return True
