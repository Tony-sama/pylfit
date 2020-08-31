#-----------------------
# @author: Tony Ribeiro
# @created: 2019/11/05
# @updated: 2019/11/05
#
# @desc: simple Synchronizer implementation, for Learning from states transitions from ANY semantic.
#   - INPUT: a set of pairs of discrete multi-valued states
#   - OUTPUT: a logic program with constraints that realize only the input transitions
#   - THEORY:
#       - NEW (MLJ 2020?)
#   - COMPLEXITY:
#       - Variables: exponential
#       - Values: exponential
#       - Observations: polynomial
#-----------------------

from utils import eprint
from rule import Rule
from logicProgram import LogicProgram
from gula import GULA
from pride import PRIDE
import csv
import numpy as np

import itertools

class Synchronizer:
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
    HEURISTIC_PARTIAL_IMPOSSIBLE_STATE = False

    @staticmethod
    def fit(data, features, targets, complete=True): #variables, values, transitions, conclusion_values=None, complete=True):
        """
        Preprocess transitions and learn rules for all variables/values.

        Args:
            variables: list of string
                variables of the system
            values: list of list of string
                possible value of each variable
            transitions: list of (list of int, list of int)
                state transitions of a dynamic system

        Returns:
            LogicProgram
                - each rules are minimals
                - the output set explains/reproduces the input transitions
        """
        #eprint("Start LUST learning...")

        # Nothing to learn
        #if len(transitions) == 0:
        #    return LogicProgram(variables, values, [])

        #if conclusion_values == None:
        #    conclusion_values = values

        # 1) Use GULA to learn local possibilities
        #------------------------------------------
        if complete:
            model = GULA.fit(data, features, targets)#variables, values, transitions, conclusion_values)
        else:
            model = PRIDE.fit(data, features, targets)#variables, values, transitions)

        # 2) Learn constraints
        #------------------------------------------

        #negatives = [list(i)+list(j) for i,j in data] # next state value appear before current state
        #extended_variables = variables + variables # current state variables id are now += len(variables)
        #extended_values = conclusion_values + values

        # DBG
        #eprint("variables:\n", extended_variables)
        #eprint("values:\n", extended_values)
        #eprint("positives:\n", positives)
        #eprint("negatives:\n", negatives)

        negatives = np.array([tuple(s1)+tuple(s2) for s1,s2 in data])
        if len(negatives) > 0:
            negatives = negatives[np.lexsort(tuple([negatives[:,col] for col in reversed(range(0,len(features)))]))]

        if complete:
            constraints = GULA.fit_var_val(features+targets, -1, -1, negatives)
        else:
            # Extract occurences of each transition
            next_states = dict()
            for (i,j) in data:
                s_i = tuple(i)
                s_j = tuple(j)
                # new init state
                if s_i not in next_states:
                    next_states[s_i] = (s_i,[])
                # new next state
                next_states[s_i][1].append(s_j)

            # DBG
            #eprint("Transitions grouped:\n", next_states)

            impossible = set()
            for i in next_states:
                # Extract all possible value of each variable in next state
                domains = [set() for var in features]

                for s in next_states[i][1]:
                    for var in range(0,len(s)):
                        domains[var].add(s[var])
                # DBG
                #eprint("domain: ", domains)
                combinations = Synchronizer.partial_combinations(next_states[i][1], domains)
                #eprint("output: ", combinations)
                #exit()
                # DBG

                # Extract unobserved combinations
                if Synchronizer.HEURISTIC_PARTIAL_IMPOSSIBLE_STATE:
                    missings = [(next_states[i][0], tuple(j)) for j in combinations]
                else:
                    missings = [(next_states[i][0], j) for j in list(itertools.product(*domains)) if j not in next_states[i][1]]

                if missings != []:
                    impossible.update(missings)
                #eprint("Missings: ", missings)
            # DBG
            #eprint("Synchronous impossible transitions:\n", impossible)

            # convert impossible transition for PRIDE input
            positives = [list(i)+list(j) for i,j in impossible]

            constraints = PRIDE.fit_var_val(-1, -1, len(features)+len(targets), positives, negatives)

        # DBG
        #eprint("Learned constraints:\n", [r.logic_form(variables, values, None, len(variables)) for r in constraints])

        # 3) Discard non-applicable constraints
        #---------------------------------------
        necessary_constraints = []
        if not complete:
            necessary_constraints = constraints
        else:
            #eprint()
            for constraint in constraints:
                #eprint(features)
                #eprint(targets)
                #eprint(constraint, " => ", constraint.logic_form(features+targets,targets))
                applicable = True
                for (var,val) in constraint.get_body():
                    #eprint(var)
                    # Each condition on targets must be achievable by a rule head
                    if var >= len(features):
                        head_var = var-len(features)
                        #eprint(var," ",val)
                        matching_rule = False
                        # The conditions of the rule must be in the constraint
                        for rule in model.get_rules():
                            #eprint(rule)
                            if rule.get_head_variable() == head_var and rule.get_head_value() == val:
                                matching_conditions = True
                                for (cond_var,cond_val) in rule.get_body():
                                    if constraint.has_condition(cond_var) and constraint.get_condition(cond_var) != cond_val:
                                        matching_conditions = False
                                        #eprint("conflict on: ",cond_var,"=",cond_val)
                                        break
                                if matching_conditions:
                                    matching_rule = True
                                    break
                        if not matching_rule:
                            #eprint("USELESS")
                            applicable = False
                            break
                if applicable:
                    #eprint("OK")
                    necessary_constraints.append(constraint)



        output = LogicProgram(features, targets, model.get_rules(), necessary_constraints)#, necessary_constraints)

        return output

    @staticmethod
    def partial_combinations(states, domains):
        output = []
        Synchronizer.__partial_combinations(states, domains, [], output)
        return output

    @staticmethod
    def __partial_combinations(states, domains, current, output):
        #eprint("current: ", current)
        # full state constructed
        if len(current) >= len(domains):
            # Check if cover observation
            for s in states:
                if Synchronizer.cover(current, s):
                    return

            output.append(current.copy())
            #eprint("output: ", output)
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
                Synchronizer.__partial_combinations(states, domains, current, output)

            current.pop()

    # Check covering of a state by a partial state
    @staticmethod
    def cover(partial_state, complete_state):
        for i in range(0, len(partial_state)):
            if partial_state[i] == -1 or partial_state[i] != complete_state[i]:
                return False
        return True
