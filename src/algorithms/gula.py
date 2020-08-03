#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/15
# @updated: 2019/05/03
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

from utils import eprint
from rule import Rule
from logicProgram import LogicProgram
import csv
import numpy as np

class GULA:
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
    def load_input_from_csv(filepath, nb_features):
        """
        Load transitions from a csv file

        Args:
            filepath: String
                Path to csv file encoding transitions
        """
        output = []
        with open(filepath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            x_size = nb_features

            for row in csv_reader:
                if line_count == 0:
                    #x_size = row.index("y0")
                    x = row[:x_size]
                    y = row[x_size:]
                    #eprint("x: "+str(x))
                    #eprint("y: "+str(y))
                else:
                    row = [int(i) for i in row] # integer convertion
                    output.append([row[:x_size], row[x_size:]]) # x/y split
                line_count += 1

            #eprint(f'Processed {line_count} lines.')
        return output

    @staticmethod
    def fit(data, features, targets): #variables, values, transitions, conclusion_values=None, program=None): #, partial_heuristic=False):
        """
        Preprocess transitions and learn rules for all observed variables/values.

        Args:
            data: list of tuple (list of int, list of int)
                state transitions of a the system
            features: list of (String, list of String)
                feature variables of the system and their values
            targets: list of (String, list of String)
                targets variables of the system and their values

        Returns:
            LogicProgram
                A logic program whose rules:
                    - explain/reproduce all the input transitions
                    - are minimals
        """
        #eprint("Start GULA learning...")

        rules = []

        #if conclusion_values == None:
        #    conclusion_values = values

        #eprint(transitions)
        eprint("\nConverting transitions to nparray...")
        processed_transitions = np.array([tuple(s1)+tuple(s2) for s1,s2 in data])

        if len(processed_transitions) > 0:
            #eprint("flattened: ", processed_transitions)
            eprint("Sorting transitions...")
            processed_transitions = processed_transitions[np.lexsort(tuple([processed_transitions[:,col] for col in reversed(range(0,len(features)))]))]
            #for i in range(0,len(variables)):
            #processed_transitions = processed_transitions[np.argsort(processed_transitions[:,i])]
            #eprint("sorted: ", processed_transitions)

            eprint("Grouping transitions by initial state...")
            #processed_transitions = np.array([ (row[:len(variables)], row[len(variables):]) for row in processed_transitions])

            processed_transitions_ = []
            s1 = processed_transitions[0][:len(features)]
            S2 = []
            for row in processed_transitions:
                if not np.array_equal(row[:len(features)], s1): # New initial state
                    #eprint("new state: ", s1)
                    processed_transitions_.append((s1,S2))
                    s1 = row[:len(features)]
                    S2 = []

                #eprint("adding ", row[len(features):], " to ", s1)
                S2.append(row[len(features):]) # Add new next state

            # Last state
            processed_transitions_.append((s1,S2))

            processed_transitions = processed_transitions_

        # Learn rules for each observed variable/value
        for var in range(0, len(targets)):
            for val in range(0, len(targets[var][1])):
                negatives = GULA.interprete(processed_transitions, var, val)#, partial_heuristic)
                # DBG
                #eprint(negatives)
                eprint("\nStart learning of var=", var+1,"/", len(targets), ", val=", val+1, "/", len(targets[var][1]))
                rules += GULA.fit_var_val(features, var, val, negatives) #variables, values, var, val, negatives, program)#, partial_heuristic)

        # Instanciate output logic program
        output = LogicProgram(features, targets, rules)

        return output


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

        #positives = []
        negatives = []
        for s1, S2 in transitions:
            negative = True
            for s2 in S2:
                if s2[variable] == value:
                    negative = False
                    break
            if negative:
                negatives.append(s1)
            #elif partial_heuristic:
            #    positives.append(s1)

        return negatives


    @staticmethod
    def fit_var_val(features, variable, value, negatives): #variables, values, variable, value, negatives, program=None):#, partial_heuristic=False):
        """
        Learn minimal rules that explain positive examples while consistent with negatives examples

        Args:
            features: list of (name, list of int)
                Features variables
            variable: int
                variable id
            value: int
                variable value id
            negatives: list of (list of int)
                States of the system where the variable cannot take this value in the next state
        """

        # 0) Initialize program as most the general rule
        #------------------------------------------------
        minimal_rules = [Rule(variable, value, len(features), [])]

        #if program is not None:
        #    minimal_rules = program.get_rules_of(variable, value)

        # DBG
        neg_count = 0

        # Revise learned rules against each negative example
        for neg in negatives:

            neg_count += 1
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
                for var in range(len(features)):
                    for val in range(len(features[var][1])):

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
