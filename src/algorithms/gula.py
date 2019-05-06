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
    def load_input_from_csv(filepath):
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
            x_size = 0

            for row in csv_reader:
                if len(row) == 0:
                    continue
                if line_count == 0:
                    x_size = row.index("y0")
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
    def fit(variables, values, transitions, program=None):
        """
        Preprocess transitions and learn rules for all observed variables/values.

        Args:
            variables: list of string
                variables of the system
            values: list of list of string
                possible value of each variable
            transitions: list of tuple (list of int, list of int)
                state transitions of dynamic system
            program: LogicProgram
                A logic program to be fitted (kind of background knowledge)

        Returns:
            LogicProgram
                A logic program whose rules:
                    - explain/reproduce all the input transitions
                    - are minimals
        """
        #eprint("Start GULA learning...")

        rules = []

        # Learn rules for each observed variable/value
        for var in range(0, len(variables)):
            for val in range(0, len(values[var])):
                positives, negatives = GULA.interprete(transitions, var, val)
                rules += GULA.fit_var_val(variables, values, var, val, positives, negatives, program)

        # Instanciate output logic program
        output = LogicProgram(variables, values, rules)

        return output


    @staticmethod
    def interprete(transitions, variable, value):
        """
        Split transitions into positive/negatives states for the given variable/value

        Args:
            transitions: list of tuple (list of int, list of int)
                state transitions of dynamic system
            variable: int
                variable id
            value: int
                variable value id
        """
        #eprint("Interpreting transitions...")
        positives = [t1 for t1,t2 in transitions if t2[variable] == value]
        negatives = [t1 for t1,t2 in transitions if t1 not in positives]

        return positives, negatives


    @staticmethod
    def fit_var_val(variables, values, variable, value, positives, negatives, program=None):
        """
        Learn minimal rules that explain positive examples while consistent with negatives examples

        Args:
            variable: int
                variable id
            value: int
                variable value id
            positive: list of (list of int)
                States of the system where the variable can take this value in the next state
            negative: list of (list of int)
                States of the system where the variable cannot take this value in the next state
        """
        #eprint("\rStart learning of var="+str(variable)+", val="+str(value), end='')

        # 0) Initialize program as most the general rule
        #------------------------------------------------
        minimal_rules = [Rule(variable, value)]

        if program is not None:
            minimal_rules = program.get_rules_of(variable, value)

        # Revise learned rules agains each negative example
        for neg in negatives:

            # 1) Extract unconsistents rules
            #--------------------------------
            unconsistents = [ rule for rule in minimal_rules if rule.matches(neg) ]
            minimal_rules = [ rule for rule in minimal_rules if rule not in unconsistents ]

            for unconsistent in unconsistents:

                # Generates all least specialisation of the rule
                ls = []
                for var in range(len(variables)):
                    for val in range(len(values[var])):

                        # Variable availability
                        if unconsistent.has_condition(var):
                            continue

                        # Value validity
                        if val == neg[var]:
                            continue

                        # Create least specialization of r on var/val
                        least_specialization = unconsistent.copy()
                        least_specialization.add_condition(var,val)

                        # Discard if subsumed by a consistent minimal rule
                        subsumed = False
                        for minimal_rule in minimal_rules:
                            if minimal_rule.subsumes(least_specialization):
                                subsumed = True
                                break

                        # New consistent minimal rule
                        if not subsumed:
                            minimal_rules = [ minimal_rule for minimal_rule in minimal_rules if not least_specialization.subsumes(minimal_rule) ]
                            minimal_rules.append(least_specialization)

        #DBG
        #eprint("\r",end='')

        return minimal_rules
