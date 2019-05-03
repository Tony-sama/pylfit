#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2019/04/10
# @updated: 2019/05/03
#
# @desc: simple multi-valued LF1T implementation.
#   - INPUT: a set of pairs of discrete multi-valued states
#   - OUTPUT: all minimal rules that realizes the input
#   - THEORY:
#       - MLJ 2014: Learning from Interpretation Transition
#           http://link.springer.com/article/10.1007%2Fs10994-013-5353-8
#       - ILP 2014: Learning Prime Implicant Conditions From Interpretation Transition
#           http://link.springer.com/chapter/10.1007%2F978-3-319-23708-4_8
#       - PhD Thesis 2015: Studies on Learning Dynamics of Systems from State Transitions
#           http://www.tonyribeiro.fr/material/thesis/phd_thesis.pdf
#   - COMPLEXITY:
#       - Variables: exponential
#       - Values: exponential
#       - Observations: polynomial
#       - about O( |observations| * |values|^|variables| )
#-------------------------------------------------------------------------------

from utils import eprint
from rule import Rule
from logicProgram import LogicProgram
import csv

class LF1T:
    """
    Define a simple complete version of the LF1T algorithm.
    Learn logic rules that explain state transitions
    of a dynamic system:
        - discrete
        - deterministic
    INPUT: a set of pairs of states
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
    def fit(variables, values, transitions):
        """
        Preprocess transitions and learn rules for all observed variables/values.
        Assume deterministics transitions: only one future for each state.

        Args:
            variables: list of string
                variables of the system
            values: list of list of string
                possible value of each variable
            transitions: list of tuple (list of int, list of int)
                state transitions of a the system

        Returns:
            LogicProgram
                A logic program whose rules:
                    - explain/reproduce all the input transitions
                        - are minimals
        """
        #eprint("Start LF1T learning...")

        rules = []

        # Learn rules for each variable/value
        for var in range(0, len(variables)):
            for val in range(0, len(values[var])):
                rules += LF1T.fit_var_val(variables, values, transitions, var, val)

        # Instanciate output logic program
        output = LogicProgram(variables, values, rules)

        return output

    @staticmethod
    def fit_var_val(variables, values, transitions, variable, value):
        """
        Learn minimal rules that explain positive examples while consistent with negatives examples

        Args:
            variables: list of string
                variables of the system
            values: list of list of string
                possible value of each variable
            transitions: list of tuple (list of int, list of int)
                state transitions of a the system
            variable: int
                variable id
            value: int
                variable value id
        """
        #eprint("\rLearning var="+str(variable+1)+"/"+str(len(variables))+", val="+str(value+1)+"/"+str(len(values[variable])), end='')

        # 0) Start with the empty rule
        minimal_rules = [Rule(variable, value)]

        # Revise learned rules agains each transition
        for state_1, state_2 in transitions:

            # 1) Extract unconsistents rules
            #--------------------------------
            unconsistents = [ rule for rule in minimal_rules if state_2[variable] != value and rule.matches(state_1) ]
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
                        if val == state_1[var]:
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
