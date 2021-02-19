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

from ..utils import eprint
from ..objects.rule import Rule
from ..objects.logicProgram import LogicProgram
from ..algorithms.algorithm import Algorithm

import csv

class LF1T (Algorithm):
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
    def fit(data, feature_domains, target_domains):
        """
        Preprocess transitions and learn rules for all observed variables/values.
        Assume deterministics transitions: only one future for each state.

        Args:
            data: list of tuple (list of int, list of int)
                state transitions of a the system
            feature_domains: list of (String, list of String)
                feature variables of the system and their values
            target_domains: list of (String, list of String)
                target variables of the system and their values

        Returns:
            LogicProgram
                A logic program whose rules:
                    - explain/reproduce all the input transitions
                    - are minimals
        """
        #eprint("Start LF1T learning...")

        rules = []

        # Learn rules for each variable/value
        for var in range(0, len(target_domains)):
            for val in range(0, len(target_domains[var][1])):
                rules += LF1T.fit_var_val(feature_domains, var, val, data)

        # Instanciate output logic program
        output = LogicProgram(feature_domains, target_domains, rules)

        return output

    @staticmethod
    def fit_var_val(feature_domains, variable, value, transitions):
        """
        Learn minimal rules that explain positive examples while consistent with negatives examples

        Args:
            feature_domains: list of string
                variables of the system
            variable: int
                variable id
            value: int
                variable value id
            transitions: list of tuple (list of int, list of int)
                state transitions of a the system
        """
        #eprint("\rLearning var="+str(variable+1)+"/"+str(len(variables))+", val="+str(value+1)+"/"+str(len(values[variable])), end='')

        # 0) Start with the empty rule
        minimal_rules = [Rule(variable, value, len(feature_domains))]

        # Revise learned rules agains each transition
        for state_1, state_2 in transitions:

            # 1) Extract unconsistents rules
            #--------------------------------
            unconsistents = [ rule for rule in minimal_rules if state_2[variable] != value and rule.matches(state_1) ]
            minimal_rules = [ rule for rule in minimal_rules if rule not in unconsistents ]

            for unconsistent in unconsistents:

                # Generates all least specialisation of the rule
                ls = []
                for var in range(len(feature_domains)):
                    for val in range(len(feature_domains[var][1])):

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
