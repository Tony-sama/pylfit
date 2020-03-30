#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/15
# @updated: 2019/05/03
#
# @desc: simple LFkT implementation, extension of LF1T for learning delayed influences.
#   - INPUT: time series of discrete muli-valued states
#   - OUTPUT: all minimal delayed rules that realizes the input
#   - THEORY:
#       - Frontiers 2015: Learning delayed influences of biological systems
#           http://www.frontiersin.org/Journal/Abstract.aspx?s=1267&name=bioinformatics_and_computational_biology&ART_DOI=10.3389/fbioe.2014.00081
#       - ILP 2015: Learning Multi-Valued Biological Models with Delayed Influence from Time-Series Observations
#           http://www.ilp2015.jp/papers/ILP2015_submission_44.pdf
#       - ICMLA 2015: Learning Multi-Valued Biological Models with Delayed Influence from Time-Series Observations
#           https://ieeexplore.ieee.org/document/7424281
#       - PhD Thesis 2015: Studies on Learning Dynamics of Systems from State Transitions
#           http://www.tonyribeiro.fr/material/thesis/phd_thesis.pdf
#   - COMPLEXITY:
#       - Variables: exponential
#       - Values: exponential
#       - delay: exponential
#       - Observations: polynomial
#       - about O(|observations| * |values|^(|variables| * max(delay)))
#-----------------------

from utils import eprint
from rule import Rule
from logicProgram import LogicProgram
from gula import GULA
import csv

class LFkT:
    """
    Define a simple complete version of the LFkT algorithm based on the
    GULA algorithm. Learn delayed logic rules that explain sequences
    of state transitions of a dynamic system:
        - discrete
        - deterministic
        - delayed
    INPUT: a set of sequences of states
    OUTPUT: a logic program with delayed influences
    """

    @staticmethod
    def fit(variables, values, time_series):
        """
        Preprocess transitions and learn rules for all variables/values.

        Args:
            variables: list of string
                variables of the system
            values: list of list of string
                possible value of each variable
            time_series: list of list (list of int, list of int)
                sequences of state transitions of the system

        Returns:
            LogicProgram
                A logic program whose rules:
                    - explain/reproduce all the input transitions
                        - are minimals
        """
        #eprint("Start LFkT learning...")

        # Nothing to learn
        if len(time_series) == 0:
            return LogicProgram(variables, values, [])

        rules = []

        # Learn rules for each variable/value
        for var in range(0, len(variables)):
            for val in range(0, len(values[var])):
                positives, negatives, delay = LFkT.interprete(variables, values, time_series, var, val)

                # Extend Herbrand Base
                extended_variables = variables.copy()
                extended_values = values.copy()
                for d in range(1,delay):
                    extended_variables += [var+"_"+str(d) for var in variables]
                    extended_values += values

                rules += GULA.fit_var_val(extended_variables, extended_values, var, val, positives, negatives)

        # Instanciate output logic program
        output = LogicProgram(variables, values, rules)

        return output


    @staticmethod
    def interprete(variables, values, time_series, variable, value): #TODO: factorise delay detection to variable level
        """
        Split the time series into positive/negatives meta-states for the given variable/value

        Args:
            variables: list of string
                variables of the system
            values: list of list of string
                possible value of each variable
            time_series: list of list (list of int, list of int)
                sequences of state transitions of the system
            variable: int
                variable id
            value: int
                variable value id
        """
        # DBG
        #eprint("Interpreting transitions...")
        positives = []
        negatives = []

        # 0) detect the delay of the variable
        #-------------------------------------
        global_delay = 1
        for serie_1 in time_series:
            for id_state_1 in range(len(serie_1)-1):
                state_1 = serie_1[id_state_1]
                next_1 = serie_1[id_state_1+1]
                # search duplicate with different future
                for serie_2 in time_series:
                    for id_state_2 in range(len(serie_2)-1):
                        state_2 = serie_2[id_state_2]
                        next_2 = serie_2[id_state_2+1]

                        # Non-determinism detected
                        if state_1 == state_2 and next_1[variable] != next_2[variable]:
                            local_delay = 2
                            id_1 = id_state_1
                            id_2 = id_state_2
                            while id_1 > 0 and id_2 > 0:
                                previous_1 = serie_1[id_1-1]
                                previous_2 = serie_2[id_2-1]
                                if previous_1 != previous_2:
                                    break
                                local_delay += 1
                                id_1 -= 1
                                id_2 -= 1

                            global_delay = max(global_delay, local_delay)

        #eprint("Delay found: ", global_delay)

        # 1) Aggregate states according to delay
        #----------------------------------------
        for serie in time_series:
            state_id = len(serie)-1
            while state_id >= global_delay:
                state = serie[state_id-global_delay:state_id].copy() # extract states
                state.reverse()
                state = [y for x in state for y in x] # fusion states
                if serie[state_id][variable] == value:
                    positives.append(state)
                else:
                    negatives.append(state)
                state_id -= 1

        return positives, negatives, global_delay
