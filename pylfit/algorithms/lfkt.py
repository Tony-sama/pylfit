#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/15
# @updated: 2019/05/03
#
# @desc: simple LFkT implementation, extension of LF1T for learning delayed influences.
#   - INPUT: time series of discrete muli-valued states
#   - OUTPUT: all minimal delayed rules that realizes the input
#       - Frontiers 2015: Learning delayed influences of biological systems
#           http://www.frontiersin.org/Journal/Abstract.aspx?s=1267&name=bioinformatics_and_computational_biology&ART_DOI=10.3389/fbioe.2014.00081
#       - ILP 2015: Learning Multi-Valued Biological Models with Delayed Influence from Time-Series Observations
#       - ICMLA 2015: Learning Multi-Valued Biological Models with Delayed Influence from Time-Series Observations
#       - PhD Thesis 2015: Studies on Learning Dynamics of Systems from State Transitions
#           http://www.tonyribeiro.fr/material/thesis/phd_thesis.pdf
#   - COMPLEXITY:
#       - Variables: exponential
#       - Values: exponential
#       - delay: exponential
#       - Observations: polynomial
#       - about O(|observations| * |values|^(|variables| * max(delay)))
#-----------------------

from ..utils import eprint
from ..objects.rule import Rule
from ..objects.logicProgram import LogicProgram
from ..algorithms.gula import GULA
from ..algorithms.algorithm import Algorithm

import csv

class LFkT (Algorithm):
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
    def fit(data, feature_domains, target_domains): #variables, values, time_series):
        """
        Preprocess transitions and learn rules for all variables/values.

        Args:
            data: list of list (list of int, list of int)
                sequences of state transitions of the system
            variables: list of string
                variables of the system
            values: list of list of string
                possible value of each variable

        Returns:
            LogicProgram
                A logic program whose rules:
                    - explain/reproduce all the input transitions
                        - are minimals
        """
        #eprint("Start LFkT learning...")

        # Nothing to learn
        if len(data) == 0:
            return LogicProgram(feature_domains, target_domains, [])

        rules = []

        final_feature_domains = feature_domains

        # Learn rules for each variable/value
        for var in range(0, len(target_domains)):
            for val in range(0, len(target_domains[var][1])):
                positives, negatives, delay = LFkT.interprete(data, feature_domains, target_domains, var, val)

                # Extend Herbrand Base
                extended_feature_domains = []
                for d in range(1,delay+1):
                    extended_feature_domains = [(var+"_"+str(d),vals) for (var,vals) in feature_domains] + extended_feature_domains

                rules += GULA.fit_var_val(extended_feature_domains, var, val, negatives)

                if len(extended_feature_domains) > len(final_feature_domains):
                    final_feature_domains = extended_feature_domains

        # Instanciate output logic program
        output = LogicProgram(final_feature_domains, target_domains, rules)

        return output


    @staticmethod
    def interprete(time_series, feature_domains, target_domains, variable, value):
        """
        Split the time series into positive/negatives meta-states for the given variable/value

        Args:
            time_series: list of list (list of int, list of int)
                sequences of state transitions of the system
            feature_domains: list of (String, list of String)
                feature variables of the system and their values
            target_domains: list of (String, list of String)
                target variables of the system and their values
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
                #state.reverse()
                state = [y for x in state for y in x] # fusion states
                if serie[state_id][variable] == value:
                    positives.append(state)
                else:
                    negatives.append(state)
                state_id -= 1

        return positives, negatives, global_delay
