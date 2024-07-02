#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/15
# @updated: 2023/12/27
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
from ..objects.legacyAtom import LegacyAtom
from ..algorithms.gula import GULA
from ..algorithms.algorithm import Algorithm

class LFkT (Algorithm):
    """
    Define a simple complete version of the LFkT algorithm based on the
    GULA algorithm. Learn delayed logic rules that explain sequences
    of state transitions of a dynamic system:
        - discrete
        - deterministic
        - delayed
    INPUT: a set of sequences of states
    OUTPUT: a list of logic rules with delayed influences
    """

    @staticmethod
    def fit(time_series, features, targets):
        """
        Preprocess transitions and learn rules for all variables/values.

        Args:
            time_series: list of list (list of any, list of any)
                sequences of state transitions of the system.
            features: list of pair (string, list of any)
                features variables of the system and their domain.
            targets: list of pair (string, list of any)
                targets variables of the system and their domain.

        Returns:
            list of Rule
                A logic program whose rules:
                    - explain/reproduce all the input transitions
                        - are minimals
        """
        rules = []

        final_features = features

        # Learn rules for each variable/value
        for var_id, (var,vals) in enumerate(targets):
            for val in vals:
                head = LegacyAtom(var, set(vals), val, var_id)
                positives, negatives, delay = LFkT.interprete(time_series, head)

                # Extend Herbrand Base
                extended_features = dict()
                for d in range(1,delay+1):
                    for var_id_, (var_,vals_) in enumerate(features):
                        extended_features[var_+"_t-"+str(d)] = LegacyAtom(var_+"_t-"+str(d),vals_,None,var_id_).void_atom()

                rules += GULA.fit_var_val(head, extended_features, negatives)

                if len(extended_features) > len(final_features):
                    final_features = extended_features

        return rules


    @staticmethod
    def interprete(time_series, head):
        """
        Split the time series into positive/negatives meta-states for the given variable/value

        Args:
            time_series: list of list of pair (list of any, list of any)
                sequences of state transitions of the system
            head: LegacyAtom
                the target atom
        """
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
                        if state_1 == state_2 and head.matches(next_1) != head.matches(next_2):
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

        # 1) Aggregate states according to delay
        #----------------------------------------
        for serie in time_series:
            state_id = len(serie)-1
            while state_id >= global_delay:
                state = serie[state_id-global_delay:state_id].copy() # extract states
                state = [y for x in state for y in x] # fusion states
                if head.matches(serie[state_id]):
                    positives.append(state)
                else:
                    negatives.append(state)
                state_id -= 1

        return positives, negatives, global_delay
