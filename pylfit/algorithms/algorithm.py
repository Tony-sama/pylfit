#-----------------------
# @author: Tony Ribeiro
# @created: 2020/12/14
# @updated: 2020/12/14
#
# @desc: LFIT algorithm interface class, define common methods.
#-----------------------

from ..utils import eprint

class Algorithm:
    """
    Define the common methods to all pylfit algorithms.
    """
    @staticmethod
    def encode_transitions_set(data, feature_domains, target_domains): # TODO: catch exception iterable of pairs, len feat/targ
        """
        Convert string data into int corresponding to feature/target variable domain value id
        """
        output = [Algorithm.encode_transition(t, feature_domains, target_domains) for t in data]

        return output

    @staticmethod
    def encode_transition(transition, feature_domains, target_domains): # TODO: catch exception len == 2, len feat/targ
        s1 = Algorithm.encode_state(transition[0], feature_domains)
        s2 = Algorithm.encode_state(transition[1], target_domains)
        return (s1, s2)

    @staticmethod
    def encode_state(state, variable_domains): # TODO: catch exception len domain == state
        output = [str(i) for i in state]
        for var, val in enumerate(output):
            val_id = variable_domains[var][1].index(val) #TODO: catch exception, not in domain
            output[var] = val_id
        return tuple(output)

    # TODO: encode time serie
    # - consider serie of pair of feat/targ states but not meaning t-1/t in this case
