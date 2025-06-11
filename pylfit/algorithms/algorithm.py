#-----------------------
# @author: Tony Ribeiro
# @created: 2020/12/14
# @updated: 2023/12/27
#
# @desc: LFIT algorithm interface class, define common methods.
#-----------------------

from ..objects.legacyAtom import LegacyAtom

class Algorithm:
    """
    Define the common methods to all pylfit algorithms.
    """

    @staticmethod
    def equality_likeliness(state_1,state_2,domains):
        if len(state_1) != len(state_2):
            return 0
        
        likeliness = 1
        for i, _ in enumerate(state_1):
            # One difference => not equal
            if state_1[i] != LegacyAtom._UNKNOWN_VALUE and state_2[i] != LegacyAtom._UNKNOWN_VALUE and state_1[i] != state_2[i]:
                return 0
            
            # Same value => no likeliness change
            if state_1[i] != LegacyAtom._UNKNOWN_VALUE and state_2[i] != LegacyAtom._UNKNOWN_VALUE and state_1[i] == state_2[i]:
                continue
            
            chances = 1 # Only one unknown
            possibilities = len(domains[i][1])

            # Both unknown
            if (state_1[i] == LegacyAtom._UNKNOWN_VALUE and state_2[i] == LegacyAtom._UNKNOWN_VALUE):
                chances = len(domains[i][1])
                possibilities = len(domains[i][1])**2
                
            likeliness *= chances/possibilities

        return likeliness