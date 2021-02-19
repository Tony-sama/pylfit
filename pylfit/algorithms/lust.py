#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/23
# @updated: 2019/05/03
#
# @desc: simple LUST implementation, for Learning from Uncertain States Transitions.
#   - INPUT: a set of pairs of discrete multi-valued states
#   - OUTPUT: multiple set of minimal rules which together realize only the input transitions
#   - THEORY:
#       - ICLP 2015: Learning probabilistic action models from interpretation transitions
#           http://www.tonyribeiro.fr/material/publications/iclp_2015.pdf
#       - PhD Thesis 2015: Studies on Learning Dynamics of Systems from State Transitions
#           http://www.tonyribeiro.fr/material/thesis/phd_thesis.pdf
#   - COMPLEXITY:
#       - Variables: exponential
#       - Values: exponential
#       - non-determinism: polynomial
#       - Observations: polynomial
#       - about O(|observations| * max(non-deterministism) * |values|^(|variables| * max(delay)))
#-----------------------

from ..utils import eprint
from ..objects.rule import Rule
from ..objects.logicProgram import LogicProgram
from ..algorithms.gula import GULA
from ..algorithms.algorithm import Algorithm
import csv

class LUST (Algorithm):
    """
    Define a simple complete version of the LUST algorithm.
    Learn logic rules that explain sequences of state transitions
    of a dynamic system:
        - discrete
        - non-deterministic
        - no delay
    INPUT: a set of pairs of states
    OUTPUT: a set of logic programs
    """

    @staticmethod
    def fit(data, features, targets):
        """
        Preprocess transitions and learn rules for all variables/values.

        Args:
            data: list of tuple (list of int, list of int)
                state transitions of a the system
            features: list of (String, list of String)
                feature variables of the system and their values
            targets: list of (String, list of String)
                targets variables of the system and their values

        Returns:
            list of LogicProgram
                    - each rules are minimals
                    - the output set explains/reproduces only the input transitions
        """
        #eprint("Start LUST learning...")

        # Nothing to learn
        if len(data) == 0:
            return [LogicProgram(features, targets, [])]

        rules = []

        # Extract strictly determinists states and separate non-determinist ones
        deterministic_core, deterministic_sets = LUST.interprete(data)
        output = []
        #common = GULA.fit(deterministic_core, features, targets)

        # deterministic input
        if len(deterministic_sets) == 0:
            output = [GULA.fit(deterministic_core, features, targets)]
        else:
            for s in deterministic_sets:
                eprint("GULA input: ",deterministic_core+s)
                p = GULA.fit(deterministic_core+s, features, targets)
                output.append(p)

        return output


    @staticmethod
    def interprete(transitions):
        """
        Split the time series into positive/negatives meta-states for the given variable/value

        Args:
            transitions: list of tuple (list of int, list of int)
                state transitions of a dynamic system
        """
        # DBG
        #eprint("Interpreting transitions...")

        placed = []
        deterministic_core = []
        deterministic_sets = []

        # 0) Place transitions into deterministic sets
        #----------------------------------------------

        # convert numpy array to list of tuple
        transitions = [(list(s1),list(s2)) for s1,s2 in transitions]

        for s1, s2 in transitions:

            # avoid duplicate
            if [s1, s2] in placed:
                continue

            # 1) Deterministic Core
            #-----------------------

            deterministic = True
            for s3, s4 in transitions:
                if s1 == s3 and s2 != s4:
                    deterministic = False
                    break

            # Deterministic transition, go in deterministic core
            if deterministic:
                deterministic_core.append([s1,s2])
                placed.append([s1,s2])
                continue

            # 2) Deterministic sets
            #-----------------------

            # Search for a set which is consistent with the transition
            added = False
            for s in deterministic_sets:
                deterministic = True
                for s3, s4 in s:
                    if s1 == s3 and s2 != s4:
                        deterministic = False
                        break

                # Compatible with this set
                if deterministic:
                    s.append([s1,s2])
                    placed.append([s1,s2])
                    added = True
                    break

            # No consistent set, create new one
            if not added:
                deterministic_sets.append([ [s1,s2] ])
                placed.append([s1,s2])

        # 3) Complete the deterministic sets
        #------------------------------------
        for s1, s2 in placed:
            if [s1,s2] not in deterministic_core: # each origin state must appears
                for s in deterministic_sets: # in each deterministic set
                    occurs = False
                    for s3, s4 in s:
                        if s1 == s3:
                            occurs = True
                            break
                    if not occurs:
                        s.append([s1,s2])


        return deterministic_core, deterministic_sets
