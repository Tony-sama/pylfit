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

from utils import eprint
from rule import Rule
from logicProgram import LogicProgram
from gula import GULA
import csv

class LUST:
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
    def fit(variables, values, transitions):
        """
        Preprocess transitions and learn rules for all variables/values.

        Args:
            variables: list of string
                variables of the system
            values: list of list of string
                possible value of each variable
            transitions: list of (list of int, list of int)
                state transitions of a dynamic system

        Returns:
            list of LogicProgram
                    - each rules are minimals
                    - the output set explains/reproduces only the input transitions
        """
        #eprint("Start LUST learning...")

        # Nothing to learn
        if len(transitions) == 0:
            return [LogicProgram(variables, values, [])]

        rules = []

        # Extract strictly determinists states and separate non-determinist ones
        deterministic_core, deterministic_sets = LUST.interprete(variables, values, transitions)

        output = []
        common = GULA.fit(variables, values, deterministic_core)

        # deterministic input
        if len(deterministic_sets) == 0:
            return [common]

        for s in deterministic_sets:
            p = GULA.fit(variables, values, s, common)
            output.append(p)

        return output


    @staticmethod
    def interprete(variables, values, transitions):
        """
        Split the time series into positive/negatives meta-states for the given variable/value

        Args:
            time_series: list of list (list of int, list of int)
                state transitions of a dynamic system
            variable: int
                variable id
            value: int
                variable value id
        """
        # DBG
        #eprint("Interpreting transitions...")

        placed = []
        deterministic_core = []
        deterministic_sets = []

        # 0) Place transitions into deterministic sets
        #----------------------------------------------

        for s1, s2 in transitions:

            # avoid duplicate
            if [s1,s2] in placed:
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
