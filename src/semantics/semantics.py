#-----------------------
# @author: Tony Ribeiro
# @created: 2020/07/14
# @updated: 2020/07/14
#
# @desc: simple implementation of asynchronous semantic over LogicProgram
#   - Update atmost one variables at a time
#   - Can generate non-deterministic transitions
#-----------------------

from abc import abstractmethod

from utils import eprint
from rule import Rule
from logicProgram import LogicProgram

class Semantics:
    """
    Define the abstract class semantic over discrete multi-valued logic program
    """

    @abstractmethod
    def next(program, state, default=None):
        """
        Compute the next state according to the rules of the program.

        Args:
            program: LogicProgram
                A multi-valued logic program
            state: list of int
                A state of the system
            default: list of list of int
                Optional default values for each variable if no rule match
                If not given state value will be considered

        Returns:
            list of (list of int)
                the possible next states according to the rules of the program.
        """

    @abstractmethod
    def transitions(program, default=None):
        """
        """

    @staticmethod
    def transitions_to_csv(filepath, transitions, features, targets):
        """
        Convert a set of transitions to a csv file

        Args:
            filepath: String
                File path to where the csv file will be saved.
            transitions: list of tuple (list of int, list of int)
                transitions of the logic program
        """
        output = ""

        for var in range(0,len(features)):
            output += str(features[var][0])+","
        for var in range(0,len(targets)):
            output += str(targets[var][0])+","

        output = output[:-1] + "\n"

        for s1, s2 in transitions:
            for val in s1:
                output += str(val)+","
            for val in s2:
                output += str(val)+","
            output = output[:-1] + "\n"

        f = open(filepath, "w")
        f.write(output)
        f.close()
