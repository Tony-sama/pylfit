#-----------------------
# @author: Tony Ribeiro
# @created: 2020/07/14
# @updated: 2021/02/17
#
# @desc: simple implementation of asynchronous semantic over LogicProgram
#   - Update atmost one variables at a time
#   - Can generate non-deterministic transitions
#-----------------------

from abc import abstractmethod

class Semantics:
    """
    Define the abstract class semantic over discrete multi-valued logic program
    """

    @staticmethod
    @abstractmethod
    def next(feature_state, targets, rules):
        """
        Compute the next state according to the rules of the program.

        Args:
            feature_state: list of int.
                A state of the system.
            targets: list of (String, list of String).
                Targets variables domains.
            rules: list of Rule.
                A list of multi-valued logic rules.

        Returns:
            list of (list of int)
                the possible next states according to the rules.
        """
        raise NotImplementedError("Must be implemented by subclass")
