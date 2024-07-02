#-----------------------
# @author: Tony Ribeiro
# @created: 2019/10/29
# @updated: 2023/12/27
#
# @desc: simple implementation synchronous semantic over DMVLP
#   - Update all variables at the same time
#   - Can generate non-deterministic transitions
#-----------------------

from ..utils import eprint
from ..objects import rule
from . import Semantics
from . import Synchronous

import itertools

class SynchronousConstrained(Semantics):
    """
    Define the synchronous constrained semantic over discrete multi-valued logic program
    """
    @staticmethod
    def next(feature_state, targets, rules, constraints):
        """
        Compute the next state according to the rules and the synchronous semantics.

        Args:
            feature_state: list of int.
                A state of the system.
            targets: list of (String, list of String).
                Targets variables domains.
            rules: list of Rule.
                A list of multi-valued logic rules.
            constraints: list of Rule.
                A list of multi-valued logic constraints

        Returns:
            dict of list of any:list of rules.
                the possible next states and the rules that produce it according to the semantics.
        """
        # Apply synchronous semantics
        candidates = Synchronous.next(feature_state, targets, rules)

        # Apply constraints
        output = dict()
        for s, rules in candidates.items():
            valid = True
            for c in constraints:
                if c.matches(list(feature_state)+list(s)):
                    valid = False
                    break
            if valid:
                output[s] = rules

        return output
