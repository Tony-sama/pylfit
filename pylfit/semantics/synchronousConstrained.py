#-----------------------
# @author: Tony Ribeiro
# @created: 2019/10/29
# @updated: 2019/10/29
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
            list of (list of int).
                the possible next states according to the rules.
        """
        # Apply synchronous semantics
        candidates = Synchronous.next(feature_state, targets, rules)

        # Apply constraints
        output = []
        for s in candidates:
            valid = True
            for c in constraints:
                if c.matches(list(feature_state)+list(s)):
                    valid = False
                    #eprint(c, " matches ", feature_state, ", ", s)
                    break
            if valid:
                # Decode state with domain values
                output.append(s)

        # DBG
        #eprint("constrainted: ", output)

        return output
