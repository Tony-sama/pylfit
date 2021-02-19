#-----------------------
# @author: Tony Ribeiro
# @created: 2020/07/14
# @updated: 2021/02/17
#
# @desc: simple implementation of general semantic over DMVLP
#   - Update any number of variables at a time
#   - Can generate non-deterministic transitions
#-----------------------

from .. import utils
from ..objects import rule
from ..semantics.semantics import Semantics
from ..utils import eprint

import itertools

class General(Semantics):
    """
    Define the general semantic over discrete multi-valued logic program
    Assume feature=targets
    """

    def next(feature_state, targets, rules):
        """
        Compute the next state according to the rules and the synchronous semantics.

        Args:
            feature_state: list of int.
                A state of the system.
            targets: list of (String, list of String).
                Targets variables domains.
            rules: list of Rule.
                A list of multi-valued logic rules.

        Returns:
            list of (list of int).
                the possible next states according to the rules.
        """

        output = []
        domains = [set() for var in targets]

        # extract conclusion of all matching rules
        for r in rules:
            if(r.matches(feature_state)):
                domains[r.head_variable].add(r.head_value)

        # Add current value as possibility
        for var in range(0,len(targets)):
            if var < len(feature_state):
                domains[var].add(feature_state[var])
            if len(domains[var]) == 0:
                domains[var] = set([-1])

        output = [list(i) for i in list(itertools.product(*domains))]

        return output
