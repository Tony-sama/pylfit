#-----------------------
# @author: Tony Ribeiro
# @created: 2019/10/29
# @updated: 2021/02/17
#
# @desc: simple implementation synchronous semantic over DMVLP
#   - Update all variables at the same time
#   - Can generate non-deterministic transitions
#-----------------------

from .. import utils
from ..objects import rule
from ..semantics.semantics import Semantics
from ..utils import eprint

import itertools

class Synchronous(Semantics):
    """
    Define the synchronous semantic over discrete multi-valued logic program
    """
    @staticmethod
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
        #eprint("-- Args --")
        #eprint("state: ", state)
        #eprint("targets: ", targets)
        #eprint("rules: ", rules)

        output = []
        domains = [set() for var in targets]

        # extract conclusion of all matching rules
        for r in rules:
            if(r.matches(feature_state)):
                domains[r.head_variable].add(r.head_value)

        # DBG
        #eprint("domains: ", domains)

        # Check variables without next value
        for i,domain in enumerate(domains):
            if len(domain) == 0:
                domains[i] = set([-1])

        # generate all combination of domains
        output = [list(i) for i in list(itertools.product(*domains))]

        # DBG
        #eprint("possible: ", possible)

        #eprint(output)

        return output
