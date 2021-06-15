#-----------------------
# @author: Tony Ribeiro
# @created: 2019/10/29
# @updated: 2021/06/15
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
    def next(feature_state, targets, rules, default=None):
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
                the possible next states according to the rules and the synchronous semantics.
        """
        #eprint("-- Args --")
        #eprint("state: ", state)
        #eprint("targets: ", targets)
        #eprint("rules: ", rules)

        domains = [set() for var in targets]
        matching_rules = []

        # extract conclusion of all matching rules
        for r in rules:
            if(r.matches(feature_state)):
                domains[r.head_variable].add(r.head_value)
                matching_rules.append(r)

        # DBG
        #eprint("domains: ", domains)

        # Check variables without next value
        for i,domain in enumerate(domains):
            if len(domain) == 0:
                if default == None:
                    domains[i] = set([-1])
                else:
                    domains[i] = set(default[i][1])

        # generate all combination of domains
        target_states = [list(i) for i in list(itertools.product(*domains))]

        output = dict()
        for s in target_states:
            realised_by = []
            for r in matching_rules:
                if r.head_value == s[r.head_variable]:
                    realised_by.append(r)
            output[tuple(s)] = realised_by

        # DBG
        #eprint("possible: ", possible)

        return output
