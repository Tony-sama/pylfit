#-----------------------
# @author: Tony Ribeiro
# @created: 2019/10/29
# @updated: 2023/12/27
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
            default: list of pair (string, any)
                default value for each variables

        Returns:
            dict of list of any:list of rules.
                the possible next states and the rules that produce it according to the semantics.
        """

        domains = [set() for var in targets]
        matching_rules = []

        # extract conclusion of all matching rules
        for r in rules:
            if(r.matches(feature_state)):
                domains[r.head.state_position].add(r.head.value)
                matching_rules.append(r)

        # Check variables without next value
        for i,domain in enumerate(domains):
            if len(domain) == 0:
                if default is None:
                    domains[i] = set(["?"])
                else:
                    domains[i] = set(default[i][1])

        # generate all combination of domains
        target_states = [list(i) for i in list(itertools.product(*domains))]

        output = dict()
        for s in target_states:
            realised_by = []
            for r in matching_rules:
                if r.head.matches(s):
                    realised_by.append(r)
            output[tuple(s)] = realised_by

        return output
