#-----------------------
# @author: Tony Ribeiro
# @created: 2020/07/14
# @updated: 2021/06/15
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
                the possible next states according to the rules.
        """

        output = []
        domains = [set() for var in targets]
        matching_rules = []

        # extract conclusion of all matching rules
        for r in rules:
            if(r.matches(feature_state)):
                domains[r.head_variable].add(r.head_value)
                matching_rules.append(r)

        # Check variables without next value
        for i,domain in enumerate(domains):
            if len(domain) == 0:
                if default == None:
                    domains[i] = set([-1])
                else:
                    domains[i] = set(default[i][1])

        # Add current value as possibility
        for var in range(0,len(targets)):
            if var < len(feature_state):
                domains[var].add(feature_state[var])

        target_states = [list(i) for i in list(itertools.product(*domains))]

        output = dict()
        for s in target_states:
            realised_by = []
            for r in matching_rules:
                if r.head_value == s[r.head_variable]:
                    realised_by.append(r)
            output[tuple(s)] = realised_by

        return output
