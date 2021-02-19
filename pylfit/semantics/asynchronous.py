#-----------------------
# @author: Tony Ribeiro
# @created: 2020/07/14
# @updated: 2021/02/17
#
# @desc: simple implementation of asynchronous semantic over LogicProgram
#   - Update atmost one variables at a time
#   - Can generate non-deterministic transitions
#-----------------------

from ..utils import eprint
from ..objects.rule import Rule
from ..semantics.semantics import Semantics

import itertools

class Asynchronous(Semantics):
    """
    Define the asynchronous semantic over discrete multi-valued logic program
    Assume feature=targets
    """

    @staticmethod
    def next(feature_state, targets, rules):
        """
        Compute the next state according to the rules and the asynchronous semantics.

        Args:
            feature_state: list of int.
                A state of the system.
            targets: list of (String, list of String).
                Targets variables domains.
            rules: list of Rules.
                A list of multi-valued logic rules.

        Returns:
            list of (list of int).
                the possible next states according to the rules.
        """

        # TODO: add projection to argument to handle different features/targets

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
        output = []

        # next is state with one replacement
        for var_id in range(0,len(domains)):
            for val in domains[var_id]:
                if feature_state[var_id] != val:
                    s2 = list(feature_state)
                    s2[var_id] = val
                    output.append(s2)

        # Self loop if no possible transitions
        if len(output) == 0:
            output.append(feature_state)

        # DBG
        #eprint("possible: ", possible)

        #eprint(output)

        return output
