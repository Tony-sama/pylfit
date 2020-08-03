#-----------------------
# @author: Tony Ribeiro
# @created: 2020/07/14
# @updated: 2020/07/14
#
# @desc: simple implementation of asynchronous semantic over LogicProgram
#   - Update atmost one variables at a time
#   - Can generate non-deterministic transitions
#-----------------------

from utils import eprint
from rule import Rule
from logicProgram import LogicProgram
from semantics import Semantics

import itertools

class General(Semantics):
    """
    Define the general semantic over discrete multi-valued logic program
    Assume feature=targets
    """

    @staticmethod
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

        output = []
        domains = [set() for var in program.get_targets()]

        # extract conclusion of all matching rules
        for r in program.get_rules():
            if(r.matches(state)):
                domains[r.get_head_variable()].add(r.get_head_value())

        # Add default for variables without matching rules
        if default != None:
            if len(default) != len(program.get_targets()):
                raise ValueError("default must be None or must give a list of value (empty included) for each target variable")

            for var in range(0,len(program.get_targets())):
                if len(domains[var]) == 0:
                    domains[var] = set(default[var])

        # Add current value as possibility
        for var in range(0,len(program.get_targets())):
            if var < len(program.get_features()):
                domains[var].add(state[var])

        output = set([i for i in list(itertools.product(*domains))])

        return output

    @staticmethod
    def transitions(program, default=None):
        output = []
        for s1 in program.states():
            next_states = General.next(program, s1, default)
            for s2 in next_states:
                output.append([s1,s2])
        return output
