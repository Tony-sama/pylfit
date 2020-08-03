#-----------------------
# @author: Tony Ribeiro
# @created: 2019/10/29
# @updated: 2019/10/29
#
# @desc: simple implementation synchronous semantic over LogicProgram
#   - Update all variables at the same time
#   - Can generate non-deterministic transitions
#-----------------------

from semantics import Semantics
from utils import eprint
from rule import Rule
from logicProgram import LogicProgram

import itertools

class Synchronous(Semantics):
    """
    Define the synchronous semantic over discrete multi-valued logic program
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
        # Check arguments
        if default != None:
            if len(default) != len(program.get_targets()) or [] in default:
                raise ValueError("default must be None or must give a list of values for each target variable")

        output = []
        domains = [set() for var in program.get_targets()]

        # extract conclusion of all matching rules
        for r in program.get_rules():
            if(r.matches(state)):
                domains[r.get_head_variable()].add(r.get_head_value())

        # Check variables without next value
        for i,domain in enumerate(domains):
            if len(domain) == 0:
                if default == None:
                    domains[i] = set([0])
                    if i < len(state):
                        domains[i] = set([state[i]])
                else:
                    domains[i] = set(default[i])

        # generate all combination of conclusions
        possible = set([i for i in list(itertools.product(*domains))])

        # DBG
        #eprint("state: ", state)
        #eprint("possible: ", possible)

        # apply constraints
        output = []
        for s in possible:
            valid = True
            for c in program.get_constraints():
                if c.matches(list(state)+list(s)):
                    valid = False
                    break
            if valid:
                #s = [program.get_conclusion_values()[var][s[var]] for var in range(0,len(s))]
                output.append(list(s))

        # DBG
        #eprint("constrainted: ", output)

        return output

    @staticmethod
    def transitions(program, default=None):
        output = []
        for s1 in program.states():
            next_states = Synchronous.next(program, s1, default)
            for s2 in next_states:
                output.append([list(s1),list(s2)])
        return output
