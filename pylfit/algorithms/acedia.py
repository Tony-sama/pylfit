#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2019/04/30
# @updated: 2019/05/03
#
# @desc: simple ACEDIA implementation:
#   - INPUT: a set of pairs of continuous valued state transitions
#   - OUTPUT: the optimal continuum logic program that realizes the input
#   - THEORY:
#       - ILP 2017:
#       Inductive Learning from State Transitions over Continuous Domains
#       https://hal.archives-ouvertes.fr/hal-01655644
#   - COMPLEXITY:
#       - Variables: exponential
#       - Observations: exponential
#       - about O( (|Observations| ^ (2 * |variables|)) * |variables| ^ 5 )
#-------------------------------------------------------------------------------

from ..utils import eprint
from ..objects.continuum import Continuum
from ..objects.continuumRule import ContinuumRule
from ..objects.continuumLogicProgram import ContinuumLogicProgram

import warnings

import csv

class ACEDIA: # TODO link to Algorithm
    """
    Define a simple complete version of the ACEDIA algorithm.
    Learn logic rules that explain state transitions
    of a dynamic system:
        - continuous valued
        - continuum deterministic
    INPUT: a set of pairs of continuous valued states
    OUTPUT: a continuum logic program
    """

    @staticmethod
    def load_input_from_csv(filepath):
        """
        Load transitions from a csv file

        Args:
            filepath: String
                Path to csv file encoding transitions
        """
        output = []
        with open(filepath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            x_size = 0

            for row in csv_reader:
                if line_count == 0:
                    x_size = row.index("y0")
                    x = row[:x_size]
                    y = row[x_size:]
                    #eprint("x: "+str(x))
                    #eprint("y: "+str(y))
                else:
                    row = [float(i) for i in row] # float convertion
                    output.append([row[:x_size], row[x_size:]]) # x/y split
                line_count += 1

            #eprint(f'Processed {line_count} lines.')
        return output

    @staticmethod
    def fit(variables, domains, transitions):
        """
        Preprocess transitions and learn rules for all observed variables/values.
        Assume deterministics transitions: only one future for each state.

        Args:
            variables: list of string
                variables of the system
            domains: list of Continuum
                domains of value of each variable
            transitions: list of tuple (list of float, list of float)
                state transitions of continuous valued dynamic system

        Returns:
            ContinuumLogicProgram
                A continuum logic program whose rules:
                    - explain/reproduce all the input transitions
                    - are minimals
        """
        #eprint("Start ACEDIA learning...")

        rules = []

        # Learn rules for each variable
        for var in range(0, len(variables)):
                rules += ACEDIA.fit_var(variables, domains, transitions, var)

        # Instanciate output logic program
        output = ContinuumLogicProgram(variables, domains, rules)

        return output

    @staticmethod
    def fit_var(variables, domains, transitions, variable):
        """
        Learn minimal rules that realizes the given transitions

        Args:
            variables: list of string
                variables of the system
            domains: list of Continuum
                domains of value of each variable
            transitions: list of (list of float, list of float)
                states transitions of the system
            variable: int
                variable id
        """
        #eprint("\rLearning var="+str(variable+1)+"/"+str(len(variables)), end='')

        # 0) Initialize undominated rule
        #--------------------------------
        body = [(var, domains[var]) for var in range(len(domains))]
        minimal_rules = [ContinuumRule(variable, Continuum(), body)]

        # Revise learned rules against each transition
        for state_1, state_2 in transitions:

            # 1) Extract unconsistents rules
            #--------------------------------
            unconsistents = [ rule for rule in minimal_rules if rule.matches(state_1) and not rule.get_head_value().includes(state_2[variable]) ]
            minimal_rules = [ rule for rule in minimal_rules if rule not in unconsistents ]

            for unconsistent in unconsistents:

                revisions = ACEDIA.least_revision(unconsistent, state_1, state_2)

                for revision in revisions:

                    # Check domination
                    dominated = False

                    for r in minimal_rules:
                        if r.dominates(revision):
                            dominated = True
                            break

                    # Remove dominated rules
                    if not dominated:
                        minimal_rules = [ r for r in minimal_rules if not revision.dominates(r) ]
                        minimal_rules.append(revision)

        # 2) remove domains wize conditions
        #-----------------------------------

        output = []

        for r in minimal_rules:
            if r.get_head_value().is_empty():
                continue

            r_ = r.copy()

            for var, val in r.get_body():
                if val == domains[var]:
                    r_.remove_condition(var)

            output.append(r_)

        #DBG
        #eprint("\r",end='')

        return output

    @staticmethod
    def least_revision(rule, state_1, state_2):
        """
        Compute the least revision of rule w.r.t. the transition (state_1, state_2)

        Agrs:
            rule: ContinuumRule
            state_1: list of float
            state_2: list of float

        Returns: list of ContinuumRule
            The least generalisation of the rule over its conclusion and
            the least specializations of the rule over each condition
        """

        # 0) Check Consistence
        #----------------------

        #Consistent rule
        if not rule.matches(state_1):
            raise ValueError("Attempting to revise a consistent rule, revision would be itself, this call is useless in ACEDIA and must be an error")

        # Consistent rule
        if rule.get_head_value().includes(state_2[rule.get_head_variable()]):
            raise ValueError("Attempting to revise a consistent rule, revision would be itself, this call is useless in ACEDIA and must be an error")


        # 1) Revise conclusion
        #----------------------
        head_var = rule.get_head_variable()
        next_value = state_2[head_var]
        revisions = []

        head_revision = rule.copy()
        head_value = head_revision.get_head_value()

        # Empty set head case
        if head_value.is_empty():
            head_value = Continuum(next_value, next_value, True, True)
        elif next_value <= head_value.get_min_value():
            head_value.set_lower_bound(next_value, True)
        else:
            head_value.set_upper_bound(next_value, True)

        head_revision.set_head_value(head_value)

        revisions.append(head_revision)

        # 2) Revise each condition
        #---------------------------
        for var, val in rule.get_body():
            state_value = state_1[var]

            # min revision
            min_revision = rule.copy()
            new_val = val.copy()
            new_val.set_lower_bound(state_value, False)
            if not new_val.is_empty():
                min_revision.set_condition(var, new_val)
                revisions.append(min_revision)

            # max revision
            max_revision = rule.copy()
            new_val = val.copy()
            new_val.set_upper_bound(state_value, False)
            if not new_val.is_empty():
                max_revision.set_condition(var, new_val)
                revisions.append(max_revision)

        return revisions
