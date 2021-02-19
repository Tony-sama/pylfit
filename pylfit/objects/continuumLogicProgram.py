#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2019/04/29
# @updated: 2019/05/03
#
# @desc: class ContinuumLogicProgram python source code file
# The program is assumed consistent, see the ILP 2017 paper:
#    https://hal.archives-ouvertes.fr/hal-01655644
#-------------------------------------------------------------------------------

from ..utils import eprint
from ..objects.continuum import Continuum
from ..objects.continuumRule import ContinuumRule

import random
import warnings

class ContinuumLogicProgram:
    """
    Define a Continuum Logic Program (CLP): a set of continuum rules
    over variables/values encoding the dynamics of a
    continuum deterministic system.
    """

    """ Variables of the CLP: list of string """
    __variables: []

    """ domains of values of each variable: list of Continuum """
    __domains: []

    """ Rules of the CLP: list of ContinuumRule """
    __rules: []

    # Constants
    __MIN_DOMAIN_SIZE = 0.0001

#--------------
# Constructors
#--------------

    def __init__(self, variables, domains, rules):
        """
        Create a ContinuumLogicProgram instance from given variables, variables values and rules

        Args:
            variables: list of String
                variables of the represented system
            domains: list of Continuum
                domain of values that each variable can take
            rules: list of ContinuumRule
                rules that define the system dynamics
        """
        if len(variables) != len(domains):
            raise ValueError("The number of domains does not correspond to the number of variables!")

        self.__variables = variables.copy()
        self.__domains = domains.copy()
        self.__rules = rules.copy()

    @staticmethod
    def random(variables, domains, rule_min_size, rule_max_size, epsilon, delay=1):
        """
        Generate a epsilon-complete ContinuumLogicProgram with a random dynamics.
        For each variable of the system, each possible epsilon state of the system is matched by at least one rule.

        Args:
            variables: list of String
                variables of the represented system
            domains: list of Continuum
                domain of values that each variable can take
            rule_min_size: int
                minimal number of conditions in each rule
            rule_max_size: int
                maximal number of conditions in each rule
            epsilon: float in ]0,1]
                precision of the completness of the program
            delay: int
                maximal delay of the conditions of each rule

        Returns:
            ContinuumLogicProgram
                an epsilon-complete CLP with a random dynamics
        """

        #eprint("Start random CLP generation: var ", len(variables), " delay: ", delay)
        extended_variables = variables.copy()
        extended_domains = domains.copy()

        # Delayed logic program: extend local herbrand base
        if delay > 1:
            for d in range(1,delay):
                extended_variables += [var+"_"+str(d) for var in variables]
                extended_domains += domains

        rules = []
        states = ContinuumLogicProgram.states(extended_domains, epsilon) # aggregated reversed time serie of size delay

        for s in states:
            #eprint(s)
            for var in range(len(variables)):
                matching = False
                for r in rules: # check if matched
                    if r.get_head_variable() == var and r.matches(s):
                        matching = True
                        break

                if not matching: # need new rule
                    val = Continuum()
                    while val.is_empty():
                        # Enumerate each possible value
                        min = domains[var].get_min_value()
                        max = domains[var].get_max_value()
                        step = epsilon * (max - min)
                        values = [min+(step*i) for i in range( int(1.0 / epsilon) )]
                        if values[-1] != max:
                            values.append(max)

                        min = random.choice(values)
                        max = random.choice(values)
                        while min > max:
                            min = random.choice(values)
                            max = random.choice(values)

                        val = Continuum(min, max, random.choice([True,False]), random.choice([True,False]))

                    body_size = random.randint(rule_min_size, rule_max_size)

                    new_rule = ContinuumRule(var, val, [])

                    # Prevent cross-match
                    # not necessarry, since next(state) assume determinism

                    # Complete the rule body if needed
                    while (new_rule.size() < body_size): # create body
                        cond_var = random.randint(0, len(s)-1)
                        if new_rule.has_condition(cond_var):
                            continue

                        # Enumerate each possible value
                        min = extended_domains[cond_var].get_min_value()
                        max = extended_domains[cond_var].get_max_value()
                        step = epsilon * (max - min)
                        values = [min+(step*i) for i in range( int(1.0 / epsilon) )]
                        if values[-1] != max:
                            values.append(max)

                        cond_val = Continuum()
                        while not cond_val.includes(s[cond_var]):
                            min = random.choice(values)
                            max = random.choice(values)
                            while min > max:
                                min = random.choice(values)
                                max = random.choice(values)
                            cond_val = Continuum(min, max, random.choice([True,False]), random.choice([True,False]))

                        new_rule.set_condition(cond_var, cond_val)

                    rules.append(new_rule)

        return ContinuumLogicProgram(variables, domains, rules)

#--------------
# Operators
#--------------

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

#--------------
# Methods
#--------------

    def to_string(self):
        """
        Convert the object to a readable string

        Returns:
            String
                a readable representation of the object
        """
        output = "{"
        output += "\nVariables: " + str(self.__variables)
        output += "\nDomains: " + str(self.__domains)
        output += "\nRules:\n"
        for r in self.__rules:
            output += r.to_string() + "\n"
        output += "}"

        return output

    def logic_form(self):
        """
        Convert the CLP to a logic programming string format

        Returns:
            String
                a logic programming representation of the CLP
        """
        output = ""

        # Variables declaration
        i = 0
        for var in range(len(self.__variables)):
            domain = self.__domains[i]
            output += "VAR "+ str(self.__variables[var]) + " " + str(domain.get_min_value()) + " " + str(domain.get_max_value())
            output += "\n"
            i += 1

        output += "\n"

        for r in self.__rules:
            output += r.logic_form(self.__variables) + "\n"

        return output

    def next(self, state):
        """
        Compute the next state according to the rules of the CLP

        Args:
            state: list of Continuum
                A Continuum state of the system

        Returns:
            list of Continuum
                the range of value that each variable can takes after the given state
        """
        output = [None for i in state]

        for r in self.__rules:
            if(r.matches(state)):
                # More precise conclusion
                if output[r.get_head_variable()] is None or output[r.get_head_variable()].includes(r.get_head_value()):
                    output[r.get_head_variable()] = r.get_head_value()

        return output

    @staticmethod
    def states(domains, epsilon):
        """
        Generates all states with atleast an epsilon distance

        Args:
            epsilon: float in ]0,1]
                the precision ratio of each state value

        Returns: list of (list of float)
            All possible state of the CLP with atleast an epsilon distance
        """
        if epsilon <= 0.0 or epsilon > 1.0:
            raise ValueError("Epsilon must be in ]0,1], got " + str(epsilon))

        if epsilon < 0.1:
            raise ValueError("Calling states of a CLP with an epsilon < 0.1, can generate a lot of states and takes a very long time")

        state = []
        for d in domains:
            if d is not None:
                state.append(d.get_min_value())
            else:
                state.append(None) # no value

        output = []
        ContinuumLogicProgram.__states(domains, epsilon, 0, state, output)
        return output

    @staticmethod
    def __states(domains, epsilon, variable, state, output):
        """
        Recursive sub-function of state(self, epsilon)

        Args:
            domains: list of Continuum
                domains of value of each variable
            epsilon: float in ]0,1]
                the precision ratio of each state value
            variable: int
                A variable id
            state: list of float
                A system state
            states: list of (list of float)
                the set of all states generated so far
        """

        # All variable are assigned
        if variable >= len(domains):
            excluded_bound = False
            for idx, val in enumerate(state):
                if val is None or not domains[idx].includes(val):
                    excluded_bound = True
                    break
            if not excluded_bound:
                output.append( state.copy() )
            return

        # No known value
        if domains[variable] is None:
            state[variable] = None
            ContinuumLogicProgram.__states(domains, epsilon, variable+1, state, output)
        else:
            # Enumerate each possible value
            min = domains[variable].get_min_value()
            max = domains[variable].get_max_value()
            step = epsilon * (max - min)
            values = [min+(step*i) for i in range( int(1.0 / epsilon) )]
            if values[-1] != max:
                values.append(max)

            #eprint(values)

            # bound exclusion
            if not domains[variable].min_included():
                values[0] = values[0] + ContinuumLogicProgram.__MIN_DOMAIN_SIZE * 0.5

            # bound exclusion
            if not domains[variable].max_included():
                values[-1] = values[-1] - ContinuumLogicProgram.__MIN_DOMAIN_SIZE * 0.5

            for val in values:
                state[variable] = val
                ContinuumLogicProgram.__states(domains, epsilon, variable+1, state, output)

    def generate_all_transitions(self, epsilon):
        """
        Generate all possible state of the program and their corresponding transition

        epsilon: float in ]0,1]
            the precision ratio of each state value

        Returns: list of tuple (list of int, list of int)
            The set of all transitions of the logic program
        """
        if epsilon <= 0 or epsilon > 1:
            raise ValueError("Epsilon must be in ]0,1], got " + str(epsilon))

        if epsilon < 0.1:
            raise ValueError("Calling states of a CLP with an epsilon < 0.1, can generate a lot of states and takes a very long time")

        output = []
        for s1 in ContinuumLogicProgram.states(self.__domains, epsilon):
            s2 = self.next(s1)
            S = ContinuumLogicProgram.states(s2, epsilon)
            for s2 in S:
                output.append( [s1.copy(), s2.copy()] )

        return output

    def transitions_to_csv(self, filepath, transitions):
        """
        Convert a set of transitions to a csv file

        Args:
            filepath: String
                File path to where the csv file will be saved
            transitions: list of tuple (list of int, list of int)
                transitions of the CLP
        """
        output = ""

        for var in range(0,len(self.__variables)):
            output += "x"+str(var)+","
        for var in range(0,len(self.__variables)):
            output += "y"+str(var)+","

        output = output[:-1] + "\n"

        for s1, s2 in transitions:
            for val in s1:
                output += str(val)+","
            for val in s2:
                output += str(val)+","
            output = output[:-1] + "\n"

        f = open(filepath, "w")
        f.write(output)
        f.close()

#--------
# Static
#--------

    @staticmethod
    def precision(expected, predicted):
        """
        Args:
            expected: list of tuple (list of float, list of float)
                originals transitions of a system
            predicted: list of (list of Continuum)
                predicted continuum states

        Returns:
            float in [0,1]
                the error ratio between expected and predicted
        """
        if len(expected) == 0:
            return 1.0

        # Predict each variable for each state
        total = len(expected) * len(expected[0][0])
        error = 0

        for i in range(len(expected)):
            s1, s2 = expected[i]
            s1_, s2_ = predicted[i]

            if s1 != s1_ or len(s2) != len(s2_):
                raise ValueError("Invalid prediction set")

            for var in range(len(s2)):
                if not s2_[var].includes(s2[var]):
                    error += 1

        precision = 1.0 - (error / total)

        return precision


#--------------
# Accessors
#--------------

    def get_variables(self):
        """
        variables accessor method

        Returns:
            list of string
                variables of the CLP
        """
        return self.__variables

    def get_domains(self):
        """
        domains accessor method

        Returns:
            list of Continuum
                domains of the variable of the CLP
        """
        return self.__domains

    def get_rules(self):
        """
        rules accessor method

        Returns:
            list of ContinuumRule
                rules of the CLP
        """
        return self.__rules

    def get_rules_of(self, variable):
        """
        specific variable rules accessor method

        Args:
            variable: int
                variable id

        Returns:
            list of ContinuumRule
                rules of the CLP whose head variable is var
        """
        output = []
        for r in self.__rules:
            if r.get_head_variable() == variable:
                output.append(r.copy())
        return output
