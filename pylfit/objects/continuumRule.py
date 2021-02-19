#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2019/04/26
# @updated: 2019/05/03
#
# @desc: Class ContinuumRule python source code file
#-------------------------------------------------------------------------------

import random

from ..objects.continuum import Continuum

class ContinuumRule:
    """
    Define a continuum logic rule, conclusion and conditions are pairs (variable, continuum)
        - one conclusion
        - one conjonction of conditions
            - atmost one condition per variable, i.e. atmost one continuum is specified for a variable
        - a rule holds when all conditions continuum includes the values of the system state, i.e. the rule matches the state
    """

    """ Conclusion variable id: int """
    __head_variable = 0

    """ Conclusion value: Continuum """
    __head_value = Continuum()

    """ Conditions values: list of (int,Continuum) """
    __body = []

#--------------
# Constructors
#--------------

    def __init__(self, head_variable, head_value, body=None):
        """
        Constructor of a continuum logic rule

        Args:
            head_variable: int
                id of the head variable
            head_value: Continuum
                values of the head variable
            body: list of tuple (int,Continuum)
                list of conditions as pairs of variable id, continuum of values
        """
        self.__head_variable = head_variable
        self.__head_value = head_value.copy()
        self.__body = []

        if body is not None:
            for var, val in body:
                self.set_condition(var,val)

    def copy(self):
        """
        copy method

        Returns:
            Rule
                A copy of the rule
        """
        return ContinuumRule(self.__head_variable, self.__head_value, self.__body)

    @staticmethod
    def random(head_variable, head_value, variables, domains, min_body_size, max_body_size):
        """
        Generates a valid continuum rule of given size randomly.

        Args:
            head_variable: int
                id of the head variable
            head_value: Continuum
                range of values of the head variable
            variables: list of String
                labels of the variable of a dynamic system
            domains: list of pairs of (int, Continuum)
                domains of values of each variable
            min_body_size: int
                minimal number of conditions to appear in the generated rule
            max_body_size: int
                maximal number of conditions to appear in the generated rule

        Returns: ContinuumRule
            A random valid continuum rule
        """

        if min_body_size > max_body_size:
            raise ValueError("min_body_size must be inferior or equal to max_body_size")

        if min_body_size > len(variables):
            raise ValueError("min_body_size can't exceed the number of variables")

        if max_body_size > len(variables):
            raise ValueError("max_body_size can't exceed the number of variables")

        size = random.randint(min_body_size, max_body_size)

        locked = []

        r = ContinuumRule(head_variable, head_value)

        while r.size() < size:
            var = random.randint(0, len(variables)-1)
            val = Continuum.random(domains[var].get_min_value(), domains[var].get_max_value())

            if var not in locked:
                r.set_condition(var,val)
                locked.append(var)

        return r



#--------------
# Observers
#--------------

    def size(self):
        """
        Gives the number of conditions in the rule

        Returns:
            int
                the number of conditions in the rule body
        """
        return len(self.__body)

    def get_condition(self, variable):
        """
        Accessor to the condition value over the given variable

        Args:
            variable: int
                a variable id

        Returns:
            Continuum
                The value of the condition over the variable if it exists
                None if no condition exists on the given variable
        """
        for (var, val) in self.__body:
            if (var == variable):
                return val
        return None

    def has_condition(self, variable):
        """
        Observer to condition existence over the given variable

        Args:
            variable: int
                a variable id

        Returns:
            Bool
                True if a condition exists over the given variable
                False otherwize
        """
        return self.get_condition(variable) is not None



#--------------
# Operators
#--------------


    def __eq__(self, rule):
        """
        Compare equallity with other rule

        Args:
            rule: ContinuumRule

        Returns:
            Boolean
                True if the other rule is equal
                False otherwize
        """
        if isinstance(rule, ContinuumRule):
            # Different head
            if (self.get_head_variable() != rule.get_head_variable()) or (self.get_head_value() != rule.get_head_value()):
                return False

            # Different size
            if len(self.get_body()) != len(rule.get_body()):
                return False

            # Check conditions
            for c in self.get_body():
                if c not in rule.get_body():
                    return False

            # Same head, same number of conditions and all conditions appear in the other rule
            return True

        return False

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
        out = str(self.__head_variable) + "=" + self.__head_value.to_string()

        out += " :- "
        for var, val in self.__body:
            out += str(var) + "=" + val.to_string() + ", "
        if len(self.__body) > 0:
            out = out[:-2]
        out += "."
        return out

    def logic_form(self, variables):
        """
        Convert the rule to a logic programming string format,
        using given variables labels

        Args:
            variables: list of string
                labels of the variables

        Returns:
            String
                a readable logic programmong representation of the rule
        """
        var_label = variables[self.__head_variable % len(variables)]
        out = str(var_label) + "(" + self.__head_value.to_string() + ",T) :- "

        for var, val in self.__body:
            var_label = variables[var % len(variables)]
            delay = int(var / len(variables)) + 1
            out += str(var_label) + "(" + val.to_string() + ",T-" + str(delay) + "), "
        if len(self.__body) > 0:
            out = out[:-2]
        out += "."
        return out

    def matches(self, state):
        """
        Check if the conditions of the rules holds in the given state

        Args:
            state: list of float
                a state of the system

        Returns:
            Boolean
                True if all conditions holds in the given state
                False otherwize
        """
        for (var,val) in self.__body:
            # delayed condition
            if(var >= len(state)):
                return False

            if(not val.includes(state[var])):
                return False
        return True

    def dominates(self, rule):
        """
        Check if the rule is more general and more precise:
            - conclusion is smaller (included in the other)
            - conditions are all bigger (they includes the others)

        Args:
            rule: ContinuumRule

        Returns:
            Boolean
                True if the rule dominates the other one
                False otherwize
        """

        # Different variable
        if self.get_head_variable() != rule.get_head_variable():
            return False

        # Conclusion more specific
        if not rule.get_head_value().includes(self.get_head_value()):
            return False

        # Conditions more general
        for var, val in self.__body:
            if rule.get_condition(var) is None:
                return False

            if not val.includes(rule.get_condition(var)):
                return False

        # Dominates
        return True

    def remove_condition(self, variable):
        """
        Remove a condition from the body of the rule

        Args:
            variable: int
                id of a variable
        """
        index = 0
        for (var, val) in self.__body:
            if (var == variable):
                self.__body.pop(index)
                return
            index += 1

#--------------
# Accessors
#--------------

    def get_head_variable(self):
        """
        Accessor to __head_variable

        Returns:
            int
                the conclusion variable id
        """
        return self.__head_variable

    def get_head_value(self):
        """
        Accessor to __head_value

        Returns:
            Continuum
                the value range of the conclusion
        """
        return self.__head_value

    def get_body(self):
        """
        Accessor to __body

        Returns:
            list of pair (int, Continuum)
                list of conditions of the rule
        """
        return self.__body

#--------------
# Mutatators
#--------------

    def set_head_variable(self, variable):
        """
        Head variable mutator method

        Args:
            variable: int
                id of the new head variable
        """
        self.__head_variable = variable

    def set_head_value(self, value):
        """
        Head value mutator method

        Args:
            value: Continuum
                continuum of the new head value
        """
        self.__head_value = value.copy()

    def set_condition(self, variable, value):
        """
        Condition mutator method

        Args:
            variable: int
                id of the variable
            value: Continuum
                new value of the condition over the given variable
        """
        for i, (var,val) in enumerate(self.__body):
            # new condition variable
            if var > variable:
                self.__body.insert(i, (variable, value))
                return

            # condition found
            if var == variable:
                self.__body[i] = (variable, value)
                return

        # new condition on variable id bigger than biggest knowned
        self.__body.append( (variable, value) )
