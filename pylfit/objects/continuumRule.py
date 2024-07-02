#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2019/04/26
# @updated: 2023/12/27
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
    #_head_variable = 0

    """ Conclusion value: Continuum """
    #_head_value = Continuum()

    """ Conditions values: list of (int,Continuum) """
    #_body = []

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
        self._head_variable = head_variable
        self._head_value = head_value.copy()
        self._body = []

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
        return ContinuumRule(self._head_variable, self._head_value, self._body)

    @staticmethod
    def from_string(string_format, features, targets):
        """
        Construct a ContinuumRule from a string format using features/targets to convert to variable ids.

        Returns:
            Rule
                The rule represented by the string w.r.t features/targets
        """
        tokens = string_format.split(":-")

        head_string = tokens[0].split('(')

        head_variable = head_string[0].strip()
        head_variable_id = [var for (var,vals) in targets].index(head_variable)

        head_value = head_string[1].split(')')[0].strip()
        head_min_value_included = head_value[0] == '['
        head_min_value = head_value.split(',')[0].strip()[1:]
        head_max_value = head_value.split(',')[1].strip()[:-1]
        head_max_value_included = head_value[-1] == ']'
        head_value = Continuum(head_min_value, head_max_value, head_min_value_included, head_max_value_included)

        body_string = tokens[1].split(", ")

        # Empty rule
        if len(body_string) >= 1 and "(" not in body_string[0]:
            return ContinuumRule(head_variable_id, head_value)

        body = []

        for token in body_string:
            token = token.split("(")
            variable = token[0].strip()
            value = token[1].split(")")[0].strip()
            body.append((variable, value))

        body_encoded = []
        for variable,value in body:
            variable_id = [var for (var,vals) in features].index(variable)

            min_value_included = value[0] == '['
            min_value = value.split(',')[0].strip()[1:]
            max_value = value.split(',')[1].strip()[:-1]
            max_value_included = value[-1] == ']'
            value = Continuum(min_value, max_value, min_value_included, max_value_included)

            body_encoded.append( (variable_id, value) )

        return ContinuumRule(head_variable_id, head_value, body_encoded)


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
        return len(self._body)

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
        for (var, val) in self._body:
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
            if (self.head_variable != rule.head_variable) or (self.head_value != rule.head_value):
                return False

            # Different size
            if len(self.body) != len(rule.body):
                return False

            # Check conditions
            for c in self.body:
                if c not in rule.body:
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
        out = str(self._head_variable) + "=" + self._head_value.to_string()

        out += " :- "
        for var, val in self._body:
            out += str(var) + "=" + val.to_string() + ", "
        if len(self._body) > 0:
            out = out[:-2]
        out += "."
        return out

    def logic_form(self, features, targets):
        """
        Convert the rule to a logic programming string format,
        using given variables labels

        Args:
            features: list of pair (string, continuum)
            targets: list of pair (string, continuum)

        Returns:
            String
                a readable logic programmong representation of the rule
        """
        var_label = targets[self._head_variable % len(targets)][0]
        out = str(var_label) + "(" + self._head_value.to_string() + ") :- "

        for var, val in self._body:
            var_label = features[var % len(features)][0]
            out += str(var_label) + "(" + val.to_string() + "), "
        if len(self._body) > 0:
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
        for (var,val) in self._body:
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
        if self.head_variable != rule.head_variable:
            return False

        # Conclusion more specific
        if not rule.head_value.includes(self.head_value):
            return False

        # Conditions more general
        for var, val in self._body:
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
        for (var, val) in self._body:
            if (var == variable):
                self._body.pop(index)
                return
            index += 1

#--------------
# Accessors
#--------------

    #_head_variable = 0

    """ Conclusion value: Continuum """
    #_head_value = Continuum()

    """ Conditions values: list of (int,Continuum) """
    #_body = []

    @property
    def head_variable(self):
        return self._head_variable

    @head_variable.setter
    def head_variable(self, value):
        self._head_variable = value

    @property
    def head_value(self):
        return self._head_value

    @head_value.setter
    def head_value(self, value):
        self._head_value = value.copy()

    @property
    def body(self):
        return self._body

#--------------
# Mutatators
#--------------

    def set_condition(self, variable, value):
        """
        Condition mutator method

        Args:
            variable: int
                id of the variable
            value: Continuum
                new value of the condition over the given variable
        """
        for i, (var,val) in enumerate(self._body):
            # new condition variable
            if var > variable:
                self._body.insert(i, (variable, value))
                return

            # condition found
            if var == variable:
                self._body[i] = (variable, value)
                return

        # new condition on variable id bigger than biggest knowned
        self._body.append( (variable, value) )
