#-----------------------
# @author: Tony Ribeiro
# @created: 2019/03/20
# @updated: 2019/05/03
#
# @desc: Class Rule python source code file
#-----------------------

import random

class Rule:
    """
    Define a discrete logic rule, conclusion and conditions are pairs (variable, value)
        - one conclusion
        - one conjonction of conditions
            - atmost one condition per variable, i.e. atmost one value is specified for a variable
        - a rule holds when all conditions are present in a system state, i.e. the rule matches the state
    """

    """ Conclusion variable id: int """
    __head_variable = 0

    """ Conclusion value id: int """
    __head_value = 0

    """ Conditions values: list of (int,int) """
    __body = []

#--------------
# Constructors
#--------------

    def __init__(self, head_variable, head_value, body=None):
        """
        Constructor of a discrete logic rule

        Args:
            head_variable: int
                id of the head variable
            head_value: int
                id of the value of the head variable
            body: list of tuple (int,int)
                list of conditions as pairs of variable id, value id
        """
        self.__head_variable = head_variable
        self.__head_value = head_value
        if body == None:
            self.__body = []
        else:
            self.__body = body.copy()

    def copy(self):
        """
        Copy method

        Returns:
            Rule
                A copy of the rule
        """
        return Rule(self.__head_variable, self.__head_value, self.__body)

    @staticmethod
    def random(head_variable, head_value, variables, values, min_body_size, max_body_size):
        """
        Generates a valid rule randomly of given size.

        Args:
            head_variable: int
                id of the head variable
            head_value: int
                id of the value of the head variable
            variables: list of String
                labels of the variable of a dynamic system
            values: list of (list of String)
                labels of the value of each variable
            min_body_size: int
                minimal number of conditions to appear in the generated rule (must be < #variables)
            max_body_size: int
                maximal number of conditions to appear in the generated rule (must be >= min_body_size)

        Returns: Rule
            A random valid rule
        """

        if min_body_size > max_body_size:
            raise ValueError("min_body_size must be inferior or equal to max_body_size")

        if min_body_size > len(variables):
            raise ValueError("min_body_size can't exceed the number of variables")

        if max_body_size > len(variables):
            raise ValueError("max_body_size can't exceed the number of variables")

        size = random.randint(min_body_size, max_body_size)

        body = []
        locked = []

        while len(body) < size:
            var = random.randint(0, len(variables)-1)
            val = random.randint(0, len(values[var])-1)

            if var not in locked:
                body.append( (var,val) )
                locked.append(var)


        return Rule(head_variable, head_value, body)



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

    def to_string(self):
        """
        Convert the object to a readable string

        Returns:
            String
                a readable representation of the object
        """
        out = str(self.__head_variable) + "=" + str(self.__head_value)

        out += " :- "
        for var, val in self.__body:
            out += str(var) + "=" + str(val) + ", "
        if len(self.__body) > 0:
            out = out[:-2]
        out += "."
        return out

    def logic_form(self, variables, values):
        """
        Convert the rule to a logic programming string format,
        using given variables/values labels.

        Args:
            variables: list of string
                labels of the variables
            values: list of (list of string)
                labels of each variable value

        Returns:
            String
                a logic programming string representation of the rule with original labels
        """
        var_label = variables[self.__head_variable % len(variables)]
        val_label = values[self.__head_variable % len(variables)][self.__head_value]
        out = str(var_label) + "(" + str(val_label) + ",T) :- "

        for var, val in self.__body:
            var_label = variables[var % len(variables)]
            val_label = values[var % len(variables)][val]
            delay = int(var / len(variables)) + 1
            out += str(var_label) + "(" + str(val_label) + ",T-" + str(delay) + "), "
        if len(self.__body) > 0:
            out = out[:-2]
        out += "."
        return out


    def get_condition(self, variable):
        """
        Accessor to the condition value over the given variable

        Args:
            variable: int
                a variable id

        Returns:
            int
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

    def matches(self, state):
        """
        Check if the conditions of the rules holds in the given state

        Args:
            state: list of int
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

            if(state[var] != val):
                return False
        return True

    def cross_matches(self, rule):
        """
        Check if their is a state that both rules match

        Args:
            rule: Rule

        Returns:
            Boolean
                True if the rules cross-match
                False otherwize
        """
        for var, val in self.__body:
            val_ = rule.get_condition(var)
            if val_ is not None and val_ != val:
                return False
        return True

    def subsumes(self, rule):
        """
        Check if the rule will match every states the other rule match.
        Onlly occurs when all conditions of the current rules appears in the other one.

        Args:
            rule: Rule

        Returns:
            Boolean
                True if the rule subsumes the other one
                False otherwize
        """

        for var, val in self.__body:
            if rule.get_condition(var) != val:
                return False

        return True

#--------------
# Operators
#--------------

    def __eq__(self, rule):
        """
        Compare equallity with other rule

        Args:
            rule: Rule

        Returns:
            Boolean
                True if the other rule is equal
                False otherwize
        """
        if isinstance(rule, Rule):
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

    # @warning: Expecting no condition already exist
    def add_condition(self, variable, value):
        """
        Add a condition to the body of the rule

        Args:
            variable: int
                id of a variable
            value: int
                id of a value of the variable
        """
        index = 0
        for var, val in self.__body: # Order w.r.t. variable id
            if var > variable:
                self.__body.insert(index, (variable,value))
                return
            index += 1

        self.__body.append((variable,value))


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
        Accessor to __Head_variable

        Returns:
            int
                the conclusion variable id
        """
        return self.__head_variable

    def get_head_value(self):
        """
        Accessor to __Head_value

        Returns:
            int
                the value of the conclusion
        """
        return self.__head_value

    def get_body(self):
        """
        Accessor to __body

        Returns:
            list of pair (int, int)
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
                variable id
        """
        self.__head_variable = variable

    def set_head_value(self, value):
        """
        Head value mutator method

        Args:
            value: int
                value id
        """
        self.__head_value = value
