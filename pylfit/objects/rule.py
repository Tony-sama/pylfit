#-----------------------
# @author: Tony Ribeiro
# @created: 2019/03/20
# @updated: 2019/05/03
#
# @desc: Class Rule python source code file
#-----------------------

from ..utils import eprint

import random
import array

class Rule:
    """
    Define a discrete logic rule, conclusion and conditions are pairs (variable, value):
        - one conclusion
        - one conjonction of conditions
            - atmost one condition per variable, i.e. atmost one value is specified for a variable
        - a rule holds when all conditions are present in a system state, i.e. the rule matches the state
    """

    """ Conclusion variable id: int """
    #_head_variable = 0

    """ Conclusion value id: int """
    #_head_value = 0

    """ Conditions variables: list of int """
    #_body_variables = [0,2,5]

    """ Conditions values: vector of int """
    #_body_values = [1,0,3]

#--------------
# Constructors
#--------------

    def __init__(self, head_variable, head_value, nb_body_variables, body=None):
        """
        Constructor of a discrete logic rule

        Args:
            head_variable: int
                id of the head variable
            head_value: int
                id of the value of the head variable
            nb_body_variables: int
                number of variables that can have a condition in body
            body: list of tuple (int,int)
                list of conditions as pairs of variable id, value id
        """
        self.head_variable = head_variable
        self.head_value = head_value

        self._body_variables = []
        self._body_values = array.array('i', [-1 for i in range(nb_body_variables)]) #np.full((nb_body_variables,),-1,dtype=int) # -1 encode no value
        #eprint("body_values: ", self._body_values)

        if body != None:
            for (var, val) in body:
                self.add_condition(var,val)

    def copy(self):
        """
        Copy method

        Returns:
            Rule
                A copy of the rule
        """
        return Rule(self.head_variable, self.head_value, len(self._body_values), self.body)

    @staticmethod
    def from_string(string_format, features, targets):
        """
        Construct a Rule from a string format using features/targets to convert to domain ids.

        Returns:
            Rule
                The rule represented by the string w.r.t features/targets
        """
        #eprint(string_format)
        constraint = False
        tokens = string_format.split(":-")

        if len(tokens[0]) == 0:
            constraint = True
            head_variable_id = -1
            head_value_id = -1
            features = features+targets
        else:
            head_string = tokens[0].split('(')

            head_variable = head_string[0].strip()
            head_value = head_string[1].split(')')[0].strip()

            head_variable_id = [var for (var,vals) in targets].index(head_variable)
            head_value_id = targets[head_variable_id][1].index(head_value)

        body_string = tokens[1].split(",")

        # Empty rule
        if len(body_string) >= 1 and "(" not in body_string[0]:
            return Rule(head_variable_id, head_value_id, len(features))

        body = []

        for token in body_string:
            token = token.split("(")
            variable = token[0].strip()
            value = token[1].split(")")[0].strip()
            body.append((variable, value))

        body_encoded = []
        for variable,value in body:
            variable_id = [var for (var,vals) in features].index(variable)
            value_id = features[variable_id][1].index(value)
            body_encoded.append( (variable_id, value_id) )

        return Rule(head_variable_id, head_value_id, len(features), body_encoded)

        # TODO

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
        return len(self._body_variables)

    def to_string(self):
        """
        Convert the object to a readable string

        Returns:
            String
                a readable representation of the object
        """
        out = str(self._head_variable) + "=" + str(self._head_value)

        out += " :- "
        for var, val in self.body:
            out += str(var) + "=" + str(val) + ", "
        if len(self._body_variables) > 0:
            out = out[:-2]
        out += "."
        return out

    def logic_form(self, features, targets):
        """
        Convert the rule to a logic programming string format,
        using given variables/values labels.

        Args:
            features: list of (String, list of String)
                Labels of the features variables and their values
            targets: list of (String, list of String)
                Labels of the targets variables and their values

        Returns:
            String
                a logic programming string representation of the rule with original labels
        """

        # DBG
        #eprint(conclusion_values)
        out = ""

        constraint = self.head_variable < 0

        # Not a constraint
        if not constraint:
            var_label = targets[self.head_variable][0]
            val_label = targets[self.head_variable][1][self.head_value]
            out = str(var_label) + "(" + str(val_label) + ") "

        out += ":- "

        for var, val in self.body:
            if var < 0:
                raise ValueError("Variable id cannot be negative in rule body")

            if var >= len(features):
                if not constraint:
                    raise ValueError("Variable id in rule body out of bound of given features")
                elif var >= len(features)+len(targets):
                    raise ValueError("Variable id in constraint body out of bound of given targets")
                var_label = targets[var-len(features)][0]
                val_label = targets[var-len(features)][1][val]
            else:
                var_label = features[var][0]
                val_label = features[var][1][val]

            out += str(var_label) + "(" + str(val_label) + "), "

        # Delete last ", "
        if len(self._body_variables) > 0:
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
        #if self._body_values[variable] == -1:
        #    return None

        return self._body_values[variable]
        #for (var, val) in self._body:
        #    if (var == variable):
        #        return val
        #return None

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
        return self._body_values[variable] != -1
        #return self.get_condition(variable) is not None

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
        #for (var,val) in self._body:
        for var in self._body_variables:
            val = self._body_values[var]
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
        #for var, val in self._body:
        for var in self._body_variables:
            val = self._body_values[var]
            val_ = rule.get_condition(var)
            if val_ !=-1 and val_ != val:
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
        #if self.size() > rule.size():
        #    return False

        #for var, val in self._body:
        for var in self._body_variables:
            val = self._body_values[var]
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
            if (self.head_variable != rule.head_variable) or (self.head_value != rule.head_value):
                return False

            # Different size
            #if len(self.body) != len(rule.body):
            if self.size() != rule.size():
                return False

            # Check conditions
            #for c in self.body:
            for var in self._body_variables:
                val = self._body_values[var]

                if var >= len(rule._body_values):
                    return False

                if rule.get_condition(var) != val:
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
        # DBG
        #if self.has_condition(variable):
        #    self.remove_condition(variable)

        #index = 0
        #for var, val in self._body: # Order w.r.t. variable id
        #for var in self._body_variables:
        #    if var > variable:
        #        self._body_variables.insert(index, variable)
        #        self._body_values[variable] = value
        #        return
        #    index += 1

        self._body_variables.append(variable)
        self._body_values[variable] = value

    def remove_condition(self, variable):
        """
        Remove a condition from the body of the rule

        Args:
            variable: int
                id of a variable
        """
        index = 0
        #for (var, val) in self._body:
        for var in self._body_variables:
            if (var == variable):
                self._body_variables.pop(index)
                self._body_values[variable] = -1
                return
            index += 1

    def pop_condition(self):
        """
        Remove last condition from the body of the rule
        """
        var = self._body_variables.pop()
        self._body_values[var] = -1
#--------------
# Properties
#--------------

    # TODO: check types

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
        self._head_value = value

    @property
    def body(self):
        output = []
        for var in self._body_variables:
            output.append((var,self._body_values[var]))
        return sorted(output)
