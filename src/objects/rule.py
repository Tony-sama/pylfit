#-----------------------
# @author: Tony Ribeiro
# @created: 2019/03/20
# @updated: 2019/05/03
#
# @desc: Class Rule python source code file
#-----------------------

from utils import eprint
import random
import array

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

    """ Conditions variables: list of int """
    __body_variables = []

    """ Conditions values: vector of int """
    __body_values = []

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
        self.__head_variable = head_variable
        self.__head_value = head_value

        self.__body_variables = []
        self.__body_values = array.array('i', [-1 for i in range(nb_body_variables)]) #np.full((nb_body_variables,),-1,dtype=int) # -1 encode no value
        #eprint("body_values: ", self.__body_values)

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
        return Rule(self.__head_variable, self.__head_value, len(self.__body_values), self.get_body())

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


        return Rule(head_variable, head_value, len(variables), body)



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
        return len(self.__body_variables)

    def to_string(self):
        """
        Convert the object to a readable string

        Returns:
            String
                a readable representation of the object
        """
        out = str(self.__head_variable) + "=" + str(self.__head_value)

        out += " :- "
        for var, val in self.get_body():
            out += str(var) + "=" + str(val) + ", "
        if len(self.__body_variables) > 0:
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

        constraint = self.__head_variable < 0

        # Not a constraint
        if not constraint:
            var_label = targets[self.__head_variable][0]
            val_label = targets[self.__head_variable][1][self.__head_value]
            out = str(var_label) + "(" + str(val_label) + ") "

        out += ":- "

        for var, val in self.get_body():
            if var < 0:
                raise ValueError("Variable id cannot be negative in rule body")

            if var >= len(features):
                if not constraint:
                    raise ValueError("Variable id in rule body out of bound of given features")
                elif var >= len(features)+len(targets):
                    raise ValueError("Variable id in constraint body out of bound of given targets")

                var_label = targets[var][0]
                val_label = targets[var][1][val]
            else:
                var_label = features[var][0]
                val_label = features[var][1][val]

            out += str(var_label) + "(" + str(val_label) + "), "

        # Delete last ", "
        if len(self.__body_variables) > 0:
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
        #if self.__body_values[variable] == -1:
        #    return None

        return self.__body_values[variable]
        #for (var, val) in self.__body:
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
        return self.__body_values[variable] != -1
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
        #for (var,val) in self.__body:
        for var in self.__body_variables:
            val = self.__body_values[var]
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
        #for var, val in self.__body:
        for var in self.__body_variables:
            val = self.__body_values[var]
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

        #for var, val in self.__body:
        for var in self.__body_variables:
            val = self.__body_values[var]
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
            #if len(self.get_body()) != len(rule.get_body()):
            if self.size() != rule.size():
                return False

            # Check conditions
            #for c in self.get_body():
            for var in self.__body_variables:
                val = self.__body_values[var]

                if var >= len(rule.__body_values):
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
        #for var, val in self.__body: # Order w.r.t. variable id
        #for var in self.__body_variables:
        #    if var > variable:
        #        self.__body_variables.insert(index, variable)
        #        self.__body_values[variable] = value
        #        return
        #    index += 1

        self.__body_variables.append(variable)
        self.__body_values[variable] = value

    def remove_condition(self, variable):
        """
        Remove a condition from the body of the rule

        Args:
            variable: int
                id of a variable
        """
        index = 0
        #for (var, val) in self.__body:
        for var in self.__body_variables:
            if (var == variable):
                self.__body_variables.pop(index)
                self.__body_values[variable] = -1
                return
            index += 1

    def pop_condition(self):
        """
        Remove last condition from the body of the rule
        """
        var = self.__body_variables.pop()
        self.__body_values[var] = -1
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
        output = []
        for var in self.__body_variables:
            output.append((var,self.__body_values[var]))
        return sorted(output)

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
