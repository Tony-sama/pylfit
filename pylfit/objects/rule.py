#-----------------------
# @author: Tony Ribeiro
# @created: 2019/03/20
# @updated: 2023/12/27
#
# @desc: Class Rule python source code file
#-----------------------

from ..utils import eprint
from ..objects.atom import Atom
from ..objects.legacyAtom import LegacyAtom

class Rule:
    """
    Define a logic rule, conclusion and conditions are atoms:
        - one conclusion
        - one conjonction of conditions
            - atmost one condition per variable, i.e. atmost one value is specified for a variable
        - a rule holds when all conditions are present in a system state, i.e. the rule matches the state
    """

    """ Conclusion atom: Atom """

    """ Conditions atoms: map of Atom """

    # Partial match values
    _NO_MATCH = 0
    _PARTIAL_MATCH = 1
    _FULL_MATCH = 2

#--------------
# Constructors
#--------------

    def __init__(self, head, body={}):
        """
        Constructor of a logic rule

        Args:
            head: Atom
                A logic atom
            body: dict of Atom
                A dictionary of logic atom
        """
        self.head = head
        self.body = body.copy()

    def copy(self):
        """
        Copy method

        Returns:
            Rule
                A copy of the rule
        """
        return Rule(self.head, self.body)

    @staticmethod
    def from_string(string_format):
        """
        Construct a Rule from a string format.

        Returns:
            Rule
                The rule represented by the string
        """
        #eprint(string_format)
        constraint = False
        tokens = string_format.strip().split(":-")

        if len(tokens[0]) == 0:
            #constraint = True
            head = None
            #features = features+targets
        else:
            head = LegacyAtom.from_string(tokens[0])

        body_string = tokens[1].split(",")

        # Empty rule
        if len(body_string) >= 1 and "(" not in body_string[0]:
            return Rule(head)

        body = {}

        for token in body_string:
            condition = LegacyAtom.from_string(token)
            body[condition.variable] = condition

        return Rule(head, body)

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
        return len(self.body)

    def to_string(self):
        """
        Convert the object to a readable string

        Returns:
            String
                a readable representation of the object
        """
        out = ""

        if self.head is not None:
            out += self.head.to_string() + " :- "
        else:
            out += ":- "
        variables = list(self.body.keys())
        variables.sort()
        for var in variables:
            out += self.body[var].to_string() + ", "
        if len(self.body) > 0:
            out = out[:-2]
        out += "."
        return out

    def get_condition(self, variable):
        """
        Accessor to the condition value over the given variable

        Args:
            variable: string
                a variable id
        Returns:
            atom
                The atom of the condition over the variable if it exists
            None
                if no condition exists on the given variable
        """
        if variable not in self.body:
            return None

        return self.body[variable]

    def has_condition(self, variable):
        """
        Observer to condition existence over the given variable

        Args:
            variable: string
                a variable id
        Returns:
            Bool
                True if a condition exists over the given variable
                False otherwize
        """
        return variable in self.body

    def matches(self, state):
        """
        Check if the conditions of the rules holds in the given state

        Args:
            state: list of any
                a state of the system

        Returns:
            Boolean
                True if all conditions holds in the given state
                False otherwize
        """
        for var in self.body:
            if not self.body[var].matches(state):
                return False
        return True
    
    def partial_matches(self, state, unknown_values):
        """
        Check if the conditions of the rules holds in the given state.
        Consider partial matching with unknown values.

        Args:
            state: list of any
                a state of the system

        Returns:
            int
                _NO_MATCH if a condition conflict with a state value
                _PARTIAL_MATCH if some condition have unknown value in the state
                _FULL_MATCH if all condition are present in the state
        """
        output = Rule._FULL_MATCH
        for var in self.body:
            result = self.body[var].partial_matches(state, unknown_values)

            if result == Atom._NO_MATCH:
                return Rule._NO_MATCH

            if result == Atom._PARTIAL_MATCH:
                output = Rule._PARTIAL_MATCH

            # Full match by default

        return output

    # TODO
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
        pass

    def subsumes(self, rule):
        """
        Check if the rule will match every states the other rule match.
        Only occurs when all conditions of the current rules appears in the other one.

        Args:
            rule: Rule

        Returns:
            Boolean
                True if the rule subsumes the other one
                False otherwize
        """
        if self.size() > rule.size():
            return False

        for var in self.body:
            if rule.get_condition(var) is None:
                return False
            if not self.body[var].subsumes(rule.get_condition(var)):
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
        return isinstance(rule, Rule) and self.head == rule.head and self.body == rule.body
    
    def __ne__(self, rule):
        return not (self == rule)

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    def __hash__(self):
        return hash(str(self))
    
    def __lt__(self, other):
        return self.subsumes(other)

#--------------
# Methods
#--------------

    # @warning: Expecting no condition already exist
    def add_condition(self, atom):
        """
        Add a condition to the body of the rule

        Args:
            Atom
                The atom to be added as condition
        """
        self.body[atom.variable] = atom

    def remove_condition(self, variable):
        """
        Remove a condition from the body of the rule

        Args:
            variable: string
                id of a variable
        """
        if variable in self.body:
            del self.body[variable]

    def least_specialization(self, state, features, unknown_values=[]):
        """
        Return a set of rules that matches same states as current rule beside given state.
        Rely on condition atom self least specialization correctness.

        Args:
            state: array of value
            features: dict of void atoms
            unknown_values: list of string
        """
        output = []
        for var in features:
            # Unknown value does not require spec
            #if state[features[var].state_position] == Atom._UNKNOWN_VALUE:
            #    continue
            # No condition on variable
            if not self.has_condition(var):
                new_atoms = features[var].least_specialization(state, unknown_values)
            else:
                new_atoms = self.body[var].least_specialization(state, unknown_values)
            for atom in new_atoms:
                new_rule = self.copy()
                new_rule.add_condition(atom) # don't need to remove it get replaced
                output.append(new_rule)

        return output

#--------------
# Properties
#--------------

    @property
    def head(self):
        return self._head

    @head.setter
    def head(self, value):
        self._head = value

    @property
    def body(self):
        return self._body

    @body.setter
    def body(self, value):
        self._body = value
