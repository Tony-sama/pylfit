#-----------------------
# @author: Tony Ribeiro
# @created: 2023/10/31
# @updated: 2023/12/27
#
# @desc: Class LegacyAtom python source code file
#-----------------------
from ..objects.atom import Atom

class LegacyAtom(Atom):
    """
    Define a classic LFIT discrete atom
    """

    _VOID_VALUE = -1

#--------------
# Constructors
#--------------

    def __init__(self, variable, domain, value, state_position):
        """
        Constructor of an abstract atom

        Args:
            variable: string
                name of the variable the atom refer to
            domain: array of string
                the domain of the variable
            value: int
                id of the value in the domain
            state_position: int
                id of the position of the variable in an observed state
        """
        super().__init__(variable,domain,value)
        self.state_position = state_position

    def copy(self):
        """
        Copy method

        Returns:
            LegacyAtom
                A copy of the atom
        """
        return LegacyAtom(self.variable, self.domain, self.value, self.state_position)
    
    @staticmethod
    def from_string(string_format):
        tokens = string_format.split('(')

        variable = tokens[0].strip()
        value = tokens[1].split(")")[0].strip()

        return LegacyAtom(variable, {value}, value, -1)


#--------------
# Observers
#--------------
    
    def matches(self, state):
        """
        No value or simple equality.
        Args:
            state: list of string
        Returns:
            Boolean
        """
        return (self.value == LegacyAtom._VOID_VALUE) or (state[self.state_position] == self.value)# or state[self.state_position] == Atom._UNKNOWN_VALUE
    
    def partial_matches(self, state, unknown_values):
        """
        How atom compare w.r.t to unknown.
        Args:
            state: list of string
                a state that atom can check matching upon.
            unknown_values: list of string
                the value encoding unknown value.
        Returns:
            int
                one of Atom._FULL_MATCH/Atom._PARTIAL_MATCH/Atom._NO_MATCH
        """
        if self.value == LegacyAtom._VOID_VALUE or state[self.state_position] == self.value:
            return Atom._FULL_MATCH
        
        if state[self.state_position] in unknown_values:
            return Atom._PARTIAL_MATCH
        
        return Atom._NO_MATCH
    
    def subsumes(self, atom):
        """
        Does the atom value is more general than the other atom value.
        Args:
            atom: LegacyAtom
        Returns:
            boolean
        """
        # Same domain/state position and self is void or same value
        return ((self.domain == atom.domain) and (self.state_position == atom.state_position)) and ((self.value == LegacyAtom._VOID_VALUE) or (self.value == atom.value))

#--------------
# Operators
#--------------

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value,LegacyAtom):
            return (self.variable == __value.variable) and (self.domain == __value.domain) and (self.value == __value.value) and (self.state_position == __value.state_position)

        return False

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return self.to_string()

    def __hash__(self) -> int:
        return hash(str(self))

#--------------
# Methods
#--------------

    def void_atom(self):
        """
        Returns:
            LegacyAtom
                The void atom corresponding to this atom.
        """
        return LegacyAtom(self.variable, self.domain, LegacyAtom._VOID_VALUE, self.state_position)

    def least_specialization(self, state, unknown_values=[]):
        """
        No specialization possible when the atom has a value.
        All other value of the domain are candidate otherwise.

        Args:
            state: list of string
                a state that atom can check matching upon.
            unknown_values: list of string
                the value encoding unknown value.
        """
        output = []
        if self.value == LegacyAtom._VOID_VALUE:
            for i in self.domain:
                if i != state[self.state_position]:
                    new_atom = LegacyAtom(self.variable, self.domain, i, self.state_position)
                    output.append(new_atom)
        return output

    def to_string(self):
        """
        Void atom format or variable(value) otherwize.
        Returns:
            String
                a readable format of the atom.
        """
        output = ""
        if self.value == LegacyAtom._VOID_VALUE:
            output = "LegacyAtom(var:"+ str(self.variable) + ",dom:" + str(self.domain) + ",val:" + str(self.value) +",pos:" + str(self.state_position) + ")"
        else:
            output = str(self.variable) + "(" + str(self.value) + ")"
        return output
    

#--------------
# Properties
#--------------

    @property
    def state_position(self):
        return self._state_position
    
    @state_position.setter
    def state_position(self, value):
        self._state_position = value
