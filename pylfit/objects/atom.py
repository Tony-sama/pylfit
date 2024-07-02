#-----------------------
# @author: Tony Ribeiro
# @created: 2023/10/30
# @updated: 2023/12/27
#
# @desc: Class Atom python source code file
#-----------------------

from ..utils import eprint

from abc import ABC, abstractmethod

class Atom(ABC):
    """
    Define an abstract atom
    """

    _UNKNOWN_VALUE = "?"

    # Partial match values
    _NO_MATCH = 0
    _PARTIAL_MATCH = 1
    _FULL_MATCH = 2

#--------------
# Constructors
#--------------

    def __init__(self, variable, domain, value):
        """
        Constructor of an abstract atom

        Args:
            variable: string
                name of the variable the atom refer to
            domain: any
                the domain of the variable
            value: any
                value of the atom, can be anything
        """
        self.variable = variable
        self.domain = domain
        self.value = value

    @abstractmethod
    def copy(self):
        pass

    @staticmethod
    @abstractmethod
    def from_string(string_format):
        pass

#--------------
# Observers
#--------------

    @abstractmethod
    def matches(self, state):
        pass

    @abstractmethod
    def partial_matches(self, state):
        pass

    @abstractmethod
    def subsumes(self, atom):
        pass

#--------------
# Operators
#--------------

#--------------
# Methods
#--------------

    @abstractmethod
    def void_atom(self):
        pass

    @abstractmethod
    def least_specialization(self, state, unknown_values=[]):
        pass

    @abstractmethod
    def to_string(self):
        pass
    

#--------------
# Properties
#--------------

    @property
    def variable(self):
        return self._variable

    @variable.setter
    def variable(self, value):
        self._variable = value
        
    @property
    def domain(self):
        return self._domain
    
    @domain.setter
    def domain(self, value):
        self._domain = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
