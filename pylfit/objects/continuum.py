#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2019/04/25
# @updated: 2023/12/27
#
# @desc: Class Continuum python source code file
#-------------------------------------------------------------------------------

import random

class Continuum:
    """
    Define a continuum: a continuous set of values between two values
        - can be the emptyset or have
            - a lower bound, open or closed
            - an upper bound, open or closed
    """

    """ is the empty set: boolean """
    #_empty_set = True

    """ lower bound value: float """
    #_min_value = None

    """ upper bound value: float """
    #_max_value = None

    """ lower bound open/close: boolean """
    #_min_included = None

    """ upper bound open/close: boolean """
    #_max_included = None

#--------------
# Constructors
#--------------

    def __init__(self, min_value=None, max_value=None, min_included=None, max_included=None):
        """
        Constructor of a continuum, all arguments to None for the emptyset

        Args:
            min_value: float
                lower bound value
            max_value: float
                upper bound value
            min_included: boolean
                inclusive lower bound or not
            max_included: boolean
                inclusive upper bound or not
        """
        # empty set
        if min_value is None or max_value is None or min_included is None or max_included is None:

            # Missing arguments
            if min_value is not None or max_value is not None or min_included is not None or max_included is not None:
                raise ValueError("Incomplete constructor: either no argument or all arguments must be given")

            self._empty_set = True
            self._min_value = None
            self._max_value = None
            self._min_included = None
            self._max_included = None
            return

        # check type of arguments

        # Invalid interval
        if min_value > max_value:
            raise ValueError("Continuum min value must be <= max value")

        # implicit empty set
        if min_value == max_value and (not min_included or not max_included):
            self._empty_set = True
            self._min_value = None
            self._max_value = None
            self._min_included = None
            self._max_included = None
            return

        # Regular interval
        self._empty_set = False
        self._min_value = float(min_value)
        self._max_value = float(max_value)
        self._min_included = min_included
        self._max_included = max_included

    def copy(self):
        """
        Copy methods

        Returns:
            Continuum
                A copy of the continuum
        """
        return Continuum(self._min_value, self._max_value, self._min_included, self._max_included)

#--------------
# Observers
#--------------

    def is_empty(self):
        """
        Check if the continuum is the empty set

        Returns:
            boolean
                True if the continuum is the empty set,
                False otherwize
        """
        return self._empty_set

    def size(self):
        """
        Gives the size of the interval: distance between lower/upper bound

        Returns:
            float
                the size of the continuum
        """
        if self.is_empty():
            return 0.0

        return abs(self._max_value - self._min_value)

    def includes(self, element):
        """
        Check inclusion of the given value or continuum

        Args:
            element: float / continuum

        Returns:
            boolean
                True if the element is included in the continuum,
                False otherwize
        """
        # Argument is a unique value
        if isinstance(element, float) or isinstance(element, int):
            # Empty set includes no value
            if self.is_empty():
                return False

            # Lower bound
            if element == self.min_value:
                return self.min_included

            # upper bound
            if element == self.max_value:
                return self.max_included

            # Inside interval
            return element > self.min_value and element < self.max_value

        # Argument is a continuum
        if isinstance(element, Continuum):
            # Empty set only includes itself
            if self.is_empty():
                return element.is_empty()

            # Every continuum contains the empty set
            if element.is_empty():
                return True

            min = self.min_value
            max = self.max_value
            min_other = element.min_value
            max_other= element.max_value

            # Lower bound over
            if min_other < min:
                return False

            # Lower bound value over
            if min_other == min:
                if not self.min_included and element.min_included:
                    return False

            # upper bound over
            if max_other > max:
                return False

            # upper bound value over
            if max_other == max:
                if not self.max_included and element.max_included:
                    return False

            return element.min_value >= min and element.max_value <= max

        raise TypeError("argument must be either a float, a int or a Continuum")

    def intersects(self, continuum):
        """
        Check intersection with the given continuum

        Args:
            continuum: Continuum

        Returns:
            boolean
                True if the continuums shares atleast one value,
                False otherwize
        """

        # Nothing to intersect on
        if self.is_empty() or continuum.is_empty():
            return False

        min = self.min_value
        max = self.max_value
        min_other = continuum.min_value
        max_other= continuum.max_value

        # all before lower bound
        if max < min_other:
            return False

        # touching bounds
        if max == min_other:
            if not continuum.min_included or not self.max_included:
                return False

        # all after upper bound
        if min > max_other:
            return False

        # touching bounds
        if min == max_other:
            if not continuum.max_included or not self.min_included:
                return False

        # Intervals are intersectings
        return True

#--------------
# Operators
#--------------

    def __eq__(self, continuum):
        """
        equality operator

        Args:
            continuum: Continuum

        Returns:
            boolean
                True if both continuum are equal,
                False otherwize
        """

        if not isinstance(continuum, Continuum):
            return False

        if self._empty_set != continuum._empty_set:
            return False

        if self._min_value != continuum._min_value:
            return False

        if self._min_included != continuum._min_included:
            return False

        if self._max_value != continuum._max_value:
            return False

        if self._max_included != continuum._max_included:
            return False

        return True

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    def __hash__(self):
        return hash(str(self))

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

        # Emptyset
        if self._empty_set:
            return u"\u2205"

        out = ""

        if self._min_included:
            out += "["
        else:
            out += "]"

        out += str(self._min_value) + "," + str(self._max_value)

        if self._max_included:
            out += "]"
        else:
            out += "["

        return out

#--------------
# Accessors
#--------------

    @property
    def min_value(self):
        return self._min_value

    @property
    def max_value(self):
        return self._max_value

    @property
    def min_included(self):
        return self._min_included

    @property
    def max_included(self):
        return self._max_included

#--------------
# Mutators
#--------------

    def set_lower_bound(self, value, included):
        """
        lower bound mutator method

        Args:
            value: float
                value of the new lower bound
            included: boolean
                if the new lower bound is inclusive
        """

        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError("value must be a float or a int but ", type(value), " given")

        if not isinstance(included, bool):
            raise TypeError("included must be a boolean", type(included), " given")

        # Increasing empty set
        if self.is_empty() and not included:
            raise ValueError("Continuum is the empty set, thus the new bound must be inclusive to create a new continuum")

        if not self.is_empty() and self.max_value < value:
            raise ValueError("New min value must be inferior to current max value")

        if self.is_empty():
            self._empty_set = False
            self._max_value = value
            self._max_included = included

        self._min_value = value
        self._min_included = included

        # Empty set
        if self._min_value == self._max_value and (not self._min_included or not self._max_included):
            self._empty_set = True
            self._min_value = None
            self._max_value = None
            self._min_included = None
            self._max_included = None

    def set_upper_bound(self, value, included):
        """
        upper bound mutator method

        Args:
            value: float
                value of the new upper bound
            included: boolean
                if the new upper bound is inclusive
        """

        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError("value must be a float or a int but ", type(value), " given")

        if not isinstance(included, bool):
            raise TypeError("included must be a boolean", type(included), " given")

        # Increasing empty set
        if self.is_empty() and not included:
            raise ValueError("Continuum is the empty set, thus the new bound must be inclusive to create a new continuum")

        if not self.is_empty() and self.min_value > value:
            raise ValueError("New max value must be superior to current min value")

        if self.is_empty():
            self._empty_set = False
            self._min_value = value
            self._min_included = included

        self._max_value = value
        self._max_included = included

        # Empty set
        if self._min_value == self._max_value and (not self._min_included or not self._max_included):
            self._empty_set = True
            self._min_value = None
            self._max_value = None
            self._min_included = None
            self._max_included = None
