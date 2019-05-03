#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2019/04/25
# @updated: 2019/05/03
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
    __empty_set = True

    """ lower bound value: float """
    __min_value = None

    """ upper bound value: float """
    __max_value = None

    """ lower bound open/close: boolean """
    __min_included = None

    """ upper bound open/close: boolean """
    __max_included = None

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

            self.__empty_set = True
            self.__min_value = None
            self.__max_value = None
            self.__min_included = None
            self.__max_included = None
            return

        # check type of arguments

        # Invalid interval
        if min_value > max_value:
            raise ValueError("Continuum min value must be <= max value")

        # implicit empty set
        if min_value == max_value and (not min_included or not max_included):
            self.__empty_set = True
            self.__min_value = None
            self.__max_value = None
            self.__min_included = None
            self.__max_included = None
            return

        # Regular interval
        self.__empty_set = False
        self.__min_value = float(min_value)
        self.__max_value = float(max_value)
        self.__min_included = min_included
        self.__max_included = max_included

    def copy(self):
        """
        Copy methods

        Returns:
            Continuum
                A copy of the continuum
        """
        return Continuum(self.__min_value, self.__max_value, self.__min_included, self.__max_included)

    @staticmethod
    def random(min_value, max_value, min_size=0):
        """
        Generates randomly a valid continuum inside the given range of value.

        Args:
            min_value: float
                minimal lower bound value
            max_value: float
                maximal upper bound value
            min_size: float
                minimal size of the produced Continuum

        Returns:
            Continuum
                A random valid continuum
        """

        # Invalid interval
        if min_value > max_value:
            raise ValueError("Continuum min value must be <= max value")

        if min_size < 0 or min_size > (max_value - min_value):
            raise ValueError("expected 0 <= min_size < (max_value - min_value)")

        min = random.uniform(min_value,max_value-min_size)
        max = random.uniform(min+min_size, max_value)

        min_included = random.choice([True, False])
        max_included = random.choice([True, False])

        return Continuum(min, max, min_included, max_included)

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
        return self.__empty_set

    def min_included(self):
        """
        Check if the continuum lower bound value is included

        Returns:
            boolean
                True if the lower bound is inclusive,
                False otherwize
        """
        return self.__min_included

    def max_included(self):
        """
        Check if the continuum upper bound value is included

        Returns:
            boolean
                True if the upper bound is inclusive,
                False otherwize
        """
        return self.__max_included

    def size(self):
        """
        Gives the size of the interval: distance between lower/upper bound

        Returns:
            float
                the size of the continuum
        """
        if self.is_empty():
            return 0.0

        return abs(self.__max_value - self.__min_value)

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
            if element == self.get_min_value():
                return self.min_included()

            # upper bound
            if element == self.get_max_value():
                return self.max_included()

            # Inside interval
            return element > self.get_min_value() and element < self.get_max_value()

        # Argument is a continuum
        if isinstance(element, Continuum):
            # Empty set only includes itself
            if self.is_empty():
                return element.is_empty()

            # Every continuum contains the empty set
            if element.is_empty():
                return True

            min = self.get_min_value()
            max = self.get_max_value()
            min_other = element.get_min_value()
            max_other= element.get_max_value()

            # Lower bound over
            if min_other < min:
                return False

            # Lower bound value over
            if min_other == min:
                if not self.min_included() and element.min_included():
                    return False

            # upper bound over
            if max_other > max:
                return False

            # upper bound value over
            if max_other == max:
                if not self.max_included() and element.max_included():
                    return False

            return element.get_min_value() >= min and element.get_max_value() <= max

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

        min = self.get_min_value()
        max = self.get_max_value()
        min_other = continuum.get_min_value()
        max_other= continuum.get_max_value()

        # all before lower bound
        if max < min_other:
            return False

        # touching bounds
        if max == min_other:
            if not continuum.min_included() or not self.max_included():
                return False

        # all after upper bound
        if min > max_other:
            return False

        # touching bounds
        if min == max_other:
            if not continuum.max_included() or not self.min_included():
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

        if self.__empty_set != continuum.__empty_set:
            return False

        if self.__min_value != continuum.__min_value:
            return False

        if self.__min_included != continuum.__min_included:
            return False

        if self.__max_value != continuum.__max_value:
            return False

        if self.__max_included != continuum.__max_included:
            return False

        return True

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

        # Emptyset
        if self.__empty_set:
            return u"\u2205"

        out = ""

        if self.__min_included:
            out += "["
        else:
            out += "]"

        out += str(self.__min_value) + "," + str(self.__max_value)

        if self.__max_included:
            out += "]"
        else:
            out += "["

        return out

#--------------
# Accessors
#--------------

    def get_min_value(self):
        """
        lower bound accessor method

        Returns:
            float
                lower bound value
        """
        return self.__min_value

    def get_max_value(self):
        """
        upper bound accessor method

        Returns:
            float
                upper bound value
        """
        return self.__max_value

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

        if not self.is_empty() and self.get_max_value() < value:
            raise ValueError("New min value must be inferior to current max value")

        if self.is_empty():
            self.__empty_set = False
            self.__max_value = value
            self.__max_included = included

        self.__min_value = value
        self.__min_included = included

        # Empty set
        if self.__min_value == self.__max_value and (not self.__min_included or not self.__max_included):
            self.__empty_set = True
            self.__min_value = None
            self.__max_value = None
            self.__min_included = None
            self.__max_included = None

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

        if not self.is_empty() and self.get_min_value() > value:
            raise ValueError("New max value must be superior to current min value")

        if self.is_empty():
            self.__empty_set = False
            self.__min_value = value
            self.__min_included = included

        self.__max_value = value
        self.__max_included = included

        # Empty set
        if self.__min_value == self.__max_value and (not self.__min_included or not self.__max_included):
            self.__empty_set = True
            self.__min_value = None
            self.__max_value = None
            self.__min_included = None
            self.__max_included = None
