#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/25
# @updated: 2022/08/31
#
# @desc: PyLFIT unit test script
#
#-----------------------

import unittest
import random

from pylfit.utils import eprint
from pylfit.objects.continuum import Continuum

import sys
import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from tests_generator import random_Continuum

random.seed(0)

class ContinuumTest(unittest.TestCase):
    """
        Unit test of class Continuum from continuum.py
    """

    __nb_unit_test = 100

    """ must be < __max_value"""
    __min_value = -100.0

    """ must be > __min_value"""
    __max_value = 100.0


    #------------------
    # Constructors
    #------------------

    def test_constructor_empty(self):
        eprint(">> Continuum.__init__(self)")

        for i in range(self.__nb_unit_test):
            c = Continuum()

            self.assertTrue(c.is_empty())
            self.assertEqual(c.min_value, None)
            self.assertEqual(c.max_value, None)
            self.assertEqual(c.min_included, None)
            self.assertEqual(c.max_included, None)

    def test_constructor_full(self):
        eprint(">> Continuum.__init__(self, min_value=None, max_value=None, min_included=None, max_included=None)")

        for i in range(self.__nb_unit_test):

            # Valid continuum
            #-----------------
            min = random.uniform(self.__min_value, self.__max_value)
            max = random.uniform(min, self.__max_value)
            min_included = random.choice([True, False])
            max_included = random.choice([True, False])

            c = Continuum(min, max, min_included, max_included)

            self.assertFalse(c.is_empty())
            self.assertEqual(c.min_value, min)
            self.assertEqual(c.max_value, max)
            self.assertEqual(c.min_included, min_included)
            self.assertEqual(c.max_included, max_included)

            # Implicit emptyset
            #-------------------
            min = random.uniform(self.__min_value, self.__max_value)

            c = Continuum(min, min, False, False)

            self.assertTrue(c.is_empty())
            self.assertEqual(c.min_value, None)
            self.assertEqual(c.max_value, None)
            self.assertEqual(c.min_included, None)
            self.assertEqual(c.max_included, None)

            # Invalid Continuum
            #--------------------
            max = random.uniform(self.__min_value, min-0.001)

            self.assertRaises(ValueError, Continuum, min, max, min_included, max_included)

            # Invalid number of arguments
            #-------------------------------

            self.assertRaises(ValueError, Continuum, min)
            self.assertRaises(ValueError, Continuum, min, max)
            self.assertRaises(ValueError, Continuum, min, max, min_included)
            self.assertRaises(ValueError, Continuum, min, max, min_included, max_included)

    def test_copy(self):
        eprint(">> Continuum.copy(self)")

        for i in range(self.__nb_unit_test):

            # Emptyset
            c_ = Continuum()
            c = c_.copy()

            self.assertTrue(c.is_empty())
            self.assertEqual(c.min_value, None)
            self.assertEqual(c.max_value, None)
            self.assertEqual(c.min_included, None)
            self.assertEqual(c.max_included, None)

            # Non empty
            min = random.uniform(self.__min_value, self.__max_value)
            max = random.uniform(min, self.__max_value)
            min_included = random.choice([True, False])
            max_included = random.choice([True, False])

            c_ = Continuum(min, max, min_included, max_included)
            c = c_.copy()

            self.assertFalse(c.is_empty())
            self.assertEqual(c.min_value, min)
            self.assertEqual(c.max_value, max)
            self.assertEqual(c.min_included, min_included)
            self.assertEqual(c.max_included, max_included)

    #--------------
    # Observers
    #--------------

    def test_size(self):
        eprint(">> Continuum.size(self)")

        for i in range(self.__nb_unit_test):

            # empty set
            c = Continuum()
            self.assertEqual(c.size(), 0.0)

            # regular
            c = random_Continuum(self.__min_value, self.__max_value)

            if not c.is_empty():
                self.assertEqual(c.size(),c.max_value - c.min_value)

    def test_includes(self):
        eprint(">> Continuum.includes(self, element)")

        for i in range(self.__nb_unit_test):

            # bad argument type
            c = random_Continuum(self.__min_value, self.__max_value)

            self.assertRaises(TypeError, c.includes, "test")

            # float argument
            #----------------

            # empty set includes nothing
            c = Continuum()
            value = random.uniform(self.__min_value, self.__max_value)
            self.assertFalse(c.includes(value))

            c = random_Continuum(self.__min_value, self.__max_value)

            # Before min
            value = c.min_value
            while value == c.min_value:
                value = random.uniform(c.min_value-100.0, c.min_value)

            self.assertFalse(c.includes(value))

            # on min bound
            self.assertEqual(c.includes(c.min_value), c.min_included)

            # Inside
            value = c.min_value
            while value == c.min_value or value == c.max_value:
                value = random.uniform(c.min_value, c.max_value)

            self.assertTrue(c.includes(value))

            # on max bound
            self.assertEqual(c.includes(c.max_value), c.max_included)

            # after max bound
            value = c.max_value
            while value == c.max_value:
                value = random.uniform(c.max_value, c.max_value+100.0)

            self.assertFalse(c.includes(value))

            # int argument
            #--------------

            # empty set includes nothing
            c = Continuum()
            value = random.randint(int(self.__min_value), int(self.__max_value))
            self.assertFalse(c.includes(value))

            c = random_Continuum(self.__min_value, self.__max_value)

            while int(c.max_value) - int(c.min_value) <= 1:
                min = random.uniform(self.__min_value, self.__max_value)
                max = random.uniform(min, self.__max_value)

                c = random_Continuum(min, max)

            #eprint(c.to_string())

            # Before min
            value = random.randint(int(c.min_value-100), int(c.min_value)-1)

            self.assertFalse(c.includes(value))

            # on min bound
            self.assertEqual(c.includes(c.min_value), c.min_included)

            # Inside
            value = random.randint(int(c.min_value)+1, int(c.max_value)-1)

            #eprint(value)

            self.assertTrue(c.includes(value))

            # on max bound
            self.assertEqual(c.includes(c.max_value), c.max_included)

            # after max bound
            value = random.randint(int(c.max_value)+1, int(c.max_value+100))

            self.assertFalse(c.includes(value))

            # continuum argument
            #--------------------

            # 0) c is empty set
            c = Continuum()
            c_ = Continuum()
            self.assertTrue(c.includes(c_)) # empty set VS empty set

            c_ = random_Continuum(self.__min_value, self.__max_value)
            while c_.is_empty():
                c_ = random_Continuum(self.__min_value, self.__max_value)
            self.assertFalse(c.includes(c_)) # empty set VS non empty

            # 1) c is non empty
            c = random_Continuum(self.__min_value, self.__max_value)

            self.assertTrue(c.includes(Continuum())) # non empty VS empty set
            self.assertTrue(c.includes(c)) # includes itself

            # 1.1) Lower bound over
            c_ = random_Continuum(c.min_value, self.__max_value)
            while c_.is_empty():
                c_ = random_Continuum(c.min_value, self.__max_value)

            value = c.min_value
            while value == c.min_value:
                value = random.uniform(c.min_value-100, c.min_value)

            c_.set_lower_bound(value,random.choice([True,False]))
            self.assertFalse(c.includes(c_))

            # 1.2) on min bound
            c_ = random_Continuum(c.min_value, self.__max_value)
            while c_.is_empty():
                c_ = random_Continuum(c.min_value, self.__max_value)
            c_.set_lower_bound(c.min_value,random.choice([True,False]))

            if not c.min_included and c_.min_included: # one value over
                self.assertFalse(c.includes(c_))

            # 1.3) upper bound over
            c_ = random_Continuum(self.__min_value, c.max_value)
            while c_.is_empty():
                c_ = random_Continuum(self.__min_value, c.max_value)

            value = c.max_value
            while value == c.max_value:
                value = random.uniform(c.max_value, c.max_value+100)

            c_.set_upper_bound(value,random.choice([True,False]))

            self.assertFalse(c.includes(c_))

            # 1.4) on upper bound
            c_ = random_Continuum(self.__min_value, c.max_value)
            while c_.is_empty():
                c_ = random_Continuum(self.__min_value, c.max_value)
            c_.set_upper_bound(c.max_value,random.choice([True,False]))

            if not c.max_included and c_.max_included: # one value over
                self.assertFalse(c.includes(c_))

            # 1.5) inside
            min = c.min_value
            while min == c.min_value:
                min = random.uniform(c.min_value, c.max_value)
            max = c.max_value
            while max == c.max_value:
                max = random.uniform(min, c.max_value)
            c_ = Continuum(min, max, random.choice([True, False]), random.choice([True,False]))

            self.assertTrue(c.includes(c_))
            self.assertFalse(c_.includes(c))

    def test_intersects(self):
        eprint(">> Continuum.intersects(self, continuum)")

        for i in range(self.__nb_unit_test):
            c = random_Continuum(self.__min_value, self.__max_value)
            c_ = Continuum()

            # emptyset
            self.assertFalse(c.intersects(c_))
            self.assertFalse(c_.intersects(c))
            self.assertFalse(c_.intersects(c_))

            # stricly before
            c = random_Continuum(self.__min_value, self.__max_value)
            c_ = random_Continuum(c.min_value-100, c.min_value)
            self.assertFalse(c.intersects(c_))
            self.assertFalse(c_.intersects(c))

            # touching on lower bound
            c = random_Continuum(self.__min_value, self.__max_value)
            c_ = random_Continuum(c.min_value-100, c.min_value)
            c_.set_upper_bound(c.min_value,True)

            self.assertEqual(c.intersects(c_), c.min_included)
            self.assertEqual(c_.intersects(c), c.min_included)

            c_.set_upper_bound(c.min_value,False)

            self.assertFalse(c.intersects(c_))
            self.assertFalse(c_.intersects(c))

            # strictly after
            c = random_Continuum(self.__min_value, self.__max_value)
            c_ = random_Continuum(c.max_value, c.max_value+100)
            self.assertFalse(c.intersects(c_))
            self.assertFalse(c_.intersects(c))

            # touching on lower bound
            c = random_Continuum(self.__min_value, self.__max_value)
            c_ = random_Continuum(c.max_value, c.max_value+100)
            c_.set_lower_bound(c.max_value,True)

            self.assertEqual(c.intersects(c_), c.max_included)
            self.assertEqual(c_.intersects(c), c.max_included)

            c_.set_lower_bound(c.max_value,False)

            self.assertFalse(c.intersects(c_))
            self.assertFalse(c_.intersects(c))

            # same (not empty)
            c = random_Continuum(self.__min_value, self.__max_value)
            while c.is_empty():
                c = random_Continuum(self.__min_value, self.__max_value)
            self.assertTrue(c.includes(c))

            # smaller
            c_ = random_Continuum(c.min_value, c.max_value)
            while c_.min_value == c.min_value and c_.max_value == c.max_value:
                c_ = random_Continuum(c.min_value, c.max_value)

            self.assertTrue(c.intersects(c_))
            self.assertTrue(c_.intersects(c))

            # bigger
            c_ = random_Continuum(c.min_value-100, c.max_value+100)
            while c_.min_value >= c.min_value or c_.max_value <= c.max_value:
                c_ = random_Continuum(c.min_value-100, c.max_value+100)

            #eprint(c.to_string())
            #eprint(c_.to_string())
            self.assertTrue(c.intersects(c_))
            self.assertTrue(c_.intersects(c))

    def test__eq__(self):
        eprint(">> Continuum.__eq__(self, continuum)")

        for i in range(self.__nb_unit_test):

            # emptyset
            c = Continuum()

            self.assertTrue(Continuum() == Continuum())
            self.assertTrue(c == Continuum())
            self.assertTrue(c == c)

            self.assertFalse(Continuum() != Continuum())
            self.assertFalse(c != Continuum())
            self.assertFalse(c != c)

            c = random_Continuum(self.__min_value, self.__max_value)

            self.assertTrue(c == c)
            self.assertFalse(c != c)
            self.assertEqual(c == Continuum(), c.is_empty())

            c_ = random_Continuum(self.__min_value, self.__max_value)

            if c.is_empty() and c_.is_empty():
                self.assertTrue(c == c_)
                self.assertTrue(c != c_)

            if c.is_empty() != c_.is_empty():
                self.assertFalse(c == c_)
                self.assertTrue(c != c_)

            if c.min_value != c_.min_value:
                self.assertFalse(c == c_)
                self.assertTrue(c != c_)

            if c.max_value != c_.max_value:
                self.assertFalse(c == c_)
                self.assertTrue(c != c_)

            if c.min_included != c_.min_included:
                self.assertFalse(c == c_)
                self.assertTrue(c != c_)

            if c.max_included != c_.max_included:
                self.assertFalse(c == c_)
                self.assertTrue(c != c_)

            # exaustive modifications
            if not c.is_empty():
                c_ = c.copy()
                value = random.uniform(1,100)
                c_.set_lower_bound(c.min_value-value,True)
                self.assertFalse(c == c_)
                self.assertTrue(c != c_)
                c_.set_lower_bound(c.min_value-value,False)
                self.assertFalse(c == c_)
                self.assertTrue(c != c_)

                c_ = c.copy()
                c_.set_lower_bound(c.min_value,not c.min_included)
                self.assertFalse(c == c_)
                self.assertTrue(c != c_)

                c_ = c.copy()
                value = random.uniform(1,100)
                c_.set_upper_bound(c.min_value+value,True)
                self.assertFalse(c == c_)
                self.assertTrue(c != c_)
                c_.set_upper_bound(c.min_value+value,False)
                self.assertFalse(c == c_)
                self.assertTrue(c != c_)

                c_ = c.copy()
                c_.set_upper_bound(c.max_value,not c.max_included)
                self.assertFalse(c == c_)
                self.assertTrue(c != c_)

            # different type
            self.assertFalse(c == "test")
            self.assertFalse(c == 0)
            self.assertFalse(c == True)
            self.assertFalse(c == [])


    def test_to_string(self):
        eprint(">> Continuum.to_string(self)")

        for i in range(self.__nb_unit_test):
            c = Continuum()

            self.assertEqual(c.to_string(), u"\u2205")

            c = random_Continuum(self.__min_value, self.__max_value)

            if c.is_empty():
                self.assertEqual(c.to_string(), u"\u2205")

            out = ""

            if c.min_included:
                out += "["
            else:
                out += "]"

            out += str(c.min_value) + "," + str(c.max_value)

            if c.max_included:
                out += "]"
            else:
                out += "["

            self.assertEqual(c.to_string(), out)
            self.assertEqual(c.__str__(), out)
            self.assertEqual(c.__repr__(), out)

    def test_set_lower_bound(self):
        eprint(">> Continuum.set_lower_bound(self, value, included)")

        for i in range(self.__nb_unit_test):

            # Empty set
            c = Continuum()

            self.assertRaises(TypeError, c.set_lower_bound, "string", True)
            self.assertRaises(TypeError, c.set_lower_bound, "string", False)
            self.assertRaises(TypeError, c.set_lower_bound, 0.5, 10)

            value = random.uniform(self.__min_value, self.__max_value)

            # extend with exclusion gives empty set, mistake expected from user
            # or both min and max will be changed and constructor must be used
            self.assertRaises(ValueError, c.set_lower_bound, value, False)

            c.set_lower_bound(value, True)

            # Empty set to one value interval
            self.assertEqual(c, Continuum(value, value, True, True))

            # Regular continuum

            # over max value
            c = random_Continuum(self.__min_value, self.__max_value)
            value = random.uniform(c.max_value, self.__max_value)
            while value == c.max_value:
                value = random.uniform(c.max_value, self.__max_value)

            self.assertRaises(ValueError, c.set_lower_bound, value, True)
            self.assertRaises(ValueError, c.set_lower_bound, value, False)

            # on max value
            c = random_Continuum(self.__min_value, self.__max_value)
            c_old = c.copy()
            value = c.max_value
            if not c.max_included or not c.min_included:
                c.set_lower_bound(value, False)
                self.assertEqual(c, Continuum()) # continuum reduced to empty set
            else:
                c.set_lower_bound(value, True)
                self.assertEqual(c.min_value, value)
                self.assertEqual(c.min_included, True)
                self.assertEqual(c.min_value, c.max_value)

                self.assertEqual(c.max_value, c_old.max_value)
                self.assertEqual(c.max_included, c_old.max_included)

            # other valid value
            c = random_Continuum(self.__min_value, self.__max_value)
            c_old = c.copy()
            value = random.uniform(self.__min_value, c.max_value)
            while value == c.max_value:
                value = random.uniform(self.__min_value, c.max_value)

            c.set_lower_bound(value, True)

            self.assertEqual(c.min_value, value)
            self.assertEqual(c.min_included, True)

            c = random_Continuum(self.__min_value, self.__max_value)
            c_old = c.copy()
            value = random.uniform(self.__min_value, c.max_value)
            while value == c.max_value:
                value = random.uniform(self.__min_value, c.max_value)

            c.set_lower_bound(value, False)

            self.assertEqual(c.min_value, value)
            self.assertEqual(c.min_included, False)



    def test_set_upper_bound(self):
        eprint(">> Continuum.set_upper_bound(self, value, included)")

        for i in range(self.__nb_unit_test):
            # Empty set
            c = Continuum()

            self.assertRaises(TypeError, c.set_upper_bound, "string", True)
            self.assertRaises(TypeError, c.set_upper_bound, "string", False)
            self.assertRaises(TypeError, c.set_upper_bound, 0.5, 10)

            value = random.uniform(self.__min_value, self.__max_value)

            # extend with exclusion gives empty set, mistake expected from user
            # or both min and max will be changed and constructor must be used
            self.assertRaises(ValueError, c.set_upper_bound, value, False)

            c.set_upper_bound(value, True)

            # Empty set to one value interval
            self.assertEqual(c, Continuum(value, value, True, True))

            # Regular continuum

            # over min value
            c = random_Continuum(self.__min_value, self.__max_value)
            value = random.uniform(self.__min_value, c.min_value)
            while value == c.min_value:
                value = random.uniform(self.__min_value, c.min_value)

            self.assertRaises(ValueError, c.set_upper_bound, value, True)
            self.assertRaises(ValueError, c.set_upper_bound, value, False)

            # on min value
            c = random_Continuum(self.__min_value, self.__max_value)
            c_old = c.copy()
            value = c.min_value
            if not c.max_included or not c.min_included:
                c.set_upper_bound(value, False)
                self.assertEqual(c, Continuum()) # continuum reduced to empty set
            else:
                c.set_upper_bound(value, True)
                self.assertEqual(c.max_value, value)
                self.assertEqual(c.max_included, True)
                self.assertEqual(c.max_value, c.min_value)

                self.assertEqual(c.min_value, c_old.min_value)
                self.assertEqual(c.min_included, c_old.min_included)

            # other valid value
            c = random_Continuum(self.__min_value, self.__max_value)
            c_old = c.copy()
            value = random.uniform(c.min_value, self.__max_value)
            while value == c.min_value:
                value = random.uniform(c.min_value, self.__max_value)

            c.set_upper_bound(value, True)

            self.assertEqual(c.max_value, value)
            self.assertEqual(c.max_included, True)

            c = random_Continuum(self.__min_value, self.__max_value)
            c_old = c.copy()
            value = random.uniform(c.min_value, self.__max_value)
            while value == c.min_value:
                value = random.uniform(c.min_value, self.__max_value)

            c.set_upper_bound(value, False)

            self.assertEqual(c.max_value, value)
            self.assertEqual(c.max_included, False)

    #------------------
    # Tool functions
    #------------------


'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
