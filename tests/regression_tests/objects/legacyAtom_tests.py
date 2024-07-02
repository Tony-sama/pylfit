#-----------------------
# @author: Tony Ribeiro
# @created: 2023/12/07
# @updated: 2023/12/07
#
# @desc: PyLFIT unit test script for LegacyAtom object
#-----------------------

import unittest
import random


import sys

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

import itertools
import numpy

from tests_generator import random_legacy_atom, random

from pylfit.utils import eprint
from pylfit.objects.legacyAtom import LegacyAtom

#random.seed(0)

class LegacyAtom_tests(unittest.TestCase):
    """
        Unit test of class Rule from rule.py
    """

    _nb_tests = 100
    
    _nb_variables = 20

    _nb_values = 10


    #------------------
    # Constructors
    #------------------

    def test_constructor(self):
        print(">> LegacyAtom.__init__(self, variable, domain, value, state_position)")
        for i in range(self._nb_tests):
            var = str(random.randint(0,self._nb_variables-1))
            dom = [str(i) for i in range(random.randint(2,self._nb_values-1))]
            val = random.choice(dom)
            pos = random.randint(0,self._nb_variables-1)
            r = LegacyAtom(var, dom, val, pos)
            self.assertEqual(r.variable, var)
            self.assertEqual(r.domain, dom)
            self.assertEqual(r.value, val)
            self.assertEqual(r.state_position, pos)

    def test_copy(self):
        print(">> LegacyAtom.copy(self)")
        for i in range(self._nb_tests):
            atom_1 = random_legacy_atom(self._nb_variables, self._nb_values)
            atom_2 = atom_1.copy()

            self.assertEqual(atom_1.variable, atom_2.variable)
            self.assertEqual(atom_1.domain, atom_2.domain)
            self.assertEqual(atom_1.value, atom_2.value)
            self.assertEqual(atom_1.state_position, atom_2.state_position)

    def test_from_string(self):
        print(">> LegacyAtom.from_string(string_format, domain, state_position)")
        for i in range(self._nb_tests):
            atom_1 = random_legacy_atom(self._nb_variables, self._nb_values)

            atom_2_str = atom_1.variable+"("+atom_1.value+")"
            atom_2 = LegacyAtom.from_string(atom_2_str)

            self.assertEqual(atom_1.variable, atom_2.variable)
            self.assertEqual(atom_2.domain, {atom_2.value})
            self.assertEqual(atom_1.value, atom_2.value)
            self.assertEqual(atom_2.state_position, -1)

    #--------------
    # Observers
    #--------------

    def test_matches(self):
        print(">> LegacyAtom.matches(self, state)")
        for i in range(self._nb_tests):
            atom = random_legacy_atom(self._nb_variables, self._nb_values)
            state = [random.randint(0,self._nb_values) for var in range(self._nb_variables+1)]

            if atom.matches(state):
                self.assertEqual(atom.value,state[atom.state_position])
            else:
                self.assertNotEqual(atom.value,state[atom.state_position])

    def test_subsumes(self):
        print(">> LegacyAtom.subsumes(self, atom)")
        for i in range(self._nb_tests):
            atom_1 = random_legacy_atom(self._nb_variables, self._nb_values)
            atom_2 = random_legacy_atom(self._nb_variables, self._nb_values)
            atom_3 = atom_1.void_atom()

            self.assertTrue(atom_1.subsumes(atom_1))
            self.assertTrue(atom_2.subsumes(atom_2))
            self.assertTrue(atom_3.subsumes(atom_1))
            self.assertFalse(atom_2.subsumes(atom_3)) # DBG: random never make a void atom
            self.assertFalse(atom_1.subsumes(atom_3)) # DBG: random never make a void atom
            

    #--------------
    # Operators
    #--------------

    def test___eq__(self):
        print(">> LegacyAtom.__eq__(self,__value)")
        for i in range(self._nb_tests):
            atom_1 = random_legacy_atom(self._nb_variables, self._nb_values)
            atom_2 = atom_1.copy()
            atom_2.variable = "-1"
            self.assertEqual(atom_1, atom_1)
            self.assertEqual(atom_2, atom_2)
            self.assertNotEqual(atom_1, atom_2)
            atom_2 = atom_1.copy()
            atom_2.value = -1
            self.assertNotEqual(atom_1, atom_2)
            atom_2 = atom_1.copy()
            atom_2.value = "-1"
            self.assertNotEqual(atom_1, atom_2)
            atom_2 = atom_1.copy()
            atom_2.domain = {"-1"}
            self.assertNotEqual(atom_1, atom_2)
            atom_2 = atom_1.copy()
            atom_2.state_position = -1
            self.assertNotEqual(atom_1, atom_2)
            atom_2 = "test"
            self.assertNotEqual(atom_1, atom_2)


    def test___str__(self):
        print(">> LegacyAtom.__str__(self)")
        for i in range(self._nb_tests):
            atom = random_legacy_atom(self._nb_variables, self._nb_values)
            self.assertEqual(atom.__str__(), atom.to_string())

    def test___repr__(self):
        print(">> LegacyAtom.__repr__(self)")
        for i in range(self._nb_tests):
            atom = random_legacy_atom(self._nb_variables, self._nb_values)
            self.assertEqual(atom.__repr__(), atom.to_string())

    def test___hash__(self):
        print(">> LegacyAtom.__hash__(self)")
        for i in range(self._nb_tests):
            atom = random_legacy_atom(self._nb_variables, self._nb_values)
            self.assertEqual(atom.__hash__(), hash(str(atom)))


    #--------------
    # Methods
    #--------------

    def test_void_atom(self):
        print(">> LegacyAtom.void_atom(self)")
        for i in range(self._nb_tests):
            atom = random_legacy_atom(self._nb_variables, self._nb_values)
            atom_void = atom.void_atom()

            self.assertEqual(atom.variable, atom_void.variable)
            self.assertEqual(atom.domain, atom_void.domain)
            self.assertEqual(atom.state_position, atom_void.state_position)
            self.assertEqual(atom_void.value, LegacyAtom._VOID_VALUE)

    def test_least_specialization(self):
        print(">> LegacyAtom.least_specialization(self)")
        for i in range(self._nb_tests):
            atom = random_legacy_atom(self._nb_variables, self._nb_values)
            state = [random.choice(atom.domain) for var in range(self._nb_variables+1)]
            least_spec = atom.least_specialization(state)
            
            self.assertEqual(least_spec,[])

            atom_void = atom.void_atom()
            least_spec = atom_void.least_specialization(state)

            # all value except the one in the state at variable position
            self.assertEqual(len(least_spec), len(atom_void.domain)-1)
            for a in least_spec:
                self.assertNotEqual(state[atom_void.state_position], a.value)
            

    def test_to_string(self):
        print(">> LegacyAtom.to_string(self)")
        for i in range(self._nb_tests):
            atom = random_legacy_atom(self._nb_variables, self._nb_values)
            output = atom.variable + "(" + atom.value + ")"
            self.assertEqual(atom.to_string(), output)
            
            atom = atom.void_atom()
            output = "LegacyAtom(var:"+ str(atom.variable) + ",dom:" + str(atom.domain) + ",val:" + str(atom.value) +",pos:" + str(atom.state_position) + ")"
            self.assertEqual(atom.to_string(), output)


'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
