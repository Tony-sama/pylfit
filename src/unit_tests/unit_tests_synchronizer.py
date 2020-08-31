#-----------------------
# @author: Tony Ribeiro
# @created: 2019/11/25
# @updated: 2019/11/25
#
# @desc: PyLFIT unit test script
#
#-----------------------

import sys
import unittest
import random
import os

sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')
sys.path.insert(0, 'src/semantics')

from utils import eprint
from synchronizer import Synchronizer
from gula import GULA
from rule import Rule
from logicProgram import LogicProgram
from synchronous import Synchronous

import itertools

random.seed(0)


class SynchronizerTest(unittest.TestCase):
    """
        Unit test of class Synchronizer from synchronizer.py
    """

    __nb_unit_test = 100

    __nb_features = 4

    __nb_targets = 3

    __nb_values = 3

    __body_size = 10

    #------------------
    # Test functions
    #------------------

    def test_fit(self):
        print(">> Synchronizer.fit(variables, values, transitions, conclusion_values, complete)")

        # No transitions
        p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)
        p_ = Synchronizer.fit([], p.get_features(), p.get_targets())
        self.assertEqual(p_.get_features(),p.get_features())
        self.assertEqual(p_.get_targets(),p.get_targets())
        rules = []
        for var in range(len(p.get_targets())):
            for val in range(len(p.get_targets()[var][1])):
                rules.append(Rule(var,val,len(p.get_features())))
        self.assertEqual(p_.get_rules(),rules)

        for unit_test in range(self.__nb_unit_test):
            # DBG
            eprint()
            eprint(unit_test,"/",self.__nb_unit_test)

            for arg in [True]:#, False]:
                if arg:
                    # DBG
                    #continue
                    eprint(">>> GULA version")
                else:
                    eprint(">>> PRIDE version")

                # Generate random program
                p = self.random_program(self.__nb_features, self.__nb_targets, self.__nb_values, self.__body_size)

                for full_random in [True,False]:

                    if full_random:
                        # Generate random transitions with program size
                        t = []
                        states = p.states()
                        states = random.sample(states, random.randint(1,len(states)))
                        for s1 in states:
                            nb_next = random.randint(1,5)
                            for i in range(0,nb_next):
                                s2 = [random.randint(0,len(p.get_targets()[var][1])-1) for var in range(0,len(p.get_targets()))]
                                if [s1,s2] not in t:
                                    t.append([s1,s2])
                    else:
                        # Generate transitions from the random program
                        t = Synchronous.transitions(p)


                    # DBG
                    #eprint("Transitions: ", len(t))
                    #eprint("Transitions: ", t)

                    p_ = Synchronizer.fit(t, p.get_features(), p.get_targets(), arg)
                    rules = p_.get_rules()

                    #eprint("Model: ", p_)

                    # Check prediction correspond to input
                    predictions = Synchronous.transitions(p_)
                    predictions = set([(tuple(s1),tuple(s2)) for s1,s2 in predictions])
                    t_set = set([(tuple(s1),tuple(s2)) for s1,s2 in t])
                    #eprint("t: ", t_set)
                    #eprint("pred: ", predictions)
                    self.assertEqual(t_set,predictions)

                    # Check correctness/completness/minimality of rules
                    for variable in range(len(p.get_targets())):
                        for value in range(len(p.get_targets()[variable][1])):
                            #eprint("var="+str(variable)+", val="+str(value))
                            #neg = GULA.interprete(t, variable, value)
                            pos = [s1 for s1,s2 in t if s2[variable] == value]
                            neg = [s1 for s1,s2 in t if s1 not in pos]

                            # Each positive is explained
                            for s in pos:
                                cover = False
                                for r in rules:
                                    if r.get_head_variable() == variable \
                                       and r.get_head_value() == value \
                                       and r.matches(s):
                                        cover = True
                                self.assertTrue(cover) # Atleast one rule cover the example

                            # No negative is covered
                            for s in neg:
                                cover = False
                                for r in rules:
                                    if r.get_head_variable() == variable \
                                       and r.get_head_value() == value \
                                       and r.matches(s):
                                        cover = True
                                self.assertFalse(cover) # no rule covers the example

                            # All rules are minimals
                            for r in rules:
                                if r.get_head_variable() == variable and r.get_head_value() == value:
                                    for (var,val) in r.get_body():
                                        r.remove_condition(var) # Try remove condition

                                        conflict = False
                                        for s in neg:
                                            if r.matches(s): # Cover a negative example
                                                conflict = True
                                                break

                                        # # DEBUG:
                                        if not conflict:
                                            eprint("not minimal "+r.to_string())
                                            eprint(neg)

                                        self.assertTrue(conflict)
                                        r.add_condition(var,val) # Cancel removal

                    # Check correctness/completness/minimality of constraints
                    constraints = p_.get_constraints()
                    #eprint()
                    #eprint("transitions:", t)
                    #eprint("P", p_.to_string())


                    # No observations are match by a constraint
                    for s1, s2 in t:
                        for c in constraints:
                            self.assertFalse(c.matches(list(s1)+list(s2)))

                    # All constraints are minimals
                    for c in constraints:
                            for (var,val) in c.get_body():
                                c.remove_condition(var) # Try remove condition

                                conflict = False
                                for s1, s2 in t:
                                    if c.matches(list(s1)+list(s2)): # Cover a negative example
                                        conflict = True
                                        break

                                # # DEBUG:
                                if not conflict:
                                    eprint("not minimal "+c.to_string())
                                    eprint(s1, s2)

                                self.assertTrue(conflict)
                                c.add_condition(var,val) # Cancel removal

                    # Only original transitions are produced from observed states
                    for s1, s2 in t:
                        next = Synchronous.next(p_,s1)
                        self.assertTrue(s2 in next)
                        for s in next:
                            self.assertTrue([s1,s2] in t)




    #------------------
    # Tool functions
    #------------------
    def random_rule(self, features, targets, body_size):
        head_var = random.randint(0,len(targets)-1)
        head_val = random.randint(0,len(targets[head_var][1])-1)
        body = []
        conditions = []

        for j in range(0, random.randint(0,body_size)):
            var = random.randint(0,len(features)-1)
            val = random.randint(0,len(features[var][1])-1)
            if var not in conditions:
                body.append( (var, val) )
                conditions.append(var)

        return  Rule(head_var,head_val,len(features),body)


    def random_constraint(self, features, targets, body_size):
        head_var = -1
        head_val = -1
        body = []
        conditions = []

        for j in range(0, random.randint(0,body_size)):
            var = random.randint(0,len(features)+len(targets)-1)
            if var < len(features):
                val = random.choice(range(0,len(features[var][1])))
            else:
                val = random.choice(range(0,len(targets[var-len(features)][1])))
            if var not in conditions:
                body.append( (var, val) )
                conditions.append(var)

        return  Rule(head_var,head_val,len(features)+len(targets),body)


    def random_program(self, nb_features, nb_targets, nb_values, body_size):
        features = [("x"+str(i), [val for val in range(0,random.randint(2,nb_values))]) for i in range(random.randint(1,nb_features))]
        targets = [("y"+str(i), [val for val in range(0,random.randint(2,nb_values))]) for i in range(random.randint(1,nb_targets))]
        rules = []
        constraints = []

        for j in range(random.randint(0,100)):
            r = self.random_rule(features, targets, body_size)
            rules.append(r)

        for j in range(random.randint(0,100)):
            r = self.random_constraint(features, targets, body_size)
            # Force constraint to have condition at t
            var = random.randint(len(features), len(features)+len(targets)-1)
            val = random.randint(0, len(targets[var-len(features)])-1)
            r.add_condition(var, val)
            constraints.append(r)

        return LogicProgram(features, targets, rules, constraints)


if __name__ == '__main__':
    """ Main """

    unittest.main()
