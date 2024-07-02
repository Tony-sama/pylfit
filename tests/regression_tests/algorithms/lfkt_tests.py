#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/15
# @updated: 2019/05/02
#
# @desc: PyLFIT unit test script
#
#-----------------------

import unittest
import random
import itertools
import sys
import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from pylfit.utils import eprint
from pylfit.algorithms.lfkt import LFkT
from pylfit.models.dmvlp import DMVLP
from pylfit.objects.legacyAtom import LegacyAtom
from pylfit.objects.rule import Rule
from pylfit.semantics.synchronous import Synchronous

from tests_generator import random_symmetric_DiscreteStateTransitionsDataset

random.seed(0)


class LFkTTest(unittest.TestCase):
    """
        Unit test of class LFkT from lfkt.py
    """

    _nb_unit_test = 100

    _nb_transitions = 100

    _nb_features = 3

    _nb_targets = 3

    _nb_feature_values = 2

    _nb_target_values = 2

    _max_delay = 4

    _body_size = 10

    _tmp_file_path = "tmp/unit_test_lfkt.tmp"

    #------------------
    # Test functions
    #------------------

    def test_fit(self):
        print(">> LFkT.fit(variables, values, time_series)")

        # No transitions
        dataset = random_symmetric_DiscreteStateTransitionsDataset( \
                            nb_transitions=0, \
                            nb_variables=random.randint(1,self._nb_features), \
                            max_variable_values=self._nb_feature_values)

        rules = LFkT.fit([],dataset.features, dataset.targets)
        #print(rules)
        self.assertEqual(len(rules),len([(var,val) for var,vals in dataset.targets for val in vals]))

        for i in range(self._nb_unit_test):
            #eprint("\rTest ", i+1, "/", self._nb_unit_test, end='')

            # Generate transitions
            dataset = random_symmetric_DiscreteStateTransitionsDataset( \
                            nb_transitions=self._nb_transitions, \
                            nb_variables=random.randint(1,self._nb_features), \
                            max_variable_values=self._nb_feature_values)

            min_body_size = 0
            max_body_size = random.randint(min_body_size, len(dataset.features))
            delay_original = random.randint(1, self._max_delay)

            features = [[var[:-4],vals] for var,vals in dataset.features]
            targets = dataset.targets

            p = DMVLP(dataset.features, dataset.targets)
            p.compile(algorithm="gula")
            p.fit(dataset=dataset)

            #eprint("Generating series...")
            time_series = [[s] for s in p.feature_states()]

            #eprint(delay_original)
            #eprint(p)
            #eprint(p.feature_states())
            #eprint(time_series)
            #exit()

            time_serie_size = delay_original + 2

            for serie in time_series:
                #eprint(">",serie)
                while len(serie) < time_serie_size:
                    serie_end = serie[-delay_original:]
                    serie_end = list(itertools.chain.from_iterable(serie_end))
                    #eprint(serie_end)
                    serie.append(list(Synchronous.next(serie_end, targets, p.rules).items())[0][0])
                    #eprint(serie)

            #eprint(p.logic_form())
            #for s in time_series:
            #    eprint(s)

            rules = LFkT.fit(time_series, features, targets)
            #eprint()
            #eprint(features)
            #eprint(targets)

            #eprint(p_.logic_form())

            for var_id, (var,vals) in enumerate(targets):
                for val in vals:
                    #eprint("var="+str(variable)+", val="+str(value))
                    head = LegacyAtom(var, set(vals), val, var_id)
                    pos, neg, delay = LFkT.interprete(time_series, head)

                    #eprint("pos: ", pos)

                    # Each positive is explained
                    for s in pos:
                        cover = False
                        for r in rules:
                            if r.head.variable == var and r.head.value == val and r.matches(s):
                                cover = True
                                break
                        if not cover:
                            print(features)
                            eprint(s)
                            eprint(rules)
                        self.assertTrue(cover) # One rule cover the example

                    #eprint("neg: ", neg)

                    # No negative is covered
                    for s in neg:
                        cover = False
                        for r in rules:
                            if r.head.variable == var and r.head.value == val and r.matches(s):
                                cover = True
                                print(s)
                                print(r)
                                break
                        self.assertFalse(cover) # no rule covers the example

                    # All rules are minimals
                    for r in rules:
                        if r.head.variable == var and r.head.value == val:
                            for var_ in r.body:
                                r_ = r.copy()
                                r_.remove_condition(var_) # Try remove condition

                                conflict = False
                                for s in neg:
                                    if r_.matches(s): # Cover a negative example
                                        conflict = True
                                        break

                                # # DEBUG:
                                if not conflict:
                                    eprint("not minimal "+r.to_string())
                                    eprint(neg)

                                self.assertTrue(conflict)
        #eprint()

    def test_interprete(self):
        print(">> LFkT.interprete(transitions, variable, value)")

        for i in range(self._nb_unit_test):
            #eprint("Start test ", i, "/", self._nb_unit_test)
            # Generate transitions
            dataset = random_symmetric_DiscreteStateTransitionsDataset( \
                            nb_transitions=self._nb_transitions, \
                            nb_variables=random.randint(1,self._nb_features), \
                            max_variable_values=self._nb_feature_values)

            min_body_size = 0
            max_body_size = random.randint(min_body_size, len(dataset.features))
            delay_original = random.randint(1, self._max_delay)

            features = []
            targets = dataset.targets

            for d in range(0, delay_original):
                features += [(var+"_t-"+str(d+1), vals) for var,vals in dataset.features]

            p = DMVLP(dataset.features, dataset.targets)
            p.compile(algorithm="gula")
            p.fit(dataset=dataset)

            #eprint("Generating series...")
            time_series = [[s] for s in p.feature_states()]

            #eprint(delay_original)
            #eprint(p)
            #eprint(p.feature_states())
            #eprint(time_series)
            #exit()

            time_serie_size = delay_original + 2

            for serie in time_series:
                #eprint(">",serie)
                while len(serie) < time_serie_size:
                    serie_end = serie[-delay_original:]
                    serie_end = list(itertools.chain.from_iterable(serie_end))
                    #eprint(serie_end)
                    serie.append(list(Synchronous.next(serie_end, targets, p.rules).items())[0][0])
                    #eprint(serie)

            var = random.randint(0, len(targets)-1)
            val = random.randint(0, len(targets[var][1])-1)

            #eprint("interpreting...")
            head = LegacyAtom(targets[var][0], set(targets[var][1]), targets[var][1][val], var)
            pos, neg, delay = LFkT.interprete(time_series, head)

            # DBG
            #eprint("variables: ", variables)
            #eprint("values", values)
            #eprint("delay: ", delay_original)
            #eprint(p.logic_form())
            #eprint(time_series)
            #eprint("var: ", var)
            #eprint("val: ", val)
            #eprint("pos: ", pos)
            #eprint("neg: ",neg)
            #eprint("delay detected: ", delay)

            # All pos are valid
            for s in pos:
                for serie in time_series:
                    for id in range(len(serie)-delay):
                        s1 = serie[id:id+delay].copy()
                        #s1.reverse()
                        s1 = [y for x in s1 for y in x]
                        #eprint(s1)
                        #eprint(s)
                        s2 = serie[id+delay]
                        if s1 == s:
                            self.assertEqual(s2[var], targets[var][1][val])
                            break
            # All neg are valid
            for s in neg:
                for serie in time_series:
                    for id in range(len(serie)-delay):
                        s1 = serie[id:id+delay].copy()
                        #s1.reverse()
                        s1 = [y for x in s1 for y in x]
                        s2 = serie[id+delay]
                        if s1 == s:
                            self.assertTrue(s2[var] != targets[var][1][val])
                            break

            # All transitions are interpreted
            #eprint("var/val: ", var, "/", val)
            #eprint("delay: ", delay)
            #eprint("Time serie: ", time_series)
            for serie in time_series:
                #eprint("checking: ", serie)
                for id in range(delay, len(serie)):
                    s1 = serie[id-delay:id].copy()
                    #s1.reverse()
                    s1 = [y for x in s1 for y in x]
                    s2 = serie[id]
                    #eprint("s1: ", s1, ", s2: ", s2)
                    #eprint("pos: ", pos)
                    #eprint("neg: ", neg)
                    if s2[var] == targets[var][1][val]:
                        self.assertTrue(s1 in pos)
                        self.assertFalse(s1 in neg)
                    else:
                        self.assertFalse(s1 in pos)
                        self.assertTrue(s1 in neg)

            # delay valid
            global_delay = 1
            for serie_1 in time_series:
                for id_state_1 in range(len(serie_1)-1):
                    state_1 = serie_1[id_state_1]
                    next_1 = serie_1[id_state_1+1]
                    # search duplicate with different future
                    for serie_2 in time_series:
                        for id_state_2 in range(len(serie_2)-1):
                            state_2 = serie_2[id_state_2]
                            next_2 = serie_2[id_state_2+1]

                            # Non-determinism detected
                            if state_1 == state_2 and next_1[var] != next_2[var]:
                                local_delay = 2
                                id_1 = id_state_1
                                id_2 = id_state_2
                                while id_1 > 0 and id_2 > 0:
                                    previous_1 = serie_1[id_1-1]
                                    previous_2 = serie_2[id_2-1]
                                    if previous_1 != previous_2:
                                        break
                                    local_delay += 1
                                    id_1 -= 1
                                    id_2 -= 1
                                global_delay = max(global_delay, local_delay)
                                self.assertTrue(local_delay <= delay)
            self.assertEqual(delay, global_delay)

            #eprint("FINISHED ", i, "/", self._nb_unit_test)

    #------------------
    # Tool functions
    #------------------


if __name__ == '__main__':
    """ Main """

    unittest.main()
