#-----------------------
# @author: Tony Ribeiro
# @created: 2019/03/20
# @updated: 2019/03/29
#
# @desc: PyLFIT benchmarks evaluation script
#
#-----------------------

import time
import random
import numpy as np

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

import sys
sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')

from utils import eprint
from lfkt import LFkT
from logicProgram import LogicProgram

# Constants
#------------
run_tests = 1

def test_random_LP(algorithm, nb_variables, nb_values, max_body_size, delay=1, train_size=None):

    max_body_size = max(0, max_body_size)

    results_time = []
    results_common = []
    results_missing = []
    results_over = []
    results_precision = []

    for run in range(run_tests):

        variables = ["x"+str(i) for i in range(nb_variables)]
        values = []

        for var in range(len(variables)):
            values.append([val for val in range(nb_values)])

        min_body_size = 0

        p = LogicProgram.random(variables, values, min_body_size, max_body_size, delay)
        serie_size = delay + random.randint(delay, 10)
        time_series = p.generate_all_time_series(serie_size)

        #eprint(p.logic_form())

        random.shuffle(time_series)
        train = time_series
        test = []

        if train_size is not None:
            if isinstance(train_size, float): # percentage
                last_obs = max(int(train_size * len(time_series)),1)
            else: # exact number of transitions
                last_obs = train_size
            train = time_series[:last_obs]
            test = time_series[last_obs:]

        #eprint(train)

        if run == 0:
            eprint(">>> Start Training on ", len(train), "/", len(time_series), " time series of size ", len(time_series[0]), " (", round(100 * len(train) / len(time_series), 2), "%)")

        eprint("\r>>> run: ", run+1 ,"/", run_tests, end='')
        start = time.time()
        model = algorithm.fit(p.get_variables(), p.get_values(), time_series)
        end = time.time()
        results_time.append(round(end - start,3))

        common, missing, over = p.compare(model)

        #eprint(">>> Original:")
        #eprint(P.to_string())

        #eprint(">>> Learned:")
        #eprint(model.to_string())

        #eprint(">>> Logic Program comparaison:")
        #eprint(">>>> Common: "+str(len(common))+"/"+str(len(P.get_rules()))+"("+str(round(100 * len(common) / len(P.get_rules()),2))+"%)")
        #eprint(">>>> Missing: "+str(len(missing))+"/"+str(len(P.get_rules()))+"("+str(round(100 * len(missing) / len(P.get_rules()),2))+"%)")
        #eprint(">>>> Over: "+str(len(over))+"/"+str(len(P.get_rules()))+"("+str(round(100 * len(over) / len(model.get_rules()),2))+"%)")

        results_common.append(len(common))
        results_missing.append(len(missing))
        results_over.append(len(over))

        if len(test) == 0:
            test = train

        pred = [(s[:-1], model.next_state(s[:-1])) for s in test]
        test = [(s[:-1], s[-1]) for s in test]
        precision = round(LogicProgram.precision(test,pred),2)

        #eprint(test)
        #eprint(pred)

        #eprint(">>> Prediction precision")
        #eprint(">>>> " + str(round(precision * 100,2)) + "%")

        results_precision.append(precision)

    run_time = round(sum(results_time) / run_tests, 3)
    common = sum(results_common) / run_tests
    missing = sum(results_missing) / run_tests
    over = sum(results_over) / run_tests
    precision = sum(results_precision) / run_tests

    eprint()
    eprint(">>> Run time: "+str(run_time)+"s")
    eprint(">>> Logic Program comparaison:")
    eprint(">>>> AVG Common: "+str(common)+"/"+str(len(p.get_rules()))+"("+str(round(100 * common / len(p.get_rules()),2))+"%)")
    eprint(">>>> AVG Missing: "+str(missing)+"/"+str(len(p.get_rules()))+"("+str(round(100 * missing / len(p.get_rules()),2))+"%)")
    eprint(">>>> AVG Over: "+str(over)+"/"+str(len(p.get_rules()))+"("+str(round(100 * over / len(model.get_rules()),2))+"%)")

    eprint(">>> Prediction precision")
    eprint(">>>> AVG accuracy: " + str(round(precision * 100,2)) + "%")

    return round(precision * 100,2)

'''

'''
if __name__ == '__main__':

    algorithm = LFkT

    perfect_test = True
    partial_test = False
    small_dataset = False

    tests = 100

    # Random CLP
    #------------
    if perfect_test:
        eprint("> Start random CLP evaluation: perfect test")
        for i in range(tests):
            eprint(">> Test ", i, "/", tests, ":")
            var = random.randint(5,6)
            val = random.randint(2,2)
            delay = random.randint(1,3)
            body_size = int(var * delay / 3)
            eprint(">>> variables: ", var)
            eprint(">>> values: ", val)
            eprint(">>> delay: ", delay)
            eprint(">>> body size: ", body_size)
            precision = test_random_LP(algorithm, var, val, body_size, delay)

            if precision != 100:
                eprint("ERROR: precision is not perfect!")

    # Random CLP tests 100% - 10%
    #----------------------------

    if partial_test:
        eprint("> Start random CLP evaluation: partial test")

        accuracy_evolution = []

        for i in reversed(range(10,110,10)):

            eprint("\n> Random CLP test with " + str(i) + "% training transitions")
            train_size = i / 100.0
            accuracy_evolution.append(test_random_CLP(12,2,5,train_size))

        eprint(accuracy_evolution)


    # Random CLP tests 5 - 50 transitions
    #-------------------------------------

    if small_dataset:
        eprint("> Start random CLP evaluation: small dataset test")

        accuracy_evolution = []

        for i in reversed(range(5,55,5)):

            accuracy = []
            train_size = i

            eprint("\n> Random CLP test with " + str(train_size) + " training transitions")

            eprint(">> Random CLP: 12 variables, 2 values, 5 rules")
            accuracy.append(test_random_CLP(12,2,5,train_size))

            accuracy_evolution.append(accuracy)

        train_size = 50
        for i in range(len(accuracy_evolution)):
            eprint(str(train_size) + ": " + str(accuracy_evolution[i]))
            train_size -= 5
