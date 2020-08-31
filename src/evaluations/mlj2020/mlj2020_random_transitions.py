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
import signal

import sys
sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')
sys.path.insert(0, 'src/semantics')

from utils import eprint
from gula import GULA
from synchronizer import Synchronizer
from logicProgram import LogicProgram
from synchronous import Synchronous
from asynchronous import Asynchronous
from general import General

random.seed(0)

# Constants
#------------
TIME_OUT = 1000


def handler(signum, frame):
    #print("Forever is over!")
    raise Exception("end of time")

def test_random_transitions(algorithm, nb_variables, nb_values, nb_transitions, run_tests):

    results_time = []
    results_precision = []

    for run in range(run_tests):

        features = [("x"+str(i), [val for val in range(nb_values)]) for i in range(nb_variables)]
        targets = [("y"+str(i), [val for val in range(nb_values)]) for i in range(0,nb_variables)]


        # totally random transitions
        train = [([random.randint(0,nb_values-1) for var in range(0,nb_variables)],[random.randint(0,nb_values-1) for var in range(0,nb_variables)]) for i in range(0,nb_transitions)]
        test = []

        #eprint(train)

        if run == 0:
            eprint(">>> Start Training on ", len(train), "/", len(train), " transitions")

        eprint("\r>>> run: ", run+1 ,"/", run_tests, end='')
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(TIME_OUT)
        try:
            start = time.time()
            model = algorithm.fit(train, features, targets)
            end = time.time()
            results_time.append(round(end - start,3))
        except:
            return -1

        signal.alarm(0)

    run_time = round(sum(results_time) / run_tests, 3)

    eprint()
    eprint(">>> Run time: "+str(run_time)+"s")

    return run_time

'''

'''
if __name__ == '__main__':

    run_tests = 1
    algorithm = GULA
    max_body_size = None
    #min_var = 1
    #max_var = 15

    nb_transitions_to_test = [10,100,1000,10000,100000,1000000]
    nb_var_to_test = [6,8,10,12,14,16,18,20]

    if len(sys.argv) < 2 or (sys.argv[1] != "GULA" and sys.argv[1] != "Synchronizer"):
        eprint("Please give the experiement to perform as parameter: GULA or Synchronizer, run_tests")
        exit()

    if sys.argv[1] == "Synchronizer":
        algorithm = Synchronizer

    run_tests = int(sys.argv[2])

    # 1) Random transitions, fixed number
    #-------------------------------------------
    eprint("> Start random transitions evaluation")
    run_times = {key:[] for key in nb_var_to_test}

    line = "nb_transitions/nb_variables"
    for nb_var in nb_var_to_test:
        line += "," + str(nb_var)
    print(line)

    for nb_transitions in nb_transitions_to_test:
        print(nb_transitions,end='')
        for nb_var in nb_var_to_test:
            var = nb_var
            val = 3
            eprint(">>> variables: ", var)
            eprint(">>> values: ", val)
            run_time = test_random_transitions(algorithm, var, val, nb_transitions, run_tests)

            if run_time == -1:
                break

            print(",", run_time, end='')
        print()
            #run_times[nb_var].append(run_time)

    eprint(run_times)

    #output = []
    #nb_transitions_id = 0
    #for i in range(0,len(nb_transitions_to_test)):
    #    line = str(nb_transitions_to_test[i])
    #    for nb_var in nb_var_to_test:
    #        if len(run_times[nb_var]) > i:
    #            line += "," + str(run_times[nb_var][i])
    #    output.append(line)


    #for line in output:
    #    print(line)
