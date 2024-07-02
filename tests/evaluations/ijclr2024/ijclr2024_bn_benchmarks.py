#-----------------------
# @author: Tony Ribeiro
# @created: 2020/07/13
# @updated: 2021/03/05
#
# @desc: PyLFIT benchmarks evaluation script
#
#-----------------------

import time
import random
import numpy as np

import os
from fnmatch import fnmatch
import pathlib
import signal
import sys
import multiprocessing.pool
import numpy
import math

from pylfit.utils import eprint
from pylfit.models import DMVLP, CDMVLP, WDMVLP
from pylfit.algorithms import GULA, Synchronizer, PRIDE, BruteForce
from pylfit.preprocessing import dmvlp_from_boolean_network_file
from pylfit.semantics import Synchronous, Asynchronous, General
from pylfit.datasets import DiscreteStateTransitionsDataset
from pylfit.objects.legacyAtom import LegacyAtom


# Constants
#------------
random.seed(0)
MAX_UNKNOWN_RATIO = 0
TIME_OUT = 0

class TimeoutException(Exception):
    def __init__(self, *args, **kwargs):
        pass


def handler(signum, frame):
    #print("Forever is over!")
    raise TimeoutException()

def random_unknown_values_dataset(data, max_unknown):
    output = []
    for (i,j) in data:
        replace_count = random.randint(0, math.floor(max_unknown*len(i)))
        i_ = numpy.array(i)
        if replace_count > 0:
            i_.flat[numpy.random.choice(len(i), replace_count, replace=False)] = LegacyAtom._UNKNOWN_VALUE

        replace_count = random.randint(0, math.floor(max_unknown*len(j)))
        j_ = numpy.array(j)
        if replace_count > 0:
            j_.flat[numpy.random.choice(len(j), replace_count, replace=False)] = LegacyAtom._UNKNOWN_VALUE
        output.append((list(i_),list(j_)))
    return output


def evaluate_scalability_on_bn_benchmark(algorithm, benchmark, benchmark_name, semantics, run_tests, train_size=None, full_transitions=None):
    """
        Evaluate accuracy and explainability of an algorithm
        over a given benchmark with a given number/proporsion
        of training samples.

        Args:
            algorithm: Class
                Class of the algorithm to be tested
            benchmark: String
                Label of the benchmark to be tested
            semantics: String
                Semantics to be tested
            train_size: float in [0,1] or int
                Size of the training set in proportion (float in [0,1])
                or explicit (int)
    """

    # 0) Extract logic program
    #-----------------------
    P = benchmark
    #eprint(P)
    #eprint(semantics)

    # 1) Generate transitions
    #-------------------------------------

    # Boolean network benchmarks only have rules for value 1, if none match next value is 0
    if full_transitions is None:
        eprint("Generating benchmark transitions ...")
        full_transitions = [ (feature_state, ["0" if x=="?" else "1" for x in target_state]) for feature_state in benchmark.feature_states() for target_state in benchmark.predict([feature_state], semantics)[tuple(feature_state)] ]
    #eprint(full_transitions)

    # 2) Prepare scores containers
    #---------------------------
    results_time = []

    # 3) Average over several tests
    #-----------------------------
    for run in range(run_tests):

        # 3.1 Split train/test sets
        #-----------------------
        random.shuffle(full_transitions)
        train = full_transitions
        test = []

        # Complete, Proportion or explicit?
        if train_size is not None:
            if isinstance(train_size, float): # percentage
                last_obs = max(int(train_size * len(full_transitions)),1)
            else: # exact number of transitions
                last_obs = train_size
            train = full_transitions[:last_obs]
            test = full_transitions[last_obs:]

        # Mask training data with random unknown
        if MAX_UNKNOWN_RATIO > 0:
            train = random_unknown_values_dataset(train, MAX_UNKNOWN_RATIO)

        #eprint(train)

        # DBG
        if run == 0:
            eprint(">>> Start Training on " + str(len(train)) + "/" + str(len(full_transitions)) + " transitions (" + str(round(100 * len(train) / len(full_transitions), 2)) +"%)")

        eprint(">>>> run: " + str(run+1) + "/" + str(run_tests), end='')

        dataset = DiscreteStateTransitionsDataset(train, benchmark.features, benchmark.targets)


        # csv format of results
        if train_size != None:
            expected_train_size = train_size
        else:
            expected_train_size = 1.0
        real_train_size = round(len(train)/(len(full_transitions)),2)

        common_settings = \
        algorithm + "," +\
        semantics + "," +\
        benchmark_name + "," +\
        str(len(benchmark.features)) + "," +\
        str(len(full_transitions)) + "," +\
        "random_transitions" + "," +\
        str(expected_train_size) + "," +\
        str(real_train_size) + "," +\
        str(len(train))

        # 3.2) Learn from training set
        #-------------------------

        # Define a timeout
        def experiement():
            run_time = -2
            start = time.time()

            if algorithm in ["gula", "pride", "brute-force"]:
                model = WDMVLP(features=benchmark.features, targets=benchmark.targets)
            elif algorithm in ["synchronizer"]:
                model = CDMVLP(features=benchmark.features, targets=benchmark.targets)
            else:
                eprint("Error, algorithm not accepted: "+algorithm)
                exit()

            model.compile(algorithm=algorithm)
            model.fit(dataset)

            end = time.time()
            run_time = end - start
            results_time.append(run_time)

            return run_time

            #print(common_settings+","+str(run_time))
            #eprint(" "+str(round(run_time, 3))+"s")

        with multiprocessing.pool.ThreadPool() as pool:
            try:
                run_time = pool.apply_async(experiement).get(timeout=TIME_OUT)
                print(common_settings+","+str(run_time)+","+str(MAX_UNKNOWN_RATIO))
                eprint(" "+str(round(run_time, 3))+"s")
            except multiprocessing.TimeoutError:
                eprint(" TIME OUT")
                print(common_settings+","+"-1"+","+str(MAX_UNKNOWN_RATIO))
                return len(train), -1

    # 4) Average scores
    #-------------------
    avg_run_time = sum(results_time) / run_tests

    eprint(">> AVG Run time: "+str(round(avg_run_time,3))+"s")

    return len(train), avg_run_time

if __name__ == '__main__':
    # 0: Constants
    #--------------
    SCALABILITY_EXPERIEMENT = 0
    ACCURACY_EXPERIEMENT = 1
    EXPLANATION_EXPERIEMENT = 2

    # Number of tests for averaging scores
    run_tests = 1

    # Logic program form of Boolean network from biological litteratures
    pyboolnet_bnet_files_folder = "benchmarks/boolean_networks/pyboolnet/"
    dubrova_net_files_folder = "benchmarks/boolean_networks/boolenet/"

    # Maximal variables considered
    #min_var = 0
    #max_var = 23
    #max_var_general = 15 # general semantic limitation

    # Learning algorithm, choose from: LF1T / GULA / PRIDE
    algorithm = "gula"
    semantics_classes = ["synchronous"]#, "asynchronous", "general"]

    # 1: Parameters
    #---------------
    lfit_methods = ["gula", "brute-force"]
    baseline_methods = ["baseline"]
    experiements = ["scalability"]#, "accuracy", "explanation"]
    observations = ["all_from_init_states", "random_transitions"]

    if len(sys.argv) < 9 or (sys.argv[1] not in lfit_methods+baseline_methods) or (sys.argv[6] not in experiements) or (sys.argv[7] not in observations):
        eprint("Please give the experiement to perform as parameter: gula/pride/brute-force/synchronizer/baseline and min_var, max_var, max_var_general, run_tests, scalability/accuracy/explanation, all_from_init_states/random_transitions, time_out")
        exit()

    if sys.argv[1] in lfit_methods or sys.argv[1] in baseline_methods:
        algorithm = sys.argv[1]

    min_var = int(sys.argv[2])
    max_var = int(sys.argv[3])
    max_var_general = int(sys.argv[4])
    run_tests = int(sys.argv[5])
    experiement = SCALABILITY_EXPERIEMENT
    mode = "all_from_init_states"

    if sys.argv[6] == "scalability":
        experiement = SCALABILITY_EXPERIEMENT

    if sys.argv[6] == "accuracy":
        experiement = ACCURACY_EXPERIEMENT

    if sys.argv[6] == "explanation":
        experiement = EXPLANATION_EXPERIEMENT

    if sys.argv[7] == "all_from_init_states":
        mode = "all_from_init_states"

    if sys.argv[7] == "random_transitions":
        mode = "random_transitions"

    TIME_OUT = int(sys.argv[8])

    MAX_UNKNOWN_RATIO = float(sys.argv[9])


    # 2: benchmarks extraction
    #--------------------------
    dmvlp_benchmarks = []

    eprint("> Converting PyBoolnet bnet files to logic programs")

    # Get all pyboolnet files
    benchmarks_folders = [pyboolnet_bnet_files_folder, dubrova_net_files_folder]
    bnet_files = []

    for folder in benchmarks_folders:
        for path, subdirs, files in os.walk(folder):
            for name in files:
                if fnmatch(name, "*.bnet") or fnmatch(name, "*.net"):
                    bnet_files.append( (name, pathlib.PurePath(path, name)) )

    for file_name, file_path in bnet_files:
        model = dmvlp_from_boolean_network_file(file_path)
        dmvlp_benchmarks.append( (len(model.features), os.path.splitext(file_name)[0], model) )

    dmvlp_benchmarks = sorted(dmvlp_benchmarks)

    # Subset of benchmarks (min_var <= size <= max_var)
    if max_var > 0:
        dmvlp_benchmarks = [i for i in dmvlp_benchmarks if i[0] >= min_var and i[0] <= max_var]

    eprint("Considered Boolean Network:")
    for size,name,p in dmvlp_benchmarks:
        eprint(size, " variables: ", name)

    # 3: Scalability experiements
    #-----------------------------
    if experiement == SCALABILITY_EXPERIEMENT:
        train_sizes = [0.1,0.25,0.5,0.75,1.0]
        eprint("Start benchmark scalability evaluation: Boolean Networks, partial transitions with "+algorithm)
        eprint("\nAVG over "+str(run_tests)+" runs of run time when learning from transitions of Boolean Network benchmarks from " ,min_var, " until ",max_var, " variables:")
        #eprint("Benchmark & size & synchronous & asynchronous & general\\\\")

        final_csv_output = "method,semantics,benchmark_name,benchmark_size,transitions,mode,expected_train_percent,real_train_percent,train_size,run_time,max_unknown"
        print(final_csv_output)

        for size, name, program in dmvlp_benchmarks:
            eprint()
            eprint("> ", name, ": ", len(program.features), " variables, ", len(program.rules), " rules, ", pow(2,len(program.features)), " init states.")
            #latex = str(name).replace("_","\_") + " & $" + str(size) + "$"
            for semantics in semantics_classes:
                if semantics == "general" and size > max_var_general:
                    #latex += " & M.O."
                    continue
                eprint(">> Semantics: "+semantics)

                #latex += "&"
                #full_transitions = [ (np.array(feature_state), np.array(["0" if x=="?" else "1" for x in target_state])) for feature_state in program.feature_states() for target_state in program.predict([feature_state], semantics) ]
                default = [(var, ["0"]) for var,vals in program.targets]
                full_transitions = [ ([str(i) for i in feature_state], [str(i) for i in target_state]) for feature_state in program.feature_states() for target_state in program.predict([feature_state], semantics, default)[tuple(feature_state)] ]

                #eprint(full_transitions)

                for train_size in train_sizes:
                    if train_size >= 1.0:
                        nb_transitions, run_time = evaluate_scalability_on_bn_benchmark(algorithm, program, name, semantics, run_tests, None, full_transitions)
                        #if run_time == -1:
                        #    latex += " T.O. /"
                        #else:
                        #    latex += " $" + str(round(run_time,3)) + "$s / $"
                        #latex += str(nb_transitions) + "$"
                    else:
                        nb_transitions, run_time = evaluate_scalability_on_bn_benchmark(algorithm, program, name, semantics, run_tests, train_size, full_transitions)
                        #if run_time == -1:
                        #    latex += " T.O. /"
                        #else:
                        #    latex += " $" + str(round(run_time,3)) + "$s /"



            #print(latex + "\\\\")

    # 4: Accuracy experiements
    #--------------------------
    if experiement == ACCURACY_EXPERIEMENT:
        eprint("Out of the scope for now, only scalability is available!")

    # 5: Explanation accuracy experiements
    #--------------------------------------
    if experiement == EXPLANATION_EXPERIEMENT:
        eprint("Out of the scope for now, only scalability is available!")