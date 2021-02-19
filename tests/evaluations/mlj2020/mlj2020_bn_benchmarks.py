#-----------------------
# @author: Tony Ribeiro
# @created: 2020/07/13
# @updated: 2020/07/13
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

from pylfit.utils import eprint
from pylfit.models import DMVLP
from pylfit.models import WDMVLP
from pylfit.algorithms import GULA, Synchronizer, PRIDE
from pylfit.preprocessing import dmvlp_from_boolean_network_file
from pylfit.semantics import Synchronous, Asynchronous, General
from pylfit.datasets import StateTransitionsDataset


# Constants
#------------
random.seed(0)
TIME_OUT = 1000


def handler(signum, frame):
    #print("Forever is over!")
    raise Exception("end of time")


def evaluate_on_bn_benchmark(algorithm, benchmark, semantics, run_tests, train_size=None):
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
    eprint("Generating benchmark transitions ...")
    full_transitions = [ (np.array(feature_state), np.array(["0" if x=="?" else "1" for x in target_state])) for feature_state in P.feature_states() for target_state in P.predict(feature_state, semantics) ]
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

        # DBG
        if run == 0:
            eprint(">>> Start Training on " + str(len(train)) + "/" + str(len(full_transitions)) + " transitions (" + str(round(100 * len(train) / len(full_transitions), 2)) +"%)")

        eprint(">>> run: " + str(run+1) + "/" + str(run_tests), end='')

        dataset = StateTransitionsDataset(train, P.features, P.targets)

        # 3.2) Learn from training set
        #-------------------------

        # Define a timeout
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(TIME_OUT)
        try:
            start = time.time()

            model = DMVLP(features=P.features, targets=P.targets)
            model.compile(algorithm=algorithm)
            model.fit(dataset)

            end = time.time()
            results_time.append(round(end - start,3))
        except:
            return len(train), -1

        signal.alarm(0)

    # 4) Average scores
    #-------------------
    run_time = sum(results_time) / run_tests

    eprint()
    eprint(">>> Run time: "+str(run_time)+"s")

    return len(train), run_time

def evaluate_accuracy_on_bn_benchmark(algorithm, benchmark, semantics, run_tests, train_size, mode):
    """
        Evaluate accuracy of an algorithm
        over a given benchmark with a given number/proporsion
        of training samples.

        Args:
            algorithm: Class
                Class of the algorithm to be tested
            benchmark: String
                Label of the benchmark to be tested
            semantics: Class
                Class of the semantics to be tested
            train_size: float in [0,1] or int
                Size of the training set in proportion (float in [0,1])
            mode: string
                "all_from_init_states": training contains all transitions from its initials states
                "random": training contains random transitions, 80%/20% train/test then train is reduced to train_size
    """

    # 0) Extract logic program
    #-----------------------
    P = benchmark
    #eprint(P.to_string())

    # 1) Generate transitions
    #-------------------------------------

    # Boolean network benchmarks only have rules for value 1, if none match next value is 0
    #default = [[0] for v in P.targets]
    eprint("Generating benchmark transitions...")
    full_transitions = [ (feature_state,["0" if x=="?" else "1" for x in target_state]) for feature_state in P.feature_states() for target_state in P.predict(feature_state, semantics) ]
    full_transitions_grouped = {tuple(s1) : set(tuple(s2_) for s1_,s2_ in full_transitions if s1 == s1_) for s1,s2 in full_transitions}
    #eprint("Transitions: ", full_transitions)
    #eprint("Grouped: ", full_transitions_grouped)

    #eprint(P.to_string())
    #eprint(semantics.states(P))
    #eprint(full_transitions)

    # 2) Prepare scores containers
    #---------------------------
    results_time = []
    results_accuracy = []

    # 3) Average over several tests
    #-----------------------------
    for run in range(run_tests):

        # 3.1 Split train/test sets on initial states
        #----------------------------------------------
        train_init = list(full_transitions_grouped.keys())
        random.shuffle(train_init)

        if mode == "all_from_init_states":
            train_end = max(1, int(train_size * len(train_init)))
        else:
            train_end = max(1, int(0.8 * len(train_init))) # 80%/20% train/test then train_size of train

        test_init = train_init[train_end:]
        train_init = train_init[:train_end]

        #eprint("train size: ", len(train_init))
        #eprint("end: ", train_end)
        #eprint("train_init: ", train_init)
        #eprint("test_init: ", test_init)

        # dbg
        #eprint(train_init[0])

        # extracts transitions of both set
        train = []
        for s1 in train_init:
            train.extend([(list(s1),list(s2)) for s2 in full_transitions_grouped[s1]])
        random.shuffle(train)

        if mode == "random":
            #eprint("before: ", round(len(train)/len(full_transitions),2))
            train_end = max(1, int(train_size * len(train)))
            train = train[:train_end]
            eprint("After random picks: ", round(len(train)/len(full_transitions),2))


        test = []
        for s1 in test_init:
            test.extend([(list(s1),list(s2)) for s2 in full_transitions_grouped[s1]])

        #eprint("train: ", train)
        #eprint("test: ", test)
        #exit()

        # DBG
        if run == 0:
            eprint(">>> Start Training on " + str(len(train)) + "/" + str(len(full_transitions)) + " transitions (" + str(round(100 * len(train) / len(full_transitions), 2)) +"%)")

        eprint(">>> run: " + str(run+1) + "/" + str(run_tests), end='')

        dataset = StateTransitionsDataset([ (np.array(s1), np.array(s2)) for (s1,s2) in train], P.features, P.targets)

        # 3.2) Learn from training set
        #------------------------------------------

        # possibilities
        start = time.time()
        model = WDMVLP(features=P.features, targets=P.targets)
        model.compile(algorithm=algorithm)
        model.fit(dataset)
        #model = algorithm.fit(train, P.features, P.targets, supported_only=True)
        end = time.time()

        results_time.append(round(end - start,3))

        # 3.4) Evaluate on accuracy of domain prediction on test set
        #------------------------------------------------------------
        # DBG
        if len(test) == 0:
            test = train

        test_grouped = {tuple(s1) : set(tuple(s2_) for s1_,s2_ in test if s1 == s1_) for s1,s2 in test}
        test_set = {}
        #eprint("test grouped: ", test_grouped)

        eprint("Computing test set")

        # expected output: kinda one-hot encoding of values occurences
        count = 0
        for s1, successors in test_grouped.items():
            count += 1
            eprint("\r",count,"/",len(test_grouped.items()), end='')
            occurs = {}
            for var in range(len(P.targets)):
                for val in range(len(P.targets[var][1])):
                    occurs[(var,val)] = 0.0
                    for s2 in successors:
                        if s2[var] == P.targets[var][1][val]:
                            occurs[(var,val)] = 1.0
            test_set[s1] = occurs

        #eprint("test set: ", test_set)

        eprint("\nComputing forcast set")

        # predictions
        prediction_set = {}
        count = 0
        for s1, successors in test_grouped.items():
            count += 1
            eprint("\r",count,"/",len(test_grouped.items()), end='')
            occurs = {}
            prediction = model.predict(s1)
            for var_id, (var,vals) in enumerate(P.targets):
                for val_id, val in enumerate(P.targets[var_id][1]):
                    occurs[(var_id,val_id)] = prediction[var][val][0]

            prediction_set[s1] = occurs

        #eprint("Prediction set: ", prediction_set)
        #exit()

        eprint("\nComputing accuracy score")

        # compute average accuracy
        global_error = 0
        for s1, actual in test_set.items():
            state_error = 0
            for var in range(len(P.targets)):
                for val in range(len(P.targets[var][1])):
                    forecast = prediction_set[s1]
                    state_error += abs(actual[(var,val)] - forecast[(var,val)])

            global_error += state_error / len(actual.items())

        global_error = global_error / len(test_set.items())

        #eprint(global_error)


        accuracy = 1.0 - global_error

        eprint("AVG accuracy: " + str(round(accuracy * 100,2)) + "%")

        results_accuracy.append(accuracy)

    # 4) Average scores
    #-------------------
    accuracy = sum(results_accuracy) / run_tests
    run_time = sum(results_time) / run_tests

    eprint()
    eprint(">>> Prediction precision")
    eprint(">>>> AVG accuracy: " + str(round(accuracy * 100,2)) + "%")
    eprint(">>>> AVG run time: " + str(round(run_time,3)) + "s")

    return len(train), len(test), accuracy

if __name__ == '__main__':
    # 0: Constants
    #--------------
    SCALABILITY_EXPERIEMENT = 0
    ACCURACY_EXPERIEMENT = 1

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
    semantics_classes = ["synchronous", "asynchronous", "general"]
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # 1: Parameters
    #---------------

    if len(sys.argv) < 6 or (sys.argv[1] != "GULA" and sys.argv[1] != "Synchronizer") and sys.argv[1] != "PRIDE":
        eprint("Please give the experiement to perform as parameter: GULA or Synchronizer and min_var, max_var, max_var_general, run_tests")
        exit()

    min_var = int(sys.argv[2])
    max_var = int(sys.argv[3])
    max_var_general = int(sys.argv[4])
    run_tests = int(sys.argv[5])
    experiement = SCALABILITY_EXPERIEMENT
    mode = "all_from_init_states"

    if len(sys.argv) > 6 and sys.argv[6] == "accuracy":
        experiement = ACCURACY_EXPERIEMENT

    if len(sys.argv) > 7 and sys.argv[7] == "random":
        mode = "random"

    if sys.argv[1] == "GULA":
        algorithm = "gula"

    if sys.argv[1] == "PRIDE":
        algorithm = "pride"

    # SYNCHRONIZER: asynchronous and general for non-determinism on BN
    #------------------------------------------------------------------
    if sys.argv[1] == "Synchronizer":
        algorithm = "synchronizer"
        #semantics_classes = [Asynchronous, General]
        #min_var = 0
        #max_var = 10

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
        train_sizes = [0.25,0.5,0.75]
        eprint("> Start benchmark scalability evaluation: Boolean Networks, all transitions")
        print("\nAVG over "+str(run_tests)+" runs of run time when learning from transitions of Boolean Network benchmarks from " ,min_var, " until ",max_var, " variables:")
        print("Benchmark & size & synchronous & asynchronous & general\\\\")

        for size, name, program in dmvlp_benchmarks:
            eprint(">> ", name, ": ", len(program.features), " variables, ", len(program.rules), " rules, ", pow(2,len(program.features)), " init states.")
            latex = str(name).replace("_","\_") + " & $" + str(size) + "$"
            for semantics in semantics_classes:
                if semantics == General and size > max_var_general:
                    latex += " & M.O."
                    continue

                latex += "&"
                for train_size in train_sizes:
                    #if semantics != General and train_size < 0.75:
                    #    latex += " T.O. /"
                    #    continue
                    nb_transitions, run_time = evaluate_on_bn_benchmark(algorithm, program, semantics, run_tests, train_size)
                    if run_time == -1:
                        latex += " T.O. /"
                    else:
                        latex += " $" + str(round(run_time,3)) + "$s /"

                nb_transitions, run_time = evaluate_on_bn_benchmark(algorithm, program, semantics, run_tests, None)
                if run_time == -1:
                    latex += " T.O. /"
                else:
                    latex += " $" + str(round(run_time,3)) + "$s / $"
                latex += str(nb_transitions) + "$"

            print(latex + "\\\\")

    # 4: Accuracy experiements
    #--------------------------
    if experiement == ACCURACY_EXPERIEMENT:
        eprint("> Start benchmark accuracy evaluation: Boolean Networks")
        print("\n# AVG over "+str(run_tests)+" runs of accuracy when learning from transitions of Boolean Network benchmarks transitions from " ,min_var, " until " ,max_var, " variables:")

        # 4.1: All transitions from train init states or Random transitions from train init states (80%/20% train/test then random XX% from train)
        #-----------------------------------------------
        if mode == "all_from_init_states":
            print("# 10% to 100% of all transitions as training, test on rest (for 100%, training = test)")
        else:
            print("# random 10% to 100% of the training transitions, with 80%/20% of total transitions as train/test set")
        print()
        print("benchmark, variables", end='')
        for i in train_sizes:
            print(", ", i, end='')
        print()
        for size, name, program in dmvlp_benchmarks:
            print(name, ",", size, end='')
            for train_size in train_sizes:
                eprint(">> ", name, ": ", len(program.features), " variables, ", len(program.rules), " rules, ", train_size*100, "% training.")
                train_set_size, test_set_size, accuracy = evaluate_accuracy_on_bn_benchmark(algorithm, program, "synchronous", run_tests, train_size, mode)
                eprint("Learned from: ",train_set_size, "/", test_set_size, " train/test")
                print(", ", round(accuracy,3), end='')
            print()
