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
sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')
sys.path.insert(0, 'src/semantics')
sys.path.insert(0, 'src/evaluations')

from utils import eprint
from logicProgram import LogicProgram
from gula import GULA
from synchronizer import Synchronizer
from boolean_network_converter import BooleanNetworkConverter
from synchronous import Synchronous
from asynchronous import Asynchronous
from general import General


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
            semantics: Class
                Class of the semantics to be tested
            train_size: float in [0,1] or int
                Size of the training set in proportion (float in [0,1])
                or explicit (int)
    """

    # 0) Extract logic program
    #-----------------------
    P = benchmark
    #eprint(P.to_string())

    # 1) Generate transitions
    #-------------------------------------

    # Boolean network benchmarks only have rules for value 1, if none match next value is 0
    default = [[0] for v in P.get_targets()]
    eprint("Generating benchmark transitions...")
    full_transitions = semantics.transitions(P,default) #P.generate_all_transitions()
    #eprint(P.to_string())
    #eprint(semantics.states(P))
    #eprint(full_transitions)

    # 2) Prepare scores containers
    #---------------------------
    results_time = []
    results_common = []
    results_missing = []
    results_over = []
    results_precision = []

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

        # 3.2) Learn from training set
        #-------------------------

        # Define a timeout
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(TIME_OUT)
        try:
            start = time.time()
            model = algorithm.fit(train, P.get_features(), P.get_targets())
            #if train_size is None:
                #model = algorithm.fit(train, P.get_features(), P.get_targets())
            #else:
            #    model = algorithm.fit(train, P.get_features(), P.get_targets(), supported_only=True)
            end = time.time()
            results_time.append(round(end - start,3))
        except:
            return len(train), -1

        signal.alarm(0)

        # DBG
        #eprint(model)
        #eprint(model.logic_form())

        # 3.3) Evaluate model against originals rules
        #-----------------------------------------------

        # LUST special case
        #if type(model) == list:
        #    model = model[0]

        #common, missing, over = P.compare(model)

        #eprint(">>> Original:")
        #eprint(P.to_string())

        #eprint(">>> Learned:")
        #eprint(model.to_string())

        #eprint(">>> Logic Program comparaison:")
        #eprint(">>>> Common: "+str(len(common))+"/"+str(len(P.get_rules()))+"("+str(round(100 * len(common) / len(P.get_rules()),2))+"%)")
        #eprint(">>>> Missing: "+str(len(missing))+"/"+str(len(P.get_rules()))+"("+str(round(100 * len(missing) / len(P.get_rules()),2))+"%)")
        #eprint(">>>> Over: "+str(len(over))+"/"+str(len(P.get_rules()))+"("+str(round(100 * len(over) / len(model.get_rules()),2))+"%)")

        #over_head_zero = [r for r in over if r.get_head_value() == 0]
        #over = [r for r in over if r not in over_head_zero]

        # Collect scores
        #results_common.append(len(common))
        #results_missing.append(len(missing))
        #results_over.append(len(over))

        # Perfect case: evaluate over all transitions
        #if len(test) == 0:
        #    test = train

        # 3.4) Evaluate accuracy prediction over unseen states
        #-------------------------------------------------
        #test = set([ (tuple(s1),tuple(s2)) for s1,s2 in test])
        #pred = set([ (tuple(s1), tuple(s2)) for s1,s2_ in test for s2 in semantics.next(model,s1,default) ])

        #over_pred = pred - test
        #missing_pred = test - pred

        #eprint(pred)
        #eprint(test)
        #eprint(over_pred)
        #eprint(missing_pred)

        #precision = 1.0 #- (len(over_pred) / len(pred)) - (len(missing_pred) / len(test))

        #precision = round(LogicProgram.precision(test,pred),2)
        #precision = -1 # DBG not computed

        #eprint(">>> Prediction precision")
        #eprint(">>>> " + str(round(precision * 100,2)) + "%")

        #results_precision.append(precision)

    # 4) Average scores
    #-------------------
    run_time = sum(results_time) / run_tests
    common = sum(results_common) / run_tests
    missing = sum(results_missing) / run_tests
    over = sum(results_over) / run_tests
    precision = sum(results_precision) / run_tests

    eprint()
    eprint(">>> Run time: "+str(run_time)+"s")
    #eprint(">>> Logic Program comparaison:")
    #eprint(">>>> AVG Common: "+str(common)+"/"+str(len(P.get_rules()))+"("+str(round(100 * common / len(P.get_rules()),2))+"%)")
    #eprint(">>>> AVG Missing: "+str(missing)+"/"+str(len(P.get_rules()))+"("+str(round(100 * missing / len(P.get_rules()),2))+"%)")
    #eprint(">>>> AVG Over: "+str(over)+"/"+str(len(P.get_rules()))+"("+str(round(100 * over / len(model.get_rules()),2))+"%) not count head=0")

    #eprint(">>> Prediction precision")
    #eprint(">>>> AVG accuracy: " + str(round(precision * 100,2)) + "%")

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
    default = [[0] for v in P.get_targets()]
    eprint("Generating benchmark transitions...")
    full_transitions = semantics.transitions(P,default) #P.generate_all_transitions()
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

        # 3.2) Learn from training set
        #------------------------------------------

        # possibilities
        start = time.time()
        model = algorithm.fit(train, P.get_features(), P.get_targets(), supported_only=True)
        end = time.time()

        results_time.append(round(end - start,3))

        # impossibilities
        eprint("\nLearning anti-rules")
        anti_model = {}
        #negatives = np.array([tuple(s1)+tuple(s2) for s1,s2 in train])
        #negatives = negatives[np.lexsort(tuple([negatives[:,col] for col in reversed(range(0,len(P.get_features())))]))]

        processed_transitions = np.array([tuple(s1)+tuple(s2) for s1,s2 in train])
        processed_transitions = processed_transitions[np.lexsort(tuple([processed_transitions[:,col] for col in reversed(range(0,len(P.get_features())))]))]
        processed_transitions_ = []
        s1 = processed_transitions[0][:len(P.get_features())]
        S2 = []
        for row in processed_transitions:
            if not np.array_equal(row[:len(P.get_features())], s1): # New initial state
                #eprint("new state: ", s1)
                processed_transitions_.append((s1,S2))
                s1 = row[:len(P.get_features())]
                S2 = []

            #eprint("adding ", row[len(features):], " to ", s1)
            S2.append(row[len(P.get_features()):]) # Add new next state

        # Last state
        processed_transitions_.append((s1,S2))

        processed_transitions = processed_transitions_

        for var in range(len(P.get_targets())):
            for val in P.get_targets()[var][1]:
                targets = [(var, []) for var, vals in P.get_targets()]
                targets[var] = (P.get_targets()[var][0], [val])
                #positives = np.array([tuple(s1)+tuple(s2) for s1,s2 in train if s2[var] == val])

                # DBG
                neg, pos = GULA.interprete(processed_transitions, var, val, supported_only=True)

                eprint("\nStart learning of var=", var+1, "/", len(P.get_targets()), ", val=", val, "/", len(P.get_targets()[var][1]))
                #anti_model[(var,val)] = GULA.fit_var_val(P.get_features()+targets, -1, -1, negatives, positives)
                anti_model[(var,val)] = GULA.fit_var_val(P.get_features(), var, val, pos, neg)
        #eprint(anti_model)

        # 3.3) Compute rules weights from train matchings
        #--------------------------------------------------
        eprint("\nComputing rules weights")
        weighted_model = {}
        for var in range(len(P.get_targets())):
            for val in range(len(P.get_targets()[var][1])):
                weighted_model[(var,val)] = []
                for r in model.get_rules_of(var,val):
                    weight = 0
                    for s1 in train_init:
                        if r.matches(s1):
                            weight += 1
                    if weight > 0:
                        weighted_model[(var,val)].append((weight,r))

        eprint("Computing anti-rules weights")
        #eprint("Weighted model: ", weighted_model)
        weighted_anti_model = {}
        for var in range(len(P.get_targets())):
            for val in range(len(P.get_targets()[var][1])):
                weighted_anti_model[(var,val)] = []

        for var in range(len(P.get_targets())):
            for val in range(len(P.get_targets()[var][1])):
                for r in anti_model[(var,val)]:
                    #r_ = r.copy()
                    #r_.remove_condition(len(P.get_features())+var) # remove condition on target
                    weight = 0
                    for s1 in train_init:
                        if r.matches(s1):
                            weight += 1
                    if weight > 0:
                        weighted_anti_model[(var,val)].append((weight,r))

        #eprint("Weighted anti model: ", weighted_anti_model)

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
            for var in range(len(P.get_targets())):
                for val in range(len(P.get_targets()[var][1])):
                    occurs[(var,val)] = 0.0
                    for s2 in successors:
                        if s2[var] == val:
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
            for var in range(len(P.get_targets())):
                for val in range(len(P.get_targets()[var][1])):
                    # max rule weight
                    max_rule_weight = 0
                    best_rule = 0
                    for w,r in weighted_model[(var,val)]:
                        if w > max_rule_weight and r.matches(s1):
                            max_rule_weight = w
                            best_rule = r

                    #eprint("1: ", max_rule_weight)
                    #eprint("from: ", best_rule)

                    # max anti-rule weight
                    max_anti_rule_weight = 0
                    best_anti_rule = 0
                    for w,r in weighted_anti_model[(var,val)]:
                        r_ = r.copy()
                        r_.remove_condition(len(P.get_features())+var) # remove condition on target
                        if w > max_anti_rule_weight and r_.matches(s1):
                            max_anti_rule_weight = w
                            best_anti_rule = r

                    occurs[(var,val)] = round(0.5 + 0.5*(max_rule_weight - max_anti_rule_weight) / max(1,(max_rule_weight+max_anti_rule_weight)),3)

            prediction_set[s1] = occurs

        #eprint("Prediction set: ", prediction_set)

        eprint("\nComputing accuracy score")

        # compute average accuracy
        global_error = 0
        for s1, actual in test_set.items():
            state_error = 0
            for var in range(len(P.get_targets())):
                for val in range(len(P.get_targets()[var][1])):
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
    algorithm = GULA
    semantics_classes = [Synchronous, Asynchronous, General]
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # 1: Parameters
    #---------------

    if len(sys.argv) < 6 or (sys.argv[1] != "GULA" and sys.argv[1] != "Synchronizer"):
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

    # SYNCHRONIZER: asynchronous and general for non-determinism on BN
    #------------------------------------------------------------------
    if sys.argv[1] == "Synchronizer":
        algorithm = Synchronizer
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
        p = BooleanNetworkConverter.load_from_file(file_path)
        dmvlp_benchmarks.append( (len(p.get_features()), os.path.splitext(file_name)[0], p) )

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
            eprint(">> ", name, ": ", len(program.get_features()), " variables, ", len(program.get_rules()), " rules, ", pow(2,len(program.get_features())), " init states.")
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
                eprint(">> ", name, ": ", len(program.get_features()), " variables, ", len(program.get_rules()), " rules, ", train_size*100, "% training.")
                train_set_size, test_set_size, accuracy = evaluate_accuracy_on_bn_benchmark(algorithm, program, Synchronous, run_tests, train_size, mode)
                eprint("Learned from: ",train_set_size, "/", test_set_size, " train/test")
                print(", ", round(accuracy,3), end='')
            print()
