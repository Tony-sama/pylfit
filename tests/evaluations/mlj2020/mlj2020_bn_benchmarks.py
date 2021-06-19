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

from pylfit.utils import eprint
from pylfit.models import DMVLP, CDMVLP, WDMVLP
from pylfit.algorithms import GULA, Synchronizer, PRIDE, BruteForce
from pylfit.preprocessing import dmvlp_from_boolean_network_file
from pylfit.semantics import Synchronous, Asynchronous, General
from pylfit.datasets import StateTransitionsDataset
from pylfit.postprocessing import accuracy_score, accuracy_score_from_predictions, explanation_score, explanation_score_from_predictions
from pylfit.objects import Rule


# Constants
#------------
random.seed(0)
#TIME_OUT = 1

class TimeoutException(Exception):
    def __init__(self, *args, **kwargs):
        pass


def handler(signum, frame):
    #print("Forever is over!")
    raise TimeoutException()


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
        full_transitions = [ (np.array(feature_state), np.array(["0" if x=="?" else "1" for x in target_state])) for feature_state in benchmark.feature_states() for target_state in benchmark.predict([feature_state], semantics) ]
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

        eprint(">>>> run: " + str(run+1) + "/" + str(run_tests), end='')

        dataset = StateTransitionsDataset(train, benchmark.features, benchmark.targets)


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
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(TIME_OUT)
        run_time = -2
        try:
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

            signal.alarm(0)
            end = time.time()
            run_time = end - start
            results_time.append(run_time)
        except TimeoutException:
            signal.alarm(0)
            eprint(" TIME OUT")
            print(common_settings+","+"-1")
            return len(train), -1

        #signal.alarm(0)


        print(common_settings+","+str(run_time))
        eprint(" "+str(round(run_time, 3))+"s")

    # 4) Average scores
    #-------------------
    avg_run_time = sum(results_time) / run_tests

    eprint(">> AVG Run time: "+str(round(avg_run_time,3))+"s")

    return len(train), avg_run_time

def evaluate_accuracy_on_bn_benchmark(algorithm, benchmark, semantics, run_tests, train_size, mode, benchmark_name, full_transitions=None):
    """
        Evaluate accuracy of an algorithm
        over a given benchmark with a given number/proporsion
        of training samples.

        Args:
            algorithm: Class
                Class of the algorithm to be tested
            benchmark: DMVLP
                benchmark model to be tested
            semantics: Class
                Class of the semantics to be tested
            train_size: float in [0,1] or int
                Size of the training set in proportion (float in [0,1])
            mode: string
                "all_from_init_states": training contains all transitions from its initials states
                "random": training contains random transitions, 80%/20% train/test then train is reduced to train_size
            benchmark_name: string
                for csv output.
        Returns:
        train_set_size: int
        test_set_size: int
        accuracy: float
            Average accuracy score.
        csv_output: String
            csv string format of all tests run statistiques.
    """
    csv_output = ""

    # 0) Extract logic program
    #-----------------------
    #eprint(benchmark.to_string())

    # 1) Generate transitions
    #-------------------------------------

    # Boolean network benchmarks only have rules for value 1, if none match next value is 0
    #default = [[0] for v in benchmark.targets]
    if full_transitions is None:
        eprint(">>> Generating benchmark transitions...")
        full_transitions = [ (np.array(feature_state), np.array(["0" if x=="?" else "1" for x in target_state])) for feature_state in program.feature_states() for target_state in program.predict([feature_state], semantics) ]
    full_transitions_grouped = {tuple(s1) : set(tuple(s2_) for s1_,s2_ in full_transitions if tuple(s1) == tuple(s1_)) for s1,s2 in full_transitions}
    #eprint("Transitions: ", full_transitions)
    #eprint("Grouped: ", full_transitions_grouped)

    #eprint(benchmark.to_string())
    #eprint(semantics.states(P))
    #eprint(full_transitions)

    # 2) Prepare scores containers
    #---------------------------
    results_time = []
    results_score = []

    # 3) Average over several tests
    #-----------------------------
    for run in range(run_tests):

        # 3.1 Split train/test sets on initial states
        #----------------------------------------------
        all_feature_states = list(full_transitions_grouped.keys())
        random.shuffle(all_feature_states)

         # Test set: all transition from last 20% feature states
        test_begin = max(1, int(0.8 * len(all_feature_states)))
        test_feature_states = all_feature_states[test_begin:]

        test = []
        for s1 in test_feature_states:
            test.extend([(list(s1),list(s2)) for s2 in full_transitions_grouped[s1]])
        random.shuffle(test)

        # Train set
        # All transition from first train_size % feature states (over 80% include some test set part)
        if mode == "all_from_init_states":
            train_end = max(1, int(train_size * len(all_feature_states)))
            train_feature_states = all_feature_states[:train_end]
            train = []
            for s1 in train_feature_states:
                train.extend([(list(s1),list(s2)) for s2 in full_transitions_grouped[s1]])
            random.shuffle(train)
        # Random train_size % of transitions from the feature states not in test set
        elif mode == "random_transitions":
            train_feature_states = all_feature_states[:test_begin]
            train = []
            for s1 in train_feature_states:
                train.extend([(list(s1),list(s2)) for s2 in full_transitions_grouped[s1]])
            random.shuffle(train)
            train_end = int(max(1, train_size * len(train)))
            train = train[:train_end]
        else:
            raise ValueError("Wrong mode requested")

        #eprint("train: ", train)
        #eprint("test: ", test)
        #exit()

        # DBG
        if run == 0:
            eprint(">>> Start Training on " + str(len(train)) + "/" + str(len(full_transitions)) + " transitions (" + str(round(100 * len(train) / len(full_transitions), 2)) +"%)")

        eprint(">>>> run: " + str(run+1) + "/" + str(run_tests), end='')

        train_dataset = StateTransitionsDataset([ (np.array(s1), np.array(s2)) for (s1,s2) in train], benchmark.features, benchmark.targets)
        test_dataset = StateTransitionsDataset([(np.array(s1), np.array(s2)) for s1,s2 in test], benchmark.features, benchmark.targets)

        # 3.2) Learn from training set
        #------------------------------------------

        if algorithm == "gula" or algorithm == "pride":
            # possibilities
            start = time.time()
            model = WDMVLP(features=benchmark.features, targets=benchmark.targets)
            model.compile(algorithm=algorithm)
            model.fit(dataset=train_dataset)
            #model = algorithm.fit(train, benchmark.features, benchmark.targets, supported_only=True)
            end = time.time()

            results_time.append(round(end - start,3))

        # 3.4) Evaluate on accuracy of domain prediction on test set
        #------------------------------------------------------------

        # csv format of results
        expected_train_size = train_size
        expected_test_size = 0.2
        real_train_size = round(len(train)/(len(full_transitions)),2)
        real_test_size = round(len(test)/(len(full_transitions)),2)

        if mode == "random_transitions":
            expected_train_size = round(train_size*0.8,2)

        common_settings = \
        semantics + "," +\
        benchmark_name + "," +\
        str(len(benchmark.features)) + "," +\
        str(len(full_transitions)) + "," +\
        mode + "," +\
        str(expected_train_size) + "," +\
        str(expected_test_size) + "," +\
        str(real_train_size) + "," +\
        str(real_test_size) + "," +\
        str(len(train)) + "," +\
        str(len(test))

        if algorithm == "gula" or algorithm == "pride":
            accuracy = accuracy_score(model=model, dataset=test_dataset)
            print(algorithm + "," + common_settings + "," + str(accuracy))
            eprint(" accuracy: " + str(round(accuracy * 100,2)) + "%")
            results_score.append(accuracy)

        if algorithm == "baseline":
            csv_output_settings = csv_output
            predictions = {tuple(s1): {variable: {value: random.uniform(0.0, 1.0) for value in values} for (variable, values) in test_dataset.targets} for s1 in test_feature_states}
            #eprint(prediction)
            accuracy = accuracy_score_from_predictions(predictions=predictions, dataset=test_dataset)
            print("baseline_random," + common_settings + "," + str(accuracy))
            eprint()
            eprint(">>>>> accuracy: " + str(round(accuracy * 100,2)) + "% (baseline_random)")

        #if algorithm == "always_0.0":
            predictions = {tuple(s1): {variable: {value: 0.0 for value in values} for (variable, values) in test_dataset.targets} for s1 in test_feature_states}
            #eprint(predictions)
            accuracy = accuracy_score_from_predictions(predictions=predictions, dataset=test_dataset)
            print("baseline_always_0.0," + common_settings + "," + str(accuracy))
            eprint(">>>>> accuracy: " + str(round(accuracy * 100,2)) + "% (baseline_always_0.0)")

        #if algorithm == "always_0.5":
            predictions = {tuple(s1): {variable: {value: 0.5 for value in values} for (variable, values) in test_dataset.targets} for s1 in test_feature_states}
            #eprint(predictions)
            accuracy = accuracy_score_from_predictions(predictions=predictions, dataset=test_dataset)
            print("baseline_always_0.5," + common_settings + "," + str(accuracy))
            eprint(">>>>> accuracy: " + str(round(accuracy * 100,2)) + "% (baseline_always_0.5)")

        #if algorithm == "always_1.0":
            predictions = {tuple(s1): {variable: {value: 1.0 for value in values} for (variable, values) in test_dataset.targets} for s1 in test_feature_states}
            #eprint(predictions)
            accuracy = accuracy_score_from_predictions(predictions=predictions, dataset=test_dataset)
            print("baseline_always_1.0," + common_settings + "," + str(accuracy))
            eprint(">>>>> accuracy: " + str(round(accuracy * 100,2)) + "% (baseline_always_1.0)")


    # 4) Average scores
    #-------------------
    if algorithm in ["gula","pride"]:
        accuracy = sum(results_score) / run_tests
        run_time = sum(results_time) / run_tests

        eprint(">>> AVG accuracy: " + str(round(accuracy * 100,2)) + "%")
        #eprint(">>> AVG run time: " + str(round(run_time,3)) + "s")

def evaluate_explanation_on_bn_benchmark(algorithm, benchmark, expected_model, run_tests, train_size, mode, benchmark_name, semantics_name, full_transitions=None):
    """
        Evaluate accuracy of an algorithm
        over a given benchmark with a given number/proporsion
        of training samples.

        Args:
            algorithm: Class
                Class of the algorithm to be tested
            benchmark: DMVLP
                benchmark model to be tested
            expected_model: WDMVLP
                optimal WDMVLP that model the transitions of the benchmark.
            train_size: float in [0,1] or int
                Size of the training set in proportion (float in [0,1])
            mode: string
                "all_from_init_states": training contains all transitions from its initials states
                "random": training contains random transitions, 80%/20% train/test then train is reduced to train_size
            benchmark_name: string
                for csv output.
            benchmark_name: string
                for csv output.
        Returns:
        train_set_size: int
        test_set_size: int
        accuracy: float
            Average accuracy score.
        csv_output: String
            csv string format of all tests run statistiques.
    """
    csv_output = ""

    # 0) Extract logic program
    #-----------------------
    #eprint(benchmark.to_string())

    # 1) Generate transitions
    #-------------------------------------

    # Boolean network benchmarks only have rules for value 1, if none match next value is 0
    #default = [[0] for v in benchmark.targets]
    if full_transitions is None:
        eprint(">>> Generating benchmark transitions...")
        full_transitions = [ (np.array(feature_state), np.array(["0" if x=="?" else "1" for x in target_state])) for feature_state in program.feature_states() for target_state in program.predict([feature_state], semantics) ]
    full_transitions_grouped = {tuple(s1) : set(tuple(s2_) for s1_,s2_ in full_transitions if tuple(s1) == tuple(s1_)) for s1,s2 in full_transitions}
    #eprint("Transitions: ", full_transitions)
    #eprint("Grouped: ", full_transitions_grouped)

    #eprint(benchmark.to_string())
    #eprint(semantics.states(P))
    #eprint(full_transitions)

    # 2) Prepare scores containers
    #---------------------------
    results_time = []
    results_score = []

    # 3) Average over several tests
    #-----------------------------
    for run in range(run_tests):

        # 3.1 Split train/test sets on initial states
        #----------------------------------------------
        all_feature_states = list(full_transitions_grouped.keys())
        random.shuffle(all_feature_states)

         # Test set: all transition from last 20% feature states
        test_begin = max(1, int(0.8 * len(all_feature_states)))
        test_feature_states = all_feature_states[test_begin:]

        test = []
        for s1 in test_feature_states:
            test.extend([(list(s1),list(s2)) for s2 in full_transitions_grouped[s1]])
        random.shuffle(test)

        # Train set
        # All transition from first train_size % feature states (over 80% include some test set part)
        if mode == "all_from_init_states":
            train_end = max(1, int(train_size * len(all_feature_states)))
            train_feature_states = all_feature_states[:train_end]
            train = []
            for s1 in train_feature_states:
                train.extend([(list(s1),list(s2)) for s2 in full_transitions_grouped[s1]])
            random.shuffle(train)
        # Random train_size % of transitions from the feature states not in test set
        elif mode == "random_transitions":
            train_feature_states = all_feature_states[:test_begin]
            train = []
            for s1 in train_feature_states:
                train.extend([(list(s1),list(s2)) for s2 in full_transitions_grouped[s1]])
            random.shuffle(train)
            train_end = int(max(1, train_size * len(train)))
            train = train[:train_end]
        else:
            raise ValueError("Wrong mode requested")

        #eprint("train: ", train)
        #eprint("test: ", test)
        #exit()

        # DBG
        if run == 0:
            eprint(">>> Start Training on " + str(len(train)) + "/" + str(len(full_transitions)) + " transitions (" + str(round(100 * len(train) / len(full_transitions), 2)) +"%)")

        eprint(">>>> run: " + str(run+1) + "/" + str(run_tests), end='')

        train_dataset = StateTransitionsDataset([ (np.array(s1), np.array(s2)) for (s1,s2) in train], benchmark.features, benchmark.targets)

        # 3.2) Learn from training set
        #------------------------------------------

        if algorithm == "gula" or algorithm == "pride":
            # possibilities
            start = time.time()
            model = WDMVLP(features=benchmark.features, targets=benchmark.targets)
            model.compile(algorithm=algorithm)
            model.fit(dataset=train_dataset)
            #model = algorithm.fit(train, benchmark.features, benchmark.targets, supported_only=True)
            end = time.time()

            results_time.append(round(end - start,3))

        # 3.4) Evaluate on accuracy of domain prediction on test set
        #------------------------------------------------------------
        test_dataset = StateTransitionsDataset([(np.array(s1), np.array(s2)) for s1,s2 in test], benchmark.features, benchmark.targets)

        # csv format of results
        expected_train_size = train_size
        expected_test_size = 0.2
        real_train_size = round(len(train)/(len(full_transitions)),2)
        real_test_size = round(len(test)/(len(full_transitions)),2)

        if mode == "random_transitions":
            expected_train_size = round(train_size*0.8,2)

        common_settings = \
        semantics_name + "," +\
        benchmark_name + "," +\
        str(len(benchmark.features)) + "," +\
        str(len(full_transitions)) + "," +\
        mode + "," +\
        str(expected_train_size) + "," +\
        str(expected_test_size) + "," +\
        str(real_train_size) + "," +\
        str(real_test_size) + "," +\
        str(len(train)) + "," +\
        str(len(test))

        if algorithm == "gula" or algorithm == "pride":
            score = explanation_score(model=model, expected_model=expected_model, dataset=test_dataset)
            print(algorithm + "," + common_settings + "," + str(score))
            results_score.append(score)
            eprint(" explanation score: " + str(round(score * 100,2)) + "%")

        if algorithm == "baseline":
            eprint()

            # Perfect prediction random rule
            predictions = {tuple(s1): {variable: {value: (proba, \
            (int(proba*100), random_rule(var_id,val_id,test_dataset.features,test_dataset.targets)),\
            (100 - int(proba*100), random_rule(var_id,val_id,test_dataset.features,test_dataset.targets)) )\
            for val_id, value in enumerate(values) for proba in [int(val_id in set(test_dataset.targets[var_id][1].index(s2[var_id]) for s1_, s2 in test_dataset.data if tuple(s1_)==s1))]}\
            for var_id, (variable, values) in enumerate(test_dataset.targets)}\
            for s1 in test_feature_states}

            score = explanation_score_from_predictions(predictions=predictions, expected_model=expected_model, dataset=test_dataset)
            print("baseline_perfect_predictions_random_rules," + common_settings + "," + str(score))
            eprint(">>>>> explanation score: " + str(round(score * 100,2)) + "% (baseline_perfect_predictions_random_rules)")

            # Perfect prediction empty_program":
            predictions = {tuple(s1): {variable: {value: (proba, \
            (int(proba*100), None),\
            (100 - int(proba*100), None) )\
            for val_id, value in enumerate(values) for proba in [int(val_id in set(test_dataset.targets[var_id][1].index(s2[var_id]) for s1_, s2 in test_dataset.data if tuple(s1_)==s1))]}\
            for var_id, (variable, values) in enumerate(test_dataset.targets)}\
            for s1 in test_feature_states}

            score = explanation_score_from_predictions(predictions=predictions, expected_model=expected_model, dataset=test_dataset)
            print("baseline_perfect_predictions_no_rules," + common_settings + "," + str(score))
            eprint(">>>>> explanation score: " + str(round(score * 100,2)) + "% (baseline_perfect_predictions_no_rules)")

            # Perfect prediction most general rule
            predictions = {tuple(s1): {variable: {value: (proba, \
            (int(proba*100), Rule(var_id, val_id, len(test_dataset.features))),\
            (100 - int(proba*100), Rule(var_id, val_id, len(test_dataset.features))) )\
            for val_id, value in enumerate(values) for proba in [int(val_id in set(test_dataset.targets[var_id][1].index(s2[var_id]) for s1_, s2 in test_dataset.data if tuple(s1_)==s1))]}\
            for var_id, (variable, values) in enumerate(test_dataset.targets)}\
            for s1 in test_feature_states}

            score = explanation_score_from_predictions(predictions=predictions, expected_model=expected_model, dataset=test_dataset)
            print("baseline_perfect_predictions_most_general_rules," + common_settings + "," + str(score))
            eprint(">>>>> explanation score: " + str(round(score * 100,2)) + "% (baseline_perfect_predictions_most_general_rules)")

            # Perfect prediction most specific rule:
            predictions = {tuple(s1): {variable: {value: (proba, \
            (int(proba*100), most_specific_matching_rule),\
            (100 - int(proba*100), most_specific_matching_rule) )\
            for val_id, value in enumerate(values)\
            for proba in [int(val_id in set(test_dataset.targets[var_id][1].index(s2[var_id]) for s1_, s2 in test_dataset.data if tuple(s1_)==s1))] \
            for most_specific_matching_rule in [Rule(var_id,val_id,len(test_dataset.features),[(cond_var,cond_val) for cond_var,cond_val in enumerate(GULA.encode_state(s1,test_dataset.features))])]}\
            for var_id, (variable, values) in enumerate(test_dataset.targets)}\
            for s1 in test_feature_states}

            score = explanation_score_from_predictions(predictions=predictions, expected_model=expected_model, dataset=test_dataset)
            print("baseline_perfect_predictions_most_specific_rules," + common_settings + "," + str(score))
            eprint(">>>>> explanation score: " + str(round(score * 100,2)) + "% (baseline_perfect_predictions_most_specific_rules)")

            # Random prediction

            # random prediction and rules
            predictions = {tuple(s1): {variable: {value: (proba, \
            (int(proba*100), random_rule(var_id,val_id,test_dataset.features,test_dataset.targets)),\
            (100 - int(proba*100), random_rule(var_id,val_id,test_dataset.features,test_dataset.targets)) )\
            for val_id, value in enumerate(values) for proba in [round(random.uniform(0.0,1.0),2)]}\
            for var_id, (variable, values) in enumerate(test_dataset.targets)}\
            for s1 in test_feature_states}

            score = explanation_score_from_predictions(predictions=predictions, expected_model=expected_model, dataset=test_dataset)
            print("baseline_random_predictions_random_rules," + common_settings + "," + str(score))
            eprint(">>>>> explanation score: " + str(round(score * 100,2)) + "% (baseline_random_predictions_random_rules)")

            # empty_program":
            predictions = {tuple(s1): {variable: {value: (proba, \
            (int(proba*100), None),\
            (100 - int(proba*100), None) )\
            for val_id, value in enumerate(values) for proba in [round(random.uniform(0.0,1.0),2)]}\
            for var_id, (variable, values) in enumerate(test_dataset.targets)}\
            for s1 in test_feature_states}

            score = explanation_score_from_predictions(predictions=predictions, expected_model=expected_model, dataset=test_dataset)
            print("baseline_random_predictions_no_rules," + common_settings + "," + str(score))
            eprint(">>>>> explanation score: " + str(round(score * 100,2)) + "% (baseline_random_predictions_no_rules)")

            # random prediction and most general rule
            predictions = {tuple(s1): {variable: {value: (proba, \
            (int(proba*100), Rule(var_id, val_id, len(test_dataset.features))),\
            (100 - int(proba*100), Rule(var_id, val_id, len(test_dataset.features))) )\
            for val_id, value in enumerate(values) for proba in [round(random.uniform(0.0,1.0),2)]}\
            for var_id, (variable, values) in enumerate(test_dataset.targets)}\
            for s1 in test_feature_states}

            score = explanation_score_from_predictions(predictions=predictions, expected_model=expected_model, dataset=test_dataset)
            print("baseline_random_predictions_most_general_rules," + common_settings + "," + str(score))
            eprint(">>>>> explanation score: " + str(round(score * 100,2)) + "% (baseline_random_predictions_most_general_rules)")

            # random prediction and most specific rule:
            predictions = {tuple(s1): {variable: {value: (proba, \
            (int(proba*100), most_specific_matching_rule),\
            (100 - int(proba*100), most_specific_matching_rule) )\
            for val_id, value in enumerate(values)\
            for proba in [round(random.uniform(0.0,1.0),2)] \
            for most_specific_matching_rule in [Rule(var_id,val_id,len(test_dataset.features),[(cond_var,cond_val) for cond_var,cond_val in enumerate(GULA.encode_state(s1,test_dataset.features))])]}\
            for var_id, (variable, values) in enumerate(test_dataset.targets)}\
            for s1 in test_feature_states}

            score = explanation_score_from_predictions(predictions=predictions, expected_model=expected_model, dataset=test_dataset)
            print("baseline_random_predictions_most_specific_rules," + common_settings + "," + str(score))
            eprint(">>>>> explanation score: " + str(round(score * 100,2)) + "% (baseline_random_predictions_most_specific_rules)")

    # 4) Average scores
    #-------------------
    if algorithm in ["gula", "pride"]:
        score = sum(results_score) / run_tests
        #run_time = sum(results_time) / run_tests
        eprint(">>> AVG explanation score: " + str(round(score * 100,2)) + "%")

def random_rule(head_var_id, head_val_id, features, targets, size=None):
    body = []
    conditions = []
    nb_conditions = random.randint(0,(len(features)))
    if size is not None:
        nb_conditions = size
    while len(body) < nb_conditions:
        var = random.randint(0,len(features)-1)
        val = random.randint(0,len(features[var][1])-1)
        if var not in conditions:
            body.append( (var, val) )
            conditions.append(var)
    return Rule(head_var_id, head_val_id, len(features), body)

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
    semantics_classes = ["synchronous", "asynchronous", "general"]
    train_sizes = [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # 1: Parameters
    #---------------
    lfit_methods = ["gula", "pride", "brute-force", "synchronizer"]
    baseline_methods = ["baseline"]
    experiements = ["scalability", "accuracy", "explanation"]
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
        train_sizes = [0.01,0.05,0.1,0.25,0.5,0.75,1.0]
        eprint("Start benchmark scalability evaluation: Boolean Networks, partial transitions with "+algorithm)
        eprint("\nAVG over "+str(run_tests)+" runs of run time when learning from transitions of Boolean Network benchmarks from " ,min_var, " until ",max_var, " variables:")
        #eprint("Benchmark & size & synchronous & asynchronous & general\\\\")

        final_csv_output = "method,semantics,benchmark_name,benchmark_size,transitions,mode,expected_train_percent,real_train_percent,train_size,run_time"
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
                default = [(var, [0]) for var,vals in program.targets]
                full_transitions = [ (np.array(feature_state), np.array(target_state)) for feature_state in program.feature_states() for target_state in program.predict([feature_state], semantics, default) ]

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
        eprint("Start benchmark accuracy evaluation: Boolean Networks")
        eprint("AVG over "+str(run_tests)+" runs of accuracy when learning from transitions of Boolean Network benchmarks transitions from " ,min_var, " until " ,max_var, " variables:")

        # 4.1: All transitions from train init states or Random transitions from train init states (80%/20% train/test then random XX% from train)
        #-----------------------------------------------
        if mode == "all_from_init_states":
            eprint("10% to 100% of all transitions as training, test on rest (for 100%, training = test)")
        else:
            eprint("random 10% to 100% of the training transitions, with 80%/20% of total transitions as train/test set")
        eprint()

        final_csv_output = "method,semantics,benchmark_name,benchmark_size,transitions,mode,expected_train_percent,expected_test_percent,real_train_percent,real_test_percent,train_size,test_size,accuracy_score"
        print(final_csv_output)

        for size, name, program in dmvlp_benchmarks:
            #eprint("> ", name, ",", size, end='')
            eprint()
            eprint("> ", name, ": ", len(program.features), " variables, ", len(program.rules), " rules, ")
            for semantics in semantics_classes:
                if semantics == "general" and size > max_var_general:
                    #latex += " & M.O."
                    continue
                eprint(">> Semantics: "+semantics)

                #full_transitions = [ (np.array(feature_state), np.array(["0" if x=="?" else "1" for x in target_state])) for feature_state in program.feature_states() for target_state in program.predict([feature_state], semantics) ]
                default = [(var, [0]) for var,vals in program.targets]
                full_transitions = [ (np.array(feature_state), np.array(target_state)) for feature_state in program.feature_states() for target_state in program.predict([feature_state], semantics, default) ]

                for train_size in train_sizes:
                    real_train_size = train_size
                    if mode == "random_transitions":
                        real_train_size = 0.8*train_size
                    eprint()
                    eprint(">> ", round(real_train_size*100,2), "% training.")
                    evaluate_accuracy_on_bn_benchmark(algorithm, program, semantics, run_tests, train_size, mode, name, full_transitions)
        #print(final_csv_output)

    # 5: Explanation accuracy experiements
    #--------------------------------------
    if experiement == EXPLANATION_EXPERIEMENT:
        eprint("Start benchmark explanation evaluation: Boolean Networks")
        eprint("AVG over "+str(run_tests)+" runs of explanation when learning from transitions of Boolean Network benchmarks transitions from " ,min_var, " until " ,max_var, " variables:")

        # 4.1: All transitions from train init states or Random transitions from train init states (80%/20% train/test then random XX% from train)
        #-----------------------------------------------
        if mode == "all_from_init_states":
            eprint("10% to 100% of all transitions as training, test on rest (for 100%, training = test)")
        else:
            eprint("random 10% to 100% of the training transitions, with 80%/20% of total transitions as train/test set")

        final_csv_output = "method,semantics,benchmark_name,benchmark_size,transitions,mode,expected_train_percent,expected_test_percent,real_train_percent,real_test_percent,train_size,test_size,explanation_score"
        print(final_csv_output)

        for size, name, program in dmvlp_benchmarks:
            eprint()
            eprint("> ", name, ": ", len(program.features), " variables, ", len(program.rules), " rules, ")
            for semantics in semantics_classes:
                if semantics == "general" and size > max_var_general:
                    #latex += " & M.O."
                    continue
                eprint(">> Semantics: "+semantics)
                #full_transitions = [ (np.array(feature_state), np.array(["0" if x=="?" else "1" for x in target_state])) for feature_state in program.feature_states() for target_state in program.predict([feature_state], semantics) ]
                default = [(var, [0]) for var,vals in program.targets]
                full_transitions = [ (feature_state, target_state) for feature_state in program.feature_states() for target_state in program.predict([feature_state], semantics, default) ]

                # compute expected WDMVLP
                expected_model = WDMVLP(features=program.features, targets=program.targets)
                expected_model.compile(algorithm="gula")
                dataset = StateTransitionsDataset(full_transitions, program.features, program.targets)
                expected_model.fit(dataset=dataset)

                for train_size in train_sizes:
                    real_train_size = train_size
                    if mode == "random_transitions":
                        real_train_size = 0.8*train_size
                    eprint()
                    eprint(">> ", round(real_train_size*100,2), "% training.")
                    evaluate_explanation_on_bn_benchmark(algorithm, program, expected_model, run_tests, train_size, mode, name, semantics, full_transitions)
