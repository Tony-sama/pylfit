#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2021/09/16
# @updated: 2021/09/16
#
# @desc: example of learning from Boolean network of a pyboolnet file
#   - print the original Boolean functions
#   - convert to DMVLP
#   - print the DMVLP
#   - generation of all transition, extract training/test subset
#   - learning from the subset of transition using GULA
#   - print learn DMVLP
#-------------------------------------------------------------------------------


import random
import numpy as np

import pylfit
from pylfit.utils import eprint
from pylfit.datasets import DiscreteStateTransitionsDataset
from pylfit.models import DMVLP, WDMVLP
from pylfit.postprocessing import accuracy_score, explanation_score

BENCHMARK_NAME = "faure_cellcycle"
TRAIN_SIZE = 0.3
VERBOSE = 1
PRUNING_MIN_WEIGHT = 30
PRUNING_MAX_BODY_SIZE = 4

random.seed(0)

# 1: Main
#------------
if __name__ == '__main__':

    bn_file_path = "benchmarks/boolean_networks/pyboolnet/bio/"+BENCHMARK_NAME+".bnet"
    output_file_path = "tmp/"+BENCHMARK_NAME+".csv"

# 1) Print Boolean Functions
#--------------------------------
    print()
    print("Boolean network benchmark:")
    f = open(bn_file_path, 'r')
    file_contents = f.read()
    print (file_contents)
    f.close()


# 2) Convert to DMVLP
#--------------------------------
    print()
    print("Loading DMVLP from Boolean network file: "+bn_file_path)
    benchmark = pylfit.preprocessing.boolean_network.dmvlp_from_boolean_network_file(bn_file_path)
    benchmark.summary()

    print("Benchmark rules: ", len(benchmark.rules))

# 3) Generate all transitions
#--------------------------------
    print()
    print("All states transitions (synchronous): ")
    full_transitions = []
    prediction = benchmark.predict(benchmark.feature_states())
    for s1 in prediction:
        for s2 in prediction[s1]:
            full_transitions.append( (s1, np.array(["0" if x=="?" else "1" for x in s2])) ) # Use 0 as default: replace ? by 0

    print(len(full_transitions))

# 4) Compute perfect WDMVLP from all Transitions
#------------------------------------------------
    expected_model = WDMVLP(features=benchmark.features, targets=benchmark.targets)
    expected_model.compile(algorithm="gula")
    dataset = DiscreteStateTransitionsDataset(full_transitions, benchmark.features, benchmark.targets)
    expected_model.fit(dataset=dataset, verbose=VERBOSE)

# 5) Generate training/test sets
#--------------------------------
    full_transitions_grouped = {tuple(s1) : set(tuple(s2_) for s1_,s2_ in full_transitions if tuple(s1) == tuple(s1_)) for s1,s2 in full_transitions}
    all_feature_states = list(full_transitions_grouped.keys())
    random.shuffle(all_feature_states)

    test_begin = max(1, int(0.8 * len(all_feature_states)))
    test_feature_states = all_feature_states[test_begin:]

    test = []
    for s1 in test_feature_states:
        test.extend([(list(s1),list(s2)) for s2 in full_transitions_grouped[s1]])
    random.shuffle(test)

    # Train set
    # Random train_size % of transitions from the feature states not in test set
    train_feature_states = all_feature_states[:test_begin]
    train = []
    for s1 in train_feature_states:
        train.extend([(list(s1),list(s2)) for s2 in full_transitions_grouped[s1]])
    random.shuffle(train)
    train_end = int(max(1, TRAIN_SIZE * len(full_transitions)))
    train = train[:train_end]

    print()
    print(">>> Start Training on " + str(len(train)) + "/" + str(len(full_transitions)) + " transitions (" + str(round(100 * len(train) / len(full_transitions), 2)) +"%)")

    train_dataset = DiscreteStateTransitionsDataset([ (np.array(s1), np.array(s2)) for (s1,s2) in train], benchmark.features, benchmark.targets)
    test_dataset = DiscreteStateTransitionsDataset([(np.array(s1), np.array(s2)) for s1,s2 in test], benchmark.features, benchmark.targets)


# 6) Learn WDMVLP from training set
#-----------------------------------
    learned_model = WDMVLP(features=benchmark.features, targets=benchmark.targets)
    learned_model.compile(algorithm="gula")
    learned_model.fit(train_dataset, verbose=VERBOSE)

    print()
    print("Likeliness rules:", len(learned_model.rules))
    print("Unlikeliness rules:", len(learned_model.unlikeliness_rules))

# 6) Check performances: prediction accuracy
#----------------------------------------
    print()
    accuracy = accuracy_score(model=learned_model, dataset=test_dataset)
    print("Accuracy on test set: " + str(round(accuracy * 100,2)) + "%")
    score = explanation_score(model=learned_model, expected_model=expected_model, dataset=test_dataset)
    print("Explanation score on test set: " + str(round(score * 100,2)) + "%")

# 7) Prune learned rule for readability
#----------------------------------------
    print()
    print("Pruning rules for readability:")
    learned_model.rules = [(w,r) for (w,r) in learned_model.rules if w >= PRUNING_MIN_WEIGHT and r.size() <= PRUNING_MAX_BODY_SIZE]
    learned_model.unlikeliness_rules = [(w,r) for (w,r) in learned_model.unlikeliness_rules if w >= PRUNING_MIN_WEIGHT and r.size() <= PRUNING_MAX_BODY_SIZE]

    learned_model.summary()

    print("Likeliness rules:", len(learned_model.rules))
    print("Unlikeliness rules:", len(learned_model.unlikeliness_rules))
    activation_rules = [(w,r) for (w,r) in learned_model.rules if r.head_value == 1]
    print("Activations rules:", len(activation_rules))

# 8) Check pruning impact on accuracy
#----------------------------------------

    print()
    accuracy = accuracy_score(model=learned_model, dataset=test_dataset)
    print("Accuracy on test set: " + str(round(accuracy * 100,2)) + "%")
    score = explanation_score(model=learned_model, expected_model=expected_model, dataset=test_dataset)
    print("Explanation score on test set: " + str(round(score * 100,2)) + "%")

# 9) Evaluate similarity with benchmark
#----------------------------------------

    print()
    # Check VS original rules missing/additional
    print("Likeliness rules 1 in head only:",len(activation_rules))
    for (w,r) in activation_rules:
        print(str(w)+", "+r.logic_form(learned_model.features, learned_model.targets))

    missing = [r for r in benchmark.rules if r not in [r_ for (w,r_) in activation_rules]]
    spurious = [(w,r) for (w,r) in activation_rules if r not in benchmark.rules]

    print("Missing original rules: ", len(missing))
    print("Missing original rules:")
    for r in missing:
        print(r.logic_form(learned_model.features, learned_model.targets))

    print("Spurious learned rules: ", len(spurious))
    print("Spurious learned rules:")
    for (w,r) in spurious:
        print(str(w)+", "+r.logic_form(learned_model.features, learned_model.targets))
