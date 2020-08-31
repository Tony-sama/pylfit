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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

import sys
sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')
sys.path.insert(0, 'src/semantics')

from utils import eprint
from logicProgram import LogicProgram
from synchronous import Synchronous

# Constants
#------------

# Number of tests for averaging scores
run_tests = 1

# Logic program form of Boolean network from biological litteratures
benchmarks = {
    "pbnet_toy_raf": "benchmarks/logic_programs/pyboolnet/toys/raf.lp",
    "pbnet_toy_n3s1c1a": "benchmarks/logic_programs/pyboolnet/toys/n3s1c1a.lp",
    "pbnet_toy_n3s1c1b": "benchmarks/logic_programs/pyboolnet/toys/n3s1c1b.lp",
    "pbnet_toy_n5s3": "benchmarks/logic_programs/pyboolnet/toys/n5s3.lp",
    "pbnet_toy_n6s1c2": "benchmarks/logic_programs/pyboolnet/toys/n6s1c2.lp",
    "pbnet_toy_n7s3": "benchmarks/logic_programs/pyboolnet/toys/n7s3.lp",
    "pbnet_toy_n12c5": "benchmarks/logic_programs/pyboolnet/toys/n12c5.lp",

    "repressilator": "benchmarks/logic_programs/repressilator.lp",
    "mammalian": "benchmarks/logic_programs/mammalian.lp",
    "fission": "benchmarks/logic_programs/fission.lp",
    "budding": "benchmarks/logic_programs/budding.lp",
    "arabidopsis": "benchmarks/logic_programs/arabidopsis.lp"
}

def evaluate_on_bn_benchmark(algorithm, benchmark, train_size=None):
    """
        Evaluate accuracy and explainability of an algorithm
        over a given benchmark with a given number/proporsion
        of training samples.

        Args:
            algorithm: Class
                Class of the algorithm to be tested
            benchmark: String
                Label of the benchmark to be tested
            train_size: float in [0,1] or int
                Size of the training set in proportion (float in [0,1])
                or explicit (int)
    """

    # 0) Extract logic program
    #-----------------------
    benchmark = benchmarks[benchmark]
    P = LogicProgram.load_from_file(benchmark)
    #eprint(P.to_string())

    # 1) Generate transitions
    #-------------------------------------

    # Boolean network benchmarks only have rules for value 1, if none match next value is 0
    default = [[0] for v in P.get_variables()]
    full_transitions = Synchronous.transitions(P,default) #P.generate_all_transitions()
    #eprint(P.to_string())
    #eprint(Synchronous.states(P))
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

        eprint("\r>>> run: " + str(run+1) + "/" + str(run_tests), end='')

        # 3.2) Learn from training set
        #-------------------------
        start = time.time()
        model = algorithm.fit(P.get_variables(), P.get_values(), train)
        end = time.time()
        results_time.append(round(end - start,3))

        # DBG
        #eprint(model)

        # 3.3) Evaluate model against originals rules
        #-----------------------------------------------

        # LUST special case
        if type(model) == list:
            model = model[0]

        common, missing, over = P.compare(model)

        #eprint(">>> Original:")
        #eprint(P.to_string())

        #eprint(">>> Learned:")
        #eprint(model.to_string())

        #eprint(">>> Logic Program comparaison:")
        #eprint(">>>> Common: "+str(len(common))+"/"+str(len(P.get_rules()))+"("+str(round(100 * len(common) / len(P.get_rules()),2))+"%)")
        #eprint(">>>> Missing: "+str(len(missing))+"/"+str(len(P.get_rules()))+"("+str(round(100 * len(missing) / len(P.get_rules()),2))+"%)")
        #eprint(">>>> Over: "+str(len(over))+"/"+str(len(P.get_rules()))+"("+str(round(100 * len(over) / len(model.get_rules()),2))+"%)")

        # Collect scores
        results_common.append(len(common))
        results_missing.append(len(missing))
        results_over.append(len(over))

        # Perfect case: evaluate over all transitions
        if len(test) == 0:
            test = train

        # 3.4) Evaluate accuracy prediction over unseen states
        #-------------------------------------------------
        pred = [(s1, model.next(s1)) for s1, s2 in test]
        precision = round(LogicProgram.precision(test,pred),2)

        #eprint(">>> Prediction precision")
        #eprint(">>>> " + str(round(precision * 100,2)) + "%")

        results_precision.append(precision)

    # 4) Average scores
    #-------------------
    run_time = sum(results_time) / run_tests
    common = sum(results_common) / run_tests
    missing = sum(results_missing) / run_tests
    over = sum(results_over) / run_tests
    precision = sum(results_precision) / run_tests

    eprint()
    eprint(">>> Run time: "+str(run_time)+"s")
    eprint(">>> Logic Program comparaison:")
    eprint(">>>> AVG Common: "+str(common)+"/"+str(len(P.get_rules()))+"("+str(round(100 * common / len(P.get_rules()),2))+"%)")
    eprint(">>>> AVG Missing: "+str(missing)+"/"+str(len(P.get_rules()))+"("+str(round(100 * missing / len(P.get_rules()),2))+"%)")
    eprint(">>>> AVG Over: "+str(over)+"/"+str(len(P.get_rules()))+"("+str(round(100 * over / len(model.get_rules()),2))+"%)")

    eprint(">>> Prediction precision")
    eprint(">>>> AVG accuracy: " + str(round(precision * 100,2)) + "%")

    return round(precision * 100, 2)

def evaluate_on_bn_benchmark_with_NN(algorithm, benchmark, train_size, artificial_size=None):
    """
        Evaluate accuracy and explainability of an algorithm
        over a given benchmark with a given number/propertion
        of training samples.
        Additional artificial transitions are produced by Neural network.

        Args:
            name: String
                Label of the benchmark to be tested
            train_size: float in [0,1] or int
                Size of the training set in proportion (float in [0,1])
                or explicit (int)
    """

    # 0) Extract logic program
    #-----------------------
    benchmark = benchmarks[benchmark]
    P = LogicProgram.load_from_file(benchmark)
    #eprint(P.to_string())

    # 1) Generate transitions
    #-------------------------------------
    full_transitions = P.generate_all_transitions()

    # 2) Prepare scores containers
    #---------------------------
    results_time = []
    results_common = []
    results_missing = []
    results_over = []
    results_precision_NN = []
    results_precision_algo = []

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
            if artificial_size is None:
                eprint(">>>> Generating all "+str(artificial_size)+" test transitions from NN")
            else:
                if artificial_size > len(test):
                    eprint(">>>> Warning given artificial training set size is greater than total unseen transitions: " +str(artificial_size) + "/" + str(len(test)))
                    eprint(">>>> Generating all "+str(len(test))+" test transitions from NN")
                else:
                    eprint(">>>> Generating "+str(artificial_size)+" random artificial transitions from NN")

        eprint("\r>>> run: " + str(run+1) + "/" + str(run_tests), end='')

        # 3.2) Train Neural Network
        #---------------------------
        #eprint(">>>> Training NN")
        start = time.time()

        train_X = np.array([s1 for s1, s2 in train])
        train_y = np.array([s2 for s1, s2 in train])

        NN = Sequential()
        NN.add(Dense(128, activation='relu', input_dim=train_X.shape[1]))
        NN.add(Dense(64, activation='relu'))
        NN.add(Dense(32, activation='relu'))
        NN.add(Dense(train_y.shape[1], activation='sigmoid'))
        NN.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Train the model, iterating on the data in batches of 32 samples
        NN.fit(train_X, train_y, epochs=100, batch_size=32, verbose=0)

        # 3.3) Generate artificial data
        #------------------------------
        generated = []

        # All unobserved transition will be predicted by the NN
        if artificial_size is None or artificial_size >= len(test):
            for s1, s2 in full_transitions:
                unknown = True

                for s1_, s2_ in train: # predict only unknown transitions
                    if s1 == s1_:
                        unknown = False
                        break

                if unknown:
                    prediction = NN.predict(np.array( [s1] ))
                    prediction = [ int(i > 0.5) for i in prediction[0] ]
                    generated.append( (s1, prediction) )
        else: # generate given number of artificial transition
            while len(generated) < artificial_size:
                s1 = [ random.randint(0, len(P.get_values()[var])-1) for var in range(len(P.get_variables()))] # random state
                unknown = True

                for s1_, s2_ in train: # predict only unknown transitions
                    if s1 == s1_:
                        unknown = False
                        break

                if unknown:
                    prediction = NN.predict(np.array( [s1] ))
                    prediction = [ int(i > 0.5) for i in prediction[0] ]
                    generated.append( (s1, prediction) )
                    #eprint("\r"+str(len(generated))+"/"+str(artificial_size), end='')

        #eprint("NN generated: ")
        #eprint(generated)

        # Merge raw data + artificial
        train = train + generated

        # 3.4) Learn from extended training set
        #---------------------------------------
        model = algorithm.fit(P.get_variables(), P.get_values(), train)
        end = time.time()
        results_time.append(round(end - start,3))

        # 3.5) Evaluate model against originals rules
        #-----------------------------------------------
        common, missing, over = P.compare(model)

        #eprint(">>> Original:")
        #eprint(P.to_string())

        #eprint(">>> Learned:")
        #eprint(model.to_string())

        #eprint(">>> Logic Program comparaison:")
        #eprint(">>>> Common: "+str(len(common))+"/"+str(len(P.get_rules()))+"("+str(round(100 * len(common) / len(P.get_rules()),2))+"%)")
        #eprint(">>>> Missing: "+str(len(missing))+"/"+str(len(P.get_rules()))+"("+str(round(100 * len(missing) / len(P.get_rules()),2))+"%)")
        #eprint(">>>> Over: "+str(len(over))+"/"+str(len(P.get_rules()))+"("+str(round(100 * len(over) / len(model.get_rules()),2))+"%)")

        # Collect scores
        results_common.append(len(common))
        results_missing.append(len(missing))
        results_over.append(len(over))

        # Perfect case: evaluate over all transitions
        if len(test) == 0:
            test = train

        # 3.6) Evaluate accuracy prediction over unseen states
        #------------------------------------------------------

        # NN accuracy
        predictions = NN.predict(np.array( [s1 for s1, s2 in test] ))
        for s in predictions:
            for i in range(len(s)):
                s[i] = int(s[i] > 0.5)
        pred = []

        for i in range(len(test)):
            pred.append( (test[i][0], predictions[i]) )
        precision_NN = round(LogicProgram.precision(test,pred),2)

        # Algorithm accuracy
        pred = [(s1, model.next(s1)) for s1, s2 in test]
        precision_algo = round(LogicProgram.precision(test,pred),2)

        #eprint(">>> Prediction precision")
        #eprint(">>>> " + str(round(precision * 100,2)) + "%")

        results_precision_NN.append(precision_NN)
        results_precision_algo.append(precision_algo)

    # 4) Average scores
    #-------------------
    run_time = sum(results_time) / run_tests
    common = sum(results_common) / run_tests
    missing = sum(results_missing) / run_tests
    over = sum(results_over) / run_tests
    precision_NN = sum(results_precision_NN) / run_tests
    precision_algo = sum(results_precision_algo) / run_tests

    eprint()
    eprint(">>> Scores over "+ str(len(test)) +" test samples:")
    eprint(">>> Run time: "+str(run_time)+"s")
    eprint(">>>> Logic Program comparaison:")
    eprint(">>>>> AVG Common: "+str(common)+"/"+str(len(P.get_rules()))+"("+str(round(100 * common / len(P.get_rules()),2))+"%)")
    eprint(">>>>> AVG Missing: "+str(missing)+"/"+str(len(P.get_rules()))+"("+str(round(100 * missing / len(P.get_rules()),2))+"%)")
    eprint(">>>>> AVG Over: "+str(over)+"/"+str(len(P.get_rules()))+"("+str(round(100 * over / len(model.get_rules()),2))+"%)")

    eprint(">>>> Prediction precision")
    eprint(">>>>> AVG accuracy NN: " + str(round(precision_NN * 100,2)) + "%")
    eprint(">>>>> AVG accuracy algorithm: " + str(round(precision_algo * 100,2)) + "%")

    return round(precision_algo * 100, 2)
