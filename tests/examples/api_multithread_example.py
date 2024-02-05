#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2021/01/01
# @updated: 2021/06/15
#
# @desc: example of the use DMVLP model with algorithm GULA and PRIDE
#-------------------------------------------------------------------------------

BENCHMARK_NAME = "xiao_wnt5a" #"arabidopsis"

import pylfit

import multiprocessing
import numpy as np
import time

# 1: Main
#------------
if __name__ == '__main__':

    #bn_file_path = "benchmarks/boolean_networks/boolenet/"+BENCHMARK_NAME+".net"
    bn_file_path = "benchmarks/boolean_networks/pyboolnet/bio/"+BENCHMARK_NAME+".bnet"
    output_file_path = "tmp/"+BENCHMARK_NAME+".csv"

# 1) Convert to DMVLP
#--------------------------------
    print()
    print("Loading DMVLP from Boolean network file: "+bn_file_path)
    benchmark = pylfit.preprocessing.boolean_network.dmvlp_from_boolean_network_file(bn_file_path)
    #benchmark.summary()

    print("Benchmark variables: ", len(benchmark.features))
    print("Benchmark rules: ", len(benchmark.rules))

# 2) Generate all transitions
#--------------------------------
    print()
    print("All states transitions (synchronous): ")
    full_transitions = []
    prediction = benchmark.predict(benchmark.feature_states())
    for s1 in prediction:
        for s2 in prediction[s1]:
            full_transitions.append( (s1, np.array(["0" if x=="?" else "1" for x in s2])) ) # Use 0 as default: replace ? by 0

    print(len(full_transitions), "transitions")

    dataset = pylfit.datasets.DiscreteStateTransitionsDataset(full_transitions, benchmark.features, benchmark.targets)

# 3) DMVLP learning
#------------------------------------------------

    print("> GULA multi-thread at target atom level")
    # Initialize a DMVLP with the dataset variables and set GULA as learning algorithm
    model = pylfit.models.DMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm="gula")

    for i in range(1, multiprocessing.cpu_count()+1):
        start = time.time()
        model.fit(dataset=dataset, verbose=0, threads=i)
        end = time.time()
        print(">> "+str(i)+" threads: "+str(round(end - start,2))+" s")

    print("> PRIDE multi-thread at target atom level")
    # Initialize a DMVLP with the dataset variables and set PRIDE as learning algorithm
    model = pylfit.models.DMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm="pride") # model.compile(algorithm="pride")

    for i in range(1, multiprocessing.cpu_count()+1):
        start = time.time()
        model.fit(dataset=dataset, verbose=0, threads=i)
        end = time.time()
        print(">> "+str(i)+" threads: "+str(round(end - start,2))+" s")

    print("> PRIDE multi-thread at rule level")
    for i in range(1, multiprocessing.cpu_count()+1):
        start = time.time()
        model.fit(dataset=dataset, verbose=0, heuristics=["multi_thread_at_rule_level"], threads=i)
        end = time.time()
        print(">> "+str(i)+" threads: "+str(round(end - start,2))+" s")

# 3) CMVLP learning
#------------------------------------------------

    print("> Synchronizer multi-thread at target atom level")
    # Initialize a DMVLP with the dataset variables and set GULA as learning algorithm
    model = pylfit.models.CDMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm="synchronizer")

    for i in range(1, multiprocessing.cpu_count()+1):
        start = time.time()
        model.fit(dataset=dataset, verbose=0, threads=i)
        end = time.time()
        print(">> "+str(i)+" threads: "+str(round(end - start,2))+" s")


# 3) PMVLP learning
#------------------------------------------------

    print("> Probalizer multi-thread at target atom level")
    # Initialize a DMVLP with the dataset variables and set GULA as learning algorithm
    model = pylfit.models.PDMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm="gula")

    for i in range(1, multiprocessing.cpu_count()+1):
        start = time.time()
        model.fit(dataset=dataset, verbose=0, threads=i)
        end = time.time()
        print(">> "+str(i)+" threads: "+str(round(end - start,2))+" s")



    # Fit the DMVLP on the dataset
    # optional targets: model.fit(dataset=dataset, targets_to_learn={'p_t':["1"], 'r_t':["0"]})
    #model.fit(dataset=dataset, targets_to_learn=None)
    #model.summary()
