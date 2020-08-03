#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2019/05/06
# @updated: 2019/05/06
#
# @desc: example of the use of LFkT algorithm
#-------------------------------------------------------------------------------

import sys
sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')
sys.path.insert(0, 'src/semantics')

from itertools import islice

from utils import eprint
from logicProgram import LogicProgram
from lfkt import LFkT
from synchronous import Synchronous

# 1: Main
#------------
if __name__ == '__main__':

    # 0) Example from text file representing a logic program
    #--------------------------------------------------------
    eprint("Example using logic program definition file:")
    eprint("----------------------------------------------")

    # Delay are encoded as regular feature variables
    encoded_benchmark = LogicProgram.load_from_file("benchmarks/logic_programs/repressilator_delayed.lp")

    eprint("Original logic program with delay encoded in features variables: \n", encoded_benchmark.logic_form())

    time_serie_size = 5

    print("Generating decoded time series of size ", time_serie_size)

    raw_features = [("p", [0,1]),("q", [0,1]), ("r", [0,1])]
    raw_target = [("p_t", [0,1]),("q_t", [0,1]), ("r_t", [0,1])]

    #time_series = [[list(s[:len(raw_features)]),list(s[len(raw_features):])] for s in encoded_benchmark.states()]
    cut = len(raw_target)
    delay = 2
    time_series = [[list(s[cut*(d-1):cut*d]) for d in range(1,delay+1)] for s in encoded_benchmark.states()]

    for serie in time_series:
        while len(serie) < time_serie_size:
            serie.append(Synchronous.next(encoded_benchmark, serie[-2]+serie[-1])[0])

    #eprint(raw_series)

    eprint("LFkT input:")
    for s in time_series:
        eprint(s)

    model = LFkT.fit(time_series, raw_features, raw_target)

    eprint("LFkT output: \n", model.logic_form())

    expected = time_series
    predicted = [[list(s[:len(raw_features)]),list(s[len(raw_features):])] for s in model.states()]

    for serie in predicted:
        while len(serie) < time_serie_size:
            serie.append(Synchronous.next(model, serie[-2]+serie[-1])[0])

    eprint("Predicted: ")
    for s in predicted:
        eprint(s)

    missing = 0
    unexpected = 0

    for t in expected:
        if t not in predicted:
            eprint("Missing transition: ", t)
            missing += 1

    for t in predicted:
        if t not in expected:
            eprint("Unexpected transition: ", t)
            unexpected += 1

    eprint("total missing transitions: ", missing)
    eprint("total unexpected transitions: ", unexpected)

    if missing == 0 and unexpected == 0:
        eprint("SUCCESS: equivalent model learned")
    else:
        eprint("FAILURE: learned model predictions are wrong")

    serie = [[1,1,1], [0,0,0], [0,1,0]]
    next = Synchronous.next(model, serie[-2]+serie[-1])

    eprint("Next state of ", serie, " is ", next, " according to learned model")

    eprint("----------------------------------------------")
