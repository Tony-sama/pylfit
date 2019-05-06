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

from utils import eprint
from logicProgram import LogicProgram
from lfkt import LFkT

# 1: Main
#------------
if __name__ == '__main__':

    # 0) Example from text file representing a logic program
    #--------------------------------------------------------
    eprint("Example using logic program definition file:")
    eprint("----------------------------------------------")

    benchmark = LogicProgram.load_from_file("benchmarks/logic_programs/repressilator_delayed.lp")

    eprint("Original logic program: \n", benchmark.logic_form())

    time_serie_size = 10

    eprint("Generating time series of size ", time_serie_size)

    input = benchmark.generate_all_time_series(time_serie_size)

    eprint("LFkT input:")
    for s in input:
        eprint(s)

    model = LFkT.fit(benchmark.get_variables(), benchmark.get_values(), input)

    eprint("LFkT output: \n", model.logic_form())

    expected = benchmark.generate_all_time_series(time_serie_size)
    predicted = model.generate_all_time_series(time_serie_size)

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
    next = model.next_state(serie)

    eprint("Next state of ", serie, " is ", next, " according to learned model")

    eprint("----------------------------------------------")
