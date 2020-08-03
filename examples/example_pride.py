#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2019/05/06
# @updated: 2019/05/06
#
# @desc: example of the use of PRIDE algorithm
#-------------------------------------------------------------------------------

import sys
sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')
sys.path.insert(0, 'src/semantics')

from utils import eprint
from logicProgram import LogicProgram
from pride import PRIDE
from synchronous import Synchronous

# 1: Main
#------------
if __name__ == '__main__':

    # 0) Example from text file representing a logic program
    #--------------------------------------------------------
    eprint("Example using logic program definition file:")
    eprint("----------------------------------------------")

    benchmark = LogicProgram.load_from_file("benchmarks/logic_programs/repressilator.lp")

    eprint("Original logic program: \n", benchmark.logic_form())

    eprint("Generating transitions...")

    input = Synchronous.transitions(benchmark)

    eprint("PRIDE input: \n", input)

    model = PRIDE.fit(input, benchmark.get_features(), benchmark.get_targets())

    eprint("PRIDE output: \n", model.logic_form())

    expected = input
    predicted = Synchronous.transitions(model)

    precision = LogicProgram.precision(expected, predicted) * 100

    eprint("Model accuracy: ", precision, "%")

    state = [1,1,1]
    next = Synchronous.next(model, state)

    eprint("Next state of ", state, " is ", next, " according to learned model")

    eprint("----------------------------------------------")

    # 1) Example from csv file encoding transitions
    #--------------------------------------------------------
    eprint()
    eprint("Example using transition from csv file:")
    eprint("----------------------------------------------")

    input = PRIDE.load_input_from_csv("benchmarks/transitions/repressilator.csv")

    eprint("PRIDE input: \n", input)

    features = [["a_t_1",[0,1]], ["b_t_1",[0,1]], ["c_t_1",[0,1]]]
    targets = [["a_t",[0,1]], ["b_t",[1]], ["c_t",[]]] # Reduced domains of targets to only learn those

    model = PRIDE.fit(input, features, targets)

    eprint("PRIDE output: \n", model.logic_form())

    expected = input
    predicted = Synchronous.transitions(model)

    precision = LogicProgram.precision(expected, predicted) * 100

    eprint("Model accuracy: ", precision, "%")

    state = [1,1,1]
    next = Synchronous.next(model, state)

    eprint("Next state of ", state, " is ", next, " according to learned model")

    eprint("----------------------------------------------")
