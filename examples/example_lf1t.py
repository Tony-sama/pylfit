#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2019/05/06
# @updated: 2019/05/06
#
# @desc: example of the use of LF1T algorithm
#-------------------------------------------------------------------------------

import sys
sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')

from utils import eprint
from logicProgram import LogicProgram
from lf1t import LF1T

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

    input = benchmark.generate_all_transitions()

    eprint("LF1T input: \n", input)

    model = LF1T.fit(benchmark.get_variables(), benchmark.get_values(), input)

    eprint("LF1T output: \n", model.logic_form())

    expected = benchmark.generate_all_transitions()
    predicted = model.generate_all_transitions()

    precision = LogicProgram.precision(expected, predicted) * 100

    eprint("Model accuracy: ", precision, "%")

    state = [1,1,1]
    next = model.next(state)

    eprint("Next state of ", state, " is ", next, " according to learned model")

    eprint("----------------------------------------------")

    # 1) Example from csv file encoding transitions
    #--------------------------------------------------------
    eprint()
    eprint("Example using transition from csv file:")
    eprint("----------------------------------------------")

    input = LF1T.load_input_from_csv("benchmarks/transitions/repressilator.csv")

    eprint("LF1T input: \n", input)

    model = LF1T.fit(benchmark.get_variables(), benchmark.get_values(), input)

    eprint("LF1T output: \n", model.logic_form())

    expected = benchmark.generate_all_transitions()
    predicted = model.generate_all_transitions()

    precision = LogicProgram.precision(expected, predicted) * 100

    eprint("Model accuracy: ", precision, "%")

    state = [1,1,1]
    next = model.next(state)

    eprint("Next state of ", state, " is ", next, " according to learned model")

    eprint("----------------------------------------------")
