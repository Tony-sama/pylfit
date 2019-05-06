#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2019/05/06
# @updated: 2019/05/06
#
# @desc: example of the use of ACEDIA algorithm
#-------------------------------------------------------------------------------

import sys
sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')

import random

from utils import eprint
from continuum import Continuum
from continuumLogicProgram import ContinuumLogicProgram
from acedia import ACEDIA

# 1: Main
#------------
if __name__ == '__main__':

    # 0) Example from text file representing a logic program
    #--------------------------------------------------------
    eprint("Example using random logic program:")
    eprint("----------------------------------------------")

    variables = ["a", "b", "c"]
    domains = [ Continuum(0.0,1.0,True,True) for v in variables ]
    rule_min_size = 0
    rule_max_size = 3
    epsilon = 0.5
    random.seed(9999)

    benchmark = ContinuumLogicProgram.random(variables, domains, rule_min_size, rule_max_size, epsilon, delay=1)

    eprint("Original logic program: \n", benchmark.logic_form())

    eprint("Generating transitions...")

    input = benchmark.generate_all_transitions(epsilon)

    eprint("ACEDIA input: \n", input)

    model = ACEDIA.fit(benchmark.get_variables(), benchmark.get_domains(), input)

    eprint("ACEDIA output: \n", model.logic_form())

    expected = benchmark.generate_all_transitions(epsilon)
    predicted = [(s1,model.next(s1)) for s1,s2 in expected]

    precision = ContinuumLogicProgram.precision(expected, predicted) * 100

    eprint("Model accuracy: ", precision, "%")

    state = [0.75,0.33,0.58]
    next = model.next(state)

    eprint("Next state of ", state, " is ", next, " according to learned model")

    eprint("----------------------------------------------")

    # 1) Example from csv file encoding transitions
    #--------------------------------------------------------
    eprint()
    eprint("Example using transition from csv file:")
    eprint("----------------------------------------------")

    input = ACEDIA.load_input_from_csv("benchmarks/transitions/repressilator_continuous.csv")

    eprint("ACEDIA input: \n", input)

    variables = ["p", "q", "r"]
    domains = [ Continuum(0.0,1.0,True,True) for v in variables ]
    model = ACEDIA.fit(variables, domains, input)

    eprint("ACEDIA output: \n", model.logic_form())

    expected = input
    predicted = [(s1,model.next(s1)) for s1,s2 in expected]

    precision = ContinuumLogicProgram.precision(expected, predicted) * 100

    eprint("Model accuracy: ", precision, "%")

    state = [0.75,0.33,0.58]
    next = model.next(state)

    eprint("Next state of ", state, " is ", next, " according to learned model")

    eprint("----------------------------------------------")
