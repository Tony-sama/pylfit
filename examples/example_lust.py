import sys
sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')

from utils import eprint
from logicProgram import LogicProgram
from gula import GULA
from lust import LUST

# 1: Main
#------------
if __name__ == '__main__':

    eprint("Example with disjonctive Boolean Network")

    transitions = LUST.load_input_from_csv("benchmarks/transitions/disjonctive_boolean_network.csv")

    eprint("transitions: \n", transitions)

    variables = ["p","q","r"]
    values = [ [0,1] for i in variables ]

    eprint("variables: ", variables)
    eprint("values: ", values)

    programs = LUST.fit(variables, values, transitions)

    eprint()
    eprint("Model learned: ")
    for p in programs:
        eprint(p.logic_form())

    predicted = []

    for p in programs:
        predicted += p.generate_all_transitions()

    missing = 0
    unexpected = 0

    for t in transitions:
        if t not in predicted:
            eprint("Missing transition: ", t)
            missing += 1

    for t in predicted:
        if t not in transitions:
            eprint("Unexpected transition: ", t)
            unexpected += 1

    eprint("total missing transitions: ", missing)
    eprint("total unexpected transitions: ", unexpected)

    if missing == 0 and unexpected == 0:
        eprint("SUCCESS: equivalent model learned")
    else:
        eprint("FAILURE: learned model predictions are wrong")
