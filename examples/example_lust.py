import sys
sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')
sys.path.insert(0, 'src/semantics')

from utils import eprint
from logicProgram import LogicProgram
from gula import GULA
from lust import LUST
from synchronous import Synchronous

# 1: Main
#------------
if __name__ == '__main__':

    eprint("Example with disjonctive Boolean Network")

    features = [("a",[0,1]), ("b",[0,1]), ("c",[0,1])]
    targets = [("a_t",[0,1]), ("b_t",[0,1]), ("c_t",[0,1])]

    transitions = GULA.load_input_from_csv("benchmarks/transitions/disjonctive_boolean_network.csv", len(features))

    eprint("transitions: \n", transitions)


    eprint("features: ", features)
    eprint("targets: ", targets)

    programs = LUST.fit(transitions, features, targets)

    eprint()
    eprint("Model learned: ")
    for p in programs:
        eprint(p.logic_form())

    predicted = []

    for p in programs:
        predicted += Synchronous.transitions(p)

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
