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

import sys
sys.path.insert(0, 'src/')
sys.path.insert(0, 'src/algorithms')
sys.path.insert(0, 'src/objects')
sys.path.insert(0, 'src/semantics')

from utils import eprint
from gula import GULA
from synchronizer import Synchronizer
from logicProgram import LogicProgram
from rule import Rule
from synchronous import Synchronous
from asynchronous import Asynchronous
from general import General

# Constants
#------------
MAX_NB_CONSTRAINTS = 100

def random_constraint(variables, values, body_size):
    var = -1
    val = -1
    body = []
    conditions = []

    for j in range(0, random.randint(0,body_size)):
        var = random.randint(0,len(variables)*2-1)
        val = random.choice(values[var%len(variables)])
        if var not in conditions:
            body.append( (var, val) )
            conditions.append(var)

    return  Rule(var,val,len(variables)*2,body)

def random_program(variables, values, max_body_size, generate_constraints):
    rules = []
    constraints = []

    features = [var, [val for val in range(0,values[var])]) for var in variables)]
    targets = [var, [val for val in range(0,random.randint(2,nb_values))]) for i in range(random.randint(1,nb_targets))]


    rules = LogicProgram.random(features, targets, 0, max_body_size).get_rules()

    if generate_constraints:
        for j in range(random.randint(0,MAX_NB_CONSTRAINTS)):
            r = random_constraint(variables, values, max_body_size)
            # Force constraint to have condition at t
            var = random.randint(0, len(variables)-1)
            val = random.randint(0, len(values[var])-1)
            r.add_condition(var, val)
            constraints.append(r)

    return LogicProgram(variables, values, rules, constraints)

def test_random_LP(algorithm, nb_variables, nb_values, semantics, run_tests, max_body_size=None, generate_constraints=False):

    results_time = []
    results_common = []
    results_missing = []
    results_over = []
    results_precision = []

    for run in range(run_tests):

        variables = ["x"+str(i) for i in range(nb_variables)]
        values = []

        for var in range(len(variables)):
            values.append([val for val in range(nb_values)])

        min_body_size = 0
        max_body_size = len(variables)
        if not max_body_size is None:
            max_body_size = min(max_body_size,len(variables))

        if max_body_size is None:
            max_body_size = len(variables)

        #p = LogicProgram.random(variables, values, min_body_size, max_body_size)
        full_transitions = []
        while len(full_transitions) <= 0:
            p = random_program(variables, values, max_body_size, generate_constraints)
            default = [[0] for v in p.get_variables()]
            full_transitions = semantics.transitions(p,default)

        #eprint(p.logic_form())

        random.shuffle(full_transitions)
        train = full_transitions
        test = []

        #eprint(train)

        if run == 0:
            eprint(">>> Start Training on ", len(train), "/", len(full_transitions), " transitions (", round(100 * len(train) / len(full_transitions), 2), "%)")

        eprint("\r>>> run: ", run+1 ,"/", run_tests, end='')
        start = time.time()
        model = algorithm.fit(full_transitions, p.get_features(), p.get_targets())
        end = time.time()
        results_time.append(round(end - start,3))

        common, missing, over = p.compare(model)

        #eprint(">>> Original:")
        #eprint(P.to_string())

        #eprint(">>> Learned:")
        #eprint(model.to_string())

        #eprint(">>> Logic Program comparaison:")
        #eprint(">>>> Common: "+str(len(common))+"/"+str(len(P.get_rules()))+"("+str(round(100 * len(common) / len(P.get_rules()),2))+"%)")
        #eprint(">>>> Missing: "+str(len(missing))+"/"+str(len(P.get_rules()))+"("+str(round(100 * len(missing) / len(P.get_rules()),2))+"%)")
        #eprint(">>>> Over: "+str(len(over))+"/"+str(len(P.get_rules()))+"("+str(round(100 * len(over) / len(model.get_rules()),2))+"%)")

        results_common.append(len(common))
        results_missing.append(len(missing))
        results_over.append(len(over))

        if len(test) == 0:
            test = train

        #pred = [(s[:-1], model.next_state(s[:-1])) for s in test]
        #test = [(s[:-1], s[-1]) for s in test]
        #precision = round(LogicProgram.precision(test,pred),2)

        test = set([ (tuple(s1),tuple(s2)) for s1,s2 in test])
        pred = set([ (tuple(s1), tuple(s2)) for s1,s2_ in test for s2 in semantics.next(model,s1,default) ])

        over_pred = pred - test
        missing_pred = test - pred

        #eprint(pred)
        #eprint(test)
        #eprint(over_pred)
        #eprint(missing_pred)

        precision = 1.0 - (len(over_pred) / len(pred)) - (len(missing_pred) / len(test))


        #eprint(test)
        #eprint(pred)

        #eprint(">>> Prediction precision")
        #eprint(">>>> " + str(round(precision * 100,2)) + "%")

        results_precision.append(precision)

    run_time = round(sum(results_time) / run_tests, 3)
    common = sum(results_common) / run_tests
    missing = sum(results_missing) / run_tests
    over = sum(results_over) / run_tests
    precision = sum(results_precision) / run_tests

    eprint()
    eprint(">>> Run time: "+str(run_time)+"s")
    eprint(">>> Logic Program comparaison:")
    eprint(">>>> AVG Common: "+str(common)+"/"+str(len(p.get_rules()))+"("+str(round(100 * common / len(p.get_rules()),2))+"%)")
    eprint(">>>> AVG Missing: "+str(missing)+"/"+str(len(p.get_rules()))+"("+str(round(100 * missing / len(p.get_rules()),2))+"%)")
    eprint(">>>> AVG Over: "+str(over)+"/"+str(len(p.get_rules()))+"("+str(round(100 * over / len(model.get_rules()),2))+"%)")

    eprint(">>> Prediction precision")
    eprint(">>>> AVG accuracy: " + str(round(precision * 100,2)) + "%")

    return run_time

'''

'''
if __name__ == '__main__':

    run_tests = 1
    algorithm = GULA
    max_body_size = None
    generate_constraints = False
    #min_var = 1
    #max_var = 15

    if len(sys.argv) < 6 or (sys.argv[1] != "GULA" and sys.argv[1] != "Synchronizer"):
        eprint("Please give the experiement to perform as parameter: GULA or Synchronizer, min_var, max_var, max_var_general, run_tests, max_body_size*")
        exit()

    if sys.argv[1] == "Synchronizer":
        algorithm = Synchronizer
        generate_constraints = True

    min_var = int(sys.argv[2])
    max_var = int(sys.argv[3])
    max_var_general = int(sys.argv[4])
    run_tests = int(sys.argv[5])
    if len(sys.argv) >= 7:
        max_body_size = int(sys.argv[6])

    # 1) Random logic program, full transitions
    #-------------------------------------------
    semantics_classes = [("synchronous", Synchronous), ("asynchronous", Asynchronous), ("general", General)]
    run_times = {"synchronous":[], "asynchronous":[], "general":[]}

    # Random dmvlp
    #--------------
    eprint("> Start random logic program evaluation: perfect test")

    for semantics_name, semantics_class in semantics_classes:
        for nb_var in range(min_var,max_var+1):
            if semantics_class == General and nb_var > max_var_general:
                run_times["general"].append("")
                continue
            var = nb_var
            val = 3
            eprint(">>> variables: ", var)
            eprint(">>> values: ", val)
            run_time = test_random_LP(algorithm, var, val, semantics_class, run_tests, max_body_size,generate_constraints)

            run_times[semantics_name].append(round(run_time,3))

    eprint(run_times)

    output = []
    nb_var = min_var
    for i in range(0,max_var-min_var+1):
        output.append(str(nb_var) + "," + str(run_times["synchronous"][i]) + "," + str(run_times["asynchronous"][i]) + "," + str(run_times["general"][i]) )
        nb_var += 1

    print("nb_var,synchronous,asynchronous,general")
    for line in output:
        print(line)
