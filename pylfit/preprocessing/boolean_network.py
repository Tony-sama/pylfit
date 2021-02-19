""" pylfit boolean networks loading utilities """

from ..utils import eprint
from ..objects import Rule
from ..models import DMVLP

import pandas
import numpy
import itertools
import os

def dmvlp_from_boolean_network_file(file_path, compute_complementary_rules=False):
    """
    Factory methods that loads a Boolean Network from a formated text file with Boolnet format
        - Extract the list of variables of the system
        - Extract the corresponding boolean function and convert them as mvl rules

    Args: String
        File path to text file encoding a Boolean Network

    Returns: LogicProgram
        The logic program equivalent to the encoded Boolean Network in the input file
    """

    file_type = os.path.splitext(file_path)

    if file_type[1] not in [".bnet", ".net"]:
        raise ValueError("Unsuported file format", file_type[1])

    if file_type[1] == ".bnet":
        eprint("Converting '.bnet' file:", file_path)
        features, targets, rules = dmvlp_from_bnet_file(file_path)

    if file_type[1]== ".net":
        eprint("Converting '.net' file:", file_path)
        features, targets, rules = dmvlp_from_net_file(file_path)

    if compute_complementary_rules:
        # Group rules bodies (DNF)
        grouped_rules = {var_id: [] for var_id, var in enumerate(features)}
        for r in rules:
            grouped_rules[r.head_variable].append(r.body)

        #eprint(grouped_rules)
        #eprint()

        # Negation: inverse values (CNF)
        for var_id, bodies in grouped_rules.items():
            reversed_bodies = [ [(var,0) if val == 1 else (var,1) for var,val in body] for body in bodies]
            grouped_rules[var_id] = reversed_bodies

        #eprint(grouped_rules)
        #eprint()

        # CNF to DNF
        eprint("computing complementary rules...")
        for var_id, bodies in grouped_rules.items():
            #combinations = list(itertools.product(*bodies))

            nb_combinations = numpy.prod([len(l) for l in bodies])
            done = 0

            valid_bodies = []
            for body in itertools.product(*bodies):
                done += 1
                eprint("\r", var_id+1, "/", len(variables), " ",done,"/",nb_combinations, end='')
                # delete doublon
                body = set(body)

                # Only one value per variable
                valid = True
                for var,val in body:
                    if val == 0 and (var,1) in body:
                        valid = False
                        break
                    if val == 1 and (var,0) in body:
                        valid = False
                        break
                if valid:
                    valid_bodies.append(list(body))

            #eprint(variables[var_id]," valids: ", valid_bodies)

            new_rules = []
            for body in valid_bodies:
                new_rule = Rule(var_id, 0, len(variables), body)
                #eprint()
                #eprint(">>>", body)
                #eprint(">>>", new_rule)
                # Check subsumption of the new rule
                subsumed = False
                for r in new_rules:
                    #eprint(new_rule, " VS ", r)
                    #eprint("--", new_rule.body, " VS ", r.body)
                    if r.subsumes(new_rule):
                        subsumed = True
                        #eprint("SUBSUMED")
                        break
                    #eprint()

                if subsumed:
                    continue

                # Remove rules subsumed by new rule
                index=0
                while index < len(new_rules):
                    if new_rule.subsumes(new_rules[index]):
                        #eprint("SUBSUMES ", new_rules[index])
                        del new_rules[index]
                        continue
                    index+=1

                new_rules.append(new_rule)
            #eprint(">>", new_rules)

            rules += new_rules

    model = DMVLP(features, targets, rules)
    model.compile(algorithm="gula")

    return model


def dmvlp_from_bnet_file(file_path):
    # 1) Extract functions lines
    #----------------------------

    f = open(file_path,"r")

    functions_lines = []

    for line in f:

        line_list = line.strip()

        if len(line_list) == 0: # ignore empty lines
            continue
        if line_list[0] == "#": # ignore comment lines
            continue
        if len(line_list) > 7 and line_list[:7] == "targets": # ignore type line
            continue

        functions_lines.append(line_list)

    f.close()

    #eprint(functions_lines)

    # 2) Extract functions
    #----------------------------
    functions = []

    for line in functions_lines:
        # Extract variable
        beg = 0
        end = line.index(',')

        target = line[beg:end]

        #eprint(target)

        factors = line[end+1:]
        factors = factors.replace('(', '')
        factors = factors.replace(')', '')
        factors = factors.replace(' ','')
        factors = factors.replace('\t','')

        factors = factors.split('|')
        factors = [i.split("&") for i in factors]

        #eprint(factors)

        functions.append((target,factors))

    # 3) Convert DNF to mvl rules
    #-----------------------------

    variables = [target for target, factors in functions]
    rules = []
    head_var_id = 0

    for target, factors in functions:
        # Special function without clauses
        if len(factors) == 1:
            if factors == [["0"]]: # Function always false
                rules.append(Rule(head_var_id,0,len(variables)))
                head_var_id += 1
                continue
            if factors == [["1"]]: # Function always true
                rules.append(Rule(head_var_id,1,len(variables)))
                head_var_id += 1
                continue
        for clause in factors:
            #eprint(clause)
            rule = Rule(head_var_id,1,len(variables))
            for condition in clause:

                if condition[0] == '!':
                    var_id = variables.index(condition[1:])
                    val_id = 0
                else:
                    var_id = variables.index(condition[0:])
                    val_id = 1

                rule.add_condition(var_id, val_id)

            rules.append(rule)
            #eprint(rule)
            #eprint()
        head_var_id += 1

    #eprint(rules)

    #eprint(variables)
    features = [(var+"_t_1", ["0","1"]) for var in variables]
    targets = [(var+"_t", ["0","1"]) for var in variables]

    return features, targets, rules

def dmvlp_from_net_file(file_path):
    # 1) Extract functions lines
    #----------------------------

    f = open(file_path,"r")
    lines = [line for line in f]
    f.close()

    # 1) Extract number of variables
    #--------------------------------

    index = 0
    for line in lines:

        line = line.strip()

        if len(line) == 0 or line[0] == '#':
            index += 1
            continue

        line = line.split()

        if line[0] != ".v":
            raise ValueError("Expected .v number_of_vertices but got '", line, "'")

        nb_var = int(line[1])
        #eprint("Number of variables extracted: ", nb_var)
        break

    lines = lines[index+1:]
    index = 0

    # 2) Extract variables labels
    #--------------------------------

    variables = []

    for line in lines:

        line = line.strip()

        if len(line) == 0:
            index += 1
            continue

        if '=' in line:
            tokens = line.split()
            variable = tokens[-1].replace(",", "_")
            if "," in tokens[-1]:
                eprint("WARNING: "+tokens[-1]+" renamed "+variable+" for compatability with logic format")
            variables.append(variable)
            index += 1

        if len(variables) == nb_var:
            break

    #eprint(variables)

    lines = lines[index:]
    index = 0

    # 3) Extract Boolean functions
    #--------------------------------

    rules = []

    head_var_id = 0
    for var in variables:
        regulators = []

        # Go to next variable function
        for line in lines:

            line = line.strip()

            # Ignore empty and comment lines
            if len(line) == 0 or line[0] == '#':
                index += 1
                continue

            tokens = line.split()

            if tokens[0] != ".n":
                raise ValueError("Syntax error: expected .n got ", tokens[0])

            # Extracts regulators
            var_id = int(tokens[1])-1
            nb_regulators = int(tokens[2])
            regulators = tokens[3:]
            regulators = [int(i) for i in regulators]

            if var_id != head_var_id:
                raise ValueError("Syntax error: expected var id ',head_var_id,' (ordered) got ", var_id)

            if len(regulators) != nb_regulators:
                raise ValueError("Syntax error: expected ',nb_regulators,' regulators got ", regulators)

            #eprint(regulators)
            break

        lines = lines[index+1:]
        index = 0

        # Extracts DNF clauses
        for line in lines:
            line = line.strip()

            # end of function
            if len(line) == 0:
                index += 1
                break

            head_val = int(line[-1])
            rule = Rule(head_var_id,head_val,len(variables))

            for reg_id in range(0,len(regulators)):
                if line[reg_id] == '0':
                    rule.add_condition(regulators[reg_id]-1,0)
                if line[reg_id] == '1':
                    rule.add_condition(regulators[reg_id]-1,1)

            #eprint(rule)
            rules.append(rule)

            index += 1

        lines = lines[index:]
        index = 0

        head_var_id += 1

    #values = [[0,1] for var in variables]
    features = [(var+"_t_1", ["0","1"]) for var in variables]
    targets = [(var+"_t", ["0","1"]) for var in variables]

    return features, targets, rules
