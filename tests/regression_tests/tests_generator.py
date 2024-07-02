#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/15
# @updated: 2023/12/20
#
# @desc: regression tests generator script.
# Provide random factory function for pylfit regression tests.
#   Random generators:
#   - DiscreteStateTransitionsDataset
#   - Rule
#
#-----------------------

import random
import numpy

from pylfit.objects import Rule, Continuum, ContinuumRule
from pylfit.datasets import DiscreteStateTransitionsDataset, ContinuousStateTransitionsDataset
from pylfit.models import DMVLP, CDMVLP, WDMVLP, PDMVLP, CLP
from pylfit.objects.legacyAtom import LegacyAtom

#--------------
# DMVLP
#--------------

def random_legacy_atom(nb_variables, nb_values):
    var = str(random.randint(0,nb_variables-1))
    if nb_values <= 2:
        dom = ["0","1"]
    else:
        dom = [str(i) for i in range(random.randint(2,nb_values))]
    val = random.choice(dom)
    pos = random.randint(0,nb_variables-1)
    return LegacyAtom(var, dom, val, pos)

def random_rule(nb_features, nb_targets, nb_values, max_body_size):
    head = random_legacy_atom(nb_targets, nb_values)
    body = {}
    nb_conditions = random.randint(0,max_body_size)
    while len(body) < nb_conditions:
        atom = random_legacy_atom(nb_features, nb_values)
        valid = True
        for var in body:
            if body[var].state_position == atom.state_position:
                valid = False
                break
        if valid:
            body[atom.variable] = atom
        
    r = Rule(head,body)

    return r

def random_constraint(nb_features, nb_targets, nb_values, max_body_size):
    head_var = -1
    head_val = -1
    body = []
    conditions = []
    nb_conditions = random.randint(0,max_body_size)
    while len(body) < nb_conditions:
        var = random.randint(0,nb_features+nb_targets-1)
        val = random.randint(0,nb_values-1)
        if var not in conditions:
            body.append( (var, val) )
            conditions.append(var)
    r = Rule(head_var,head_val,nb_features+nb_targets,body)

    return r

def random_features(nb_features,max_feature_values):
    return [("x"+str(i), ["val_"+str(val) for val in range(0,random.randint(1,max_feature_values))]) for i in range(nb_features)]

def random_targets(nb_targets,max_targets_values):
    return [("y"+str(i), ["val_"+str(val) for val in range(0,random.randint(1,max_targets_values))]) for i in range(nb_targets)]

def random_DiscreteStateTransitionsDataset(nb_transitions, nb_features, nb_targets, max_feature_values, max_target_values):
    features = [("x"+str(i), ["val_"+str(val) for val in range(0,random.randint(1,max_feature_values))]) for i in range(nb_features)]
    targets = [("y"+str(i), ["val_"+str(val) for val in range(0,random.randint(1,max_target_values))]) for i in range(nb_targets)]

    data = []

    for i in range(nb_transitions):
        s1 = [random.choice(vals) for (var, vals) in features]
        s2 = [random.choice(vals) for (var, vals) in targets]
        data.append( (s1,s2) )
        if random.choice([True,False]):
            s3 = [random.choice(vals) for (var, vals) in targets]
            data.append( (s1,s3) )
            i+=1

    return DiscreteStateTransitionsDataset(data, features, targets)

def random_symmetric_DiscreteStateTransitionsDataset(nb_transitions, nb_variables, max_variable_values):
    features = [("v_"+str(i)+"_t_1", ["val_"+str(val) for val in range(0,random.randint(1,max_variable_values))]) for i in range(nb_variables)]
    targets = [("y_"+str(i)+"_t", vals) for i,(var,vals) in enumerate(features)]

    data = []

    for i in range(nb_transitions):
        s1 = [random.choice(vals) for (var, vals) in features]
        s2 = [random.choice(vals) for (var, vals) in targets]
        data.append( (s1,s2) )

    return DiscreteStateTransitionsDataset(data, features, targets)

def random_DMVLP(nb_features, nb_targets, max_feature_values, max_target_values,algorithm):
    dataset = random_DiscreteStateTransitionsDataset(100,nb_features, nb_targets, max_feature_values, max_target_values)

    model = DMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm=algorithm)
    model.fit(dataset=dataset)

    return model

def random_CDMVLP(nb_features, nb_targets, max_feature_values, max_target_values,algorithm):
    dataset = random_DiscreteStateTransitionsDataset(100,nb_features, nb_targets, max_feature_values, max_target_values)

    model = CDMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm=algorithm)
    model.fit(dataset=dataset)

    return model

def random_WDMVLP(nb_features, nb_targets, max_feature_values, max_target_values,algorithm):
    dataset = random_DiscreteStateTransitionsDataset(100,nb_features, nb_targets, max_feature_values, max_target_values)

    model = WDMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm=algorithm)
    model.fit(dataset=dataset)

    return model

def random_PDMVLP(nb_features, nb_targets, max_feature_values, max_target_values,algorithm):
    dataset = random_DiscreteStateTransitionsDataset(100,nb_features, nb_targets, max_feature_values, max_target_values)

    model = PDMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm=algorithm)
    model.fit(dataset=dataset)

    return model

#---------
# CLP
#---------

def random_Continuum(min_value, max_value, min_size=0):
    # Invalid interval
    if min_value > max_value:
        raise ValueError("Continuum min value must be <= max value")

    if min_size < 0 or min_size > (max_value - min_value):
        raise ValueError("expected 0 <= min_size < (max_value - min_value)")

    min = random.uniform(min_value,max_value-min_size)
    max = random.uniform(min+min_size, max_value)

    min_included = random.choice([True, False])
    max_included = random.choice([True, False])

    return Continuum(min, max, min_included, max_included)

def random_ContinuumRule(features, targets, min_continuum_size):
    head_variable = random.randint(0,len(targets)-1)
    head_value = random_Continuum(targets[head_variable][1].min_value, targets[head_variable][1].max_value, min_continuum_size)
    size = random.randint(0, len(features))

    locked = []

    r = ContinuumRule(head_variable, head_value)

    while r.size() < size:
        var = random.randint(0, len(features)-1)
        val = random_Continuum(features[var][1].min_value, features[var][1].max_value, min_continuum_size)

        if var not in locked:
            r.set_condition(var,val)
            locked.append(var)

    return r

def random_ContinuousStateTransitionsDataset(nb_transitions, nb_features, nb_targets, min_value, max_value, min_continuum_size):
    features = [("x"+str(i), random_Continuum(min_value,max_value,min_continuum_size)) for i in range(nb_features)]
    targets = [("y"+str(i), random_Continuum(min_value,max_value,min_continuum_size)) for i in range(nb_targets)]

    data = []

    for i in range(nb_transitions):
        s1 = [random.uniform(val.min_value, val.max_value) for (var, val) in features]
        s2 = [random.uniform(val.min_value, val.max_value) for (var,val) in targets]
        data.append( (s1,s2) )
        if random.choice([True,False]):
            s3 = [random.uniform(val.min_value, val.max_value) for (var,val) in targets]
            data.append( (s1,s3) )
            i+=1

    return ContinuousStateTransitionsDataset(data, features, targets)

def random_CLP(nb_features, nb_targets, algorithm):
    dataset = random_ContinuousStateTransitionsDataset(0, nb_features, nb_targets, -100, 100, 1)

    rules = []
    for i in range(random.randint(0,10)):
        rules.append(random_ContinuumRule(dataset.features, dataset.targets, 1))

    model = CLP(features=dataset.features, targets=dataset.targets, rules=rules)
    model.compile(algorithm=algorithm)
    #model.fit(dataset=dataset)

    return model

def random_continuous_state(variables):
    return [random.uniform(variables[var][1].min_value,variables[var][1].max_value) for var in range(len(variables))]


#----------
# Unknowns
#----------

def random_unknown_values_dataset(data):
    output = []
    for (i,j) in data:
        replace_count = random.randint(0, len(i))
        i_ = i.copy()
        if replace_count > 0:
            i_.flat[numpy.random.choice(len(i), replace_count, replace=False)] = LegacyAtom._UNKNOWN_VALUE

        replace_count = random.randint(0, len(j))
        j_ = j.copy()
        if replace_count > 0:
            j_.flat[numpy.random.choice(len(j), replace_count, replace=False)] = LegacyAtom._UNKNOWN_VALUE
        output.append((i_,j_))
    return output