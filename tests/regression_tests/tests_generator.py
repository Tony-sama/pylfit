#-----------------------
# @author: Tony Ribeiro
# @created: 2019/04/15
# @updated: 2021/06/15
#
# @desc: regression tests generator script.
# Provide random factory function for pylfit regression tests.
#   Random generators:
#   - StateTransitionsDataset
#   - Rule
#
#-----------------------

import random
import numpy

from pylfit.objects import Rule
from pylfit.datasets import StateTransitionsDataset
from pylfit.models import DMVLP, CDMVLP, WDMVLP

def random_rule(nb_features, nb_targets, nb_values, max_body_size):
    head_var = random.randint(0,nb_targets-1)
    head_val = random.randint(0,nb_values-1)
    body = []
    conditions = []
    nb_conditions = random.randint(0,max_body_size)
    while len(body) < nb_conditions:
        var = random.randint(0,nb_features-1)
        val = random.randint(0,nb_values-1)
        if var not in conditions:
            body.append( (var, val) )
            conditions.append(var)
    r = Rule(head_var,head_val,nb_features,body)

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

def random_StateTransitionsDataset(nb_transitions, nb_features, nb_targets, max_feature_values, max_target_values):
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

    return StateTransitionsDataset(data, features, targets)

def random_symmetric_StateTransitionsDataset(nb_transitions, nb_variables, max_variable_values):
    features = [("v_"+str(i)+"_t_1", ["val_"+str(val) for val in range(0,random.randint(1,max_variable_values))]) for i in range(nb_variables)]
    targets = [("y_"+str(i)+"_t", vals) for i,(var,vals) in enumerate(features)]

    data = []

    for i in range(nb_transitions):
        s1 = [random.choice(vals) for (var, vals) in features]
        s2 = [random.choice(vals) for (var, vals) in targets]
        data.append( (s1,s2) )

    return StateTransitionsDataset(data, features, targets)

def random_DMVLP(nb_features, nb_targets, max_feature_values, max_target_values,algorithm):
    dataset = random_StateTransitionsDataset(100,nb_features, nb_targets, max_feature_values, max_target_values)

    model = DMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm=algorithm)
    model.fit(dataset=dataset)

    return model

def random_CDMVLP(nb_features, nb_targets, max_feature_values, max_target_values,algorithm):
    dataset = random_StateTransitionsDataset(100,nb_features, nb_targets, max_feature_values, max_target_values)

    model = CDMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm=algorithm)
    model.fit(dataset=dataset)

    return model

def random_WDMVLP(nb_features, nb_targets, max_feature_values, max_target_values,algorithm):
    dataset = random_StateTransitionsDataset(100,nb_features, nb_targets, max_feature_values, max_target_values)

    model = WDMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm=algorithm)
    model.fit(dataset=dataset)

    return model
