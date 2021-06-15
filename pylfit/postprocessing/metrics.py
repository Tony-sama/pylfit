#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2021/03/03
# @updated: 2021/06/15
#
# @desc: pylfit metrics functions
#-------------------------------------------------------------------------------

import pylfit

from pylfit.utils import eprint

def accuracy_score(model, dataset):
    """
    Args:
        model: WDMVLP
            The model whose prediction will be scored.
        dataset: StateTransitionsDataset
            The state transitions to be predicted
    Returns:
        The prediction accuracy of the model over the possibility of occurence of each target variable value for each feature states of the dataset.
    """
    # Empty dataset
    if len(dataset.data) == 0:
        raise ValueError("Requesting accuracy score over empty dataset")

    init_states = [list(s) for s in set(tuple(s1) for s1,s2 in dataset.data)]
    predictions = model.predict(init_states)
    predictions = {s1: {variable: {value: item[variable][value][0] for value in values} for (variable, values) in model.targets} for s1, item in predictions.items()}
    return accuracy_score_from_predictions(predictions, dataset)

def accuracy_score_from_predictions(predictions, dataset):
    """
    Args:
        prediction: dict {tuple of string: {string: {string: float}}}
            Predictions of probability of each target variable value occurence for each feature_state of the dataset,
            i.e. dict {feature_state: {target_variable: {domain_value: probability}}}, with 0.0 <= probability <= 1.0.
        dataset: StateTransitionsDataset
            The state transitions to be predicted
    Returns:
        The average prediction accuracy over the possibility of occurence of each target variable value for each feature states of the dataset.
    """
    # Empty dataset
    if len(dataset.data) == 0:
        raise ValueError("Requesting accuracy score over empty dataset")

    # Domains are not consistent
    dataset_init_states = set(tuple(s1) for s1,s2 in dataset.data)
    dataset_domain = set((var,val) for var, values in dataset.targets for val in values )

    for s in dataset_init_states:
        if s not in predictions:
            raise ValueError("Missing dataset initial state in given predictions")

    for s in predictions:
        if s not in dataset_init_states:
            raise ValueError("Predicted state not in dataset")
        predictions_domain = set((var,val) for var, values in predictions[s].items() for val, _ in values.items())
        if dataset_domain != predictions_domain:
            raise ValueError("Predictions and dataset targets domains are different")


    grouped_transitions = {tuple(s1) : set(tuple(s2_) for s1_,s2_ in dataset.data if tuple(s1) == tuple(s1_)) for s1,s2 in dataset.data}

    #eprint(grouped_transitions)

    #eprint("Extracting expected values")

    # expected output: kinda one-hot encoding of values occurences
    expected = {}
    count = 0
    for s1, successors in grouped_transitions.items():
        count += 1
        occurs = {}
        for var_id, (var, values) in enumerate(dataset.targets):
            for val_id, val in enumerate(values):
                occurs[(var,val)] = 0.0
                for s2 in successors:
                    if s2[var_id] == val:
                        occurs[(var,val)] = 1.0
                        break
        expected[s1] = occurs

    #eprint("Expected: ", expected)
    #eprint("Computing predicted values")

    # predictions
    predicted = {}
    count = 0
    for s1, successors in grouped_transitions.items():
        count += 1
        #eprint("\r",count,"/",len(grouped_transitions.items()), end='')
        occurs = {}
        for var_id, (var,values) in enumerate(dataset.targets):
            for val_id, val in enumerate(values):
                occurs[(var,val)] = predictions[s1][var][val]

        predicted[s1] = occurs

    #eprint("Prediction: ", predicted)

    #eprint("\nComputing accuracy score")

    # compute average accuracy
    global_error = 0
    for s1, actual in expected.items():
        state_error = 0
        for var, values in dataset.targets:
            for val in values:
                forecast = predicted[s1]
                state_error += abs(actual[(var,val)] - forecast[(var,val)])

        global_error += state_error / len(actual.items())

    global_error = global_error / len(expected.items())

    accuracy = 1.0 - global_error

    #eprint("AVG accuracy: " + str(round(accuracy * 100,2)) + "%")

    return accuracy


def explanation_score(model, expected_model, dataset):
    """
    Args:
        model: WDMVLP
            The model whose prediction explanation over feature_states will be scored.
        expected_model: WDMVLP
            The model with the expected rules to be used to explain prediction from feature_states.
        dataset: StateTransitionsDataset
            The transitions to explain.
    """
    if len(dataset.data) == 0:
        raise ValueError("Requesting explanation score with an empty dataset")

    if model.targets != expected_model.targets or model.targets != dataset.targets:
        raise ValueError("Arguments targets domains are differents.")

    grouped_transitions = {tuple(s1) : set(tuple(s2_) for s1_,s2_ in dataset.data if tuple(s1) == tuple(s1_)) for s1,s2 in dataset.data}

    # expected output: kinda one-hot encoding of values occurences
    expected = {}
    count = 0
    for s1, successors in grouped_transitions.items():
        count += 1
        occurs = {}
        for var_id, (var, values) in enumerate(model.targets):
            for val_id, val in enumerate(values):
                occurs[(var,val)] = 0.0
                for s2 in successors:
                    if s2[var_id] == val:
                        occurs[(var,val)] = 1.0
                        break
        expected[s1] = occurs

    sum_explanation_score = 0.0
    for feature_state, actual in expected.items():
        #eprint("Feature state: ", feature_state)

        prediction = model.predict(feature_states=[list(feature_state)], raw_rules=True)
        prediction = prediction[feature_state]
        #eprint(">> prediction: ", prediction)
        #eprint(">> actual: ", actual)

        sum_score = 0.0
        nb_targets = 0
        for var_id, (variable, values) in enumerate(model.targets):
            #eprint(" "+variable+": ")
            for val_id, (value, (proba, (_, r1), (_, r2))) in enumerate(prediction[variable].items()):
                #eprint(" "+value+" "+str(round(proba*100.0,2))+"%")

                # No decision or bad prediction implies wrong explanation
                if proba == 0.5 or (proba > 0.5 and actual[(variable,value)] == 0.0) or (proba < 0.5 and actual[(variable,value)] == 1.0):
                    score = 0.0
                    sum_score += score
                    nb_targets += 1
                    continue

                encoded_feature_state = pylfit.algorithms.GULA.encode_state(feature_state, model.features)

                # Predicted likely
                if proba > 0.5:
                    expected_rules = [r for (w,r) in expected_model.rules \
                    if r.head_variable == var_id and r.head_value == val_id and r.matches(encoded_feature_state)]
                    explanation_rule = r1

                # Predicted unlikely
                if proba < 0.5:
                    expected_rules = [r for (w,r) in expected_model.unlikeliness_rules \
                    if r.head_variable == var_id and r.head_value == val_id and r.matches(encoded_feature_state)]
                    explanation_rule = r2

                min_distance = len(model.features)
                nearest_expected = None
                for r in expected_rules:
                    distance = pylfit.postprocessing.hamming_distance(explanation_rule,r)
                    if distance <= min_distance:
                        min_distance = distance
                        nearest_expected = r

                score = 1.0 - (min_distance / len(model.features))

                #eprint(explanation_type + " explanation evaluation")
                #eprint("Explanation rule: " + explanation)
                #eprint("Explanation score: ", end='')
                #eprint(str(round(score, 2)) + " (nearest expected " + explanation_type + " rule: " + nearest_expected.logic_form(model.features, model.targets) + " distance: " + str(min_distance) + ")")
                sum_score += score
                nb_targets += 1
        sum_explanation_score += sum_score / nb_targets

    return sum_explanation_score / len(expected)

def explanation_score_from_predictions(predictions, expected_model, dataset):
    """
    Args:
        prediction: dict {tuple of string: {string: {string: (float,(float,Rule),(float,Rule))}}}
            Predictions of probability and explanation of each target variable value occurence for each feature_state of the dataset,
            i.e. dict {feature_state: {target_variable: {domain_value: (probability, (likeliness_weight, likliness_rule), unlikeliness_weight, unlikeliness_rule))}}}, with 0.0 <= probability <= 1.0.
        expected_model: WDMVLP
            The model with the expected rules to be used to explain prediction from feature_states.
        dataset: StateTransitionsDataset
            The transitions to explain.
    """
    if len(dataset.data) == 0:
        raise ValueError("Requesting explanation score with an empty dataset")

    # Domains are not consistent
    dataset_init_states = set(tuple(s1) for s1,s2 in dataset.data)
    dataset_domain = set((var,val) for var, values in dataset.targets for val in values )

    for s in dataset_init_states:
        if s not in predictions:
            raise ValueError("Missing dataset initial state in given predictions")

    for s in predictions:
        if s not in dataset_init_states:
            raise ValueError("Predicted state not in dataset")
        predictions_domain = set((var,val) for var, values in predictions[s].items() for val, _ in values.items())
        if dataset_domain != predictions_domain:
            raise ValueError("Predictions and dataset targets domains are different")

    # TODO: check feature states are same

    grouped_transitions = {tuple(s1) : set(tuple(s2_) for s1_,s2_ in dataset.data if tuple(s1) == tuple(s1_)) for s1,s2 in dataset.data}

    # expected output: kinda one-hot encoding of values occurences
    expected = {}
    count = 0
    for s1, successors in grouped_transitions.items():
        count += 1
        occurs = {}
        for var_id, (var, values) in enumerate(dataset.targets):
            for val_id, val in enumerate(values):
                occurs[(var,val)] = 0.0
                for s2 in successors:
                    if s2[var_id] == val:
                        occurs[(var,val)] = 1.0
                        break
        expected[s1] = occurs

    sum_explanation_score = 0.0
    for feature_state, actual in expected.items():
        #eprint("Feature state: ", feature_state)

        prediction = predictions[feature_state]
        #eprint(">> prediction: ", prediction)
        #eprint(">> actual: ", actual)

        sum_score = 0.0
        nb_targets = 0
        for var_id, (variable, values) in enumerate(dataset.targets):
            #eprint(" "+variable+": ")
            for val_id, (value, (proba, (w1, r1), (w2, r2))) in enumerate(prediction[variable].items()):
                #eprint(" "+value+" "+str(round(proba*100.0,2))+"%")

                # No decision or bad prediction implies wrong explanation
                if proba == 0.5 or (proba > 0.5 and actual[(variable,value)] == 0.0) or (proba < 0.5 and actual[(variable,value)] == 1.0):
                    score = 0.0

                # Extract explanation
                else:
                    explanation_rule = None
                    encoded_feature_state = pylfit.algorithms.GULA.encode_state(feature_state, dataset.features)

                    # Predicted likely
                    if proba > 0.5:
                        expected_rules = [r for (w,r) in expected_model.rules \
                        if r.head_variable == var_id and r.head_value == val_id and r.matches(encoded_feature_state)]
                        explanation_rule = r1

                    # Predicted unlikely
                    if proba < 0.5:
                        expected_rules = [r for (w,r) in expected_model.unlikeliness_rules \
                        if r.head_variable == var_id and r.head_value == val_id and r.matches(encoded_feature_state)]
                        explanation_rule = r2

                    if explanation_rule == None:
                        score = 0.0

                    # Find nearest expected rule
                    else:
                        min_distance = len(dataset.features)
                        nearest_expected = None
                        for r in expected_rules:
                            distance = hamming_distance(explanation_rule,r)
                            if distance <= min_distance:
                                min_distance = distance
                                nearest_expected = r

                        score = 1.0 - (min_distance / len(dataset.features))

                sum_score += score
                nb_targets += 1
        sum_explanation_score += sum_score / nb_targets

    return sum_explanation_score / len(expected)


def hamming_distance(rule_1, rule_2):
    """
    Args:
        rule_1: Rule
        rule_2: Rule
    Returns:
        the number of differents conditions between the two rules.
    """
    cond_var = set()
    for var,val in rule_1.body:
        cond_var.add(var)
    for var, val in rule_2.body:
        cond_var.add(var)

    distance = 0
    for var in cond_var:
        if rule_1.get_condition(var) != rule_2.get_condition(var):
            distance += 1

    return distance
