#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2022/08/24
# @updated: 2022/08/31
#
# @desc: example of the use CLP model with algorithm ACEDIA
#-------------------------------------------------------------------------------

import pylfit

# 1: Main
#------------
if __name__ == '__main__':

    # Array data
    # list of tuple (list of String/int, list of String/int) encoded states transitions
    # integer will be interpreted as string by pylfit api
    data = [ \
    ([0,0,0],[0,0,1]), \
    ([0,0,0],[1,0,0]), \
    ([1,0,0],[0,0,0]), \
    ([0,1,0],[1,0,1]), \
    ([0,0,1],[0,0,1]), \
    ([1,1,0],[1,0,0]), \
    ([1,0,1],[0,1,0]), \
    ([0,1,1],[1,0,1]), \
    ([1,1,1],[1,1,0])]

    # Convert array data as a ContinuousStateTransitionsDataset using pylfit.preprocessing
    dataset = pylfit.preprocessing.continuous_state_transitions_dataset_from_array(data=data, \
    feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])
    dataset.summary()
    print()

    # Initialize a CLP with the dataset variables and set ACEDIA as learning algorithm
    model = pylfit.models.CLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm="acedia")
    model.summary()
    print()

    # Fit the CLP on the dataset
    # optional targets: model.fit(dataset=dataset, targets_to_learn={'p_t':["1"], 'r_t':["0"]})
    model.fit(dataset=dataset)#, threads=4)
    model.summary()

    # Predict from [0,0,0] and [0,0,1] (default: continuum synchronous deterministic)
    states = [[0,0,0], [0,0,1]]
    prediction = model.predict(states)

    print()
    print("> Model predictions")
    for state in prediction:
        print()
        print(">> Feature state:", state)
        for target in prediction[state]:
            print(">>> Target state continuums:", target)
            print(">>> Matching rules:")
            for r in prediction[state][target]:
                print(">>>> "+r.logic_form(model.features, model.targets))

    # All states transitions of the model (continuum synchronous deterministic)
    print()
    transitions = []
    prediction = model.predict(model.feature_states(epsilon=1.0))
    for s1 in prediction:
        for s2 in prediction[s1]:
            transitions.append( (s1, s2) )

    print("All transitions:", transitions)
