#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2021/01/01
# @updated: 2021/06/15
#
# @desc: example of the use DMVLP model with algorithm GULA and PRIDE
#-------------------------------------------------------------------------------

import pylfit

# 1: Main
#------------
if __name__ == '__main__':

    # Array data
    # list of tuple (list of String/int, list of String/int) encoded states transitions
    # integer will be interpreted as string by pylfit api
    data = [ \
    (["0","0","0"],["0","0","1"]), \
    #(["0","0","0"],["1","0","0"]), \
    (["1","0","0"],["0","0","0"]), \
    (["0","1","0"],["1","0","1"]), \
    (["0","0","1"],["0","0","1"]), \
    (["1","1","0"],["1","0","0"]), \
    (["1","0","1"],["0","1","0"]), \
    (["0","1","1"],["1","0","1"]), \
    (["1","1","1"],["1","1","0"])]

    # Convert array data as a DiscreteStateTransitionsDataset using pylfit.preprocessing
    dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=data, \
    feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])
    dataset.summary()
    print()

    # Initialize a DMVLP with the dataset variables and set GULA as learning algorithm
    model = pylfit.models.DMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm="gula") # model.compile(algorithm="pride")
    model.summary()
    print()

    # Fit the DMVLP on the dataset
    # optional targets: model.fit(dataset=dataset, targets_to_learn={p_t:["1"], r_t:["0"]})
    model.fit(dataset=dataset)
    model.summary()

    # Predict from ['0','0','0'] (default: synchronous)
    state = ("0","0","0")
    prediction = model.predict([state])
    print("Synchronous:", [s for s in prediction[tuple(state)]])

    # Predict from ['1','0','1'] (synchronous)
    state = ("1","0","1")
    prediction = model.predict([state], semantics="synchronous", default=None)
    print("Synchronous:", [s for s in prediction[state]])

    # Predict from ['1','0','1'] (asynchronous)
    prediction = model.predict([state], semantics="asynchronous")
    print("Asynchronous:", [s for s in prediction[state]])

    # Predict from ['1','0','1'] (general)
    prediction = model.predict([state], semantics="general")
    print("General:", [s for s in prediction[state]])

    # All states transitions of the model (synchronous)
    transitions = []
    prediction = model.predict(model.feature_states())
    for s1 in prediction:
        for s2 in prediction[s1]:
            transitions.append( (s1, s2) )

    print("All transitions:", transitions)

    dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=transitions, feature_names=[var for var,vals in dataset.features], target_names=[var for var,vals in dataset.targets])

    print("Saving transitions to csv...")
    dataset.to_csv('tmp/output.csv')
    print("Saved to tmp/output.csv")
