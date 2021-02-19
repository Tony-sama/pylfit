#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2021/01/01
# @updated: 2021/02/18
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

    print("Convert array data as a StateTransitionsDataset using pylfit.preprocessing")
    dataset = pylfit.preprocessing.transitions_dataset_from_array(data=data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])
    dataset.summary()
    print()

    print("Initialize a DMVLP with the dataset variables and set GULA as learning algorithm")
    model = pylfit.models.DMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm="gula") # model.compile(algorithm="pride")
    model.summary()
    print()

    print("Fit the DMVLP on the dataset")
    model.fit(dataset=dataset) # optional targets: model.fit(dataset=dataset, targets_to_learn={p_t:["1"], r_t:["0"]})

    print("Trained model:")
    model.summary()

    print("Predict from ['0','0','0'] (default: synchronous): ", end='')
    prediction = model.predict(["0","0","0"])
    print(prediction)

    print("Predict from ['1','0','1'] (synchronous): ", end='')
    prediction = model.predict(["1","0","1"])
    print(prediction)

    print("Predict from ['1','0','1'] (asynchronous): ", end='')
    prediction = model.predict(["1","0","1"], semantics="asynchronous")
    print(prediction)

    print("Predict from ['1','0','1'] (general): ", end='')
    prediction = model.predict(["1","0","1"], semantics="general")
    print(prediction)

    print("All states transitions of the model (synchronous): ")
    transitions = []
    for s1 in model.feature_states():
        prediction = model.predict(s1)
        for s2 in prediction:
            transitions.append( (s1, s2) )

    print(transitions)

    dataset = pylfit.preprocessing.transitions_dataset_from_array(data=transitions, feature_names=[var for var,vals in dataset.features], target_names=[var for var,vals in dataset.targets])

    print("Saving transitions to csv...")
    dataset.to_csv('tmp/output.csv')
    print("Saved to tmp/output.csv")
