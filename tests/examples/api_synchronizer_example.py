#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2021/02/03
# @updated: 2021/06/15
#
# @desc: example of the use of CDMVLP model with the Synchronizer algorithm
#-------------------------------------------------------------------------------

import pylfit

ALGORITHM = "synchronizer" # "gula"

# 1: Main
#------------
if __name__ == '__main__':

    # 1) Example from csv file encoding transitions
    #--------------------------------------------------------
    print()
    print("Example using transition from array:")

    # Array data
    # list of tuple (list of String/int, list of String/int) encoded states transitions
    # integer will be interpreted as string by pylfit api
    data = [ \
    (["0","0","0"],["0","0","1"]), \
    (["0","0","0"],["1","0","0"]), \
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

    print("Initialize a CDMVLP with the dataset variables and set Synchronizer as learning algorithm")
    model = pylfit.models.CDMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm=ALGORITHM)
    model.summary()
    print()

    print("Fit the CDMVLP on the dataset")
    model.fit(dataset=dataset) # optional targets

    print("Trained model:")
    model.summary()

    state = ['0','0','0']
    print("Predict from",state,": ", end='')
    prediction = model.predict([state])
    print(prediction)

    state = ['0','0','0']
    print("Predict from",state,": ", end='')
    prediction = model.predict([state])
    print(prediction)

    print("Check model dynamics: ")
    errors = 0
    expected = set((tuple(s1),tuple(s2)) for s1,s2 in dataset.data)
    predicted = set()


    predicted = [(tuple(s1), tuple(s2)) for s1,S2 in model.predict(model.feature_states()) for s2 in S2]

    print()
    done = 0
    for s1,s2 in expected:
        done += 1
        print("\rChecking transitions ",done,"/",len(expected),end='')
        if (s1,s2) not in predicted:
            print()
            print("missing transition: ", (s1,s2))
            errors +=1

    print()
    done = 0
    for s1,s2 in predicted:
        done += 1
        print("\rChecking transitions ",done,"/",len(predicted),end='')
        if (s1,s2) not in expected:
            print()
            print("spurious transition: ", (s1,s2))
            errors += 1
    print()

    if errors == 0:
        print("Perfect reproduction of data")
    else:
        print("Errors: ", errors)
