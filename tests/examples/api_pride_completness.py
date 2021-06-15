#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2021/03/01
# @updated: 2021/06/15
#
# @desc: example of using extend on WDMVLP
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
    (["1","0","0"],["0","0","0"]), \
    (["0","1","0"],["1","0","1"]), \
    (["0","0","1"],["0","0","1"]), \
    (["1","1","0"],["1","0","0"]), \
    (["1","0","1"],["0","1","0"]), \
    (["0","1","1"],["1","0","1"]), \
    (["1","1","1"],["1","1","0"])\
    ]

    full_dataset = pylfit.preprocessing.transitions_dataset_from_array(data=data,\
    feature_domains=[("p_t_1",["0","1"]),("q_t_1",["0","1"]),("r_t_1",["0","1"])],\
    target_domains=[("p_t",["0","1"]),("q_t",["0","1"]),("r_t",["0","1"])])

    data = [ \
    (["0","0","0"],["0","0","1"]), \
    (["1","0","0"],["0","0","0"]), \
    (["0","1","0"],["1","0","1"]), \
    #(["0","0","1"],["0","0","1"]), \
    #(["1","1","0"],["1","0","0"]), \
    #(["1","0","1"],["0","1","0"]), \
    #(["0","1","1"],["1","0","1"]), \
    #(["1","1","1"],["1","1","0"])\
    ]

    train_dataset = pylfit.preprocessing.transitions_dataset_from_array(data=data,\
    feature_domains=full_dataset.features,\
    target_domains=full_dataset.targets)

    print("Train dataset")
    train_dataset.summary()
    print()

    model = pylfit.models.WDMVLP(features=train_dataset.features, targets=train_dataset.targets)
    model.compile(algorithm="gula")
    #model.fit(train_dataset)
    print()

    data = [ \
    #(["0","0","0"],["0","0","1"]), \
    #(["0","0","0"],["1","0","0"]), \
    #(["1","0","0"],["0","0","0"]), \
    (["0","1","0"],["1","0","1"]), \
    (["0","0","1"],["0","0","1"]), \
    (["1","1","0"],["1","0","0"]), \
    (["1","0","1"],["0","1","0"]), \
    (["0","1","1"],["1","0","1"]), \
    (["1","1","1"],["1","1","0"])\
    ]

    test_dataset = pylfit.preprocessing.transitions_dataset_from_array(data=data,\
    feature_domains=full_dataset.features,\
    target_domains=full_dataset.targets)

    print("Test dataset")
    test_dataset.summary()
    print()

    #print("Fit the WDMVLP on the partial dataset")
    #model.fit(dataset=train_dataset)

    print("Uncomplete model:")
    model.summary()

    print("Predict as usual")
    s = [["1","1","1"]]
    print(model.predict(s))

    accuracy_score = pylfit.postprocessing.accuracy_score(model=model, dataset=test_dataset)
    print()
    print("AVG accuracy score: " + str(round(accuracy_score, 2)))

    print("Extend model to cover requested states")
    feature_states = [list(s1) for s1,s2 in test_dataset.data]
    model.extend(train_dataset, feature_states)
    print(model.predict(s))

    accuracy_score = pylfit.postprocessing.accuracy_score(model=model, dataset=test_dataset)
    print()
    print("AVG accuracy score: " + str(round(accuracy_score, 2)))

    model.extend(train_dataset, [list(s1) for s1,s2 in train_dataset.data])

    model.summary()
