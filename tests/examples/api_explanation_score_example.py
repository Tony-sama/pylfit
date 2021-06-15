#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2021/03/01
# @updated: 2021/06/15
#
# @desc: example of scoring WDMVLP explanation
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
    (["0","0","0"],["1","0","0"]), \
    (["1","0","0"],["0","0","0"]), \
    (["0","1","0"],["1","0","1"]), \
    (["0","0","1"],["0","0","1"]), \
    (["1","1","0"],["1","0","0"]), \
    #(["1","0","1"],["0","1","0"]), \
    (["0","1","1"],["1","0","1"]), \
    (["1","1","1"],["1","1","0"])\
    ]

    print("Convert array data as a StateTransitionsDataset using pylfit.preprocessing")
    full_dataset = pylfit.preprocessing.transitions_dataset_from_array(data=data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])
    full_dataset.summary()

    print("Optimal WDMVLP learn by GULA from all possible transitions")
    optimal_model = pylfit.models.WDMVLP(features=full_dataset.features, targets=full_dataset.targets)
    optimal_model.compile(algorithm="gula") # model.compile(algorithm="pride")
    optimal_model.fit(dataset=full_dataset)
    optimal_model.summary()
    print()

    print("Train dataset")

    data = [ \
    (["0","0","0"],["0","0","1"]), \
    (["0","0","0"],["1","0","0"]), \
    #(["1","0","0"],["0","0","0"]), \
    #(["0","1","0"],["1","0","1"]), \
    #(["0","0","1"],["0","0","1"]), \
    #(["1","1","0"],["1","0","0"]), \
    #(["1","0","1"],["0","1","0"]), \
    #(["0","1","1"],["1","0","1"]), \
    (["1","1","1"],["1","1","0"])\
    ]

    train_dataset = pylfit.preprocessing.transitions_dataset_from_array(data=data,\
    feature_domains=[("p_t_1",["0","1"]),("q_t_1",["0","1"]),("r_t_1",["0","1"])],\
    target_domains=[("p_t",["0","1"]),("q_t",["0","1"]),("r_t",["0","1"])])
    train_dataset.summary()
    print()

    model = pylfit.models.WDMVLP(features=train_dataset.features, targets=train_dataset.targets)
    model.compile(algorithm="gula")
    print()

    print("Fit the WDMVLP on the partial dataset")
    model.fit(dataset=train_dataset)

    print("Trained model:")
    model.summary()

    print("Test dataset")

    data = [ \
    #(["0","0","0"],["0","0","1"]), \
    #(["0","0","0"],["1","0","0"]), \
    (["1","0","0"],["0","0","0"]), \
    (["0","1","0"],["1","0","1"]), \
    (["0","0","1"],["0","0","1"]), \
    (["1","1","0"],["1","0","0"]), \
    #(["1","0","1"],["0","1","0"]), \
    (["0","1","1"],["1","0","1"]), \
    #(["1","1","1"],["1","1","0"])\
    ]

    print("Convert array data as a StateTransitionsDataset using pylfit.preprocessing")
    test_dataset = pylfit.preprocessing.transitions_dataset_from_array(data=data,\
    feature_domains=[("p_t_1",["0","1"]),("q_t_1",["0","1"]),("r_t_1",["0","1"])],\
    target_domains=[("p_t",["0","1"]),("q_t",["0","1"]),("r_t",["0","1"])])
    test_dataset.summary()

    explanation_score = pylfit.postprocessing.explanation_score(model=model, expected_model=optimal_model, dataset=test_dataset)
    accuracy_score = pylfit.postprocessing.accuracy_score(model=model, dataset=test_dataset)
    print()
    print("AVG explanation score: " + str(round(explanation_score, 2)))
    print("AVG accuracy score: " + str(round(accuracy_score, 2)))
