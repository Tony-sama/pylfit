#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2021/02/18
# @updated: 2021/06/15
#
# @desc: example of the use WDMVLP model with algorithm GULA and PRIDE
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
    (["1","1","1"],["1","1","0"])]

    # Convert array data as a DiscreteStateTransitionsDataset using pylfit.preprocessing
    dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=data, \
    feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])
    dataset.summary()
    print()

    # Initialize a DMVLP with the dataset variables and set GULA as learning algorithm
    model = pylfit.models.WDMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm="gula") # model.compile(algorithm="pride")
    model.summary()
    print()

    # Fit the WDMVLP on the dataset
    # optional targets: model.fit(dataset=dataset, targets_to_learn={'p_t':["1"], 'r_t':["0"]})
    model.fit(dataset=dataset)
    model.summary()

    # Predict from ['1','0','1'] (unseen data)
    prediction = model.predict([["1","0","1"]])
    print(prediction)

    # {variable: {value: (proba, (weight, rule), (weight, rule))}}
    # first rule is the best of the model for likeliness,
    # second is for unlikeliness w.r.t. the given feature state

    for variable, values in model.targets:
        print(variable+": ")
        for value, (proba, _, _) in prediction[("1","0","1")][variable].items():
            print(" "+value+" "+str(proba*100.0)+"%")

    print()
    print("Explanations")
    for variable, values in model.targets:
        print(variable+": ")
        for value, (proba, \
        (best_likeliness_weight, best_likeliness_rule),\
        (best_unlikeliness_weight, best_unlikeliness_rule))\
        in prediction[("1","0","1")][variable].items():
            print(" Value ''"+value+"'")
            print("  likely: "+str((best_likeliness_weight, best_likeliness_rule)))
            print("  unlikely: "+str((best_unlikeliness_weight, best_unlikeliness_rule)))
            print("  => "+str(proba*100.0)+"% likely / "+str((1-proba)*100.0)+"% unlikely")
            if proba > 0.5:
                print("  Conclusion: likely")
            if proba < 0.5:
                print("  Conclusion: unlikely")
            if proba == 0.5:
                print("  Conclusion: unconclusive")
