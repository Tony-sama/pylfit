#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2022/01/10
# @updated: 2022/01/10
#
# @desc: example of the use of PDMVLP model with algorithm probalizer
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
    (["0","0","0"],["0","0","1"]), \
    (["0","0","0"],["0","0","1"]), \

    (["0","0","0"],["1","0","0"]), \
    (["0","0","0"],["1","0","1"]), \

    (["0","0","0"],["0","0","0"]), \
    (["0","0","0"],["0","0","0"]), \
    (["0","0","0"],["0","0","0"]), \


    (["0","0","1"],["0","0","0"]), \
    (["0","0","1"],["0","0","1"]), \

    (["0","1","0"],["0","0","0"]), \
    #(["0","1","0"],["0","0","1"]), \

    (["1","0","0"],["0","0","0"]), \
    (["0","1","1"],["0","0","0"]), \
    (["1","0","1"],["0","0","0"]), \
    (["1","1","0"],["0","0","0"]), \
    (["1","1","1"],["0","0","0"]), \
    ]

    # Convert array data as a DiscreteStateTransitionsDataset using pylfit.preprocessing
    dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=data, \
    feature_names=["x0","x1","x2"], target_names=["y0","y1","y2"])
    dataset.summary()
    print()

# 1) Independant probabilities
#----------------------------------

    # Initialize a DMVLP with the dataset variables and set GULA as learning algorithm (suppose probability are synchronous independant)
    model = pylfit.models.PDMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm="gula") # model.compile(algorithm="pride")
    model.fit(dataset=dataset)
    model.summary()

    state = ("0","0","0")
    prediction = model.predict([state])
    print("Predict from ",state,":")
    for s in prediction[tuple(state)]:
        proba, rules = prediction[tuple(state)][s]
        print(" "+str(s))
        print("  probability: "+str(proba*100)+"%")
        print("  Rules: ")
        for r_str in [r.logic_form(model.features, model.targets) for r in rules]:
            print("   "+r_str)

# 1) Non-independant probabilities
#----------------------------------

    # Initialize a DMVLP with the dataset variables and set Synchronizer as learning algorithm (for non synchronous independant probability)
    model = pylfit.models.PDMVLP(features=dataset.features, targets=dataset.targets)
    model.compile(algorithm="synchronizer") # model.compile(algorithm="pride")
    model.fit(dataset=dataset, threads=2)
    model.summary()

    state = ("0","0","0")
    prediction = model.predict([state])
    print("Predict from ",state,":")
    for s in prediction[tuple(state)]:
        proba, rules = prediction[tuple(state)][s]
        print(" "+str(s))
        print("  probability: "+str(proba*100)+"%")
        print("  Rules: ")
        for r_str in [r.logic_form(model.features, model.targets) for r in rules]:
            print("   "+r_str)
