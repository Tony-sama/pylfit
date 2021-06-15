#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2021/02/02
# @updated: 2021/06/15
#
# @desc: example of loading of Boolean network from pyboolnet file
#-------------------------------------------------------------------------------


import pylfit

# 1: Main
#------------
if __name__ == '__main__':

    benchmark_name = "faure_cellcycle"
    bn_file_path = "benchmarks/boolean_networks/pyboolnet/bio/"+benchmark_name+".bnet"
    output_file_path = "tmp/"+benchmark_name+".csv"

    print("Loading DMVLP from Boolean network file")
    model = pylfit.preprocessing.boolean_network.dmvlp_from_boolean_network_file(bn_file_path)
    model.summary()

    print("All states transitions (synchronous): ")
    transitions = []
    prediction = model.predict(model.feature_states())
    for s1 in prediction:
        for s2 in prediction[s1]:
            transitions.append( (s1, s2) )

    print(len(transitions))

    dataset = pylfit.preprocessing.transitions_dataset_from_array(
    data=transitions,
    feature_names=[var for var,vals in model.features],
    target_names=[var for var,vals in model.targets])

    print("Saving transitions to csv...")
    dataset.to_csv(output_file_path)
    print("Saved to "+output_file_path)
