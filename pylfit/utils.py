import sys
import random
import copy
import pandas

def eprint(*args, **kwargs):
    """
        Debug print function, prints to standard error stream
    """
    print(*args, file=sys.stderr, **kwargs, sep="")

def load_tabular_data_from_csv(path, feature_names, target_names):
    """
    Load tabular data from a csv file into a list of pair of feature/target state given the features/target columns names

    Parameters
    ----------
    filepath: String
        Path to csv file encoding transitions
    feature_names: list of String
        Header label of the feature variables columns
    target_names: list of String
        Header label of the target variables columns

    Returns
    -------
    list
        a list of pair of tuple representing a multiset of pair of feature/target states
    """
    df = pandas.read_csv(path)
    data = [(x,y) for x, y in zip(df[feature_names].values, df[target_names].values)]
    feature_domains = [ (str(col), sorted([str(val) for val in df[col].unique()])) for col in feature_names]
    target_domains = [ (str(col), sorted([str(val) for val in df[col].unique()])) for col in target_names]

    return data, feature_domains, target_domains


# DBG: DEPRECATED
def precision(expected, predicted):
    """
    Evaluate prediction precision on deterministic sets of transitions
    Args:
        expected: list of tuple (list of int, list of int)
            originals transitions of a system
        predicted: list of tuple (list of int, list of int)
            predicted transitions of a system

    Returns:
        float in [0,1]
            the error ratio between expected and predicted
    """
    eprint("DEPRECATED")

    if len(expected) == 0:
        return 1.0

    # Predict each variable for each state
    total = len(expected) * len(expected[0][0])
    error = 0

    #eprint("comparing: ")
    #eprint(test)
    #eprint(pred)

    for i in range(len(expected)):
        s1, s2 = expected[i]

        for j in range(len(predicted)):
            s1_, s2_ = predicted[j]

            if len(s1) != len(s1_) or len(s2) != len(s2_):
                raise ValueError("Invalid prediction set")

            if s1 == s1_:
                #eprint("Compare: "+str(s2)+" VS "+str(s2_))

                for var in range(len(s2)):
                    if s2_[var] != s2[var]:
                        error += 1
                break

        #eprint("new error: "+str(error))

    precision = 1.0 - (error / total)

    return precision
