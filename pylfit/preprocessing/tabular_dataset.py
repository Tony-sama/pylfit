""" pylfit tabular dataset loading utilities """

from ..utils import eprint
from ..datasets import StateTransitionsDataset

import pandas
import numpy

def transitions_dataset_from_csv(path, feature_names, target_names):
    """ Load tabular data from a csv file into a list of pair of feature/target state

    Each line of the file must gives a value for each feature/target variables given in parameter

    Args:
        path: String
            Path to a csv file
        feature_names: list of String
            Header label of the feature variables columns
        target_names: list of String
            Header label of the target variables columns

    Returns:
        StateTransitionsDataset
            A pylfit dataset api encoding of the state transitions, ready to be used with pylfit model api.
    """
    df = pandas.read_csv(path)

    feature_domains = [(var, [str(i) for i in df[var].unique()]) for var in feature_names]
    target_domains = [(var, [str(i) for i in df[var].unique()]) for var in target_names]
    data = [(numpy.array([str(i) for i in x]), numpy.array([str(i) for i in y])) for x, y in zip(df[feature_names].values, df[target_names].values)]

    # Order domains alphabetically
    for var_id, (var, domain) in enumerate(feature_domains):
        feature_domains[var_id] = (var, sorted(domain))
    for var_id, (var, domain) in enumerate(target_domains):
        target_domains[var_id] = (var, sorted(domain))

    dataset = StateTransitionsDataset(data=data, features=feature_domains, targets=target_domains)

    return dataset

def transitions_dataset_from_array(data, feature_names=None, target_names=None):
    """ Create a StateTransitionsDataset from given data and variables names.

    Args:
        data: list of tuple (list of String, list of String).
            Multiset of state transitions.
        feature_names: list of String
            Names of the feature variables (optional).
            Extracted from data when not given.
            Domain values will be ordered alphabetically in this case.
        target_names: list of String
            Names of the target variables (optional).
            Extracted from data when not given.
            Domain values will be ordered alphabetically in this case.
    Returns:
        StateTransitionsDataset.
            A pylfit dataset api encoding of the state transitions, ready to be used with pylfit model api.
    """

    # Check data type
    if not isinstance(data, (list)):
        raise TypeError("Argument data must be a list.")
    if not all(isinstance(i, tuple) for i in data):
        raise TypeError("Argument data must only contains tuples.")
    if not all(len(i) == 2 for i in data):
        raise TypeError("Argument data tuples has to be of size 2.")
    if not all(isinstance(i, (str,int)) for s1,s2 in data for i in s1 ):
        raise ValueError("Argument data feature states values must be int or string")
    if not all(isinstance(i, (str,int)) for s1,s2 in data for i in s2 ):
        raise ValueError("Argument data target states values must be int or string")

    # Check feature_names type
    if feature_names is not None:
        if not isinstance(feature_names, (list)):
            raise TypeError("Argument feature_names must be a list.")
        if not all(isinstance(i, str) for i in feature_names):
            raise ValueError("Argument feature_names must only contains String.")

    # Check target_names type
    if target_names is not None:
        if not isinstance(target_names, (list)):
            raise TypeError("Argument target_names must be a list.")
        if not all(isinstance(i, str) for i in target_names):
            raise ValueError("Argument target_names must only contains String.")

    # Initialize feature/target variables domain
    feature_domains = None
    target_domains = None

    if feature_names is not None:
        feature_domains = [(str(i), []) for i in feature_names]
    if target_names is not None:
        target_domains = [(str(i), []) for i in target_names]

    # Empty dataset
    if len(data) == 0:
        return StateTransitionsDataset(data=[], features=feature_domains, targets=target_domains)

    if feature_domains is None:
        feature_domains = [("x"+str(i), []) for i in range(len(data[0][0]))]
    if target_domains is None:
        target_domains = [("y"+str(i), []) for i in range(len(data[0][1]))]

    # Check values of data
    nb_features = len(data[0][0])
    nb_targets = len(data[0][1])

    if not all(len(s1) == nb_features for (s1,s2) in data):
        raise ValueError("data feature states must be of same size.")
    if not all(len(s2) == nb_targets for (s1,s2) in data):
        raise ValueError("data target states must be of same size.")

    if not nb_features == len(feature_domains):
        raise ValueError("Size of argument feature_names must be same as argument data feature states size.")
    if not nb_targets == len(target_domains):
        raise ValueError("Size of argument target_names must be same as argument data target states size.")

    # Convert data format to StateTransitionsDataset format
    data_encoded = [(numpy.array([str(i) for i in x]), numpy.array([str(i) for i in y])) for x, y in data]

    # Extract feature/target variables domain from data
    for (s1, s2) in data_encoded:
        for var_id, value in enumerate(s1):
            if value not in feature_domains[var_id][1]:
                feature_domains[var_id][1].append(value)

        for var_id, value in enumerate(s2):
            if value not in target_domains[var_id][1]:
                target_domains[var_id][1].append(value)

    # Order domains alphabetically
    for var_id, (var, domain) in enumerate(feature_domains):
        feature_domains[var_id] = (var, sorted(domain))
    for var_id, (var, domain) in enumerate(target_domains):
        target_domains[var_id] = (var, sorted(domain))

    dataset = StateTransitionsDataset(data=data_encoded, features=feature_domains, targets=target_domains)

    return dataset
