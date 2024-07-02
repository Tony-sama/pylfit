#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2021/03/03
# @updated: 2023/12/27
#
# @desc: pylfit tabular dataset utility functions
#-------------------------------------------------------------------------------

""" pylfit tabular dataset loading utilities """

from ..utils import eprint
from ..datasets import DiscreteStateTransitionsDataset, ContinuousStateTransitionsDataset
from ..objects import Continuum

import pandas
import numpy

def discrete_state_transitions_dataset_from_csv(path, feature_names, target_names, unknown_values=[DiscreteStateTransitionsDataset._UNKNOWN_VALUE]):
    """ Load tabular data from a csv file into a list of pair of feature/target state

    Each line of the file must gives a value for each feature/target variables given in parameter

    Args:
        path: String
            Path to a csv file
        feature_names: list of String
            Header label of the feature variables columns
        target_names: list of String
            Header label of the target variables columns
        unknown_values: list of string
            List of value representing unknown value.

    Returns:
        DiscreteStateTransitionsDataset
            A pylfit dataset api encoding of the state transitions, ready to be used with pylfit model api.
    """
    # Check path
    if not isinstance(path,str):
        raise TypeError("Argument path must be a string.")

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
        
    # Check unknown_values type
    #if unknown_values is not None:
    if not isinstance(unknown_values, list):
        raise TypeError("Argument unknown_values must be a list")
    #if not all(isinstance(i, str) for i in unknown_values):
    #   raise TypeError("Argument unknown_values must be a list of string")

    df = pandas.read_csv(path)

    if unknown_values is None:
        unknown_values = []

    feature_domains = [(var, [str(i) for i in df[var].unique() if i not in unknown_values]) for var in feature_names]
    target_domains = [(var, [str(i) for i in df[var].unique() if i not in unknown_values]) for var in target_names]
    data = [(numpy.array([DiscreteStateTransitionsDataset._UNKNOWN_VALUE if i in unknown_values else str(i) for i in x]),
             numpy.array([DiscreteStateTransitionsDataset._UNKNOWN_VALUE if i in unknown_values else str(i) for i in y])) for x, y in zip(df[feature_names].values, df[target_names].values)]

    # Order domains alphabetically
    for var_id, (var, domain) in enumerate(feature_domains):
        feature_domains[var_id] = (var, sorted(domain))
    for var_id, (var, domain) in enumerate(target_domains):
        target_domains[var_id] = (var, sorted(domain))

    dataset = DiscreteStateTransitionsDataset(data=data, features=feature_domains, targets=target_domains)

    return dataset

def discrete_state_transitions_dataset_from_array(data, feature_domains=None, target_domains=None, feature_names=None, target_names=None, unknown_values=[DiscreteStateTransitionsDataset._UNKNOWN_VALUE]):
    """ Create a DiscreteStateTransitionsDataset from given data according to variables domains if given or variable names if given.

    Feature/target variables names are automatically generated if not given:
        x_0, x_1, ... for features.
        y_0, y_1, ... for targets.
    Feature/target variables domains are extracted from data if not given as argument.

    Args:
        data: list of tuple (list of String, list of String).
            Multiset of state transitions.
        feature_domains: list of (String, list of String)
            Name and domain of each feature variable (optional)
        target_domains: list of (String, list of String)
            Name and domain of each target variable (optional)
        feature_names: list of String
            Names of the feature variables (optional).
            Should not be given if feature_domains is given.
        target_names: list of String
            Names of the target variables (optional).
            Should not be given if target_domains is given.
        unknown_values: list of string
            List of value representing unknown value.
    Returns:
        DiscreteStateTransitionsDataset.
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

    # Check feature_domains type
    if feature_domains is not None:
        if not isinstance(feature_domains, list):
            raise TypeError("Argument feature_domains must be a list")
        if not all(isinstance(i, tuple) for i in feature_domains):
            raise TypeError("Argument feature_domains must be a list of tuple")
        if not all(len(i) == 2 for i in feature_domains):
            raise TypeError("Argument feature_domains must be a list of tuple of size 2")
        if not all(isinstance(var, str) for var,vals in feature_domains):
            raise TypeError("Argument feature_domains must be a list of tuple (String, list of String)")
        if not all(isinstance(vals, list) for var,vals in feature_domains):
            raise TypeError("Argument feature_domains must be a list of tuple (String, list of String)")
        if not all(isinstance(val, str) for var,vals in feature_domains for val in vals):
            raise TypeError("Argument feature_domains must be a list of tuple (String, list of String)")
        if not len([var for var, vals in feature_domains]) == len(set([var for var, vals in feature_domains])):
            raise ValueError("Argument feature_domains, each variable name must be unique")
        if not all(len(vals) == len(set(vals)) for var, vals in feature_domains):
            raise ValueError("Argument feature_domains, each value name in a domain must be unique")

    # Check target_domains type
    if target_domains is not None:
        if not isinstance(target_domains, list):
            raise TypeError("Argument target_domains must be a list")
        if not all(isinstance(i, tuple) for i in target_domains):
            raise TypeError("Argument target_domains must be a list of tuple")
        if not all(len(i) == 2 for i in target_domains):
            raise TypeError("Argument target_domains must be a list of tuple of size 2")
        if not all(isinstance(var, str) for var,vals in target_domains):
            raise TypeError("Argument target_domains must be a list of tuple (String, list of String)")
        if not all(isinstance(vals, list) for var,vals in target_domains):
            raise TypeError("Argument target_domains must be a list of tuple (String, list of String)")
        if not all(isinstance(val, str) for var,vals in target_domains for val in vals):
            raise TypeError("Argument target_domains must be a list of tuple (String, list of String)")
        if not len([var for var, vals in target_domains]) == len(set([var for var, vals in target_domains])):
            raise ValueError("Argument target_domains, each variable name must be unique")
        if not all(len(vals) == len(set(vals)) for var, vals in target_domains):
            raise ValueError("Argument target_domains, each value name in a domain must be unique")

    # Check feature_names type
    if feature_names is not None:
        if feature_domains is not None:
            raise ValueError("Argument feature_names should not be given when feature_domains is specified")
        if not isinstance(feature_names, (list)):
            raise TypeError("Argument feature_names must be a list.")
        if not all(isinstance(i, str) for i in feature_names):
            raise ValueError("Argument feature_names must only contains String.")

    # Check target_names type
    if target_names is not None:
        if target_domains is not None:
            raise ValueError("Argument target_names should not be given when target_domains is specified")
        if not isinstance(target_names, (list)):
            raise TypeError("Argument target_names must be a list.")
        if not all(isinstance(i, str) for i in target_names):
            raise ValueError("Argument target_names must only contains String.")
        
    # Check unknown_values type
    #if unknown_values is not None:
    if not isinstance(unknown_values, list):
        raise TypeError("Argument unknown_values must be a list")
    #if not all(isinstance(i, str) for i in unknown_values):
    #    raise TypeError("Argument unknown_values must be a list of string")
        
    # Empty dataset
    if len(data) == 0:
        if feature_domains is None:
            raise ValueError("Features domain must not be None if data is empty")
        if target_domains is None:
            raise ValueError("Features and targets domain must not be None if data is empty")

        return DiscreteStateTransitionsDataset(data=[], features=feature_domains, targets=target_domains)

    # Initialize feature/target variables domain
    if feature_names is not None and feature_domains is None:
        feature_domains = [(str(i), []) for i in feature_names]

    if target_names is not None and target_domains is None:
        target_domains = [(str(i), []) for i in target_names]

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
        raise ValueError("Size of argument feature_domains and feature_names must be same as argument data feature states size.")
    if not nb_targets == len(target_domains):
        raise ValueError("Size of argument target_domains and target_names must be same as argument data target states size.")

    # Convert data format to DiscreteStateTransitionsDataset format
    data_encoded = [(numpy.array([DiscreteStateTransitionsDataset._UNKNOWN_VALUE if i in unknown_values else str(i) for i in x]),
                     numpy.array([DiscreteStateTransitionsDataset._UNKNOWN_VALUE if i in unknown_values else str(i) for i in y])) for x, y in data]

    # Initialize unknown values
    if unknown_values is None:
        unknown_values = []

    # Extract feature/target variables domain from data
    feature_domains = feature_domains.copy()
    target_domains = target_domains.copy()
    for (s1, s2) in data_encoded:
        for var_id, value in enumerate(s1):
            if value not in unknown_values and value not in feature_domains[var_id][1]:
                feature_domains[var_id][1].append(value)

        for var_id, value in enumerate(s2):
            if value not in unknown_values and value not in target_domains[var_id][1]:
                target_domains[var_id][1].append(value)

    # Order domains alphabetically
    for var_id, (var, domain) in enumerate(feature_domains):
        feature_domains[var_id] = (var, sorted(domain))
    for var_id, (var, domain) in enumerate(target_domains):
        target_domains[var_id] = (var, sorted(domain))

    dataset = DiscreteStateTransitionsDataset(data=data_encoded, features=feature_domains, targets=target_domains)

    return dataset

def continuous_state_transitions_dataset_from_csv(path, feature_names, target_names):
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
        ContinuousStateTransitionsDataset
            A pylfit dataset api encoding of the state transitions, ready to be used with pylfit model api.
    """
    df = pandas.read_csv(path)

    feature_domains = [(var, Continuum(df[var].min(), df[var].max(), True, True)) for var in feature_names]
    target_domains = [(var, Continuum(df[var].min(), df[var].max(), True, True)) for var in target_names]
    data = [(numpy.array([float(i) for i in x]), numpy.array([float(i) for i in y])) for x, y in zip(df[feature_names].values, df[target_names].values)]

    dataset = ContinuousStateTransitionsDataset(data=data, features=feature_domains, targets=target_domains)

    return dataset


def continuous_state_transitions_dataset_from_array(data, feature_domains=None, target_domains=None, feature_names=None, target_names=None):
    """ Create a ContinuousStateTransitionsDataset from given data according to variables domains if given or variable names if given.

    Feature/target variables names are automatically generated if not given:
        x_0, x_1, ... for features.
        y_0, y_1, ... for targets.
    Feature/target variables domains are extracted from data if not given as argument.

    Args:
        data: list of tuple (list of float, list of float).
            Multiset of state transitions.
        feature_domains: list of (String, Continuum)
            Name and domain of each feature variable (optional)
        target_domains: list of (String, Continuum)
            Name and domain of each target variable (optional)
        feature_names: list of String
            Names of the feature variables (optional).
            Should not be given if feature_domains is given.
        target_names: list of String
            Names of the target variables (optional).
            Should not be given if target_domains is given.
    Returns:
        ContinuousStateTransitionsDataset.
            A pylfit dataset api encoding of the state transitions, ready to be used with pylfit model api.
    """

    # Check data type
    if not isinstance(data, (list)):
        raise TypeError("Argument data must be a list.")
    if not all(isinstance(i, tuple) for i in data):
        raise TypeError("Argument data must only contains tuples.")
    if not all(len(i) == 2 for i in data):
        raise TypeError("Argument data tuples has to be of size 2.")
    if not all(isinstance(i, (float,int)) for s1,s2 in data for i in s1 ):
        raise ValueError("Argument data feature states values must be int or float")
    if not all(isinstance(i, (float,int)) for s1,s2 in data for i in s2 ):
        raise ValueError("Argument data target states values must be int or float")

    # Check feature_domains type
    if feature_domains is not None:
        if not isinstance(feature_domains, list):
            raise TypeError("Argument feature_domains must be a list")
        if not all(isinstance(i, tuple) for i in feature_domains):
            raise TypeError("Argument feature_domains must be a list of tuple")
        if not all(len(i) == 2 for i in feature_domains):
            raise TypeError("Argument feature_domains must be a list of tuple of size 2")
        if not all(isinstance(var, str) for var,vals in feature_domains):
            raise TypeError("Argument feature_domains must be a list of tuple (String, Continuum)")
        if not all(isinstance(vals, Continuum) for var,vals in feature_domains):
            raise TypeError("Argument feature_domains must be a list of tuple (String, Continuum)")
        if not len([var for var, vals in feature_domains]) == len(set([var for var, vals in feature_domains])):
            raise ValueError("Argument feature_domains, each variable name must be unique")

    # Check target_domains type
    if target_domains is not None:
        if not isinstance(target_domains, list):
            raise TypeError("Argument target_domains must be a list")
        if not all(isinstance(i, tuple) for i in target_domains):
            raise TypeError("Argument target_domains must be a list of tuple")
        if not all(len(i) == 2 for i in target_domains):
            raise TypeError("Argument target_domains must be a list of tuple of size 2")
        if not all(isinstance(var, str) for var,vals in target_domains):
            raise TypeError("Argument target_domains must be a list of tuple (String, Continuum)")
        if not all(isinstance(vals, Continuum) for var,vals in target_domains):
            raise TypeError("Argument target_domains must be a list of tuple (String, Continuum)")
        if not len([var for var, vals in target_domains]) == len(set([var for var, vals in target_domains])):
            raise ValueError("Argument target_domains, each variable name must be unique")

    # Check feature_names type
    if feature_names is not None:
        if feature_domains is not None:
            raise ValueError("Argument feature_names should not be given when feature_domains is specified")
        if not isinstance(feature_names, (list)):
            raise TypeError("Argument feature_names must be a list.")
        if not all(isinstance(i, str) for i in feature_names):
            raise ValueError("Argument feature_names must only contains String.")

    # Check target_names type
    if target_names is not None:
        if target_domains is not None:
            raise ValueError("Argument target_names should not be given when target_domains is specified")
        if not isinstance(target_names, (list)):
            raise TypeError("Argument target_names must be a list.")
        if not all(isinstance(i, str) for i in target_names):
            raise ValueError("Argument target_names must only contains String.")

    # Empty dataset
    if len(data) == 0:
        if feature_domains is None:
            raise ValueError("Features domain must not be None if data is empty")
        if target_domains is None:
            raise ValueError("Features and targets domain must not be None if data is empty")

        return ContinuousStateTransitionsDataset(data=[], features=feature_domains, targets=target_domains)

    # Initialize feature/target variables domain
    if feature_names is not None and feature_domains is None:
        feature_domains = [(str(i), Continuum()) for i in feature_names]

    if target_names is not None and target_domains is None:
        target_domains = [(str(i), Continuum()) for i in target_names]

    if feature_domains is None:
        feature_domains = [("x"+str(i), Continuum()) for i in range(len(data[0][0]))]
    if target_domains is None:
        target_domains = [("y"+str(i), Continuum()) for i in range(len(data[0][1]))]

    # Check values of data
    nb_features = len(data[0][0])
    nb_targets = len(data[0][1])

    if not all(len(s1) == nb_features for (s1,s2) in data):
        raise ValueError("data feature states must be of same size.")
    if not all(len(s2) == nb_targets for (s1,s2) in data):
        raise ValueError("data target states must be of same size.")

    if not nb_features == len(feature_domains):
        raise ValueError("Size of argument feature_domains and feature_names must be same as argument data feature states size.")
    if not nb_targets == len(target_domains):
        raise ValueError("Size of argument target_domains and target_names must be same as argument data target states size.")

    # Convert data format to ContinuousStateTransitionsDataset format
    data_encoded = [(numpy.array([float(i) for i in x]), numpy.array([float(i) for i in y])) for x, y in data]

    # Extract feature/target variables domain from data
    feature_domains = feature_domains.copy()
    target_domains = target_domains.copy()
    for (s1, s2) in data_encoded:
        for var_id, value in enumerate(s1):
            if not feature_domains[var_id][1].includes(value):
                if feature_domains[var_id][1].is_empty():
                    feature_domains[var_id] = (feature_domains[var_id][0], Continuum(value,value, True, True))
                elif value < feature_domains[var_id][1].min_value:
                    feature_domains[var_id][1].set_lower_bound(value, True)
                else:
                    feature_domains[var_id][1].set_upper_bound(value, True)

        for var_id, value in enumerate(s2):
            if not target_domains[var_id][1].includes(value):
                if target_domains[var_id][1].is_empty():
                    target_domains[var_id] = (target_domains[var_id][0], Continuum(value,value, True, True))
                elif value < target_domains[var_id][1].min_value:
                    target_domains[var_id][1].set_lower_bound(value, True)
                else:
                    target_domains[var_id][1].set_upper_bound(value, True)

    dataset = ContinuousStateTransitionsDataset(data=data_encoded, features=feature_domains, targets=target_domains)

    return dataset
