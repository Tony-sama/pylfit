#-----------------------
# @author: Tony Ribeiro
# @created: 2021/01/13
# @updated: 2021/01/13
#
# @desc: Dataset container class for state transitions data
#   - Features/target variables labels and domain
#   - Data is a list of pair of feature/target states
#       - State are nparray of int following the order of features/targets
#       - Integer value correspond to domain id of corresponding feature/target variable
#-----------------------

from ..utils import eprint

from ..datasets import Dataset

import numpy
import pandas
import csv

class StateTransitionsDataset(Dataset):
    """ Container class for state transitions dataset

    Parameters:
        data: list of tuple (nparray of string, nparray of string)
            State transitions as pair of feature/target states
        features: list of tuple (string, list of string)
            Feature variables name and domain.
            Can have values not appearing in data.
            Missing data values will be added.
        targets: list of tuple (string, list of string)
            Target variables name and domain.
            Can have values not appearing in data.
            Missing data values will be added.
    """

#--------------
# Constructors
#--------------

    def __init__(self, data, features, targets):
        """
        Constructor of a state transitions dataset

        Args:
            data: list of tuple (nparray of string, nparray of string)
                State transitions as pair of feature/target states
            features: list of tuple (string, list of string)
                Feature variables name and domain.
                Can have values not appearing in data.
                Missing data values will be added.
            targets: list of tuple (string, list of string)
                Target variables name and domain.
                Can have values not appearing in data.
                Missing data values will be added.
        """

        self.data = data
        self.features = features
        self.targets = targets

        # Check data values are in features/targets domains
        for transition_id, (s1,s2) in enumerate(data):

            # Check size of states
            if len(s1) != len(self.features):
                raise ValueError("Transition " + str((s1,s2)) + ": feature state of wrong size, features size is " + str(len(self.features)))
            if len(s2) != len(self.targets):
                raise ValueError("Transition " + str((s1,s2)) + ": target state of wrong size, targets size is " + str(len(self.targets)))

            for var_id, val in enumerate(s1):
                if str(val) not in features[var_id][1]:
                    raise ValueError("Transition " + str((s1,s2)) + ": value not in features for variable " + str(var_id))
            for var_id, val in enumerate(s2):
                if str(val) not in targets[var_id][1]:
                    raise ValueError("Transition " + str((s1,s2)) + ": value not in targets for variable " + str(var_id))

#--------------
# Methods
#--------------

    def summary(self, line_length=None, print_fn=None):
        """
        Prints a string summary of the model.

        Args:
            line_length: int
                Total length of printed lines (e.g. set this to adapt the display to different terminal window sizes).
            print_fn: function
                Print function to use. Defaults to print.
                You can set it to a custom function in order to capture the string summary.
        """

        # TODO: proper log, check Keras style

        if print_fn == None:
            print_fn = print

        print_fn(str(self.__class__.__name__) + " summary:")
        print_fn(" Features: ")
        for var in self.features:
            print_fn('  ' + str(var[0]) + ': ' + str(list(var[1])))
        print_fn(" Targets: ")
        for var in self.targets:
            print_fn('  ' + str(var[0]) + ': ' + str(list(var[1])))
        if len(self.data) == 0:
            print_fn(' Data: []')
        else:
            print_fn(" Data:")
            for d in self.data:
                print_fn("  "+str( ([i for i in d[0]], [i for i in d[1]] )))

    def to_string(self):
        """
        Convert the object to a readable string

        Returns:
            String
                a readable representation of the object
        """
        output = "{"
        output += "Features: " + str(self.features)
        output += "\nTargets: " + str(self.targets)
        output += "\nData: " + str(self.data)
        output += "}"

        return output

    def to_csv(self, path_to_file):
        """
        Save the dataset content to csv format.

        Args:
            path_to_file: String.
                Path to the file.
        """
        with open(path_to_file, mode='w') as output_file:
            output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            output_writer.writerow([var for var,vals in self.features]+[var for var,vals in self.targets])
            for s1,s2 in self.data:
                output_writer.writerow(list(s1)+list(s2))

#--------------
# Operators
#--------------

#--------------
# Statics methods
#--------------

#--------------
# Accessors
#--------------

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):

        msg = "Constructing a "+self.__class__.__name__+" with wrong argument format: data must be a list of tuple (numpy.ndarray of string, numpy.ndarray of string)"

        if not isinstance(value, list): # data must be a list
            raise TypeError(msg)
        for transition in value:
            if not isinstance(transition, tuple) or len(transition) != 2: # transitions must be pairs
                raise TypeError(msg)
            feature_state = transition[0]
            target_state = transition[1]
            if not isinstance(feature_state, numpy.ndarray) or not isinstance(target_state, numpy.ndarray): # States must be ndarray
                raise ValueError(msg)
            if not all(isinstance(val, (str,int)) for val in feature_state) or not all(isinstance(val, (str,int)) for val in target_state): # Values must be string or int
                raise ValueError(msg)

        self._data = [(numpy.array([str(i) for i in s1]), numpy.array([str(i) for i in s2])) for s1, s2 in value]

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        if not isinstance(value, list) \
        or not all(isinstance(vals, list) for (var,vals) in value) \
        or not all(isinstance(val, str) for (var,vals) in value for val in vals):
            raise TypeError("Features must be a list of pair of (string, list of string)")
        self._features = value

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        if not isinstance(value, list) \
        or not all(isinstance(vals, list) for (var,vals) in value) \
        or not all(isinstance(val, str) for (var,vals) in value for val in vals):
            raise TypeError("Targets must be a list of pair of (string, list of string)")
        self._targets = value
