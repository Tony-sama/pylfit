#-----------------------
# @author: Tony Ribeiro
# @created: 2022/08/24
# @updated: 2023/12/27
#
# @desc: Dataset container class for contiuous state transitions data
#   - Features/target variables labels and continuum domain
#   - Data is a list of pair of feature/target states
#       - State are nparray of float following the order of features/targets
#-----------------------

from ..utils import eprint

from ..datasets import Dataset

from ..objects import Continuum

import numpy
import csv
import collections

class ContinuousStateTransitionsDataset(Dataset):
    """ Container class for state transitions dataset

    Parameters:
        data: list of tuple (nparray of float, nparray of float)
            State transitions as pair of feature/target states
        features: list of tuple (float, Continuum)
            Feature variables name and domain.
        targets: list of tuple (float, Continuum)
            Target variables name and domain.
    """

#--------------
# Constructors
#--------------

    def __init__(self, data, features, targets):
        """
        Constructor of a continuous state transitions dataset

        Args:
            data: list of tuple (nparray of float, nparray of float)
                State transitions as pair of feature/target states
            features: list of tuple (float, Continuum)
                Feature variables name and domain.
                Can be extended to cover all observed values.
            targets: list of tuple (float, Continuum)
                Target variables name and domain.
                Can be extended to cover all observed values.
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
                if not features[var_id][1].includes(val):
                    raise ValueError("Transition " + str((s1,s2)) + ": value not in domain of feature variable " + str(var_id))
            for var_id, val in enumerate(s2):
                if not targets[var_id][1].includes(val):
                    raise ValueError("Transition " + str((s1,s2)) + ": value not in domain of feature variable " + str(var_id))

    def copy(self):
        return ContinuousStateTransitionsDataset(self.data, self.features, self.targets)
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
            print_fn('  ' + str(var[0]) + ': ' + str(var[1]))
        print_fn(" Targets: ")
        for var in self.targets:
            print_fn('  ' + str(var[0]) + ': ' + str(var[1]))
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
        with open(path_to_file, mode='w', newline='') as output_file:
            output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            output_writer.writerow([var for var,vals in self.features]+[var for var,vals in self.targets])
            for s1,s2 in self.data:
                output_writer.writerow(list(s1)+list(s2))

#--------------
# Operators
#--------------

    def __eq__(self, dataset):
        if (self.features != dataset.features):
            return False
        if (self.targets != dataset.targets):
            return False
        if len(self.data) != len(dataset.data):
            return False
        for id, (i,j) in enumerate(self.data):
            if (dataset.data[id][0] != i).any() or (dataset.data[id][1] != j).any():
                return False
        return True

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

        msg = "Constructing a "+self.__class__.__name__+" with wrong argument format: data must be a list of tuple (list of float, list of float)"

        if not isinstance(value, list): # data must be a list
            raise TypeError(msg)
        for transition in value:
            if not isinstance(transition, tuple) or len(transition) != 2: # transitions must be pairs
                raise TypeError(msg)
            feature_state = transition[0]
            target_state = transition[1]
            if not all(isinstance(val, (float,int)) for val in feature_state) or not all(isinstance(val, (float,int)) for val in target_state): # Values must be float or int
                raise ValueError(msg)

        self._data = [(numpy.array([float(i) for i in s1]), numpy.array([float(i) for i in s2])) for s1, s2 in value]

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        if not isinstance(value, list) \
        or not all(isinstance(i, tuple) for i in value) or not all(len(i) == 2 for i in value) \
        or not all(isinstance(vals, Continuum) for (var,vals) in value):
            raise TypeError("Features must be a list of pair of (string, Continuum)")

        duplicate = [item for item, count in collections.Counter([var for var,vals in value]).items() if count > 1]
        if(len(duplicate) > 0):
            raise ValueError("Feature variables name must be unique: "+str(duplicate)+" are duplicated")
        self._features = value.copy()

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        if not isinstance(value, list) \
        or not all(isinstance(i, tuple) for i in value) or not all(len(i) == 2 for i in value) \
        or not all(isinstance(vals, Continuum) for (var,vals) in value):
            raise TypeError("Targets must be a list of pair of (string, Continuum)")

        duplicate = [item for item, count in collections.Counter([var for var,vals in value]).items() if count > 1]
        if(len(duplicate) > 0):
            raise ValueError("Target variables name must be unique: "+str(duplicate)+" are duplicated")
        self._targets = value.copy()
