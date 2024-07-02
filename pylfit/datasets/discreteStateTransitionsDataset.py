#-----------------------
# @author: Tony Ribeiro
# @created: 2021/01/13
# @updated: 2023/12/27
#
# @desc: Dataset container class for state transitions data
#   - Features/target variables labels and domain
#   - Data is a list of pair of feature/target states
#       - State are nparray of int following the order of features/targets
#       - Integer value correspond to domain id of corresponding feature/target variable
#-----------------------

from ..utils import eprint

from ..datasets import Dataset
from ..objects.atom import Atom
from ..objects.legacyAtom import LegacyAtom

import numpy
import csv
import collections

class DiscreteStateTransitionsDataset(Dataset):
    """ Container class for state transitions dataset

    Parameters:
        data: list of tuple (nparray of string, nparray of string)
            State transitions as pair of feature/target states
        features: list of tuple (string, list of string)
            Feature variables name and domain.
            Can have values not appearing in data.
            Missing data values will be added.
        features_atoms: list of LegacyAtoms
            For each variable the corresponding void atom.
        targets: list of tuple (string, list of string)
            Target variables name and domain.
            Can have values not appearing in data.
            Missing data values will be added.
        unknown_values: list of string
            List of value representing unknown value.
    """

    """ Value representing unknown """
    _UNKNOWN_VALUE = Atom._UNKNOWN_VALUE

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
        self.features_void_atoms = {}
        self.targets_void_atoms = {}
        self.targets = targets

        # Compute void atoms
        for var_id, (var, vals) in enumerate(features):
            self.features_void_atoms[var] = LegacyAtom(var, vals, LegacyAtom._VOID_VALUE, var_id)
        for var_id, (var, vals) in enumerate(targets):
            self.targets_void_atoms[var] = LegacyAtom(var, vals, LegacyAtom._VOID_VALUE, var_id)

        # Check data values are in features/targets domains
        for transition_id, (s1,s2) in enumerate(data):

            # Check size of states
            if len(s1) != len(self.features):
                raise ValueError("Transition " + str((s1,s2)) + ": feature state of wrong size, features size is " + str(len(self.features)))
            if len(s2) != len(self.targets):
                raise ValueError("Transition " + str((s1,s2)) + ": target state of wrong size, targets size is " + str(len(self.targets)))

            for var_id, val in enumerate(s1):
                if val != self._UNKNOWN_VALUE and str(val) not in features[var_id][1]:
                    raise ValueError("Transition " + str((s1,s2)) + ": value not in features for variable " + str(var_id))
            for var_id, val in enumerate(s2):
                if val != self._UNKNOWN_VALUE and str(val) not in targets[var_id][1]:
                    raise ValueError("Transition " + str((s1,s2)) + ": value not in targets for variable " + str(var_id))

    def copy(self):
        return DiscreteStateTransitionsDataset(self.data, self.features, self.targets)
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
                print_fn("  "+str( ([i for i in d[0]],
                                    [i for i in d[1]] )))
        print_fn(" Unknown values: "+ str(self.nb_unknown_values))

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
        output += "\nData: " #+ str(self.data)
        for d in self.data:
            output += "\n"+str( ([i for i in d[0]],
                                [i for i in d[1]] ))
        output += "\n}"

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
# Observers
#--------------

    def has_unknown_values(self):
        return self.nb_unknown_values > 0

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

        msg = "Constructing a "+self.__class__.__name__+" with wrong argument format: data must be a list of tuple (list of string, list of string)"

        if not isinstance(value, list): # data must be a list
            raise TypeError(msg)
        for transition in value:
            if not isinstance(transition, tuple) or len(transition) != 2: # transitions must be pairs
                raise TypeError(msg)
            feature_state = transition[0]
            target_state = transition[1]
            if not all(isinstance(val, (str)) for val in feature_state) or not all(isinstance(val, (str)) for val in target_state): # Values must be string or int
                raise ValueError(msg)

        self._data = [(numpy.array([str(i) if str(i) != DiscreteStateTransitionsDataset._UNKNOWN_VALUE else DiscreteStateTransitionsDataset._UNKNOWN_VALUE for i in s1]),
                       numpy.array([str(i) if str(i) != DiscreteStateTransitionsDataset._UNKNOWN_VALUE else DiscreteStateTransitionsDataset._UNKNOWN_VALUE for i in s2]))
                       for s1, s2 in value]
        
        self.nb_unknown_values = 0
        for (i,j) in self.data:
            self.nb_unknown_values += numpy.count_nonzero(i == DiscreteStateTransitionsDataset._UNKNOWN_VALUE)
            self.nb_unknown_values += numpy.count_nonzero(j == DiscreteStateTransitionsDataset._UNKNOWN_VALUE)

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        if not isinstance(value, list) \
        or not all(isinstance(i, tuple) for i in value) or not all(len(i) == 2 for i in value) \
        or not all(isinstance(vals, list) for (var,vals) in value) \
        or not all(isinstance(val, str) for (var,vals) in value for val in vals):
            raise TypeError("Features must be a list of pair of (string, list of string)")

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
        or not all(isinstance(vals, list) for (var,vals) in value) \
        or not all(isinstance(val, str) for (var,vals) in value for val in vals):
            raise TypeError("Targets must be a list of pair of (string, list of string)")

        duplicate = [item for item, count in collections.Counter([var for var,vals in value]).items() if count > 1]
        if(len(duplicate) > 0):
            raise ValueError("Target variables name must be unique: "+str(duplicate)+" are duplicated")
        self._targets = value.copy()

    @property
    def features_void_atoms(self):
        return self._features_void_atoms
    
    @features_void_atoms.setter
    def features_void_atoms(self, value):
        self._features_void_atoms = value

    @property
    def targets_void_atoms(self):
        return self._targets_void_atoms
    
    @targets_void_atoms.setter
    def targets_void_atoms(self, value):
        self._targets_void_atoms = value

    @property
    def nb_unknown_values(self):
        return self._nb_unknown_values

    @nb_unknown_values.setter
    def nb_unknown_values(self, value):
        self._nb_unknown_values = value

