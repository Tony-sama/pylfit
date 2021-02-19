#-----------------------
# @author: Tony Ribeiro
# @created: 2020/12/23
# @updated: 2021/01/13
#
# @desc: Interface class of LFIT datasets
#   - Features/target variables labels and domain
#   - State values are encode by their value id
#-----------------------

from ..utils import eprint

import abc

class Dataset():
    """ Abstract class of LFIT algorithm input datasets

    """

    """ Dataset class must implement a data attribute """
    #_data = []

    """ Feature variables name and domain: list of pair (string, list of string) """
    #_features = []

    """ Target variables name and domain: list of pair (string, list of string) """
    #_targets = []

#--------------
# Constructors
#--------------

    def __init__(self, data, features, targets):
        """
        Constructor of an empty dataset

        Args:
            features: list of pair (string, list of objects)
                Feature variables name and domain
            targets: list of pair (string, list of objects)
                Target variables name and domain
        """

        raise NotImplementedError('Must be implemented in subclasses.')

#--------------
# Operators
#--------------

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    @abc.abstractmethod
    def to_string(self):
        """
        Convert the object to a readable string

        Returns:
            String
                a readable representation of the object
        """
        raise NotImplementedError('Must be implemented in subclasses.')

#--------------
# Accessors
#--------------

    @property
    def data(self):
        raise NotImplementedError('Must be implemented in subclasses.')

    @data.setter
    def data(self, value):
        raise NotImplementedError('Must be implemented in subclasses.')

    @property
    def features(self):
        raise NotImplementedError('Must be implemented in subclasses.')

    @features.setter
    def features(self, value):
        raise NotImplementedError('Must be implemented in subclasses.')

    @property
    def targets(self):
        raise NotImplementedError('Must be implemented in subclasses.')

    @targets.setter
    def targets(self, value):
        raise NotImplementedError('Must be implemented in subclasses.')
