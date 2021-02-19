#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2021/01/14
# @updated: 2021/01/14
#
# @desc: class Model python source code file
#-------------------------------------------------------------------------------

from ..utils import eprint

from ..datasets import Dataset
from ..algorithms import Algorithm

import abc

class Model:
    """
    Define a pylfit model.

    Args:
        algorithm: pylfit.algorithm.Algorithm subclass,
            the algorithm to be used to fit the model.
    """

    """ Dataset types compatible with the model: list of pylfit.datasets.Dataset subclasses. """
    _COMPATIBLE_DATASETS = []

    """ Names of learning algorithms that can be used to fit this model: list of String. """
    _ALGORITHMS = []

    """ Optimization parameters. """
    _OPTIMIZERS = []

#--------------
# Constructors
#--------------

    def __init__(self):
        """
        Create an empty model instance.
        Ensure class attribute have correct values w.r.t. pylfit API.
        """

        if not isinstance(self._COMPATIBLE_DATASETS, list) or not all(issubclass(i, Dataset) for i in self._COMPATIBLE_DATASETS):
            raise ValueError('_COMPATIBLE_DATASETS wrong type: must be a list of pylfit.dataset.Dataset subclasses.')

        if len(self._COMPATIBLE_DATASETS) == 0:
            raise ValueError('_COMPATIBLE_DATASETS is empty: should atleast contain one pylfit.dataset.Dataset subclass.')

        if not isinstance(self._ALGORITHMS, list) or not all(isinstance(i, str) for i in self._ALGORITHMS):
            raise ValueError('_ALGORITHMS wrong type: must be a list of String.')

        # TODO check optimizers

        self.algorithm = None

        #raise NotImplementedError('Must be implemented in subclasses.')

#--------------
# Operators
#--------------

    def __str__(self):
        return self.to_string()

    def _repr__(self):
        return self.to_string()

#--------------
# Methods
#--------------

    @abc.abstractmethod
    def compile(self, algorithm):
        """

        """
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def fit(self, dataset):
        """

        """
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
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
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def to_string(self):
        """
        Convert the object to a readable string

        Returns:
            String
                a readable representation of the object
        """
        raise NotImplementedError('Must be implemented in subclasses.')

#--------
# Static
#--------


#--------------
# Accessors
#--------------

    # TODO: check param type/format

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        self._algorithm = value
