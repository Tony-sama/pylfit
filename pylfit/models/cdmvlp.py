#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2021/02/03
# @updated: 2021/06/15
#
# @desc: class CDMVLP python source code file
#-------------------------------------------------------------------------------

from . import DMVLP

from ..utils import eprint
from ..objects import Rule

from ..datasets import StateTransitionsDataset

from ..algorithms import Synchronizer

from ..semantics import SynchronousConstrained

import itertools
import random
import numpy

class CDMVLP(DMVLP):
    """
    Define a Constrained Dynamic Multi-Valued Logic Program (CDMVLP), a set of rules and constraints over features/target variables/values
    that can encode the dynamics of a discrete dynamic system (also work for static systems).

    Args:
        features: list of (string, list of string).
            Variables and their values that appear in body of rules
        targets: list of (string, list of string).
            Variables that appear in body of rules.
        rules: list of pylfit.objects.Rule.
            Logic rules of the program.
        constraints: list of pylfit.objects.Rule.
            Logic constraints of the program.
        algorithm: pyflfit.algorithm.Algorithm subclass.
            The algorithm to be used for fiting the model.
    """


    """ Dataset types compatible with dmvlp """
    _COMPATIBLE_DATASETS = [StateTransitionsDataset]

    """ Learning algorithms that can be use to fit this model """
    _ALGORITHMS = ["synchronizer","synchronizer-pride"]

    """ Optimization """
    _OPTIMIZERS = []

#--------------
# Constructors
#--------------

    def __init__(self, features, targets, rules=[], constraints=[]):
        """
        Create a CDMVLP instance from given features/targets variables and optional rules/constraints

        Args:
            features: list of pairs (String, list of String),
                labels of the features variables and their values (appear only in body of rules and constraints).
            targets: list of pairs (String, list of String),
                labels of the targets variables and their values (appear in head of rules and body of constraint).
            rules: list of pylfit.objects.Rule,
                rules that define logic program dynamics: influences of feature variables values over target variables values.
            cosntraints: list of pylfit.objects.Rule,
                constraints that restrict the logic program dynamics: prevent some combination of target to appear in a transition.
        """
        super().__init__(features, targets, rules)
        self.constraints = constraints

#--------------
# Operators
#--------------

#--------------
# Methods
#--------------

    def compile(self, algorithm="synchronizer"):
        """
        Set the algorithm to be used to fit the model.
        Supported algorithms:
            - "synchronizer", use GULA to learn rules and GULA again for constraints (TODO)

        """

        if algorithm not in CDMVLP._ALGORITHMS:
            raise ValueError('algorithm parameter must be one element of CDMVLP._COMPATIBLE_ALGORITHMS: '+str(CDMVLP._ALGORITHMS)+'.')

        if algorithm == "synchronizer":
            self.algorithm = "synchronizer"
        elif algorithm == "synchronizer-pride":
            self.algorithm = "synchronizer-pride"
        else:
            raise NotImplementedError('<DEV> algorithm="'+str(algorithm)+'" is in CDMVLP._COMPATIBLE_ALGORITHMS but no behavior implemented.')

    def fit(self, dataset, verbose=0):
        """
        Use the algorithm set by compile() to fit the rules to the dataset.
            - Learn a model from scratch using the chosen algorithm.
            - update model (TODO).

        Check and encode dataset to be used by the desired algorithm.

        Raises:
            ValueError if the dataset can't be used with the algorithm.

        """

        if not any(isinstance(dataset, i) for i in self._COMPATIBLE_DATASETS):
            msg = 'Dataset type (' + str(dataset.__class__.__name__)+ ') not suported by CDMVLP model.'
            raise ValueError(msg)

        # TODO: add time serie management
        #eprint("algorithm set to " + str(self.algorithm))

        if self.algorithm not in CDMVLP._ALGORITHMS:
            raise ValueError('algorithm property must be one element of CDMVLP._COMPATIBLE_ALGORITHMS: '+str(CDMVLP._ALGORITHMS)+'.')

        msg = 'Dataset type (' + str(dataset.__class__.__name__) + ') not supported \
        by the algorithm (' + str(self.algorithm.__class__.__name__) + '). \
        Dataset must be of type ' + str(StateTransitionsDataset.__class__.__name__)

        if self.algorithm == "synchronizer":
            if not isinstance(dataset, StateTransitionsDataset):
                raise ValueError(msg)
            if verbose > 0:
                eprint("Starting fit with Synchronizer using GULA")
            self.rules, self.constraints = Synchronizer.fit(dataset=dataset, verbose=verbose)
        elif self.algorithm == "synchronizer-pride":
            if not isinstance(dataset, StateTransitionsDataset):
                raise ValueError(msg)
            if verbose > 0:
                eprint("Starting fit with Synchronizer using PRIDE")
            self.rules, self.constraints = Synchronizer.fit(dataset=dataset, complete=False, verbose=verbose)
        else:
            raise NotImplementedError('<DEV> self.algorithm="'+str(self.algorithm)+'" is in CDMVLP._COMPATIBLE_ALGORITHMS but no behavior implemented.')

    def predict(self, feature_states, semantics="synchronous-constrained"):
        """
        Predict the possible target states of the given feature states according to the model rules and constraints.

        Args:
            feature_states: list of list of String
                Feature states from wich target state must be predicted.
            semantics: String (optional)
                The dynamic semantics used to generate the target states.
        Returns:
            A list of pair (feature state, targets states): list of (list of String, list of list of String)
        """
        if not isinstance(feature_states, list):
            raise TypeError("Argument feature_states must be a list of list of strings")
        if not all(isinstance(i,(list,tuple,numpy.ndarray)) for i in feature_states):
            raise TypeError("Argument feature_states must be a list of list of strings")
        if not all(isinstance(j,str) for i in feature_states for j in i):
            raise TypeError("Argument feature_states must be a list of list of strings")
        if not all(len(i) == len(self.features) for i in feature_states):
            raise TypeError("Features state must correspond to the model feature variables (bad length)")

        output = []
        for feature_state in feature_states:

            # Encode feature state with domain value id
            feature_state_encoded = []
            for var_id, val in enumerate(feature_state):
                val_id = self.features[var_id][1].index(str(val))
                feature_state_encoded.append(val_id)

            #eprint(feature_state_encoded)

            target_states = SynchronousConstrained.next(feature_state_encoded, self.targets, self.rules, self.constraints)

            # Decode target states
            local_output = []
            for s in target_states:
                target_state = []
                for var_id, val_id in enumerate(s):
                    #eprint(var_id, val_id)
                    if val_id == -1:
                        target_state.append("?")
                    else:
                        target_state.append(self.targets[var_id][1][val_id])
                local_output.append(target_state)
            output.append((list(feature_state), local_output))

        return output

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
        super().summary(line_length, print_fn)

        if print_fn == None:
            print_fn = print

        if len(self.constraints) == 0:
            print_fn(" Constraints: []")
        else:
            print_fn(" Constraints:")
            for r in self.constraints:
                print_fn("  "+r.logic_form(self.features, self.targets))

    def to_string(self):
        """
        Convert the object to a readable string

        Returns:
            String
                a readable representation of the object
        """
        output = super().to_string()[:-1]
        for r in self.constraints:
            output += r.logic_form(self.features, self.targets) + "\n"
        output += "}"

        return output

#--------
# Static
#--------


#--------------
# Properties
#--------------


    # TODO: check param type/format

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, value):
        self._constraints = value.copy()

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        self._algorithm = value
