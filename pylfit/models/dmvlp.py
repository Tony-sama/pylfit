#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2021/01/01
# @updated: 2021/02/05
#
# @desc: class DMVLP python source code file
#-------------------------------------------------------------------------------

from ..models import Model

from ..utils import eprint
from ..objects import Rule

from ..datasets import StateTransitionsDataset

from ..algorithms import GULA
from ..algorithms import PRIDE

from ..semantics import Synchronous, Asynchronous, General

import itertools
import random
import numpy

class DMVLP(Model):
    """
    Define a Dynamic Multi-Valued Logic Program (DMVLP), a set of rules over features/target variables/values
    that can encode the dynamics of a discrete dynamic system (also work for static systems).

    Args:
        features: list of (string, list of string).
            Variables and their values that appear in body of rules
        targets: list of (string, list of string).
            Variables that appear in body of rules.
        rules: list of pylfit.objects.Rule.
            Logic rules of the program.
        algorithm: pyflfit.algorithm.Algorithm subclass.
            The algorithm to be used for fiting the model.
    """


    """ Dataset types compatible with dmvlp """
    _COMPATIBLE_DATASETS = [StateTransitionsDataset]

    """ Learning algorithms that can be use to fit this model """
    _ALGORITHMS = ["gula", "pride", "lf1t"]

    """ Optimization """
    _OPTIMIZERS = []

#--------------
# Constructors
#--------------

    def __init__(self, features, targets, rules=[]):
        """
        Create a DMVLP instance from given features/targets variables and optional rules

        Args:
            features: list of pairs (String, list of String),
                labels of the features variables and their values (appear only in body of rules and constraints).
            targets: list of pairs (String, list of String),
                labels of the targets variables and their values (appear in head of rules and body of constraint).
            rules: list of Rule,
                rules that define logic program dynamics: influences of feature variables values over target variables values.
        """
        super().__init__()

        self.features = features
        self.targets = targets
        self.rules = rules

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

    def compile(self, algorithm="gula"):
        """
        Set the algorithm to be used to fit the model.
        Supported algorithms:
            - "gula", General Usage LFIT Algorithm (TODO)
            - "pride", (TODO)
            - "lf1t", (TODO)

        """

        if algorithm not in DMVLP._ALGORITHMS:
            raise ValueError('algorithm parameter must be one element of DMVLP._COMPATIBLE_ALGORITHMS: '+str(DMVLP._ALGORITHMS)+'.')

        if algorithm == "gula":
            self.algorithm = GULA
        elif algorithm == "pride":
            self.algorithm = PRIDE
        else:
            raise NotImplementedError('<DEV> algorithm="'+str(algorithm)+'" is in DMVLP._COMPATIBLE_ALGORITHMS but no behavior implemented.')

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
            msg = 'Dataset type (' + str(dataset.__class__.__name__)+ ') not suported by DMVLP model.'
            raise ValueError(msg)

        # TODO: add time serie management
        #eprint("algorithm set to " + str(self.algorithm))

        msg = 'Dataset type (' + str(dataset.__class__.__name__) + ') not supported \
        by the algorithm (' + str(self.algorithm.__class__.__name__) + '). \
        Dataset must be of type ' + str(StateTransitionsDataset.__class__.__name__)

        if self.algorithm == GULA:
            if not isinstance(dataset, StateTransitionsDataset):
                raise ValueError(msg)
            if verbose > 0:
                eprint("Starting fit with GULA")
            self.rules = GULA.fit(dataset=dataset,verbose=verbose) #, targets_to_learn={'y1': ['1']})
        elif self.algorithm == PRIDE:
            if not isinstance(dataset, StateTransitionsDataset):
                raise ValueError(msg)
            if verbose > 0:
                eprint("Starting fit with PRIDE")
            self.rules = PRIDE.fit(dataset=dataset,verbose=verbose)
        else:
            raise NotImplementedError("Algorithm usage not implemented yet")

        # TODO
        #raise NotImplementedError('Not implemented yet')

    def predict(self, feature_state, semantics="synchronous"):
        """
        Predict the possible target states of the given feature state according to the model rules.

        Args:
            feature_state: list of String
                Feature state from wich target state must be predicted.
            semantics: String (optional)
                The dynamic semantics used to generate the target states.
        """

        # Encode feature state with domain value id
        feature_state_encoded = []
        for var_id, val in enumerate(feature_state):
            val_id = self.features[var_id][1].index(str(val))
            feature_state_encoded.append(val_id)

        #eprint(feature_state_encoded)

        target_states = []
        if semantics == "synchronous":
            target_states = Synchronous.next(feature_state_encoded, self.targets, self.rules)
        if semantics == "asynchronous":
            if len(self.features) != len(self.targets):
                raise ValueError("Asynchronous semantics can only be used if features and targets variables are the same (for now).")
            target_states = Asynchronous.next(feature_state_encoded, self.targets, self.rules)
        if semantics == "general":
            if len(self.features) != len(self.targets):
                raise ValueError("General semantics can only be used if features and targets variables are the same (for now).")
            target_states = General.next(feature_state_encoded, self.targets, self.rules)


        # Decode target states
        output = []
        for s in target_states:
            target_state = []
            for var_id, val_id in enumerate(s):
                #eprint(var_id, val_id)
                if val_id == -1:
                    target_state.append("?")
                else:
                    target_state.append(self.targets[var_id][1][val_id])
            output.append(target_state)

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
        if self.algorithm is None:
            raise ValueError('Model has not been built: compile(algorithm) must be called before using summary.')

        # TODO: proper log, check Keras style

        if print_fn == None:
            print_fn = print
        print_fn(str(self.__class__.__name__) + " summary:")
        print_fn(" Algorithm: " + str(self.algorithm.__name__) + ' (' + str(self.algorithm) + ')')
        print_fn(" Features: ")
        for var in self.features:
            print_fn('  ' + str(var[0]) + ': ' + str(list(var[1])))
        print_fn(" Targets: ")
        for var in self.targets:
            print_fn('  ' + str(var[0]) + ': ' + str(list(var[1])))
        if len(self.rules) == 0:
            print_fn(' Rules: []')
        else:
            print_fn(" Rules:")
            for r in self.rules:
                print_fn("  "+r.logic_form(self.features, self.targets))

    def to_string(self):
        """
        Convert the object to a readable string

        Returns:
            String
                a readable representation of the object
        """
        output = "{\n"
        output += "Algorithm: " + str(self.algorithm.__name__)
        output += "\nFeatures: " + str(self.features)
        output += "\nTargets: " + str(self.targets)
        output += "\nRules:\n"
        for r in self.rules:
            output += r.logic_form(self.features, self.targets) + "\n"
        output += "}"

        return output

    def feature_states(self, value_id_encoded=False):
        """
        Compute all possible feature states of the logic program:
        all combination of variables values

        Returns:
            - list of (list of string) if value_id_encoded=False
                All possible feature state of the logic program with their domain string label
            - list of (list of int) if value_id_encoded=True
                All possible feature state of the logic program with their domain value id
        """
        if value_id_encoded:
            values_ids = [[j for j in range(0,len(self.features[i][1]))] for i in range(0,len(self.features))]
        else:
            values_ids = [[self.features[i][1][j] for j in range(0,len(self.features[i][1]))] for i in range(0,len(self.features))]
        output = [list(i) for i in list(itertools.product(*values_ids))]
        return output

    def target_states(self, value_id_encoded=False):
        """
        Compute all possible target state of the logic program:
        all combination of variables values

        Returns:
            - list of (list of string) if value_id_encoded=False
                All possible target state of the logic program with their domain string label
            - list of (list of int) if value_id_encoded=True
                All possible target state of the logic program with their domain value id
        """
        if value_id_encoded:
            values_ids = [[j for j in range(0,len(self.targets[i][1]))] for i in range(0,len(self.targets))]
        else:
            values_ids = [[self.targets[i][1][j] for j in range(0,len(self.targets[i][1]))] for i in range(0,len(self.targets))]
        output = [list(i) for i in list(itertools.product(*values_ids))]
        return output

#--------
# Static
#--------


#--------------
# Properties
#--------------


    # TODO: check param type/format

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        if not isinstance(value, list):
            raise TypeError("features must be a list")
        if not all(isinstance(i, tuple) for i in value):
            raise TypeError("features must contain tuples")
        if not all(len(i)==2 for i in value):
            raise TypeError("features tuples must be of size 2")
        if not all(isinstance(domain, list) for (var,domain) in value):
            raise TypeError("features domains must be a list")
        if not all(isinstance(val, str) for (var,domain) in value for val in domain):
            raise ValueError("features domain values must be String")

        self._features = value.copy()

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, value):
        if not isinstance(value, list):
            raise TypeError("features must be a list")
        if not all(isinstance(i, tuple) for i in value):
            raise TypeError("features must contain tuples")
        if not all(len(i)==2 for i in value):
            raise TypeError("features tuples must be of size 2")
        if not all(isinstance(domain, list) for (var,domain) in value):
            raise TypeError("features domains must be a list")
        if not all(isinstance(val, str) for (var,domain) in value for val in domain):
            raise ValueError("features domain values must be String")

        self._targets = value.copy()

    @property
    def rules(self):
        return self._rules

    @rules.setter
    def rules(self, value):
        self._rules = value.copy()

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        self._algorithm = value
