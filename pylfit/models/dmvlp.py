#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2021/01/01
# @updated: 2023/12/27
#
# @desc: class DMVLP python source code file
#-------------------------------------------------------------------------------

from ..utils import eprint
from ..models import Model
from ..objects import LegacyAtom
from ..objects import Rule
from ..datasets import DiscreteStateTransitionsDataset
from ..algorithms import GULA, PRIDE
from ..semantics import Synchronous, Asynchronous, General

import itertools
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
    _COMPATIBLE_DATASETS = [DiscreteStateTransitionsDataset]

    """ Learning algorithms that can be use to fit this model """
    _ALGORITHMS = ["gula", "pride"]

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

    def copy(self):
        output = DMVLP(self.features, self.targets, self.rules)
        output.algorithm = self.algorithm
        return output
#--------------
# Operators
#--------------

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

#--------------
# Methods
#--------------

    def compile(self, algorithm="gula"):
        """
        Set the algorithm to be used to fit the model.
        Supported algorithms:
            - "gula", General Usage LFIT Algorithm
            - "pride", Polynomial heuristic version of GULA

        """

        if algorithm not in DMVLP._ALGORITHMS:
            raise ValueError('algorithm parameter must be one element of DMVLP._COMPATIBLE_ALGORITHMS: '+str(DMVLP._ALGORITHMS)+'.')

        if algorithm == "gula":
            self.algorithm = "gula"
        elif algorithm == "pride":
            self.algorithm = "pride"
        else:
            raise NotImplementedError('<DEV> algorithm="'+str(algorithm)+'" is in DMVLP._COMPATIBLE_ALGORITHMS but no behavior implemented.')

    def fit(self, dataset, options=None): # targets_to_learn=None, verbose=0, heuristics=None, threads=1):
        """
        Use the algorithm set by compile() to fit the rules to the dataset.
            - Learn a model from scratch using the chosen algorithm.
            - update model (TODO).

        Check and encode dataset to be used by the desired algorithm.

        Raises:
            ValueError if the dataset can't be used with the algorithm.

        Args:
            dataset: pylfit.datasets.DiscreteStateTransitionsDataset
                state transitions of a the system
            options: dict string => any
                targets_to_learn: dict of {String: list of String}
                    target variables values of the dataset for wich we want to learn rules.
                    If not given, all targets values will be learned.
                verbose: int (0 or 1)
                heuristics: list of string (only for PRIDE)
                threads: int (>=1)
        """
        # Options
        targets_to_learn = None
        verbose = 0
        heuristics = None
        threads = 1

        if options is not None:
            if "targets_to_learn" in options:
                targets_to_learn = options["targets_to_learn"]
            if "verbose" in options:
                verbose = options["verbose"]
            if "heuristics" in options:
                heuristics = options["heuristics"]
            if "threads" in options:
                threads = options["threads"]
            
        # Check parameters
        if not any(isinstance(dataset, i) for i in self._COMPATIBLE_DATASETS):
            msg = 'Dataset type (' + str(dataset.__class__.__name__)+ ') not suported by DMVLP model, must be one of '+ \
            str([i for i in self._COMPATIBLE_DATASETS])
            raise ValueError(msg)

        if targets_to_learn is None:
            targets_to_learn = dict()
            for a, b in dataset.targets:
                targets_to_learn[a] = b
        elif not isinstance(targets_to_learn, dict) \
            or not all(isinstance(key, str) and isinstance(value, list) for key, value in targets_to_learn.items()) \
            or not all(isinstance(v, str) for key, value in targets_to_learn.items() for v in value):
            raise ValueError('targets_to_learn must be a dict of format {String: list of String}')
        else:
            for key, values in targets_to_learn.items():
                targets_names = [var for var, vals in dataset.targets]
                if key not in targets_names:
                    raise ValueError('targets_to_learn keys must be dataset target variables')
                var_id = targets_names.index(key)
                for val in values:
                    if val not in dataset.targets[var_id][1]:
                        raise ValueError('targets_to_learn values must be in target variable domain')

        if self.algorithm not in DMVLP._ALGORITHMS:
            raise ValueError('algorithm property must be one element of DMVLP._COMPATIBLE_ALGORITHMS: '+str(DMVLP._ALGORITHMS)+'.')

        msg = 'Dataset type (' + str(dataset.__class__.__name__) + ') not supported \
        by the algorithm (' + str(self.algorithm.__class__.__name__) + '). \
        Dataset must be of type ' + str(DiscreteStateTransitionsDataset.__class__.__name__)

        if self.algorithm == "gula":
            if not isinstance(dataset, DiscreteStateTransitionsDataset):
                raise ValueError(msg)
            if verbose > 0:
                eprint("Starting fit with GULA")
            self.rules = GULA.fit(dataset=dataset, targets_to_learn=targets_to_learn, verbose=verbose, threads=threads)
        elif self.algorithm == "pride":
            if not isinstance(dataset, DiscreteStateTransitionsDataset):
                raise ValueError(msg)
            if verbose > 0:
                eprint("Starting fit with PRIDE")
            self.rules = PRIDE.fit(dataset=dataset, options={"targets_to_learn":targets_to_learn, "verbose":verbose, "heuristics":heuristics, "threads":threads})
        else:
            raise NotImplementedError("Algorithm usage not implemented yet")

    def extend(self, dataset, feature_states, verbose=0):
        """
        Complete the model with additional optimal rules of the given dataset that also match the features states of feature_states if there exists.

        Args:
            dataset: DiscreteStateTransitionsDataset
                State transitions to learn from.
            feature_states: list of (list of string)
                Features states that must be matched by the new rules to be found.
            verbose: int (0 or 1)
        """
        if not isinstance(feature_states, list):
            raise TypeError("Argument feature_states must be a list of list of strings")
        if not all(isinstance(i,(list,tuple,numpy.ndarray)) for i in feature_states):
            raise TypeError("Argument feature_states must be a list of list of strings")
        if not all(isinstance(j,str) for i in feature_states for j in i):
            raise TypeError("Argument feature_states must be a list of list of strings")
        if not all(len(i) == len(self.features) for i in feature_states):
            raise TypeError("Features state must correspond to the model feature variables (bad length)")

        for feature_state in feature_states:

            for var_id, (var,vals) in enumerate(dataset.targets):
                for val_id, val in enumerate(vals):

                    # Check if new rules are needed
                    head = LegacyAtom(var,set(vals),val,var_id)
                    for r in self.rules:
                        if r.head == head:
                            if r.matches(feature_state):
                                continue

                    # usual data conversion
                    positives, negatives = PRIDE.interprete(dataset, head)

                    # Search for likeliness rules
                    new_rule = PRIDE.find_one_optimal_rule_of(head, dataset, positives, negatives, feature_state, verbose)
                    if new_rule is not None:
                        self.rules.append(new_rule)
                    else:
                        if verbose > 0:
                            eprint("Requested state "+str(feature_state)+\
                            " cannot be matched by a likeliness rule of "+self.targets[var_id][0]+"("+self.targets[var_id][1][val_id]+") consistent with given dataset")


    def predict(self, feature_states, semantics="synchronous", default=None):
        """
        Predict the possible target states of the given feature state according to the model rules.

        Args:
            feature_states: list of list of String
                Feature states from wich target states must be predicted.
            semantics: String (optional)
                The dynamic semantics used to generate the target states.
            default: list(String, list of String)
                Default value for each variable when no rule match.
                Will be '?' if not given.
        """
        if not isinstance(feature_states, list):
            raise TypeError("Argument feature_states must be a list of list of strings")
        if not all(isinstance(i,(list,tuple,numpy.ndarray)) for i in feature_states):
            raise TypeError("Argument feature_states must be a list of list of strings")
        if not all(isinstance(j,str) for i in feature_states for j in i):
            raise TypeError("Argument feature_states must be a list of list of strings")
        if not all(len(i) == len(self.features) for i in feature_states):
            raise TypeError("Features state must correspond to the model feature variables (bad length)")

        output = dict()
        for feature_state in feature_states:
            target_states = []
            if semantics == "synchronous":
                target_states = Synchronous.next(feature_state, self.targets, self.rules, default)
            elif semantics == "asynchronous":
                if len(self.features) != len(self.targets):
                    raise ValueError("Asynchronous semantics can only be used if features and targets variables are the same (for now).")
                target_states = Asynchronous.next(feature_state, self.targets, self.rules, default)
            elif semantics == "general":
                if len(self.features) != len(self.targets):
                    raise ValueError("General semantics can only be used if features and targets variables are the same (for now).")
                target_states = General.next(feature_state, self.targets, self.rules, default)
            else:
                raise ValueError("Parameter semantics of DMVLP.predict must be one element of ['synchronous', 'asynchronous', 'general']")

            output[tuple(feature_state)] = target_states
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
        print_fn(" Algorithm: " + str(self.algorithm))
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
                print_fn("  "+str(r))

    def to_string(self):
        """
        Convert the object to a readable string

        Returns:
            String
                a readable representation of the object
        """
        output = "{\n"
        output += "Algorithm: " + str(self.algorithm)
        output += "\nFeatures: " + str(self.features)
        output += "\nTargets: " + str(self.targets)
        output += "\nRules:\n"
        for r in self.rules:
            output += r.to_string() + "\n"
        output += "}"

        return output

    def feature_states(self):
        """
        Compute all possible feature states of the logic program:
        all combination of variables values

        Returns:
            - list of (list of any)
                All possible feature state of the logic program with their domain string label
        """
        values_ids = [[self.features[i][1][j] for j in range(0,len(self.features[i][1]))] for i in range(0,len(self.features))]
        output = [list(i) for i in list(itertools.product(*values_ids))]
        return output

    def target_states(self):
        """
        Compute all possible target state of the logic program:
        all combination of variables values

        Returns:
            - list of (list of any)
                All possible target state of the logic program with their domain string label
        """
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
        if not isinstance(value, list):
            raise TypeError("rules must be a list")
        if not all(isinstance(i, Rule) for i in value):
            raise TypeError("rules must be of type pylfit.objects.Rule")
        self._rules = value.copy()

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        self._algorithm = value
