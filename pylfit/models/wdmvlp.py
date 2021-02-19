#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2021/01/01
# @updated: 2021/02/05
#
# @desc: class DMVLP python source code file
#-------------------------------------------------------------------------------

from . import DMVLP

from ..utils import eprint
from ..objects import Rule

from ..datasets import StateTransitionsDataset

from ..algorithms import GULA
from ..algorithms import PRIDE

from ..semantics import Synchronous, Asynchronous, General

import itertools
import random
import numpy

class WDMVLP(DMVLP):
    """
    Define a Weighted Dynamic Multi-Valued Logic Program (W-DMVLP), a set of rules over features/target variables/values
    that can encode the dynamics of a discrete dynamic system (also work for static systems).

    Args:
        features: list of (string, list of string).
            Variables and their values that appear in body of rules
        targets: list of (string, list of string).
            Variables that appear in body of rules.
        rules: list of pylfit.objects.Rule.
            Logic rules of the program giving possibility of value.
        unlikeliness_rules: list of pylfit.objects.Rule.
            Logic rules of the program giving impossibility of value.
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

    def __init__(self, features, targets, rules=[], unlikeliness_rules=[]):
        """
        Create a WDMVLP instance from given features/targets variables and optional rules

        Args:
            features: list of pairs (String, list of String),
                labels of the features variables and their values (appear only in body of rules and constraints).
            targets: list of pairs (String, list of String),
                labels of the targets variables and their values (appear in head of rules and body of constraint).
            rules: list of pair (Rule, int),
                possibility rules with their weight.
            unlikeliness_rules: list of pair (Rule, int),
                impossibility rules with their weight.
        """
        super().__init__(features, targets)
        self.rules = rules
        self.unlikeliness_rules = unlikeliness_rules

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

        if algorithm not in WDMVLP._ALGORITHMS:
            raise ValueError('algorithm parameter must be one element of DMVLP._COMPATIBLE_ALGORITHMS: '+str(WDMVLP._ALGORITHMS)+'.')

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
                eprint("Learning possibilities...")
            rules = GULA.fit(dataset=dataset) #, targets_to_learn={'y1': ['1']})

            if verbose > 0:
                eprint("Learning impossibilities...")
            unlikeliness_rules = GULA.fit(dataset=dataset, impossibility_mode=True)

        elif self.algorithm == PRIDE:
            if not isinstance(dataset, StateTransitionsDataset):
                raise ValueError(msg)
            if verbose > 0:
                eprint("Starting fit with PRIDE")
                eprint("Learning likeliness...")
            rules = PRIDE.fit(dataset=dataset) #, targets_to_learn={'y1': ['1']})

            if verbose > 0:
                eprint("Learning unlikeliness...")
            unlikeliness_rules = PRIDE.fit(dataset=dataset, impossibility_mode=True)
        else:
            raise NotImplementedError("Algorithm usage not implemented yet")


        if verbose > 0:
            eprint("computing likeliness rules weights...")
        weighted_rules = {}
        train_init = [GULA.encode_state(s1, dataset.features) for s1,s2 in dataset.data]

        for var in range(len(self.targets)): # TODO: check if useless
            for val in range(len(self.targets[var][1])): # TODO: check if useless
                weighted_rules[(var,val)] = []
                for r in rules:
                    if r.head_variable != var or r.head_value != val:
                        continue
                    weight = 0
                    for s1 in train_init:
                        if r.matches(s1):
                            weight += 1
                    if weight > 0:
                        weighted_rules[(var,val)].append((weight,r))

        self.rules = [(w,r) for key,values in weighted_rules.items() for w,r in values]

        if verbose > 0:
            eprint("Computing unlikeliness rules weights")
        #eprint("Weighted model: ", weighted_rules)
        weighted_rules = {}
        for var in range(len(self.targets)): # TODO: check if useless
            for val in range(len(self.targets[var][1])): # TODO: check if useless
                weighted_rules[(var,val)] = []
                for r in unlikeliness_rules:
                    if r.head_variable != var or r.head_value != val:
                        continue
                    weight = 0
                    for s1 in train_init:
                        if r.matches(s1):
                            weight += 1
                    if weight > 0:
                        weighted_rules[(var,val)].append((weight,r))

        self.unlikeliness_rules = [(w,r) for key,values in weighted_rules.items() for w,r in values]

    def predict(self, feature_state, raw_rules=False):
        """
        Predict the possible target states of the given feature state according to the model rules.

        Args:
            feature_state: list of String
                Feature state from wich target state must be predicted.
            raw_rules: Boolean (optional)
                By default rules are output under string logic format.
                If True, the raw Rule objects will be output.
        """

        encoded_feature_state = GULA.encode_state(feature_state, self.features)
        output = {}
        for var_id, (var, vals) in enumerate(self.targets):
            output[var] = {}
            for val_id, val in enumerate(vals):
                # max rule weight
                max_rule_weight = 0
                best_rule = None
                for w,r in self.rules:
                    if r.head_variable == var_id and r.head_value == val_id:
                        if w > max_rule_weight and r.matches(encoded_feature_state):
                            max_rule_weight = w
                            best_rule = r

                # max anti-rule weight
                max_anti_rule_weight = 0
                best_anti_rule = None
                for w,r in self.unlikeliness_rules:
                    if r.head_variable == var_id and r.head_value == val_id:
                        if w > max_anti_rule_weight and r.matches(encoded_feature_state):
                            max_anti_rule_weight = w
                            best_anti_rule = r

                prediction = round(0.5 + 0.5*(max_rule_weight - max_anti_rule_weight) / max(1,(max_rule_weight+max_anti_rule_weight)),3)

                if not raw_rules:
                    if best_rule is not None:
                        best_rule = best_rule.logic_form(self.features, self.targets)
                    if best_anti_rule is not None:
                        best_anti_rule = best_anti_rule.logic_form(self.features, self.targets)

                output[var][val] = (prediction, (max_rule_weight, best_rule), (max_anti_rule_weight, best_anti_rule))

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
            print_fn(' Likeliness rules: []')
        else:
            print_fn(" Likeliness rules:")
            for w,r in self.rules:
                print_fn("  "+ str(w) + ", " +r.logic_form(self.features, self.targets))
        if len(self.unlikeliness_rules) == 0:
            print_fn(' Unlikeliness rules: []')
        else:
            print_fn(" Unlikeliness rules:")
            for w,r in self.unlikeliness_rules:
                print_fn("  "+ str(w) + ", " +r.logic_form(self.features, self.targets))

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
        output += "\nLikeliness rules:\n"
        for w,r in self.rules:
            output += "(" + str(w) + ", " + r.logic_form(self.features, self.targets) +")\n"
        output += "\nUnlikeliness rules:\n"
        for w,r  in self.unlikeliness_rules:
            output += "(" + str(w) + ", " + r.logic_form(self.features, self.targets) + ")\n"
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
    def rules(self):
        return self._rules

    @rules.setter
    def rules(self, value):
        self._rules = value.copy()

    @property
    def unlikeliness_rules(self):
        return self._unlikeliness_rules

    @unlikeliness_rules.setter
    def unlikeliness_rules(self, value):
        self._unlikeliness_rules = value.copy()

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        self._algorithm = value
