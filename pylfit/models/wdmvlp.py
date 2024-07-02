#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2021/01/01
# @updated: 2023/12/27
#
# @desc: class DMVLP python source code file
#-------------------------------------------------------------------------------

from . import DMVLP

from ..utils import eprint
from ..objects import LegacyAtom
from ..datasets import DiscreteStateTransitionsDataset
from ..algorithms import GULA
from ..algorithms import PRIDE
from ..algorithms import BruteForce

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
    _COMPATIBLE_DATASETS = [DiscreteStateTransitionsDataset]

    """ Learning algorithms that can be use to fit this model """
    _ALGORITHMS = ["gula", "pride", "brute-force"]

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
            - "pride",
            - "brute-force",

        """

        if algorithm not in WDMVLP._ALGORITHMS:
            raise ValueError('algorithm parameter must be one element of DMVLP._COMPATIBLE_ALGORITHMS: '+str(WDMVLP._ALGORITHMS)+'.')

        if algorithm == "gula":
            self.algorithm = "gula"
        elif algorithm == "pride":
            self.algorithm = "pride"
        elif algorithm == "brute-force":
            self.algorithm = "brute-force"
        else:
            raise NotImplementedError('<DEV> algorithm="'+str(algorithm)+'" is in DMVLP._COMPATIBLE_ALGORITHMS but no behavior implemented.')

    def fit(self, dataset, verbose=0, heuristics=None, threads=1):
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

        msg = 'Dataset type (' + str(dataset.__class__.__name__) + ') not supported \
        by the algorithm (' + str(self.algorithm.__class__.__name__) + '). \
        Dataset must be of type ' + str(DiscreteStateTransitionsDataset.__class__.__name__)

        if self.algorithm == "gula":
            if not isinstance(dataset, DiscreteStateTransitionsDataset):
                raise ValueError(msg)
            if verbose > 0:
                eprint("Starting fit with GULA")
                eprint("Learning possibilities...")
            rules = GULA.fit(dataset=dataset, verbose=verbose, threads=threads)

            if verbose > 0:
                eprint("Learning impossibilities...")
            unlikeliness_rules = GULA.fit(dataset=dataset, impossibility_mode=True, threads=threads)

        elif self.algorithm == "pride":
            if not isinstance(dataset, DiscreteStateTransitionsDataset):
                raise ValueError(msg)
            if verbose > 0:
                eprint("Starting fit with PRIDE")
                eprint("Learning likeliness...")
            rules = PRIDE.fit(dataset=dataset, options={"verbose":verbose, "heuristics":heuristics, "threads":threads})

            if verbose > 0:
                eprint("Learning unlikeliness...")
            unlikeliness_rules = PRIDE.fit(dataset=dataset, options={"impossibility_mode":True, "heuristics":heuristics, "threads":threads})
        elif self.algorithm == "brute-force":
                if not isinstance(dataset, DiscreteStateTransitionsDataset):
                    raise ValueError(msg)
                if verbose > 0:
                    eprint("Starting fit with BruteForce")
                    eprint("Learning possibilities...")
                rules = BruteForce.fit(dataset=dataset, verbose=verbose)

                if verbose > 0:
                    eprint("Learning impossibilities...")
                unlikeliness_rules = BruteForce.fit(dataset=dataset, impossibility_mode=True, verbose=verbose)
        else:
            raise NotImplementedError("Algorithm usage not implemented yet")

        if verbose > 0:
            eprint("computing likeliness rules weights...")
        weighted_rules = {}
        train_init = set(tuple(s1) for s1,s2 in dataset.data)

        for var, vals in self.targets:
            for val in vals:
                weighted_rules[(var,val)] = []

        for r in rules:
            weight = 0
            for s1 in train_init:
                if r.matches(s1):
                    weight += 1
            #if weight > 0:
            weighted_rules[(var,val)].append((weight,r))

        self.rules = [(w,r) for key,values in weighted_rules.items() for w,r in values]

        if verbose > 0:
            eprint("Computing unlikeliness rules weights")
        weighted_rules = {}
        for var, vals in self.targets:
            for val in vals:
                weighted_rules[(var,val)] = []

        for r in unlikeliness_rules:
            weight = 0
            for s1 in train_init:
                if r.matches(s1):
                    weight += 1
            #if weight > 0:
            weighted_rules[(var,val)].append((weight,r))

        self.unlikeliness_rules = [(w,r) for key,values in weighted_rules.items() for w,r in values]

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

        dataset_feature_states = set(tuple(s1) for s1,s2 in dataset.data)

        for feature_state in feature_states:
            for var_id, (var,vals) in enumerate(dataset.targets):
                for val_id, val in enumerate(vals):

                    # Check if new rules are needed
                    need_likeliness = True
                    need_unlikeliness = True
                    for (w,r) in self.rules:
                        if r.head.variable == var and r.head.value == val:
                            if r.matches(feature_state):
                                need_likeliness = False
                                break
                    for (w,r) in self.unlikeliness_rules:
                        if r.head.variable == var and r.head.value == val:
                            if r.matches(feature_state):
                                need_unlikeliness = False
                                break

                    if not need_likeliness and not need_unlikeliness:
                        continue

                    head = LegacyAtom(var, dataset.targets[var_id][1], val, var_id)
                    positives, negatives = PRIDE.interprete(dataset, head)

                    # Search for likeliness rules
                    if need_likeliness:
                        new_rule = PRIDE.find_one_optimal_rule_of(head, dataset, positives, negatives, feature_state, verbose)
                        if new_rule is not None:
                            # compute weight
                            weight = 0
                            for s1 in dataset_feature_states:
                                if new_rule.matches(s1):
                                    weight += 1
                            self.rules.append((weight,new_rule))
                        else:
                            if verbose > 0:
                                eprint("Requested state "+str(feature_state)+\
                                " cannot be matched by a likeliness rule of "+self.targets[var_id][0]+"("+self.targets[var_id][1][val_id]+") consistent with given dataset")

                    # Search for unlikeliness rules
                    if need_unlikeliness:
                        new_rule = PRIDE.find_one_optimal_rule_of(head, dataset, negatives, positives, feature_state, verbose)
                        if new_rule is not None:
                            # compute weight
                            weight = 0
                            for s1 in dataset_feature_states:
                                if new_rule.matches(s1):
                                    weight += 1
                            self.unlikeliness_rules.append((weight,new_rule))
                        else:
                            if verbose > 0:
                                eprint("Requested state "+str(feature_state)+\
                                " cannot be matched by a unlikeliness rule of "+self.targets[var_id][0]+"("+self.targets[var_id][1][val_id]+") consistent with given dataset")

    def predict(self, feature_states, raw_rules=False):
        """
        Predict the possible target states of the given feature state according to the model rules.

        Args:
            feature_states: list of list of String
                Feature states from wich target values must be predicted.
            raw_rules: Boolean (optional)
                By default rules are output under string format.
                If True, the raw Rule objects will be output.
        Returns:
            dictionary of target state with corresponding rules that predict it
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
            prediction = dict()
            for (var, vals) in self.targets:
                prediction[var] = {}
                for val in vals:
                    # max rule weight
                    max_rule_weight = 0
                    best_rule = None
                    for w,r in self.rules:
                        if r.head.variable == var and r.head.value == val:
                            if w > max_rule_weight and r.matches(feature_state):
                                max_rule_weight = w
                                best_rule = r
                            elif w == max_rule_weight and r.matches(feature_state):
                                if best_rule == None or r.size() < best_rule.size():
                                    max_rule_weight = w
                                    best_rule = r

                    # max anti-rule weight
                    max_anti_rule_weight = 0
                    best_anti_rule = None
                    for w,r in self.unlikeliness_rules:
                        if r.head.variable == var and r.head.value == val:
                            if w > max_anti_rule_weight and r.matches(feature_state):
                                max_anti_rule_weight = w
                                best_anti_rule = r
                            elif w == max_anti_rule_weight and r.matches(feature_state):
                                if best_anti_rule == None or r.size() < best_anti_rule.size():
                                    max_anti_rule_weight = w
                                    best_anti_rule = r

                    proba = round(0.5 + 0.5*(max_rule_weight - max_anti_rule_weight) / max(1,(max_rule_weight+max_anti_rule_weight)),3)

                    if not raw_rules:
                        if best_rule is not None:
                            best_rule = best_rule.to_string()
                        if best_anti_rule is not None:
                            best_anti_rule = best_anti_rule.to_string()

                    prediction[var][val] = (proba, (max_rule_weight, best_rule), (max_anti_rule_weight, best_anti_rule))
            output[tuple(feature_state)] = prediction

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
            print_fn(' Likeliness rules: []')
        else:
            print_fn(" Likeliness rules:")
            for w,r in self.rules:
                print_fn("  "+ str(w) + ", " +r.to_string())
        if len(self.unlikeliness_rules) == 0:
            print_fn(' Unlikeliness rules: []')
        else:
            print_fn(" Unlikeliness rules:")
            for w,r in self.unlikeliness_rules:
                print_fn("  "+ str(w) + ", " +r.to_string())

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
        output += "\nLikeliness rules:\n"
        for w,r in self.rules:
            output += "(" + str(w) + ", " + r.to_string() +")\n"
        output += "\nUnlikeliness rules:\n"
        for w,r  in self.unlikeliness_rules:
            output += "(" + str(w) + ", " + r.to_string() + ")\n"
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
