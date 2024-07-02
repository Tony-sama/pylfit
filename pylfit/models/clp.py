#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2022/08/24
# @updated: 2023/12/27
#
# @desc: class CLP python source code file
#-------------------------------------------------------------------------------

from ..utils import eprint
from ..models import Model
from ..objects import Continuum, ContinuumRule
from ..datasets import ContinuousStateTransitionsDataset
from ..algorithms import ACEDIA

import numpy

class CLP(Model):
    """
    Define a Continuum Logic Program.

    Args:
        algorithm: pylfit.algorithm.Algorithm subclass,
            the algorithm to be used to fit the model.
    """

    """ Dataset types compatible with dmvlp """
    _COMPATIBLE_DATASETS = [ContinuousStateTransitionsDataset]

    """ Learning algorithms that can be use to fit this model """
    _ALGORITHMS = ["acedia"]

    """ Optimization """
    _OPTIMIZERS = []

    _MIN_DOMAIN_SIZE = 0.0001

#--------------
# Constructors
#--------------

    def __init__(self, features, targets, rules=[]):
        """
        Create a CLP instance from given features/targets variables and optional rules

        Args:
            features: list of pairs (String, Continuum),
                labels of the features variables and their continuum of values (appear only in body of rules).
            targets: list of pairs (String, Continuum),
                labels of the targets variables and their continuum of values (appear in head of rules).
            rules: list of ContinuumRule,
                rules that define logic program dynamics: influences of feature variables values over target variables values.
        """
        super().__init__()

        self.features = features
        self.targets = targets
        self.rules = rules

    def copy(self):
        output = CLP(self.features, self.targets, self.rules)
        output.algorithm = self.algorithm
        return output

#--------------
# Operators
#--------------

#--------------
# Methods
#--------------

    def compile(self, algorithm="acedia"):
        """
        Set the algorithm to be used to fit the model.
        Supported algorithms:
            - "acedia", Abstraction-free Continuum Environment Dynamics Inference Algorithm
        """

        if algorithm not in CLP._ALGORITHMS:
            raise ValueError('algorithm parameter must be one element of CLP._COMPATIBLE_ALGORITHMS: '+str(CLP._ALGORITHMS)+'.')

        if algorithm == "acedia":
            self.algorithm = "acedia"
        else:
            raise NotImplementedError('<DEV> algorithm="'+str(algorithm)+'" is in CLP._COMPATIBLE_ALGORITHMS but no behavior implemented.')

    def fit(self, dataset, targets_to_learn=None, verbose=0, heuristics=None, threads=1):
        """
        Use the algorithm set by compile() to fit the rules to the dataset.
            - Learn a model from scratch using the chosen algorithm.
            - update model (TODO).

        Check and encode dataset to be used by the desired algorithm.

        Raises:
            ValueError if the dataset can't be used with the algorithm.

        """

        # Check parameters
        if not any(isinstance(dataset, i) for i in CLP._COMPATIBLE_DATASETS):
            msg = 'Dataset type (' + str(dataset.__class__.__name__)+ ') not suported by CLP model, must be one of '+ \
            str([i for i in CLP._COMPATIBLE_DATASETS])
            raise ValueError(msg)

        if targets_to_learn is None:
            targets_to_learn = [var for var, vals in dataset.targets]
        elif not isinstance(targets_to_learn, list) or not all(isinstance(var, str) for var in targets_to_learn):
            raise ValueError('targets_to_learn must be a list of string')
        else:
            for target in targets_to_learn:
                targets_names = [var for var, vals in dataset.targets]
                if target not in targets_names:
                    raise ValueError('targets_to_learn values must be dataset target variables')

        if self.algorithm not in CLP._ALGORITHMS:
            raise ValueError('algorithm property must be one element of CLP._COMPATIBLE_ALGORITHMS: '+str(CLP._ALGORITHMS)+'.')

        msg = 'Dataset type (' + str(dataset.__class__.__name__) + ') not supported \
        by the algorithm (' + str(self.algorithm.__class__.__name__) + '). \
        Dataset must be of type ' + str(ContinuousStateTransitionsDataset.__class__.__name__)

        if self.algorithm == "acedia":
            if not isinstance(dataset, ContinuousStateTransitionsDataset):
                raise ValueError(msg)
            if verbose > 0:
                eprint("Starting fit with ACEDIA")
            self.rules = ACEDIA.fit(dataset=dataset, targets_to_learn=targets_to_learn, verbose=verbose, threads=threads)
        else:
            raise NotImplementedError("Algorithm usage not implemented yet")

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
            print_fn('  ' + str(var[0]) + ': ' + str(var[1]))
        print_fn(" Targets: ")
        for var in self.targets:
            print_fn('  ' + str(var[0]) + ': ' + str(var[1]))
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
        output += "Algorithm: " + str(self.algorithm)
        output += "\nFeatures: " + str(self.features)
        output += "\nTargets: " + str(self.targets)
        output += "\nRules:\n"
        for r in self.rules:
            output += r.logic_form(self.features, self.targets) + "\n"
        output += "}"

        return output

    def predict(self, feature_states):
        """
        Compute the next state according to the rules of the CLP

        Args:
            state: list of Continuum
                A Continuum state of the system

        Returns:
            list of Continuum
                the range of value that each variable can takes after the given state
        """
        if not isinstance(feature_states, list):
            raise TypeError("Argument feature_states must be a list of list of float")
        if not all(isinstance(i,(list,tuple,numpy.ndarray)) for i in feature_states):
            raise TypeError("Argument feature_states must be a list of list of float")
        if not all(isinstance(j,(float,int)) for i in feature_states for j in i):
            raise TypeError("Argument feature_states must be a list of list of float")
        if not all(len(i) == len(self.features) for i in feature_states):
            raise TypeError("Features state must correspond to the model feature variables (bad length)")

        output = dict()
        for feature_state in feature_states:

            target_state = [None for i in self.targets]
            rules = []

            for r in self.rules:
                if(r.matches(feature_state)):
                    # More precise conclusion
                    if target_state[r.head_variable] is None or target_state[r.head_variable].includes(r.head_value):
                        target_state[r.head_variable] = r.head_value

                    rules.append(r)
            output[tuple(feature_state)] = {tuple(target_state):rules}

        return output

    def feature_states(self, epsilon):
        """
        Generates all features states with atleast an epsilon distance

        Args:
            epsilon: float in ]0,1]
                the precision ratio of each state value

        Returns: list of (list of float)
            All possible feature state of the CLP with atleast an epsilon distance
        """
        if epsilon <= 0.0 or epsilon > 1.0:
            raise ValueError("Epsilon must be in ]0,1], got " + str(epsilon))

        if epsilon < 0.1:
            raise ValueError("Calling states of a CLP with an epsilon < 0.1, can generate a lot of states and takes a very long time")

        return self._variables_states(self.features, epsilon)

    def target_states(self, epsilon):
        """
        Generates all target states with atleast an epsilon distance

        Args:
            epsilon: float in ]0,1]
                the precision ratio of each state value

        Returns: list of (list of float)
            All possible target state of the CLP with atleast an epsilon distance
        """
        if epsilon <= 0.0 or epsilon > 1.0:
            raise ValueError("Epsilon must be in ]0,1], got " + str(epsilon))

        if epsilon < 0.1:
            raise ValueError("Calling states of a CLP with an epsilon < 0.1, can generate a lot of states and takes a very long time")

        return self._variables_states(self.targets, epsilon)

    def _variables_states(self, variables, epsilon):
        """
        Generates all states with atleast an epsilon distance

        Args:
            epsilon: float in ]0,1]
                the precision ratio of each state value

        Returns: list of (list of float)
            All possible state of the CLP with atleast an epsilon distance
        """

        state = []
        for var, d in variables:
            if not d.is_empty():
                state.append(d.min_value)
            else:
                state.append(None) # no value

        output = []
        self._variables_states_(variables, epsilon, 0, state, output)
        return output


    def _variables_states_(self, variables, epsilon, variable, state, output):
        """
        Recursive sub-function of state(self, epsilon)

        Args:
            variables: list of pairs (String, Continuum),
                labels of the variables and their continuum of values.
            epsilon: float in ]0,1]
                the precision ratio of each state value
            variable: int
                A variable id
            state: list of float
                A system state
        """

        # All variable are assigned
        if variable >= len(variables):
            excluded_bound = False
            for idx, val in enumerate(state):
                if val is None or not variables[idx][1].includes(val):
                    excluded_bound = True
                    break
            if not excluded_bound:
                output.append( state.copy() )
            return

        # No known value
        if variables[variable][1].is_empty():
            state[variable] = None
            self._variables_states_(variables, epsilon, variable+1, state, output)
        else:
            # Enumerate each possible value
            min = variables[variable][1].min_value
            max = variables[variable][1].max_value
            step = epsilon * (max - min)
            values = [min+(step*i) for i in range( int(1.0 / epsilon) )]
            if values[-1] != max:
                values.append(max)

            #eprint(values)

            # bound exclusion
            if not variables[variable][1].min_included:
                values[0] = values[0] + self._MIN_DOMAIN_SIZE * 0.5

            # bound exclusion
            if not variables[variable][1].max_included:
                values[-1] = values[-1] - self._MIN_DOMAIN_SIZE * 0.5

            for val in values:
                state[variable] = val
                self._variables_states_(variables, epsilon, variable+1, state, output)

#--------
# Static
#--------




#--------------
# Accessors
#--------------

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
        if not all(isinstance(domain, Continuum) for (var,domain) in value):
            raise TypeError("features domains must be a pylfit.objects.Continuum")

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
        if not all(isinstance(domain, Continuum) for (var,domain) in value):
            raise TypeError("features domains must be a pylfit.objects.Continuum")

        self._targets = value.copy()

    @property
    def rules(self):
        return self._rules

    @rules.setter
    def rules(self, value):
        if not isinstance(value, list):
            raise TypeError("rules must be a list")
        if not all(isinstance(i, ContinuumRule) for i in value):
            raise TypeError("rules must be of type pylfit.objects.ContinuuumRule")
        self._rules = value.copy()

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value):
        self._algorithm = value
