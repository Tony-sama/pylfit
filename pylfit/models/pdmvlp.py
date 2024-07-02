#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2021/01/01
# @updated: 2023/12/27
#
# @desc: class PDMVLP python source code file
#-------------------------------------------------------------------------------

from ..models import CDMVLP

from ..utils import eprint
from ..datasets import DiscreteStateTransitionsDataset
from ..algorithms import Probalizer
from ..semantics import SynchronousConstrained

import numpy

class PDMVLP(CDMVLP):
    """
    Define a Probalistic Dynamic Multi-Valued Logic Program (PDMVLP), a set of rules over features/target variables/values
    that can encode the dynamics of a discrete dynamic system (also work for static systems).
    Probabilities being encoded into targets value.

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


    """ Dataset types compatible with PDMVLP """
    _COMPATIBLE_DATASETS = [DiscreteStateTransitionsDataset]

    """ Learning algorithms that can be use to fit this model """
    _ALGORITHMS = ["gula", "pride", "synchronizer"]

    """ Optimization """
    _OPTIMIZERS = []

#--------------
# Constructors
#--------------

    def copy(self):
        output = PDMVLP(self.features, self.targets, self.rules)
        output.algorithm = self.algorithm
        return output
#--------------
# Operators
#--------------

#--------------
# Methods
#--------------

    def compile(self, algorithm="gula"):
        """
        Set the algorithm to be used to fit the model.
        Supported algorithms:
            - "gula", General Usage LFIT Algorithm
            - "pride", Polynomial heuristic version of GULA
            - "synchronizer", GULA + constraint learning

        """

        if algorithm not in PDMVLP._ALGORITHMS:
            raise ValueError('algorithm parameter must be one element of PDMVLP._COMPATIBLE_ALGORITHMS: '+str(PDMVLP._ALGORITHMS)+'.')

        if algorithm == "gula":
            self.algorithm = "gula"
        elif algorithm == "pride":
            self.algorithm = "pride"
        elif algorithm == "synchronizer":
            self.algorithm = "synchronizer"
        else:
            raise NotImplementedError('<DEV> algorithm="'+str(algorithm)+'" is in PDMVLP._COMPATIBLE_ALGORITHMS but no behavior implemented.')

    def fit(self, dataset, verbose=0, threads=1):
        """
        Use the algorithm set by compile() to fit the rules to the dataset.
            - Learn a model from scratch using the chosen algorithm.
            - update model (TODO).

        Check and encode dataset to be used by the desired algorithm.

        Raises:
            ValueError if the dataset can't be used with the algorithm.

        """

        if not any(isinstance(dataset, i) for i in self._COMPATIBLE_DATASETS):
            msg = 'Dataset type (' + str(dataset.__class__.__name__)+ ') not suported by PDMVLP model.'
            raise ValueError(msg)

        if self.algorithm not in PDMVLP._ALGORITHMS:
            raise ValueError('algorithm property must be one element of PDMVLP._COMPATIBLE_ALGORITHMS: '+str(PDMVLP._ALGORITHMS)+'.')

        msg = 'Dataset type (' + str(dataset.__class__.__name__) + ') not supported \
        by the algorithm (' + str(self.algorithm.__class__.__name__) + '). \
        Dataset must be of type ' + str(DiscreteStateTransitionsDataset.__class__.__name__)

        if self.algorithm == "gula":
            if not isinstance(dataset, DiscreteStateTransitionsDataset):
                raise ValueError(msg)
            if verbose > 0:
                eprint("Starting fit with GULA")
            self.targets, self.rules, _ = Probalizer.fit(dataset=dataset, complete=True, verbose=0, threads=threads)
        elif self.algorithm == "pride":
            if not isinstance(dataset, DiscreteStateTransitionsDataset):
                raise ValueError(msg)
            if verbose > 0:
                eprint("Starting fit with PRIDE")
            self.targets, self.rules, _ = Probalizer.fit(dataset=dataset, complete=False, verbose=0, threads=threads)
        elif self.algorithm == "synchronizer":
            if not isinstance(dataset, DiscreteStateTransitionsDataset):
                raise ValueError(msg)
            if verbose > 0:
                eprint("Starting fit with Synchronizer")
            self.targets, self.rules, self.constraints = Probalizer.fit(dataset=dataset, complete=True, synchronous_independant=False, verbose=0, threads=threads)
        else:
            raise NotImplementedError("Algorithm usage not implemented yet")

    def predict(self, feature_states):
        """
        Predict the possible target states and their probability of the given feature state according to the model rules.

        Args:
            feature_states: list of list of String
                Feature states from wich target states must be predicted.
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
            target_states = SynchronousConstrained.next(feature_state, self.targets, self.rules, self.constraints)

            # Decode target states
            local_output = dict()

            for s, rules in target_states.items():
                target_state = list(s)

                # proba of target state
                if self.algorithm == "synchronizer":
                    val_proba = target_state[0].split(",")[1]
                    target_state_proba =  int(val_proba.split("/")[0]) / int(val_proba.split("/")[1])
                else:
                    target_state_proba = 1.0
                    for var_id, val in enumerate(target_state):
                        val_label = val.split(",")[0]
                        val_proba = val.split(",")[1]
                        val_proba = int(val_proba.split("/")[0]) / int(val_proba.split("/")[1])
                        target_state_proba *= val_proba
                        target_state[var_id] = val_label

                local_output[tuple(target_state)] = (target_state_proba, rules)

            output[tuple(feature_state)] = local_output


        return output

#--------
# Static
#--------


#--------------
# Properties
#--------------
