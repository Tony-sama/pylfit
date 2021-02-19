#-----------------------
# @author: Tony Ribeiro
# @created: 2019/03/20
# @updated: 2021/01/26
#
# @desc: simple approximated version of GULA implementation.
#    - extract patern from pair of interpretation of transitions
#
#-----------------------

from ..utils import eprint
from ..objects.rule import Rule
from ..algorithms import Algorithm
from ..datasets import StateTransitionsDataset

import csv

class PRIDE (Algorithm):
    """
    Define a simple approximative version of the GULA algorithm.
    Learn logic rules that explain state transitions of a discrete dynamic system.
    """

    def fit(dataset, targets_to_learn=None, impossibility_mode=False, verbose=0): #variables, values, transitions, conclusion_values=None, program=None): #, partial_heuristic=False):
        """
        Preprocess transitions and learn rules for all given features/targets variables/values.

        Args:
            dataset: pylfit.datasets.StateTransitionsDataset
                state transitions of a the system
            targets_to_learn: dict of {String: list of String}
                target variables values of the dataset for wich we want to learn rules.
                If not given, all targets values will be learned.

        Returns:
            list of pylfit.objects.Rule
                A set of DMVLP rules that is:
                    - correct: explain/reproduce all the transitions of the dataset.
                    - optimal: all rules are minimals
        """

        #eprint("Start PRIDE learning...")

        # Parameters checking
        if not isinstance(dataset, StateTransitionsDataset):
            raise ValueError('Dataset type not supported, PRIDE expect ' + str(StateTransitionsDataset.__name__))

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

        feature_domains = dataset.features
        target_domains = dataset.targets
        rules = []

        # Replace state variable value (string) by their domain id (int)
        encoded_data = Algorithm.encode_transitions_set(dataset.data, dataset.features, dataset.targets)

        # Nothing to learn
        if len(encoded_data) == 0:
            return []

        # Learn rules for each observed variable/value
        #for var in range(0, len(target_domains)):
        #    for val in range(0, len(target_domains[var][1])):
        for var_id, (var_name, var_domain) in enumerate(dataset.targets):
            #eprint(var_id, (var_name, var_domain))
            for val_id, val_name in enumerate(var_domain):
                #eprint(val_id, val_name)
                if var_name not in targets_to_learn:
                    continue
                if val_name not in targets_to_learn[var_name]:
                    continue
                positives, negatives = PRIDE.interprete(encoded_data, var_id, val_id)

                if impossibility_mode:
                    rules += PRIDE.fit_var_val(var_id, val_id, len(feature_domains), negatives, positives, verbose)
                else:
                    rules += PRIDE.fit_var_val(var_id, val_id, len(feature_domains), positives, negatives, verbose)

        output = rules

        return output


    @staticmethod
    def interprete(transitions, variable, value):
        """
        Split transition into positive/negatives states for the given variable/value
        Warning: assume deterministic transitions

        Args:
            transitions: list of tuple (list of int, list of int)
                state transitions of dynamic system
            variable: int
                variable id
            value: int
                variable value id
        """
        transitions = set((tuple(s1), tuple(s2)) for s1,s2 in transitions)
        positives = set(s1 for s1,s2 in transitions if s2[variable] == value)
        negatives = set(s1 for s1,s2 in transitions if s1 not in positives)

        return list(positives), list(negatives)


    @staticmethod
    def fit_var_val(variable, value, nb_features, positives, negatives, verbose=0):
        """
        Learn minimal rules that explain positive examples while consistent with negatives examples

        Args:
            variable: int
                variable id
            value: int
                variable value id
            positive: list of (list of int)
                States of the system where the variable takes this value in the next state
            negative: list of (list of int)
                States of the system where the variable does not take this value in the next state
        """
        if verbose > 0:
            eprint("Start learning of var="+str(variable)+", val="+str(value))

        remaining = positives.copy()
        output = []

        # exausting covering loop
        while len(remaining) > 0:
            #eprint("Remaining positives: "+str(remaining))
            #eprint("Negatives: "+str(negatives))
            target = remaining[0]
            #eprint("new target: "+str(target))

            R = Rule(variable, value, nb_features)
            #eprint(R.to_string())

            # 1) Consistency: against negatives examples
            #---------------------------------------------
            for neg in negatives:
                if R.matches(neg): # Cover a negative example
                    #eprint(R.to_string() + " matches " + str(neg))
                    for var in range(0,len(target)):
                        if not R.has_condition(var) and neg[var] != target[var]: # free condition
                            #eprint("adding condition "+str(var)+":"+str(var)+"="+str(target[var]))
                            if target[var] > -1: # Valid target value (-1 encode all value for partial state)
                                R.add_condition(var,target[var]) # add value of target positive example
                                break

            # 2) Minimalize: only necessary conditions
            #-------------------------------------------

            reductible = True

            conditions = R.body.copy()

            for (var,val) in conditions:
                R.remove_condition(var) # Try remove condition

                conflict = False
                for neg in negatives:
                    if R.matches(neg): # Cover a negative example
                        conflict = True
                        R.add_condition(var,val) # Cancel removal
                        break

            # Add new minimal rule
            #eprint("New rule: "+R.to_string())
            output.append(R)
            remaining.pop(0)

            # 3) Clean new covered positives examples
            #------------------------------------------
            i = 0
            while i < len(remaining):
                if R.matches(remaining[i]):
                    #eprint("Covers "+str(remaining[i]))
                    remaining.pop(i)
                else:
                    i += 1

        return output
