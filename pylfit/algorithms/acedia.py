#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2019/04/30
# @updated: 2023/12/27
#
# @desc: simple ACEDIA implementation:
#   - INPUT: a set of pairs of continuous valued state transitions
#   - OUTPUT: the optimal continuum logic program that realizes the input
#   - THEORY:
#       - ILP 2017:
#       Inductive Learning from State Transitions over Continuous Domains
#       https://hal.archives-ouvertes.fr/hal-01655644
#   - COMPLEXITY:
#       - Variables: exponential
#       - Observations: exponential
#       - about O( (|Observations| ^ (2 * |variables|)) * |variables| ^ 5 )
#-------------------------------------------------------------------------------

from ..utils import eprint
from ..objects.continuum import Continuum
from ..objects.continuumRule import ContinuumRule
from ..algorithms.algorithm import Algorithm
from ..datasets import ContinuousStateTransitionsDataset

import multiprocessing
import itertools

class ACEDIA (Algorithm):
    """
    Define a simple complete version of the ACEDIA algorithm.
    Learn logic rules that explain state transitions
    of a dynamic system:
        - continuous valued
        - continuum deterministic
    INPUT: a set of pairs of continuous valued states
    OUTPUT: a list of continuum rules
    """

    @staticmethod
    def fit(dataset, targets_to_learn=None, verbose=0, threads=1):
        """
        Preprocess transitions and learn rules for all target_to_learn variables.
        Assume continuum deterministics transitions: only one future continuum for each state.

        Args:
            dataset: pylfit.datasets.ContinuousStateTransitionsDataset
                state transitions of a the system
            targets_to_learn: list of String
                target variables of the dataset for wich we want to learn rules.
                If not given, all targets variables will be learned.

        Returns:
            list of pylfit.objects.ContinuumRule
                A set of CLP rules that is:
                    - correct: explain/reproduce all the transitions of the dataset.
                    - complete: matches all possible feature states (even not in the dataset).
                    - optimal: all rules are minimals

        """
        #eprint("Start ACEDIA learning...")

        if not isinstance(dataset, ContinuousStateTransitionsDataset):
            raise ValueError('Dataset type not supported, ACEDIA expect ' + str(ContinuousStateTransitionsDataset.__name__))

        if targets_to_learn is None:
            targets_to_learn = [var for var, vals in dataset.targets]
        elif not isinstance(targets_to_learn, list) or not all(isinstance(var, str) for var in targets_to_learn):
            raise ValueError('targets_to_learn must be a list of string')
        else:
            for target in targets_to_learn:
                targets_names = [var for var, vals in dataset.targets]
                if target not in targets_names:
                    raise ValueError('targets_to_learn values must be dataset target variables')

        rules = []

        # Learn rules for each variable
        thread_parameters = []
        for target_id in range(0, len(dataset.targets)):
            #rules += ACEDIA.fit_var(dataset.features, dataset.data, target_id, verbose)

            if(threads == 1):
                rules += ACEDIA.fit_thread([dataset.features, dataset.targets, dataset.data, target_id, verbose])
            else:
                thread_parameters.append([dataset.features, dataset.targets, dataset.data, target_id, verbose])

        if(threads > 1):
            if(verbose):
                eprint("Start learning over "+str(threads)+" threads")
            with multiprocessing.Pool(processes=threads) as pool:
                rules = pool.map(ACEDIA.fit_thread, thread_parameters)
            rules = list(itertools.chain.from_iterable(rules))

        return rules

    @staticmethod
    def fit_thread(args):
        """
        Thread wrapper for fit_var function (see below)
        """
        feature_domains, target_domains, transitions, var_id, verbose = args
        if verbose > 0:
            eprint("\nStart learning of var=", var_id+1,"/", len(target_domains))

        rules = ACEDIA.fit_var(feature_domains, transitions, var_id, verbose)

        if verbose > 0:
            eprint("\nFinished learning of var=", var_id+1,"/", len(target_domains))

        return rules

    @staticmethod
    def fit_var(features, transitions, variable, verbose=0):
        """
        Learn minimal rules that realizes the given transitions

        Args:
            features: list of (string, list of string)
            transitions: list of pair (list of float, list of float)
            variable: string
            varbose: int
        Returns:
            list of ContinuumRule
        """
        #if verbose > 0:
        #    eprint("\rLearning var="+str(variable+1)+"/"+str(len(features)), end='')

        # 0) Initialize undominated rule
        #--------------------------------
        body = [(var, features[var][1]) for var in range(len(features))]
        minimal_rules = [ContinuumRule(variable, Continuum(), body)]

        # Revise learned rules against each transition
        for state_1, state_2 in transitions:

            # 1) Extract unconsistents rules
            #--------------------------------
            unconsistents = [ rule for rule in minimal_rules if rule.matches(state_1) and not rule.head_value.includes(state_2[variable]) ]
            minimal_rules = [ rule for rule in minimal_rules if rule not in unconsistents ]

            for unconsistent in unconsistents:

                revisions = ACEDIA.least_revision(unconsistent, state_1, state_2)

                for revision in revisions:

                    # Check domination
                    dominated = False

                    for r in minimal_rules:
                        if r.dominates(revision):
                            dominated = True
                            break

                    # Remove dominated rules
                    if not dominated:
                        minimal_rules = [ r for r in minimal_rules if not revision.dominates(r) ]
                        minimal_rules.append(revision)

        # 2) remove domains wize conditions
        #-----------------------------------

        output = []

        for r in minimal_rules:
            if r.head_value.is_empty():
                continue

            r_ = r.copy()

            for var, val in r.body:
                if val == features[var][1]:
                    r_.remove_condition(var)

            output.append(r_)

        #DBG
        #eprint("\r",end='')

        return output

    @staticmethod
    def least_revision(rule, state_1, state_2):
        """
        Compute the least revision of rule w.r.t. the transition (state_1, state_2)

        Args:
            rule: ContinuumRule
            state_1: list of float
            state_2: list of float

        Returns: list of ContinuumRule
            The least generalisation of the rule over its conclusion and
            the least specializations of the rule over each condition
        """

        # 0) Check Consistence
        #----------------------

        #Consistent rule
        if not rule.matches(state_1):
            raise ValueError("Attempting to revise a consistent rule, revision would be itself, this call is useless in ACEDIA and must be an error")

        # Consistent rule
        if rule.head_value.includes(state_2[rule.head_variable]):
            raise ValueError("Attempting to revise a consistent rule, revision would be itself, this call is useless in ACEDIA and must be an error")


        # 1) Revise conclusion
        #----------------------
        head_var = rule.head_variable
        next_value = state_2[head_var]
        revisions = []

        head_revision = rule.copy()
        head_value = head_revision.head_value

        # Empty set head case
        if head_value.is_empty():
            head_value = Continuum(next_value, next_value, True, True)
        elif next_value <= head_value.min_value:
            head_value.set_lower_bound(next_value, True)
        else:
            head_value.set_upper_bound(next_value, True)

        head_revision.head_value = head_value

        revisions.append(head_revision)

        # 2) Revise each condition
        #---------------------------
        for var, val in rule.body:
            state_value = state_1[var]

            # min revision
            min_revision = rule.copy()
            new_val = val.copy()
            new_val.set_lower_bound(state_value, False)
            if not new_val.is_empty():
                min_revision.set_condition(var, new_val)
                revisions.append(min_revision)

            # max revision
            max_revision = rule.copy()
            new_val = val.copy()
            new_val.set_upper_bound(state_value, False)
            if not new_val.is_empty():
                max_revision.set_condition(var, new_val)
                revisions.append(max_revision)

        return revisions
