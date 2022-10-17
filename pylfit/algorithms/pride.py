#-----------------------
# @author: Tony Ribeiro
# @created: 2019/03/20
# @updated: 2021/06/15
#
# @desc: simple approximated version of GULA implementation.
#    - extract patern from pair of interpretation of transitions
#
#-----------------------

from ..utils import eprint
from ..objects.rule import Rule
from ..algorithms import Algorithm
from ..datasets import DiscreteStateTransitionsDataset

import multiprocessing
import itertools

import csv

class PRIDE (Algorithm):
    """
    Define a simple approximative version of the GULA algorithm.
    Learn logic rules that explain state transitions of a discrete dynamic system.
    """

    """ Learning algorithms that can be use to fit this model """
    _HEURISTICS = ["try_all_atoms", "max_coverage_dynamic", "max_coverage_static", "max_diversity", "multi_thread_at_rule_level"]

    def fit(dataset, targets_to_learn=None, impossibility_mode=False, verbose=0, heuristics=None, threads=1): #variables, values, transitions, conclusion_values=None, program=None): #, partial_heuristic=False):
        """
        Preprocess transitions and learn rules for all given features/targets variables/values.

        Args:
            dataset: pylfit.datasets.DiscreteStateTransitionsDataset
                state transitions of a the system
            targets_to_learn: dict of {String: list of String}
                target variables values of the dataset for wich we want to learn rules.
                If not given, all targets values will be learned.
            impossibility_mode: Boolean
                if true will learn impossibility rule for each atoms, i.e. when body match head is not observed in dataset next states
            verbose: int
                When greater than 0 progress of learning will be print in stderr
            heuristics: list of string
                - "try_all_atoms": each atom of the current positive target will be tried to appear in a rule
                - "max_coverage_dynamic": every atoms are tried as specialization when matching a negative, the rule with most positive match is kept
                - "max_coverage_static": all atoms are scored regarding their occurence in positives at start.
                    These score are used in place of rule score when selecting specialization.
                    Have priority over max_coverage_dynamic if both given.
                - "max_diversity": same as max_coverage but scoring on the diversity with already learn rules
                - "multi_thread_at_rule_level": multithreading will be used at rule level in place of var/val level

        Returns:
            list of pylfit.objects.Rule
                A set of DMVLP rules that is:
                    - correct: explain/reproduce all the transitions of the dataset.
                    - optimal: all rules are minimals
        """

        #eprint("Start PRIDE learning...")

        # Parameters checking
        if not isinstance(dataset, DiscreteStateTransitionsDataset):
            raise ValueError('Dataset type not supported, PRIDE expect ' + str(DiscreteStateTransitionsDataset.__name__))

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

        if heuristics is not None:
            for h in heuristics:
                if h not in PRIDE._HEURISTICS:
                    raise ValueError(str(h)+" is not a valid heuristic option, must be one of "+str(PRIDE._HEURISTICS))

        feature_domains = dataset.features
        target_domains = dataset.targets
        rules = []
        thread_parameters = []

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
                #positives, negatives = PRIDE.interprete(encoded_data, var_id, val_id)

                #if impossibility_mode:
                #    rules += PRIDE.fit_var_val(var_id, val_id, len(feature_domains), negatives, positives, verbose, heuristics, threads)
                #else:
                #    rules += PRIDE.fit_var_val(var_id, val_id, len(feature_domains), positives, negatives, verbose, heuristics, threads)

                if(threads == 1):
                    if verbose > 0:
                        eprint("\nStart learning of var=", var_id+1,"/", len(target_domains), ", val=", val_id+1, "/", len(target_domains[var_id][1]))

                    rules += PRIDE.fit_thread([encoded_data, feature_domains, target_domains, var_id, val_id, impossibility_mode, verbose, heuristics, threads])
                    # DBG
                    #eprint(negatives)
                elif heuristics is not None and "multi_thread_at_rule_level" in heuristics:
                    rules += PRIDE.fit_thread([encoded_data, feature_domains, target_domains, var_id, val_id, impossibility_mode, verbose, heuristics, threads])
                else:
                    thread_parameters.append([encoded_data, feature_domains, target_domains, var_id, val_id, impossibility_mode, verbose, heuristics, 1])

        #pool = ThreadPool(4)
        if threads > 1 and (heuristics is None or "multi_thread_at_rule_level" not in heuristics):
            if(verbose):
                eprint("Start learning over "+str(threads)+" threads")
            with multiprocessing.Pool(processes=threads) as pool:
                rules = pool.map(PRIDE.fit_thread, thread_parameters)
            rules = list(itertools.chain.from_iterable(rules))

        output = rules

        return output

    @staticmethod
    def fit_thread(args):
        encoded_data, feature_domains, target_domains, var_id, val_id, impossibility_mode, verbose, heuristics, threads = args
        if verbose > 0:
            eprint("\nStart learning of var=", var_id+1,"/", len(target_domains), ", val=", val_id+1, "/", len(target_domains[var_id][1]))

        positives, negatives = PRIDE.interprete(encoded_data, var_id, val_id)

        if impossibility_mode:
            rules = PRIDE.fit_var_val(var_id, val_id, len(feature_domains), negatives, positives, verbose, heuristics, threads)
        else:
            rules = PRIDE.fit_var_val(var_id, val_id, len(feature_domains), positives, negatives, verbose, heuristics, threads)


        if verbose > 0:
            eprint("\nFinished learning of var=", var_id+1,"/", len(target_domains), ", val=", val_id+1, "/", len(target_domains[var_id][1]))
        return rules


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
    def fit_var_val(variable, value, nb_features, positives, negatives, verbose=0, heuristics=None, threads=1):
        """
        Choose between basic or heuristics learning

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
        if heuristics is None and threads == 1:
            return PRIDE.fit_var_val_basic(variable, value, nb_features, positives, negatives, verbose)
        else:
            return PRIDE.fit_var_val_heuristics(variable, value, nb_features, positives, negatives, heuristics, verbose, threads)

    @staticmethod
    def fit_var_val_basic(variable, value, nb_features, positives, negatives, verbose=0):
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

    @staticmethod
    def fit_var_val_heuristics(variable, value, nb_features, positives, negatives, heuristics, verbose=0, threads=1):
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

        if threads < 1:
            raise(ValueError, "threads must be >= 1")

        # heuristics
        #------------
        # "try_all_atoms": try to find an optimal rule for each atom of the positive example
        # "max_coverage": try to maximize positive matching during specialization
        # DEBUG:
        #heuristics = ["try_all_atoms", "max_coverage","max_diversity"]
        heuristic_try_all_atoms = False
        heuristic_max_coverage_dynamic = False
        heuristic_max_coverage_static = False
        heuristic_max_diversity = False

        if heuristics is not None:
            heuristic_try_all_atoms = "try_all_atoms" in heuristics
            heuristic_max_coverage_dynamic = "max_coverage_dynamic" in heuristics
            heuristic_max_coverage_static = "max_coverage_static" in heuristics
            heuristic_max_diversity = "max_diversity" in heuristics
        #------------

        # Compute coverage score of all atoms
        coverage_static_score = dict()
        if heuristic_max_coverage_static:
            heuristic_max_coverage_dynamic = False
            for pos in positives:
                for var in range(len(pos)):
                    coverage_static_score[(var,pos[var])] = coverage_static_score.get((var,pos[var]), 0) + 1

        remaining = positives.copy()
        output = []

        # exausting covering loop
        while len(remaining) > 0:
            if(verbose > 0):
                eprint("\rRemaining positives:",len(remaining))
            #eprint("Negatives: "+str(negatives))
            #eprint("new target: "+str(target))

            if threads > 1:
                thread_parameters = []
                for target_id in range(0,min(threads, len(remaining))):
                    thread_parameters.append([remaining[target_id], variable, value, nb_features, positives, negatives, heuristics, verbose, coverage_static_score])

                with multiprocessing.Pool(processes=len(thread_parameters)) as pool:
                    optimal_rules = pool.map(PRIDE.fit_var_val_heuristics_thread, thread_parameters)
                optimal_rules = list(itertools.chain.from_iterable(optimal_rules))
            else:
                optimal_rules = PRIDE.fit_var_val_heuristics_thread([remaining[0], variable, value, nb_features, positives, negatives, heuristics, verbose, coverage_static_score])

            # Add new minimal rules
            #eprint("New rule: "+R.to_string())
            for R in optimal_rules:
                output.append(R)
            #remaining.pop(0)

            # 3) Clean new covered positives examples
            #------------------------------------------
            for R in optimal_rules:
                i = 0
                while i < len(remaining):
                    if R.matches(remaining[i]):
                        #eprint("Covers "+str(remaining[i]))
                        remaining.pop(i)
                    else:
                        i += 1
        if(verbose > 0):
            eprint("\rRemaining positives:",len(remaining))

        return output

    @staticmethod
    def fit_var_val_heuristics_thread(args):
        target, variable, value, nb_features, positives, negatives, heuristics, verbose, coverage_static_score = args

        # DBG
        #eprint("Start thread, target: ",target)

        heuristic_try_all_atoms = False
        heuristic_max_coverage_dynamic = False
        heuristic_max_coverage_static = False
        heuristic_max_diversity = False


        #heuristics
        if heuristics is not None:
            heuristic_try_all_atoms = "try_all_atoms" in heuristics
            heuristic_max_coverage_dynamic = "max_coverage_dynamic" in heuristics
            heuristic_max_coverage_static = "max_coverage_static" in heuristics
            heuristic_max_diversity = "max_diversity" in heuristics

        R = Rule(variable, value, nb_features)
        candidates = [(R,None)]
        optimal_rules = set() # prevent duplicate

        # Try to make a rule from each atom of the target
        if heuristic_try_all_atoms:
            for var in range(0,len(target)):
                R_copy = R.copy()
                R_copy.add_condition(var, target[var])
                candidates.append((R_copy,var))

        #eprint(R.to_string())

        # 1) Consistency: against negatives examples
        #---------------------------------------------
        for R, forced_var in candidates:
            consistent = True
            if heuristic_max_coverage_dynamic:
                matched_positives = [s for s in positives if R.matches(s)]
            for neg in negatives:
                if R.matches(neg): # Cover a negative example
                    #eprint(R.to_string() + " matches " + str(neg))
                    specializations = []
                    for var in range(0,len(target)):
                        if not R.has_condition(var) and neg[var] != target[var]: # free condition
                            #eprint("adding condition "+str(var)+":"+str(var)+"="+str(target[var]))
                            if not heuristic_max_coverage_dynamic and not heuristic_max_coverage_static and not heuristic_max_diversity:
                                R.add_condition(var,target[var]) # add value of target positive example
                                break
                            else: # store and score all specialization
                                R_copy = R.copy()
                                R_copy.add_condition(var,target[var])
                                score = 1
                                if heuristic_max_coverage_dynamic:
                                    matched_positives = [s for s in matched_positives if R_copy.matches(s)]
                                    coverage = len(matched_positives)
                                    score *= coverage
                                if heuristic_max_coverage_static:
                                    coverage = coverage_static_score[(var,target[var])]
                                    score *= coverage
                                if heuristic_max_diversity:
                                    diversity = len([r for r in optimal_rules if not R_copy.subsumes(r) and not r.subsumes(R_copy)])
                                    score *= 1+diversity
                                specializations.append((score,R_copy))

                    if heuristic_max_coverage_dynamic:
                        matched_positives = [s for s in positives if R.matches(s)]

                    if heuristic_max_coverage_dynamic or heuristic_max_coverage_static or heuristic_max_diversity:
                        R = sorted(specializations, reverse=True, key=lambda x: x[0])[0][1]

            # 2) Minimalize: only necessary conditions
            #-------------------------------------------
            reductible = True

            conditions = R.body.copy()

            for (var,val) in conditions:
                # Keep forced_var until last moment
                if(forced_var is not None and var == forced_var):
                    continue

                R.remove_condition(var) # Try remove condition

                conflict = False
                for neg in negatives:
                    if R.matches(neg): # Cover a negative example
                        conflict = True
                        R.add_condition(var,val) # Cancel removal
                        break

            # Finally check if force_var is necessary
            if(forced_var is not None):
                var = forced_var
                val = R.get_condition(var)
                R.remove_condition(var) # Try remove condition

                conflict = False
                for neg in negatives:
                    if R.matches(neg): # Cover a negative example
                        conflict = True
                        R.add_condition(var,val) # Cancel removal
                        break

            optimal_rules.add(R)

        # DBG
        #eprint("End thread, target: ",target)

        return optimal_rules

    @staticmethod
    def find_one_optimal_rule_of(variable, value, nb_features, positives, negatives, feature_state_to_match, verbose=0):
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
            feature_state_to_match: list of int
                Feature state that must matched by the learned rule
        Returns:
            Rule or None
            An optimal rule that matches feature_state_to_match and atleast one element of positives if there exists one.
            Returns None otherwize.
        """
        if verbose > 0:
            eprint("Searching for a rule of var="+str(variable)+", val="+str(value)+" that matches "+str(feature_state_to_match)+" and positives feature states")

        if feature_state_to_match in negatives:
            return None

        remaining = positives.copy()
        output = []

        # exausting covering loop
        while len(remaining) > 0:
            #eprint("Remaining positives: "+str(remaining))
            #eprint("Negatives: "+str(negatives))
            target = list(remaining.pop(0))
            if verbose > 0:
                eprint("Check "+str(target))
            all_diff = True
            for var in range(0,len(target)):
                if target[var] != feature_state_to_match[var]:
                    target[var] = -1
                else:
                    all_diff = False
            if all_diff:
                if verbose > 0:
                    eprint("Cannot match both: no common value")
                continue

            if verbose > 0:
                eprint("Common values: "+str(target))

            R = Rule(variable, value, nb_features)
            #eprint(R.to_string())

            # 1) Consistency: against negatives examples
            #---------------------------------------------
            consistent = True
            for neg in negatives:
                if R.matches(neg): # Cover a negative example
                    #eprint(R.to_string() + " matches " + str(neg))
                    consistent = False
                    for var in range(0,len(target)):
                        if not R.has_condition(var) and neg[var] != target[var]: # free condition
                            #eprint("adding condition "+str(var)+":"+str(var)+"="+str(target[var]))
                            if target[var] > -1: # Valid target value (-1 encode all value for partial state)
                                R.add_condition(var,target[var]) # add value of target positive example
                                consistent = True
                                break
                    if not consistent:
                        if verbose > 0:
                            eprint("Cannot avoid matching negative: "+str(neg))
                        break
            if not consistent:
                continue

            if verbose > 0:
                eprint("Consistent rule found: "+R.to_string())

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

            return R

        return None
