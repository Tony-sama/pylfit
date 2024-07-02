#-----------------------
# @author: Tony Ribeiro
# @created: 2019/03/20
# @updated: 2027/12/27
#
# @desc: simple approximated version of GULA implementation.
#    - extract patern from pair of interpretation of transitions
#
#-----------------------

from ..utils import eprint
from ..objects.legacyAtom import LegacyAtom
from ..objects.rule import Rule
from ..algorithms import Algorithm
from ..datasets import DiscreteStateTransitionsDataset

import multiprocessing
import itertools

class PRIDE (Algorithm):
    """
    Define a simple approximative version of the GULA algorithm.
    Learn logic rules that explain state transitions of a discrete dynamic system.
    """

    """ Learning heuristics that can be use with the algorithm"""
    _HEURISTICS = ["try_all_atoms", "max_coverage_dynamic", "max_coverage_static", "max_diversity", "multi_thread_at_rule_level"]

    def fit(dataset, options=None):
        """
        Preprocess transitions and learn rules for all given features/targets variables/values.

        Args:
            dataset: pylfit.datasets.DiscreteStateTransitionsDataset
                state transitions of a the system
            options: dict string:any
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
                threads: int (>=1)
                    Number of CPU threads to be used

        Returns:
            list of pylfit.objects.Rule
                A set of DMVLP rules that is:
                    - correct: explain/reproduce all the transitions of the dataset.
                    - optimal: all rules are minimals
        """
        # Options
        targets_to_learn = None
        impossibility_mode = False
        verbose = 0
        heuristics = None
        threads = 1

        if options is not None:
            if "targets_to_learn" in options:
                targets_to_learn = options["targets_to_learn"]
            if "impossibility_mode" in options:
                impossibility_mode = options["impossibility_mode"]
            if "verbose" in options:
                verbose = options["verbose"]
            if "heuristics" in options:
                heuristics = options["heuristics"]
            if "threads" in options:
                threads = options["threads"]


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
                
        if threads < 1:
            raise ValueError("Argument thread must be >=1")

        target_domains = dataset.targets
        rules = []
        thread_parameters = []
        
        # Nothing to learn
        if len(dataset.data) == 0:
            return []

        # Learn rules for each observed variable/value
        for var_id, (var_name, var_domain) in enumerate(dataset.targets):
            for val_id, val_name in enumerate(var_domain):
                if var_name not in targets_to_learn:
                    continue
                if val_name not in targets_to_learn[var_name]:
                    continue

                head = LegacyAtom(var_name, set(var_domain), val_name, var_id)

                if(threads == 1):
                    if verbose > 0:
                        eprint("\nStart learning of var=", var_id+1,"/", len(target_domains), ", val=", val_id+1, "/", len(target_domains[var_id][1]))

                    rules += PRIDE.fit_thread([head, dataset, impossibility_mode, verbose, heuristics, threads])
                elif heuristics is not None and "multi_thread_at_rule_level" in heuristics:
                    rules += PRIDE.fit_thread([head, dataset, impossibility_mode, verbose, heuristics, threads])
                else:
                    thread_parameters.append([head, dataset, impossibility_mode, verbose, heuristics, 1])

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
        """
        Thread wrapper for fit_var/fit_var_val_with_unknown_values functions (see below)
        """
        head, dataset, impossibility_mode, verbose, heuristics, threads = args
        if verbose > 0:
            eprint("\nStart learning of ", head)

        positives, negatives = PRIDE.interprete(dataset, head)

        # Remove potential false negatives
        if dataset.has_unknown_values:
            certain_negatives = []
            for neg in negatives:
                uncertain_negative = False
                for pos in positives:
                    possible_same_state = True
                    for i in range(len(pos)):
                        if neg[i] != pos[i] and pos[i] != dataset._UNKNOWN_VALUE and neg[i] != dataset._UNKNOWN_VALUE: # explicit difference
                            possible_same_state = False
                            break
                    if possible_same_state:
                        uncertain_negative = True
                        break
                if not uncertain_negative:
                    certain_negatives.append(neg)

            negatives = certain_negatives

        if impossibility_mode:
            rules = PRIDE.fit_var_val(head, dataset, negatives, positives, verbose, heuristics, threads)
        else:
            rules = PRIDE.fit_var_val(head, dataset, positives, negatives, verbose, heuristics, threads)


        if verbose > 0:
            eprint("\nFinished learning of ", head)
        return rules


    @staticmethod
    def interprete(dataset, head):
        """
        Split transition into positive/negatives states for the given variable/value
        Warning: assume deterministic transitions

        Args:
            dataset: pylfit.datasets.DiscreteStateTransitionsDataset
                state transitions of a the system
            head: Atom
                target atom
        Returs:
            positives: list of list of any
            negatives: list of list of any
        """
        positives = set(tuple(s1) for s1,s2 in dataset.data if head.matches(s2) or s2[head.state_position] == LegacyAtom._UNKNOWN_VALUE)
        negatives = set(tuple(s1) for s1,s2 in dataset.data if tuple(s1) not in positives)

        return list(positives), list(negatives)

    @staticmethod
    def fit_var_val(head, dataset, positives, negatives, verbose=0, heuristics=None, threads=1):
        """
        Choose between basic or heuristics learning

        Args:
            head: Atom
                target atom
            dataset: pylfit.datasets.DiscreteStateTransitionsDataset
                state transitions of a the system
            positive: list of (list of any)
                States of the system where the head matches the next state
            negative: list of (list of any)
                States of the system where the head matches the next state
            verbose: int
                When greater than 0 progress of learning will be print in stderr
            heuristics: list of string
                see fit function above
            threads: int (>=1)
                Number of CPU threads to be used
        Returns: list of Rules
        """
        if heuristics is None and threads == 1:
            return PRIDE.fit_var_val_basic(head, dataset, positives, negatives, verbose)
        else:
            return PRIDE.fit_var_val_heuristics(head, dataset, positives, negatives, heuristics, verbose, threads)

    @staticmethod
    def fit_var_val_basic(head, dataset, positives, negatives, verbose=0):
        """
        Learn minimal rules that explain positive examples while consistent with negatives examples

        Args:
            head: Atom
                target atom
            dataset: pylfit.datasets.DiscreteStateTransitionsDataset
                state transitions of a the system
            positive: list of (list of any)
                States of the system where the head matches the next state
            negative: list of (list of any)
                States of the system where the head matches the next state
            verbose: int
                When greater than 0 progress of learning will be print in stderr
        Returns:
            list of rules
        """
        remaining = []

        # 0) Clean unexplainable positives
        #------------------------------------
        #if dataset.has_unknown_values():
        #    for pos in positives:
        #        explainable = True
        #        values = [(var,val) for var, val in enumerate(pos) if val != dataset._UNKNOWN_VALUE]
        #        pos_rule = Rule(head)
        #        for var_id,val in values:
        #            var_name = dataset.features[var_id][0]
        #            domain = set(dataset.features[var_id][1])
        #            pos_rule.add_condition(LegacyAtom(var_name,domain,val,var_id))
        #
        #        for neg in negatives:
        #            if pos_rule.matches(neg):
        #                explainable = False
        #                break
        #
        #        if explainable:
        #           remaining.append(pos)
        #else:
        #    remaining = positives.copy()
        remaining = positives.copy()

        output = []

        # exausting covering loop
        while len(remaining) > 0:
            target = remaining[0]

            R = Rule(head)

            # 1) Consistency: against negatives examples
            #---------------------------------------------
            for neg in negatives:
                if R.matches(neg): # Cover a negative example
                    specialized = False
                    for var in dataset.features_void_atoms:
                        void_atom = dataset.features_void_atoms[var]
                        if R.has_condition(void_atom.variable):
                            ls = R.get_condition(void_atom.variable).least_specialization(neg)
                        else:
                            ls = void_atom.least_specialization(neg)
                        for atom in ls:
                            if atom.matches(target):
                                R.add_condition(atom)
                                specialized = True
                                break
                        if specialized:
                            break

            # 2) Minimalize: only necessary conditions
            #-------------------------------------------
            conditions = R.body.copy().items()

            for (var,atom) in conditions:
                R.remove_condition(var) # Try remove condition

                for neg in negatives:
                    if R.matches(neg): # Cover a negative example
                        R.add_condition(atom) # Cancel removal
                        break

            # Add new minimal rule
            output.append(R)
            remaining.pop(0)

            # 3) Clean new covered positives examples
            #------------------------------------------
            i = 0
            while i < len(remaining):
                if R.matches(remaining[i]):
                    remaining.pop(i)
                else:
                    i += 1

        return output

    @staticmethod
    def fit_var_val_heuristics(head, dataset, positives, negatives, heuristics=None, verbose=0, threads=1):
        """
        Learn minimal rules that explain positive examples while consistent with negatives examples

        Args:
            head: Atom
                target atom
            dataset: pylfit.datasets.DiscreteStateTransitionsDataset
                state transitions of a the system
            positive: list of (list of any)
                States of the system where the head matches the next state
            negative: list of (list of any)
                States of the system where the head matches the next state
            heuristics: list of string
                see fit function above
            verbose: int
                When greater than 0 progress of learning will be print in stderr
            threads: int (>=1)
                Number of CPU threads to be used
        Returns:
            list of Rules
        """
        if verbose > 0:
            eprint("Start learning of ",head)

        if threads < 1:
            raise ValueError("Argument threads must be >= 1")

        # heuristics
        #------------
        # "try_all_atoms": try to find an optimal rule for each atom of the positive example
        # "max_coverage": try to maximize positive matching during specialization
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
            for var in dataset.features_void_atoms:
                atom = dataset.features_void_atoms[var].copy()
                if isinstance(atom, LegacyAtom):
                    for val in atom.domain:
                        a = atom.copy()
                        a.value = val
                        coverage_static_score[a] = 0

            for pos in positives:
                for var in dataset.features_void_atoms:
                    atom = dataset.features_void_atoms[var].copy()
                    if isinstance(atom, LegacyAtom):
                        for val in atom.domain:
                            a = atom.copy()
                            a.value = val
                        if a.matches(pos):
                            coverage_static_score[a] += 1

        remaining = []

        # 0) Clean unexplainable positives
        #------------------------------------
        if dataset.has_unknown_values():
            for pos in positives:
                explainable = True
                values = [(var,val) for var, val in enumerate(pos) if val != dataset._UNKNOWN_VALUE]
                pos_rule = Rule(head)
                for var_id,val in values:
                    var_name = dataset.features[var_id][0]
                    domain = set(dataset.features[var_id][1])
                    pos_rule.add_condition(LegacyAtom(var_name,domain,val,var_id))

                for neg in negatives:
                    if pos_rule.matches(neg):
                        explainable = False
                        break

                if explainable:
                    remaining.append(pos)
        else:
            remaining = positives.copy()

        output = []

        # exausting covering loop
        while len(remaining) > 0:
            if(verbose > 0):
                eprint("\rRemaining positives:",len(remaining))

            if threads > 1:
                thread_parameters = []
                for target_id in range(0,min(threads, len(remaining))):
                    thread_parameters.append([remaining[target_id], head, dataset, positives, negatives, heuristics, verbose, coverage_static_score])

                with multiprocessing.Pool(processes=len(thread_parameters)) as pool:
                    optimal_rules = pool.map(PRIDE.fit_var_val_heuristics_thread, thread_parameters)
                optimal_rules = list(itertools.chain.from_iterable(optimal_rules))
            else:
                optimal_rules = PRIDE.fit_var_val_heuristics_thread([remaining[0], head, dataset, positives, negatives, heuristics, verbose, coverage_static_score])

            # Add new minimal rules
            for R in optimal_rules:
                output.append(R)

            # 3) Clean new covered positives examples
            #------------------------------------------
            for R in optimal_rules:
                i = 0
                while i < len(remaining):
                    if R.matches(remaining[i]):
                        remaining.pop(i)
                    else:
                        i += 1
        if(verbose > 0):
            eprint("\rRemaining positives:",len(remaining))

        return output

    @staticmethod
    def fit_var_val_heuristics_thread(args):
        """
        thread wrapper for it_var_val_heuristics
        """
        target, head, dataset, positives, negatives, heuristics, verbose, coverage_static_score = args

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

        R = Rule(head)
        candidates = [(R,None)]
        optimal_rules = set() # prevent duplicate

        # Try to make a rule from each atom of the target
        if heuristic_try_all_atoms:
            for var in dataset.features_void_atoms:
                # TODO: maybe a check for Legacy or not?
                atom = dataset.features_void_atoms[var].copy()
                if target[atom.state_position] != dataset._UNKNOWN_VALUE:
                    atom.value = target[atom.state_position]
                    R_copy = R.copy()
                    R_copy.add_condition(atom)
                    candidates.append((R_copy,atom.variable))

        # 1) Consistency: against negatives examples
        #---------------------------------------------
        for R, forced_var in candidates:
            consistent = True
            if heuristic_max_coverage_dynamic:
                matched_positives = [s for s in positives if R.matches(s)]
            for neg in negatives:
                if R.matches(neg): # Cover a negative example
                    specializations = []
                    specialized = False
                    for var in dataset.features_void_atoms:
                        void_atom = dataset.features_void_atoms[var]
                        if R.has_condition(void_atom.variable):
                            ls = R.get_condition(void_atom.variable).least_specialization(neg)
                        else:
                            ls = void_atom.least_specialization(neg)
                        for atom in ls:
                            if atom.matches(target): # 
                                if not heuristic_max_coverage_dynamic and not heuristic_max_coverage_static and not heuristic_max_diversity:
                                    R.add_condition(atom)
                                    specialized = True
                                    break
                                else: # store and score all specialization
                                    R_copy = R.copy()
                                    R_copy.add_condition(atom)
                                    score = 1
                                    if heuristic_max_coverage_dynamic:
                                        matched_positives = [s for s in matched_positives if R_copy.matches(s)]
                                        coverage = len(matched_positives)
                                        score *= coverage
                                    if isinstance(atom, LegacyAtom) and heuristic_max_coverage_static:
                                        coverage = coverage_static_score[(atom)]
                                        score *= coverage
                                    if heuristic_max_diversity:
                                        diversity = len([r for r in optimal_rules if not R_copy.subsumes(r) and not r.subsumes(R_copy)]) # TODO: improve
                                        score *= 1+diversity
                                    specializations.append((score,R_copy))
                        if specialized:
                            break

                    if heuristic_max_coverage_dynamic or heuristic_max_coverage_static or heuristic_max_diversity:
                        R = sorted(specializations, reverse=True, key=lambda x: x[0])[0][1]
                    
            # 2) Minimalize: only necessary conditions
            #-------------------------------------------

            conditions = R.body.copy().items()

            for (var,atom) in conditions:
                # Keep forced_var until last moment
                if(forced_var is not None and var == forced_var):
                    continue
                R.remove_condition(var) # Try remove condition

                for neg in negatives:
                    if R.matches(neg): # Cover a negative example
                        R.add_condition(atom) # Cancel removal
                        break

            # Finally check if force_var is necessary
            if(forced_var is not None):
                var = forced_var
                atom = R.get_condition(var)
                R.remove_condition(var) # Try remove condition

                for neg in negatives:
                    if R.matches(neg): # Cover a negative example
                        R.add_condition(atom) # Cancel removal
                        break

            optimal_rules.add(R)

        return optimal_rules
    

    @staticmethod
    def find_one_optimal_rule_of(head, dataset, positives, negatives, feature_state_to_match, verbose=0):
        """
        Learn minimal rules that explain positive examples while consistent with negatives examples

        Args:
            head: Atom
                target atom
            dataset: pylfit.datasets.DiscreteStateTransitionsDataset
                state transitions of a the system
            positive: list of (list of any)
                States of the system where the head matches the next state
            negative: list of (list of any)
                States of the system where the head matches the next state
            feature_state_to_match: list of string
                Feature state that must matched by the learned rule
            verbose: int
                When greater than 0 progress of learning will be print in stderr
        Returns:
            Rule or None
            An optimal rule that matches feature_state_to_match and atleast one element of positives if there exists one.
            Returns None otherwize.
        """
        if verbose > 0:
            eprint("Searching for a rule of "+str(head)+" that matches "+str(feature_state_to_match)+" and positives feature states")

        if feature_state_to_match in negatives:
            return None

        remaining = positives.copy()
        output = []

        # exausting covering loop
        while len(remaining) > 0:
            target = list(remaining.pop(0))
            if verbose > 0:
                eprint("Check "+str(target))
            all_diff = True
            for var in range(0,len(target)):
                if target[var] != feature_state_to_match[var]:
                    target[var] = "<<invalid_value>>" # HACK: value not in domain
                else:
                    all_diff = False
            if all_diff:
                if verbose > 0:
                    eprint("Cannot match both: no common value")
                continue

            if verbose > 0:
                eprint("Common values: "+str(target))

            R = Rule(head)

            # 1) Consistency: against negatives examples
            #---------------------------------------------
            consistent = True
            for neg in negatives:
                if R.matches(neg): # Cover a negative example
                    specialized = False
                    for var in dataset.features_void_atoms:
                        void_atom = dataset.features_void_atoms[var]
                        if R.has_condition(void_atom.variable):
                            ls = R.get_condition(void_atom.variable).least_specialization(neg)
                        else:
                            ls = void_atom.least_specialization(neg)
                        for atom in ls:
                            if atom.matches(target):
                                R.add_condition(atom)
                                specialized = True
                                break
                        if specialized:
                            break

                    if not specialized:
                        if verbose > 0:
                            eprint("Cannot avoid matching negative: "+str(neg))
                        consistent = False
                        break
            if not consistent:
                continue

            # 2) Minimalize: only necessary conditions
            #-------------------------------------------
            conditions = R.body.copy().items()

            for (var,atom) in conditions:
                R.remove_condition(var) # Try remove condition
                for neg in negatives:
                    if R.matches(neg): # Cover a negative example
                        R.add_condition(atom) # Cancel removal
                        break

            return R

        return None
