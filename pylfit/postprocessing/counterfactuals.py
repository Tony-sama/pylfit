#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2025/04/10
# @updated: 2025/04/10
#
# @desc: pylfit counterfactual functions
#-------------------------------------------------------------------------------

import pylfit

from pylfit.utils import eprint
from pylfit.objects import LegacyAtom
from pylfit.preprocessing import discrete_state_transitions_dataset_from_array
from pylfit.algorithms import GULA

import itertools


def bruteforce_counterfactuals(dmvlp, feature_state, target_variable, excluded_values, desired_values):
    output = {value: [] for value in desired_values}

    # Rules that produce excluded values
    excluded_rules = [r for r in dmvlp.rules if r.head.variable == target_variable and r.head.value in excluded_values]

    # Rules that can produce desired values
    desired_rules = dict()
    for value in desired_values:
        desired_rules[value] = [r for r in dmvlp.rules if r.head.variable == target_variable and r.head.value == value]

    # 1) Get states that avoid excluded rules and match desired rules
    valid_states = {value: [] for value in desired_values}
    for state in dmvlp.feature_states():
        valid = True

        # State should not be matched by excluded value rule
        for r in excluded_rules:
            if r.matches(state):
                valid = False
                break
        
        if not valid:
            continue

        # State must be matched by the desired value
        for value in desired_values:
            for r in desired_rules[value]:
                if r.matches(state):
                    valid_states[value].append(state)
                    break
        
    # 2) Compute difference with the given state
    for value in desired_values:
        candidates = []
        for state in valid_states[value]:
            changes = set()
            for var_id in range(len(state)):
                if state[var_id] != feature_state[var_id]:
                    changes.add(LegacyAtom(dmvlp.features[var_id][0], set(dmvlp.features[var_id][1]), state[var_id], var_id))
            candidates.append(changes)
    
        # 3) Keep minimal changes
        solutions = []
        for s1 in candidates:
            is_minimal = True
            for s2 in candidates:
                if s1 != s2 and s2 <= s1:
                    is_minimal = False
                    break
            if is_minimal:
                solutions.append(s1)

        output[value] = solutions

    return output

def rule_counters(rule, fixed_atoms=[]):
    output = []
    for var, atom in rule.body.items():
        is_valid = True
        for other in fixed_atoms:
            if other.variable == var:
                is_valid = False
                break
    
        if is_valid:
            other_values = [val for val in atom.domain if val != atom.value]
            for val in other_values:
                new_atom = atom.copy()
                new_atom.value = val
                output += [new_atom]
    
    return output

def rules_counters(rules, feature_state, fixed_atoms=[]):
    """
    Returns
        - list of list of atoms: changes to the feature state that prevent matching any of the given rules
    """

    # Compute counters of each rule
    to_hit = [rule_counters(r, fixed_atoms) for r in rules] # if r.matches(feature_state)]

    # One rule cannot be counter
    for i in to_hit:
        if len(i) == 0:
            return []

    # Compute sets product
    candidates = set(itertools.product(*to_hit))
    candidates = [set(l) for l in candidates]

    # Keep only valid changes (one value per variable)
    solutions = []
    for s in candidates:
        is_valid = True
        variables_found = []
        for atom in s:
            if atom.variable in variables_found:
                is_valid = False
                break
            variables_found.append(atom.variable)

        if is_valid:
            solutions.append(s)

    candidates = solutions

    # Keep only minimals sets
    solutions = []
    for s1 in candidates:
        is_minimal = True
        for s2 in candidates:
            if s1 != s2 and s2 <= s1:
                is_minimal = False
                break
        if is_minimal:
            solutions.append(s1)

    return solutions

def compute_counterfactuals_1(rules, feature_state, target_variable, excluded_values, desired_values, determinist=False, verbose=0):
    """
    Compute all permutation possible of the feature state to achieve desired target values while avoiding excluded values
    
    Returns:
        - a dict of (desired target value: list of feature states)
    """
    output = dict()

    # 1) Compute exclude_values counter sets
    if not determinist:
        excluded_rules = [r for r in rules if r.head.variable == target_variable and r.head.value in excluded_values]# and r.matches(feature_state)]

        # Compute counters of each rule
        to_hit = [rule_counters(r) for r in excluded_rules]

        # Compute sets product
        candidates = set(itertools.product(*to_hit))
        candidates = [set(l) for l in candidates]

        # Keep only minimals sets
        solutions = []
        for s1 in candidates:
            is_minimal = True
            for s2 in candidates:
                if s1 != s2 and s2 <= s1:
                    is_minimal = False
                    break
            if is_minimal:
                solutions.append(s1)

        candidates = solutions

    else: # 1.1) Determinist no need to avoid other values
        candidates = [set()]
        
    common_candidates = candidates.copy()
    
    if verbose > 0:
        eprint("Excluded value counters")
        for c in candidates:
            eprint(">",c)        
    
    for value in desired_values:
        desired_rules = [r for r in rules if r.head.variable == target_variable and r.head.value == value]
        required_conditions = [[atom for _, atom in r.body.items()] for r in desired_rules]

        # 2) Remove sets that make the feature state covered by no rule of desired value
        solutions = set()
        for s in common_candidates:
            for s2 in required_conditions:
                is_valid = True
                # Check compatibility
                for atom in s:
                    for other in s2:
                        if atom.variable == other.variable and atom.value != other.value:
                            is_valid = False
                            break
                    if not is_valid:
                        break

                if is_valid:
                    solutions.add(frozenset(s.union(s2)))

        candidates = solutions
        
        if verbose > 0:
            eprint("Unified with desired values rules")
            for s in candidates:
                eprint(">>",s)

        # 3) Only keep atoms that change the feature state
        solutions = set()
        for s in candidates:
            changed_atom = set()
            for atom in s:
                if feature_state[atom.state_position] != atom.value:
                    changed_atom.add(atom)
            if changed_atom != []:
                solutions.add(frozenset(changed_atom))
        
        candidates = solutions

        if verbose > 0:
            eprint("Actual changes")
            for s in candidates:
                eprint(">>",s)

        # 4) Ensure not covered by rules of excluded target
        excluded_rules = [r for r in rules if r.head.variable == target_variable and r.head.value in excluded_values and not r.matches(feature_state)]
        solutions = []
        for s in candidates:
            is_valid = True
            new_state = feature_state.copy()
            for atom in s:
                new_state[atom.state_position] = atom.value
            # check matching
            for r in excluded_rules:
                if r.matches(new_state):
                    is_valid = False
                    break
            if is_valid:
                solutions.append(set(s))

        candidates = solutions

        if verbose > 0:
            eprint("The one without conflict with excluded rules")
            for s in candidates:
                eprint(">>",s)

        # 5) Keep only minimals sets
        solutions = []
        for s1 in candidates:
            is_minimal = True
            for s2 in candidates:
                if s1 != s2 and s2 <= s1:
                    is_minimal = False
                    break
            if is_minimal:
                solutions.append(s1)

        candidates = solutions

        if verbose > 0:
            eprint("Minimals")
            for s in candidates:
                eprint(">",s)


        output[value] = solutions

    return output


def compute_counterfactuals_2(rules, feature_state, target_variable, excluded_values, desired_values, determinist=False, verbose=0):
        """
        Compute all permutation possible of the feature state to achieve desired target values while avoiding excluded values
        
        Returns:
            - a dict of (desired target value: list of feature states)
        """
        output = dict() 
        
        for value in desired_values:
            desired_rules = [r for r in rules if r.head.variable == target_variable and r.head.value == value]
            required_conditions = [[atom for _, atom in r.body.items()] for r in desired_rules]
            common_candidates = []

            for changes in required_conditions:
                print("Current forced change:",changes)
                new_state = feature_state.copy()
                for atom in changes:
                    new_state[atom.state_position] = atom.value

                print("original state:",feature_state)
                print("new state:", new_state)
                
                excluded_rules = [r for r in rules if r.head.variable == target_variable and r.head.value in excluded_values]# and r.matches(feature_state)]
                candidates = rules_counters(excluded_rules, new_state, fixed_atoms=changes)

                candidates = [i.union(changes) for i in candidates] # add the changes of the desired rule

                # 3) Only keep atoms that change the feature state
                solutions = set()
                for s in candidates:
                    changed_atom = set()
                    for atom in s:
                        if feature_state[atom.state_position] != atom.value:
                            changed_atom.add(atom)
                    if changed_atom != []:
                        solutions.add(frozenset(changed_atom))
                
                candidates = solutions

                if verbose > 0:
                    print("> Actual changes")
                    for s in candidates:
                        print(">>",s)

                # 5) Keep only minimals sets
                solutions = []
                for s1 in candidates:
                    is_minimal = True
                    for s2 in candidates:
                        if s1 != s2 and s2 <= s1:
                            is_minimal = False
                            break
                    if is_minimal:
                        solutions.append(s1)

                candidates = solutions

                if verbose > 0:
                    print("Minimals")
                    for s in candidates:
                        print(">",s)

                common_candidates += solutions

             # 6) Keep only minimals sets
            solutions = []
            for s1 in common_candidates:
                is_minimal = True
                for s2 in common_candidates:
                    if s1 != s2 and s2 <= s1:
                        is_minimal = False
                        break
                if is_minimal:
                    solutions.append(list(s1))

            output[value] = [set(s1) for s1 in solutions]

        return output
    
     # TODO: is_determinist, just need to compare rules bodies ?

# TODO: try encoding the problem to be solve by GULA :p

def compute_counterfactuals(dmvlp, feature_state, target_variable, excluded_values, desired_values, verbose=0, determinist=False):
    """
    Compute all permutation possible of the feature state to achieve desired target values while avoiding excluded values
    
    Returns:
        - a dict of (desired target value: list of feature states)
    """
    output = dict()

    # 1) Compute exclude_values counter sets
    if not determinist:
        # 1) transform excluded values rules into negative examples
        excluded_rules = [r for r in dmvlp.rules if r.head.variable == target_variable and r.head.value in excluded_values]

        if verbose > 0:
            eprint("Excluded rules:")
            for r in excluded_rules:
                eprint(">", r)

        unknown_state = [LegacyAtom._VOID_VALUE for i in dmvlp.features]
        negatives = []
        for r in excluded_rules:
            new_state = unknown_state.copy()
            for var, atom in r.body.items():
                new_state[atom.state_position] = atom.value
            negatives.append(new_state)
        
        if verbose > 0:
            eprint("As states:")
            for i in negatives:
                eprint(">", i)

        features_void_atoms = dict()
        for var_id, (var, vals) in enumerate(dmvlp.features):
            features_void_atoms[var] = LegacyAtom(var, set(vals), LegacyAtom._VOID_VALUE, var_id)

        # Use gula to compute rules that avoid excluded rules
        rules = GULA.fit_var_val_strict(LegacyAtom("valid",{"true","false"},"true",0), features_void_atoms, negatives, verbose)
        candidates = [set([atom for _, atom in r.body.items()]) for r in rules]

        if verbose > 0:
            eprint("Rules found:")
            for r in candidates:
                eprint(r)

    else: # 1.1) Determinist no need to avoid other values
        candidates = [set()]
        
    common_candidates = candidates.copy()
    
    if verbose > 0:
        eprint("Excluded value counters")
        for c in candidates:
            eprint(">",c)        
    
    for value in desired_values:
        desired_rules = [r for r in dmvlp.rules if r.head.variable == target_variable and r.head.value == value]
        required_conditions = [[atom for _, atom in r.body.items()] for r in desired_rules]

        # 2) Remove sets that make the feature state covered by no rule of desired value
        solutions = set()
        for s in common_candidates:
            for s2 in required_conditions:
                is_valid = True
                # Check compatibility
                for atom in s:
                    for other in s2:
                        if atom.variable == other.variable and atom.value != other.value:
                            is_valid = False
                            break
                    if not is_valid:
                        break

                if is_valid:
                    solutions.add(frozenset(s.union(s2)))

        candidates = solutions
        
        if verbose > 0:
            eprint("Unified with desired values rules")
            for s in candidates:
                eprint(">>",s)

        # 3) Only keep atoms that change the feature state
        solutions = set()
        for s in candidates:
            changed_atom = set()
            for atom in s:
                if feature_state[atom.state_position] != atom.value:
                    changed_atom.add(atom)
            if changed_atom != []:
                solutions.add(frozenset(changed_atom))
        
        candidates = solutions

        if verbose > 0:
            eprint("Actual changes")
            for s in candidates:
                eprint(">>",s)

        # 4) Ensure not covered by rules of excluded target
        #excluded_rules = [r for r in rules if r.head.variable == target_variable and r.head.value in excluded_values and not r.matches(feature_state)]
        #solutions = []
        #for s in candidates:
        #    is_valid = True
        #    new_state = feature_state.copy()
        #    for atom in s:
        #        new_state[atom.state_position] = atom.value
            # check matching
        #    for r in excluded_rules:
        #        if r.matches(new_state):
        #            is_valid = False
        #            break
        #    if is_valid:
        #        solutions.append(set(s))

        #candidates = solutions

        if verbose > 0:
            eprint("The one without conflict with excluded rules")
            for s in candidates:
                eprint(">>",s)

        # 5) Keep only minimals sets
        solutions = []
        for s1 in candidates:
            is_minimal = True
            for s2 in candidates:
                if s1 != s2 and s2 <= s1:
                    is_minimal = False
                    break
            if is_minimal:
                solutions.append(s1)

        candidates = solutions

        if verbose > 0:
            eprint("Minimals")
            for s in candidates:
                eprint(">",s)

        output[value] = [set(s1) for s1 in solutions]

    return output