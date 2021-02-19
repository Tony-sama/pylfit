#-----------------------
# @author: Tony Ribeiro
# @created: 2020/01/03
# @updated: 2020/01/03
#
# @desc: simple implementation of synchronous probabilistic semantic over LogicProgram
#   - Update all variables at the same time
#   - Can generate non-deterministic transitions
#   - generate triplet (init state, next state, probability)
#-----------------------

from ..utils import eprint
from ..objects.rule import Rule
from ..objects.logicProgram import LogicProgram

import itertools
from fractions import Fraction

class SynchronousProbabilistic:
    """
    Define the synchronous probabilistic semantic over discrete multi-valued logic program
    """

    def next_heuristic(program, state, weigths):
        """
        Compute the next state according to the rules.

        Args:
            program: LogicProgram
                A multi-valued logic program
            state: list of int
                A state of the system

        Returns:
            list of (list of int, fraction)
                the possible next states and their probability according to the rules of the program.
        """
        output = []
        values_likelihood = [ [0 for val in program.get_values()[var]] for var in range(0,len(program.get_variables()))]

        # extract conclusion of all matching rules
        rule_id = 0
        for r in program.get_rules():
            if(r.matches(state)):
                val_str = program.get_conclusion_values()[r.get_head_variable()][r.get_head_value()]

                comma = val_str.find(",")
                value_id = int(val_str[:comma])
                proba_str = val_str[comma+1:]

                slash = proba_str.find("/")
                frac_top = int(proba_str[:slash])
                frac_bot = int(proba_str[slash+1:])
                proba = Fraction(frac_top, frac_bot)

                values_likelihood[r.get_head_variable()][value_id] += proba * weigths[rule_id][0]

                # DBG
                #eprint(r.to_string())
                #eprint(weigths[rule_id][1].to_string())

                if weigths[rule_id][1] != r:
                    raise Exception("ERROR: weights not correspond to program rules ordering")
            rule_id += 1

        #eprint(values_likelihood)
        interpreted_domains = [set() for var in program.get_variables()]

        # Ensure proba sum to one
        for var_id in range(0,len(values_likelihood)):
            sum_likelihood = sum(values_likelihood[var_id])
            # No matching rule for this variable, keep value
            if sum_likelihood == 0:
                values_likelihood[var_id][state[var_id]] = Fraction(1,1)
            else:
                values_likelihood[var_id] = [ i / sum_likelihood for i in values_likelihood[var_id]]
            interpreted_domains[var_id] = set((i, values_likelihood[var_id][i]) for i in range(0, len(values_likelihood[var_id])) if values_likelihood[var_id][i] > 0)

        #eprint("Interpreted domains: ", interpreted_domains)

        possible = [i for i in list(itertools.product(*interpreted_domains))]

        # DBG
        #eprint("Possible next states: ", possible)

        # Compute probability
        output = []
        for s in possible:
            state = [0 for var_id in s]
            proba = 1
            var_id = 0
            for (val,p) in s:
                proba *= p
                state[var_id] = val
                var_id += 1
            output.append((state,proba))

        # Check summ to one
        value = 0
        for (s,p) in output:
            value += p

        if value != 1:
            eprint("Error proba do not sum to one: ", value)

        # DBG
        #eprint("Output: ", output)

        return output

    def next(program, state):
        """
        Compute the next state according to the rules of the program.

        Args:
            program: LogicProgram
                A multi-valued logic program
            state: list of int
                A state of the system

        Returns:
            list of (list of int, fraction)
                the possible next states and their probability according to the rules of the program.
        """
        output = []
        domains = [set() for var in program.get_variables()]

        # extract conclusion of all matching rules
        for r in program.get_rules():
            if(r.matches(state)):
                domains[r.get_head_variable()].add(r.get_head_value())

        # Interprete conclusions and agregate probabilities
        interpreted_domains = [set() for var in domains]
        for var_id in range(0, len(domains)):
            values_likelihood = [0 for val in program.get_values()[var_id]]
            # DBG
            values = []
            for conclusion_value_id in domains[var_id]:
                val_str = program.get_conclusion_values()[var_id][conclusion_value_id]

                comma = val_str.find(",")
                value_id = int(val_str[:comma])
                proba_str = val_str[comma+1:]

                slash = proba_str.find("/")
                frac_top = int(proba_str[:slash])
                frac_bot = int(proba_str[slash+1:])
                proba = Fraction(frac_top, frac_bot)

                values_likelihood[value_id] += proba
                # DBG
                values.append((value_id,proba))


                #DBG
                #eprint(val_str, " => value_id=", value_id, ", proba=", proba)
                #exit()

            # Ensure proba sum to one
            sum_likelihood = sum(values_likelihood)
            values_likelihood = [ i / sum_likelihood for i in values_likelihood]
            interpreted_domains[var_id] = set((i,values_likelihood[i]) for i in range(0, len(values_likelihood)))
            # DBG
            #eprint(values_likelihood)
            #eprint(values)
            #eprint(interpreted_domains)

        # DBG
        #eprint("Original domains: ", domains)
        #eprint("Interpreted domains: ", interpreted_domains)

        # generate all combination of conclusions
        possible = [i for i in list(itertools.product(*interpreted_domains))]

        # DBG
        #eprint("Possible next states: ", possible)

        # Compute probability
        output = []
        for s in possible:
            state = [0 for var_id in s]
            proba = 1
            var_id = 0
            for (val,p) in s:
                proba *= p
                state[var_id] = val
                var_id += 1
            output.append((state,proba))

        # Check summ to one
        value = 0
        for (s,p) in output:
            value += p

        if value != 1:
            eprint("Error proba do not sum to one: ", value)

        # DBG
        #eprint("Output: ", output)

        return output
