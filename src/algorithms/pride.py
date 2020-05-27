#-----------------------
# @author: Tony Ribeiro
# @created: 2019/03/20
# @updated: 2019/05/03
#
# @desc: simple approximated version of GULA implementation.
#    - extract patern from pair of interpretation of transitions
#
#-----------------------

from utils import eprint
from rule import Rule
from logicProgram import LogicProgram
import csv

class PRIDE:
    """
    Define a simple approximative version of the GULA algorithm.
    Learn logic rules that explain state transitions of a discrete dynamic system.
    """

    @staticmethod
    def load_input_from_csv(filepath):
        """
        Load transitions from a csv file

        Args:
            filepath: String
                Path to csv file encoding transitions
        """
        output = []
        with open(filepath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            x_size = 0

            for row in csv_reader:
                if line_count == 0:
                    x_size = row.index("y0")
                    x = row[:x_size]
                    y = row[x_size:]
                    #eprint("x: "+str(x))
                    #eprint("y: "+str(y))
                else:
                    row = [int(i) for i in row] # integer convertion
                    output.append([row[:x_size], row[x_size:]]) # x/y split
                line_count += 1

            #eprint(f'Processed {line_count} lines.')
        return output

    @staticmethod
    def fit(variables, values, transitions):
        """
        Preprocess transitions and learn rules for all observed variables/values.

        Args:
            transitions: list of tuple (list of int, list of int)
                state transitions of dynamic system

        Returns:
            LogicProgram
                A logic program whose rules:
                    - are minimals
                    - explain/reproduce all the input transitions
        """
        #eprint("Start PRIDE learning...")

        # Nothing to learn
        if len(transitions) == 0:
            return LogicProgram(variables, values, [])

        rules = []
        nb_variables = len(transitions[0][1])

        # Extract observed values
        values = []
        for var in range(0, nb_variables):
            v = []

            for t1, t2 in transitions:
                if t2[var] not in v:
                    v.append(t2[var])

            values.append(v)

        #print("Set of values: "+str(values))

        # Learn rules for each observed variable/value
        for var in range(0, nb_variables):
            for val in values[var]:
                positives, negatives = PRIDE.interprete(transitions, var, val)
                rules += PRIDE.fit_var_val(var, val, positives, negatives)

        # Instanciate output logic program
        variables = [var for var in range(nb_variables)]
        output = LogicProgram(variables, values, rules)

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
            semantic: string in ["deterministic", "unknown"]
        """
        positives = [t1 for t1,t2 in transitions if t2[variable] == value]
        negatives = [t1 for t1,t2 in transitions if t1 not in positives]

        return positives, negatives


    @staticmethod
    def fit_var_val(variable, value, positives, negatives):
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
        #eprint("Start learning of var="+str(variable)+", val="+str(value))

        remaining = positives.copy()
        output = []

        # exausting covering loop
        while len(remaining) > 0:
            #eprint("Remaining positives: "+str(remaining))
            #eprint("Negatives: "+str(negatives))
            target = remaining[0]
            #eprint("new target: "+str(target))

            R = Rule(variable, value)
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

            conditions = R.get_body().copy()

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
