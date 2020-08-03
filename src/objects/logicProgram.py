#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2019/03/24
# @updated: 2019/05/03
#
# @desc: class LogicProgram python source code file
#-------------------------------------------------------------------------------

import itertools

from utils import eprint
from rule import Rule
import random

class LogicProgram:
    """
    Define a logic program, a set of rules over variables/values
    encoding the dynamics of a discrete dynamic system.
    """

    """ Variables of the program: list of string """
    #__variables: []

    """ Value domain of each variable condition: list of (list of string) """
    #__values: []

    """ Value domain of each variable conclusion: list of (list of string)"""
    #__conclusion_values: []

    """ Variables and their values that appear in body of rule: list of (string, list of string)"""
    __features: []

    """ Variables that appear in body of rule: list of (string, list of string) """
    __targets: []

    """ Rules of the program: list of Rule """
    __rules: []

    """ Constraints of the program: list of Rule """
    __constraints: []

#--------------
# Constructors
#--------------

    def __init__(self, features, targets, rules, constraints=[]):
        """
        Create a LogicProgram instance from given features/targets variables, rules and optional constraints

        Args:
            features: list of (String, list of String)
                Labels of the features variables and their values (appear only in body of rules and constraints)
            targets: list of (String, list of String)
                Labels of the targets variables and their values (appear in head of rules and body of constraint)
            rules: list of Rule
                Rules that define the program dynamics
        """
        self.__features = features.copy()
        self.__targets = targets.copy()
        self.__rules = rules.copy()
        self.__constraints = constraints.copy()

    @staticmethod
    def load_from_file(file_path):
        """
        Factory methods that loads a logic program from a formated text file
            - Extract the list of variables of the system and their value domain
            - Extract a list of discrete logic rules describing the system dynamics

        Args: String
            File path to text file encoding a logic program

        Returns: LogicProgram
            The logic program encoded in the input file
        """

        # 1) Extract variables
        #-----------------------

        eprint (">> Extracting variables...")

        features = []
        targets = []
        f = open(file_path,"r")

        for line in f:
            tokens = line.split()

            if len(tokens) == 0 or (tokens[0] != "FEATURE" and tokens[0] != "TARGET"):
                break

            if tokens[0] == "FEATURE":
                variable = str(tokens[1])
                values = []

                for i in range(2,len(tokens)):
                    values.append(str(tokens[i]))

                eprint(">>> Extracted feature: ",variable," domain: ",values)

                features.append((variable, values))

            elif tokens[0] == "TARGET":

                variable = tokens[1]
                values = []

                for i in range(2,len(tokens)):
                    values.append(str(tokens[i]))

                eprint(">>> Extracted target: ",variable," domain: ",values)

                targets.append((variable, values))


        # 2) Extract rules
        #------------------------

        eprint (">> Extracting rules...")

        rules = []

        for line in f:
            tokens = line.split()

            if len(tokens) == 0:
                continue

            head = str(tokens[0])

            # Extract variable
            beg = 0
            end = head.index('(')

            head_variable = head[beg:end]
            head_var_id = [name for (name,values) in targets].index(head_variable)

            # Extract value
            beg = end+1
            end = head.index(',')

            head_value = head[beg:end]
            head_val_id = targets[head_var_id][1].index(head_value)

            #eprint("Head extracted: ",head_var_id,"=",head_val_id)


            # Extract body
            body = []

            for i in range(2,len(tokens)):
                condition = str(tokens[i])

                # Extract variable
                beg = 0
                end = condition.index('(')

                variable = condition[beg:end]
                var_id = [name for (name,values) in features].index(variable)

                # Extract value
                beg = end+1
                end = condition.index(',')

                value = condition[beg:end]
                val_id = features[var_id][1].index(value)

                # TODO: delay

                #eprint("Condition extracted: ",variable,"=",value)

                body.append((var_id,val_id))

            r = Rule(head_var_id,head_val_id,len(features),body)
            rules.append(r)

            #eprint("Extracted rule:",r.to_string())

        # 2) Extract constraints TODO
        #------------------------

        f.close()

        return LogicProgram(features, targets, rules)

    @staticmethod
    def random(features, targets, rule_min_size, rule_max_size): # variables, values, rule_min_size, rule_max_size, delay=1):
        """
        Generate a complete deterministic logic program with a random dynamics.
        For each variable of the system, each possible state of the system is matched by at least one rule.

        Args:
            features: list of (String, list of String)
                Labels and values of the program feature variables
            targets: list of (String, list of String)
                Labels and values of the program target variables
            rule_min_size: int
                minimal number of conditions in each rule
            rule_max_size: int
                maximal number of conditions in each rule (can be exceded to enforce determinism)
        """

        rules = []
        p = LogicProgram(features, targets, [])
        states = p.states()

        for s in states:
            for var in range(len(targets)):

                matching = False
                for r in rules: # check if matched
                    if r.get_head_variable() == var and r.matches(s):
                        matching = True
                        break

                if not matching: # need new rule
                    val = random.randint(0, len(targets[var][1])-1)
                    body_size = random.randint(rule_min_size, rule_max_size)

                    new_rule = Rule(var, val, len(features), [])

                    # Prevent cross-match
                    for r in rules:
                        if r.get_head_variable() == var and r.cross_matches(new_rule):
                            # Search an available variable
                            # Always exists since no rule matches s yet
                            while True:
                                (cond_var, cond_val) = random.choice(r.get_body())
                                if not new_rule.has_condition(cond_var) and cond_val != s[cond_var]:
                                    new_rule.add_condition(cond_var, s[cond_var])
                                    break

                    # Complete the rule body if needed
                    while (new_rule.size() < body_size): # create body
                        cond_var = random.randint(0, len(s)-1)
                        cond_val = s[cond_var]

                        if new_rule.has_condition(cond_var):
                            continue

                        new_rule.add_condition(cond_var, cond_val)

                    rules.append(new_rule)

        return LogicProgram(features, targets, rules)

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

    def to_string(self):
        """
        Convert the object to a readable string

        Returns:
            String
                a readable representation of the object
        """
        output = "{"
        output += "\nFeatures: " + str(self.__features)
        output += "\nTargets: " + str(self.__targets)
        output += "\nRules:\n"
        for r in self.__rules:
            output += r.to_string() + "\n"
        for r in self.__constraints:
            output += r.to_string() + "\n"
        output += "}"

        return output

    def logic_form(self):
        """
        Convert the logic program to a logic programming string format

        Returns:
            String
                a logic programming representation of the logic program
        """
        output = ""

        # Variables declaration
        for var in range(len(self.__features)):
            output += "FEATURE "+str(self.__features[var][0])
            for val in self.__features[var][1]:
                output += " " + str(val)
            output += "\n"

        for var in range(len(self.__targets)):
            output += "TARGET "+str(self.__targets[var][0])
            for val in self.__targets[var][1]:
                output += " " + str(val)
            output += "\n"

        output += "\n"

        for r in self.__rules:
            output += r.logic_form(self.__features, self.__targets) + "\n"

        #eprint("DBG: ",self.__constraints)

        for r in self.__constraints:
            output += r.logic_form(self.__features+self.__targets, []) + "\n"

        return output

    def states(self):
        """
        Compute all possible state of the logic program:
        all combination of variables values

        Returns: list of (list of int)
            All possible state of the logic program
        """
        values_ids = [[j for j in range(0,len(self.__features[i][1]))] for i in range(0,len(self.__features))]
        output = [i for i in list(itertools.product(*values_ids))]
        return output

    def compare(self, other):
        """
        Compare the rules of a logic program with another.

        Args:
            other: LogicProgram
                Another logic program to be compared with

        Returns:
            common: rules in common (p1 \cup p2)
            missing: rules missing in the other (p1 \setminus p2)
            over: rules only present in the other (p2 \setminus p1)
        """
        common = [] # original rules
        missing = [] # missing original rules
        over = [] # non original rules

        P1 = self.get_rules().copy()
        P2 = other.get_rules().copy()

        #eprint("P1: "+str([r.to_string() for r in P1]))
        #eprint("P2: "+str([r.to_string() for r in P2]))

        for r in P1:
            #eprint("Checking: "+r.to_string()+" in P2")
            if r in P2:
                common.append(r)
            else:
                missing.append(r)

        for r in P2:
            #eprint("Checking: "+r.to_string()+" in P1")
            if r not in P1:
                over.append(r)

        #eprint("Logic Program comparaison:")
        #eprint("Common: "+str(len(common))+"/"+str(len(P1))+"("+str(100 * len(common) / len(P1))+"%)")
        #eprint("Missing: "+str(len(missing))+"/"+str(len(P1))+"("+str(100 * len(missing) / len(P1))+"%)")
        #eprint("Over: "+str(len(over))+"/"+str(len(P2))+"("+str(100 * len(over) / len(P2))+"%)")

        return common, missing, over

#--------
# Static
#--------

    @staticmethod
    def precision(expected, predicted):
        """
        Evaluate prediction precision on deterministic sets of transitions
        Args:
            expected: list of tuple (list of int, list of int)
                originals transitions of a system
            predicted: list of tuple (list of int, list of int)
                predicted transitions of a system

        Returns:
            float in [0,1]
                the error ratio between expected and predicted
        """
        eprint("DEPRECATED")

        if len(expected) == 0:
            return 1.0

        # Predict each variable for each state
        total = len(expected) * len(expected[0][0])
        error = 0

        #eprint("comparing: ")
        #eprint(test)
        #eprint(pred)

        for i in range(len(expected)):
            s1, s2 = expected[i]

            for j in range(len(predicted)):
                s1_, s2_ = predicted[j]

                if len(s1) != len(s1_) or len(s2) != len(s2_):
                    raise ValueError("Invalid prediction set")

                if s1 == s1_:
                    #eprint("Compare: "+str(s2)+" VS "+str(s2_))

                    for var in range(len(s2)):
                        if s2_[var] != s2[var]:
                            error += 1
                    break

            #eprint("new error: "+str(error))

        precision = 1.0 - (error / total)

        return precision

#--------------
# Accessors
#--------------

    def get_features(self):
        """
        Accessor method to __features names

        Returns:
            list of string
                feature variables of the program
        """
        return self.__features

    def get_targets(self):
        """
        Accessor method to __targets names

        Returns:
            list of string
                feature variables of the program
        """
        return self.__targets

    def get_rules(self):
        """
        Accessor method to __rules

        Returns:
            list of Rule
                rules of the program
        """
        return self.__rules

    def get_rules_of(self, var, val):
        """
        Specific head rule accessor method

        Args:
            var: int
                variable id
            val: int
                value id

        Returns:
            list of Rule
                rules of the program wich head is var=val
        """
        output = []
        for r in self.__rules:
            if r.get_head_variable() == var and r.get_head_value() == val:
                output.append(r.copy())
        return output

    def get_constraints(self):
        """
        Accessor method to __constraints

        Returns:
            list of Rule
                constraints of the program
        """
        return self.__constraints
