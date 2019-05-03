#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2019/03/24
# @updated: 2019/05/03
#
# @desc: class LogicProgram python source code file
#-------------------------------------------------------------------------------

from utils import eprint
from rule import Rule
import random

class LogicProgram:
    """
    Define a logic program, a set of rules over variables/values
    encoding the dynamics of a discrete dynamic system.
    """

    """ Variables of the program: list of string """
    __variables: []

    """ Value domain of each variable: list of (list of int) """
    __values: []

    """ Rules of the program: list of Rule """
    __rules: []

#--------------
# Constructors
#--------------

    def __init__(self, variables, values, rules):
        """
        Create a LogicProgram instance from given variables, variables values and rules

        Args:
            variables: list of String
                Labels of the program variables
            values: list of list of String
                Domain of values that each variable can take
            rules: list of Rule
                Rules that define the program dynamics
        """
        self.__variables = variables.copy()
        self.__values = values.copy()
        self.__rules = rules.copy()

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

        #eprint (">> Extracting variables...")

        variables = []
        variables_values = []
        f = open(file_path,"r")

        for line in f:
        	tokens = line.split()

        	if len(tokens) == 0 or tokens[0] != "VAR":
        		break

        	variable = tokens[1]
        	values = []

        	for i in range(2,len(tokens)):
        		values.append(tokens[i])

        	#eprint(">>> Extracted variable:",variable,"domain:",values)

        	variables.append(variable)
        	variables_values.append(values)


        # 2) Extract rules
        #------------------------

        rules = []

        for line in f:
        	tokens = line.split()

        	if len(tokens) == 0:
        		continue

        	head = tokens[0]

        	# Extract variable
        	beg = 0
        	end = head.index('(')

        	head_variable = head[beg:end]
        	head_var_id = variables.index(head_variable)

        	# Extract value
        	beg = end+1
        	end = head.index(',')

        	head_value = head[beg:end]
        	head_val_id = variables_values[head_var_id].index(head_value)

        	#eprint("Head extracted: ",head_var_id,"=",head_val_id)


        	# Extract body
        	body = []

        	for i in range(2,len(tokens)):
        		condition = tokens[i]

        		# Extract variable
        		beg = 0
        		end = condition.index('(')

        		variable = condition[beg:end]
        		var_id = variables.index(variable)

        		# Extract value
        		beg = end+1
        		end = condition.index(',')

        		value = condition[beg:end]
        		val_id = variables_values[var_id].index(value)

        		# TODO: delay

        		#eprint("Condition extracted: ",variable,"=",value)

        		body.append((var_id,val_id))

        	r = Rule(head_var_id,head_val_id,body)
        	rules.append(r)

        	#eprint("Extracted rule:",r.to_string())

        f.close()

        return LogicProgram(variables, variables_values, rules)

    @staticmethod
    def random(variables, values, rule_min_size, rule_max_size, delay=1):
        """
        Generate a deterministic complete logic program with a random dynamics.
        For each variable of the system, each possible state of the system is matched by at least one rule.

        Args:
            variables: list of String
                Labels of the program variables
            values: list of list of String
                Domain of values that each variable can take
            rule_min_size: int
                minimal number of conditions in each rule
            rule_max_size: int
                maximal number of conditions in each rule (can be exceeded for completeness)
            delay: int
                maximal delay of the conditions of each rule
        """
        extended_variables = variables.copy()
        extended_values = values.copy()

        # Delayed logic program: extend local herbrand base
        if delay > 1:
            for d in range(1,delay):
                extended_variables += [var+"_"+str(d) for var in variables]
                extended_values += values

        rules = []
        p = LogicProgram(extended_variables, extended_values, [])
        states = p.states() # aggregated reversed time serie of size delay

        for s in states:
            for var in range(len(variables)):

                matching = False
                for r in rules: # check if matched
                    if r.get_head_variable() == var and r.matches(s):
                        matching = True
                        break

                if not matching: # need new rule
                    val = random.randint(0, len(values[var])-1)
                    body_size = random.randint(rule_min_size, rule_max_size)

                    new_rule = Rule(var, val, [])

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

        return LogicProgram(variables, values, rules)

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
        output += "\nVariables: " + str(self.__variables)
        output += "\nValues: " + str(self.__values)
        output += "\nRules:\n"
        for r in self.__rules:
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
        for var in range(len(self.__variables)):
            output += "VAR "+str(self.__variables[var])
            for val in self.__values[var]:
                output += " " + str(val)
            output += "\n"

        output += "\n"

        for r in self.__rules:
            output += r.logic_form(self.__variables, self.__values) + "\n"

        return output


    def next(self, state):
        """
        Compute the next state according to the rules of the program,
        assuming a synchronous deterministic semantic.

        Args:
            state: list of int
                A state of the system

        Returns:
            list of int
                the next state according to the rules of the program.
        """
        output = [-1 for i in self.__variables]

        for r in self.__rules:
            if(r.matches(state)):
                output[r.get_head_variable()] = r.get_head_value()

        return output

    def generate_transitions(self, nb_states):
        """
        Generate randomly a given number of deterministic synchronous
        transitions from the given system.

        Args:
            nb_state: int
                number of state to generate.
        Returns:
            list of tuple (list of int, list of int)
                a set of transitions of the given system of size nb_state.
        """
        generated = 0
        transitions = []

        while generated < nb_states:

            # Random state
            s1= [random.randint(0,1) for v in self.__variables]

            # Next state according to rules
            s2 = self.next(s1)

            transitions.append((s1,s2))
            generated += 1

        return [(list(s1),list(s2)) for s1, s2 in transitions]

    def generate_all_transitions(self):
        """
        Generate all possible state of the program and their corresponding transition

        Returns: list of tuple (list of int, list of int)
            The set of all synchronous deterministic transitions of the logic program
        """
        output = []
        for s1 in self.states():
            output.append( [s1, self.next(s1)] )
        return output

    def transitions_to_csv(self, filepath, transitions):
        """
        Convert a set of transitions to a csv file

        Args:
            filepath: String
                File path to where the csv file will be saved.
            transitions: list of tuple (list of int, list of int)
                transitions of the logic program
        """
        output = ""

        for var in range(0,len(self.__variables)):
            output += "x"+str(var)+","
        for var in range(0,len(self.__variables)):
            output += "y"+str(var)+","

        output = output[:-1] + "\n"

        for s1, s2 in transitions:
            for val in s1:
                output += str(val)+","
            for val in s2:
                output += str(val)+","
            output = output[:-1] + "\n"

        f = open(filepath, "w")
        f.write(output)
        f.close()

    def next_state(self, time_serie):
        """
        Compute the next state according to the rule of the program

        Args:
            time_serie: list of list of int
                A sequence of state of the system

        Returns:
            list of int
                the next state according to the rules of the program.
        """
        output = time_serie[-1].copy()

        time_serie = time_serie.copy()
        time_serie.reverse()
        meta_state = [y for x in time_serie for y in x]

        return self.next(meta_state)

    def generate_all_time_series(self, length):
        """
        Generate all possible time series of given length
        produced by the logic program:
        all sequence of transitions starting from a state of the program.

        Returns:
            list of list of list of int
                all possible time series of given length
                according to the rules of the logic program.
        """

        output = []
        for s1 in self.states():
            serie = [s1]
            for i in range(length):
                s2 = self.next_state(serie)
                for i in range(len(s2)): # default to previous value
                    if s2[i] == -1:
                        s2[i] = serie[-1][i]
                serie.append(s2)
            output.append(serie)
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


    def states(self):
        """
        Compute all possible state of the logic program:
        all combination of variables values

        Returns: list of (list of int)
            All possible state of the logic program
        """
        state = [0 for var in self.__variables]
        output = []
        self.__states(0, state, output)
        return output

    def __states(self, variable, state, output):
        """
        Recursive sub-function of state()

        Args:
            variable: int
                A variable id
            state: list of int
                A system state
            output: list of (list of int)
                the set of all states generated so far
        """
        # All variable are assigned
        if variable >= len(self.__variables):
            output.append( state.copy() )
            return

        # Enumerate each possible value
        for val in range(len(self.__values[variable])):
            state[variable] = val
            self.__states(variable+1, state, output)

#--------
# Static
#--------

    @staticmethod
    def precision(expected, predicted):
        """
        Args:
            expected: list of tuple (list of int, list of int)
                originals transitions of a system
            predicted: list of tuple (list of int, list of int)
                predicted transitions of a system

        Returns:
            float in [0,1]
                the error ratio between expected and predicted
        """
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
            s1_, s2_ = predicted[i]

            if s1 != s1_ or len(s2) != len(s2_):
                raise ValueError("Invalid prediction set")

            #eprint("Compare: "+str(s2)+" VS "+str(s2_))

            for var in range(len(s2)):
                if s2_[var] != s2[var]:
                    error += 1

            #eprint("new error: "+str(error))

        precision = 1.0 - (error / total)

        return precision

#--------------
# Accessors
#--------------

    def get_variables(self):
        """
        Accessor method to __variables

        Returns:
            list of string
                variables of the program
        """
        return self.__variables

    def get_values(self):
        """
        Accessor method to __values

        Returns:
            list of (list of string)
                values of the program variables
        """
        return self.__values

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
