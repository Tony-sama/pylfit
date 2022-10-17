#-------------------------------------------------------------------------------
# @author: Tony Ribeiro
# @created: 2022/07/19
# @updated: 2022/07/19
#
# @desc: example of the use of heuristic with PRIDE
#-------------------------------------------------------------------------------

# Settings
#-----------------------
NB_PRODUCTS = 3
NB_CLIENTS = 1000
MIN_LOG_SIZE = 0
MAX_LOG_SIZE = 100

NB_EXPLANATIONS = 3
MIN_EXPLANATION_SIZE = 1
MAX_EXPLANATION_SIZE = 4

HEURISTICS = ["try_all_atoms", "max_coverage_dynamic", "max_coverage_static", "max_diversity"]

NB_RUNS = 10

RANDOM_SEED = 0
#-----------------------

# Equivalences
NB_EVENTS = NB_PRODUCTS
NB_SEQUENCES = NB_CLIENTS
MIN_SIZE_SEQUENCE = MIN_LOG_SIZE
MAX_SIZE_SEQUENCE = MAX_LOG_SIZE

import random
import time
from itertools import chain, combinations

import pylfit
from pylfit.objects import Rule

random.seed(RANDOM_SEED)

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

# # 1) Data Generation
# Atomic data are unique string values representique different events
def generate_events(nb_events, prefix="e"):
    return ["e"+str(i) for i in range(0,NB_EVENTS)]

# Observation data are sequences of events of different size
def generate_sequence(events, min_size, max_size):
    return random.choices(population=events,k=random.randint(min_size,max_size))

# A dataset is a set of observations data
def generate_observations(events, nb_sequences, min_size, max_size):
    return [generate_sequence(events, min_size, max_size) for i in range(0,nb_sequences)]

# Some of the observations satisfy a hidden property.
# In this example:
# - events are generalised to just the usage of a product (ei means product i is used).
# - Product 2 is a new version of product 0 but quality is bad in comparison
# - Users who used product 2 after their product 0 stop working stop buying brand product because quality decay


# Create a rule from an observation
def generate_rule(head_var, head_val, features, targets, data):
    state = [(idx,val) for idx,val in enumerate(random.choice(data))]
    rule_body = random.sample(state, random.randint(MIN_EXPLANATION_SIZE,MAX_EXPLANATION_SIZE))
    rule = Rule(head_var, head_val, len(features), rule_body)

    return rule

# # 2) Data encoding
# In order for LFIT to process the data they must be encoded properly:
# - Data must be pair of feature/target states, properties of the event sequence are the feature state and the observation class is the target state
# - temporal property must be computed to compose feature state

# ## 2.2) Apply property function on data and convert to LFIT dataset format
def encode_as_properties(events, data, property_functions):
    feature_states = []

    for idx, sequence in enumerate(data):
        encoded_sequence = []
        # Generate properties features
        for p in property_functions:
            encoded_sequence += p(events, sequence)[1]
        feature_states.append(tuple(encoded_sequence))
        print("\r>> "+str(idx+1)+"/"+str(len(data))+" sequences encoded",end="")
    print()
    return feature_states

def generate_hidden_rules(data, feature_states, targets):
    # Encode data for pylfit rule matching
    pylfit_encoded_data = [(pylfit.algorithms.Algorithm.encode_state([str(val) for val in s], features)) for s in data]

    hidden_rules = []
    tries = 100
    while(len(hidden_rules) < NB_EXPLANATIONS and tries > 0):
        tries -= 1

        rule = generate_rule(0, targets[0][1].index("pos"), features, targets, pylfit_encoded_data)

        # No subsumption
        subsumed = False
        for other in hidden_rules:
            if(rule.subsumes(other) or other.subsumes(rule)):
                subsumed = True
                break

        if(not subsumed):
            hidden_rules.append(rule)
            print(">>>", len([s for s in pylfit_encoded_data if rule.matches(s)]), rule.logic_form(features,targets))

    return hidden_rules

# Classify pos/neg states according to hidden rules
def classify_data(data, features):
    classified_data = []
    for s in data:
        label = "neg"
        state = pylfit.algorithms.Algorithm.encode_state(s,features)
        for r in hidden_rules:
            if(r.matches(state)):
                label = "pos"
                break
        classified_data.append((s,[label]))

    return classified_data

def optimize_hidden_rules(data, hidden_rules):
    negatives = [pylfit.algorithms.Algorithm.encode_state(s,features) for s,l in data if l == ["neg"]]
    optimal_hidden_rules = []
    for rule in hidden_rules:
        conditions = rule.body.copy()

        for (var,val) in conditions:
            rule.remove_condition(var) # Try remove condition

            conflict = False
            for neg in negatives:
                if rule.matches(neg): # Cover a negative example
                    conflict = True
                    rule.add_condition(var,val) # Cancel removal
                    break

        # No subsumption
        subsumed = False
        for other in optimal_hidden_rules:
            if(rule.subsumes(other) or other.subsumes(rule)):
                subsumed = True
                break

        if(not subsumed):
            optimal_hidden_rules.append(rule)
            print(">>>", rule.logic_form(features,targets))
            #print(">>>", len([s for s in pylfit_encoded_data if rule.matches(s)]), rule.logic_form(features,targets))
    return optimal_hidden_rules

# 2.1) Sequences property functions
from sequence_properties import\
existence,\
absence_2,\
choice,\
exclusive_choice,\
resp_existence,\
coexistence,\
response,\
precedence,\
succession,\
alt_response,\
alt_precedence,\
alt_succession,\
chain_response,\
chain_precedence,\
chain_succession,\
not_coexistence,\
not_succession,\
not_chain_succession,\
not_precedence

PROPERTY_FUNCTIONS = [
existence,\
absence_2,\
#choice,\
#exclusive_choice,\
resp_existence,\
coexistence,\
response,\
precedence,\
succession,\
#alt_response,\
#alt_precedence,\
#alt_succession,\
chain_response,\
chain_precedence,\
chain_succession,\
#not_coexistence,\
#not_succession,\
#not_chain_succession,\
not_precedence]

# 1: Main
#------------
if __name__ == '__main__':

    global_start = time.time()

    methods = [("pride",None)] + [("pride",subset) for subset in list(powerset(HEURISTICS))[1:] if not ("max_coverage_dynamic" in subset and "max_coverage_static" in subset)]
    accuracy = {h:[] for h in methods}
    run_time = {h:[] for h in methods}

    for run_id in range(NB_RUNS):
        print("> Run "+str(run_id+1)+"/"+str(NB_RUNS))

        # 1) Generate random sequences
        #--------------------------------
        print(">> Generating sequences data...")
        events = generate_events(NB_EVENTS)
        raw_data = generate_observations(events, NB_SEQUENCES, MIN_SIZE_SEQUENCE, MAX_SIZE_SEQUENCE)

        # Encode those sequence as their temporal properties
        property_functions = PROPERTY_FUNCTIONS
        features = [(i,["True","False"]) for f in property_functions for i in f(events,[])[0]]
        targets = [("class",["pos","neg"])]

        print(">> Encoding sequences data temporal properties ("+str([f.__name__ for f in property_functions])+")...")
        data = set(encode_as_properties(events, raw_data, property_functions))

        # 2) Generate hidden properties
        #--------------------------------

        print(">> Generate hiddden rules...")
        hidden_rules = generate_hidden_rules(data, features, targets)

        # Classify pos/neg states according to hidden rules
        print(">> Classify data according to hidden rules...")
        data = classify_data(data, features)

        # Make hidden rules into optimal rules
        print("Make hidden rules optimal w.r.t. classified data...")
        optimal_hidden_rules = optimize_hidden_rules(data, hidden_rules)

        # 3) LFIT
        #--------------------------------

        # First we create a standard pylfit dataset from the raw data using the properties encoding.
        dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=data, feature_domains=features, target_domains=targets)

        # Ensure same encoding for hidden rules
        optimal_hidden_rules = [Rule.from_string(r.logic_form(features,targets), dataset.features, dataset.targets) for r in optimal_hidden_rules]

        #dataset.summary()
        print()
        print(">> Features size:", len(dataset.features))
        print(">> Data size:", len(dataset.data))
        print(">> Positives:", len([(s,l) for s,l in data if "pos" in l]))
        print(">> Negatives:", len([(s,l) for s,l in data if "neg" in l]))

        # We can now run PRIDE with selected heuristics on this dataset to obtain explanation rules.
        for method in methods:
            print(">> Learning with "+str(method)+"...")
            start = time.time()

            model = pylfit.models.WDMVLP(features=dataset.features, targets=dataset.targets)
            model.compile(algorithm=method[0])
            model.fit(dataset=dataset, heuristics=method[1], verbose=0)
            #model.summary()
            print(">>> "+str(len(model.rules))+" rules")

            end = time.time()
            run_time[method].append(end - start)
            print(">>> "+str(round(run_time[method][-1],2))+" s")

            # # 4) Post-processing
            label = "pos"
            selection = sorted([(w,r) for w,r in model.rules if model.targets[r.head_variable][1][r.head_value] == label and w > 0], reverse=True, key=lambda x: x[0])[:10]
            print(">>> Best rules for label",label,":")
            for w,r in selection:
                print(">>> ",w,r.logic_form(model.features,model.targets))
            print()

            # Check for perfect rule
            expected_rules = optimal_hidden_rules
            found_rules = []

            for w,r in selection:
                if(r in expected_rules):
                    found_rules.append((w,r))

            print(">>> Expected rules found: "+str(len(found_rules))+"/"+str(len(expected_rules)))
            for w,r in found_rules:
                print(">>> ",w,r.logic_form(model.features,model.targets))
            print()

            accuracy[method].append(len(found_rules)/len(expected_rules))


    print("> Mean accuracy and run time over "+str(NB_RUNS)+" runs:")
    for method in methods:
        mean_accuracy = sum(accuracy[method])/len(accuracy[method])
        mean_run_time = sum(run_time[method])/len(run_time[method])
        print(">> "+str(method)+": "+str(round(mean_accuracy*100,2))+"% / "+str(round(mean_run_time,2))+"s")

    global_end = time.time()
    total_time = global_end - global_start
    print("> Total run time: "+str(round(total_time,2))+" s")
