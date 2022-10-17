#-----------------------
# @author: Tony Ribeiro
# @created: 2021/03/04
# @updated: 2021/06/15
#
# @desc: metrics.py regression test script
#
#-----------------------

import unittest
import random
import sys
import numpy as np

import pylfit
from pylfit.postprocessing import hamming_distance, accuracy_score, explanation_score, accuracy_score_from_predictions, explanation_score_from_predictions
from pylfit.models import WDMVLP
from pylfit.datasets import DiscreteStateTransitionsDataset
from pylfit.utils import eprint

import pathlib
sys.path.insert(0, str(str(pathlib.Path(__file__).parent.parent.absolute())))

from tests_generator import random_rule, random_WDMVLP, random_DiscreteStateTransitionsDataset

random.seed(0)

class metrics_tests(unittest.TestCase):
    """
        Unit test of module metrics.py
    """
    _nb_random_tests = 10

    _nb_features = 3

    _nb_targets = 2

    _nb_values = 3

    _max_body_size = 10

    _nb_transitions = 100

    def test_hamming_distance(self):
        print(">> pylfit.postprocessing.hamming_distance(rule_1, rule_2)")

        for i in range(self._nb_random_tests):
            nb_features = random.randint(1,self._nb_features)
            nb_targets = random.randint(1,self._nb_targets)
            nb_values = random.randint(1,self._nb_values)

            r1 = random_rule(nb_features, nb_targets, nb_values, min(self._max_body_size,nb_features-1))

            # Same rule
            self.assertEqual(hamming_distance(r1,r1), 0)

            # Modified rule
            r2 = r1.copy()

            distance = random.randint(0,r1.size())

            for d in range(distance):
                r2.pop_condition()

            self.assertEqual(hamming_distance(r1,r2), distance)
            self.assertEqual(hamming_distance(r2,r1), distance)

            # Empty rule
            r1 = random_rule(nb_features, nb_targets, nb_values, 0)
            r2 = random_rule(nb_features, nb_targets, nb_values, min(self._max_body_size,nb_features-1))
            self.assertEqual(hamming_distance(r1,r2), r2.size())
            self.assertEqual(hamming_distance(r2,r1), r2.size())

            # random rules
            r1 = random_rule(nb_features, nb_targets, nb_values, min(self._max_body_size,nb_features-1))
            r2 = random_rule(nb_features, nb_targets, nb_values, min(self._max_body_size,nb_features-1))

            distance = 0
            for var in range(nb_features):
                if r1.get_condition(var) != r2.get_condition(var):
                    distance += 1
            self.assertEqual(hamming_distance(r1,r2), distance)

    def test_accuracy_score(self):
        print(">> pylfit.postprocessing.accuracy_score(model, dataset)")

        # unit test
        test_data = [ \
        (["0","0","0"],["0","0","1"]), \
        (["0","0","0"],["1","0","0"]), \
        (["1","0","0"],["0","0","0"]), \
        (["0","1","0"],["1","0","1"]), \
        (["0","0","1"],["0","0","1"]), \
        (["1","1","0"],["1","0","0"]), \
        #(["1","0","1"],["0","1","0"]), \
        (["0","1","1"],["1","0","1"]), \
        (["1","1","1"],["1","1","0"])]

        test_dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=test_data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])

        train_data = [ \
        (["0","0","0"],["0","0","1"]), \
        (["0","0","0"],["1","0","0"]), \
        #(["1","0","0"],["0","0","0"]), \
        #(["0","1","0"],["1","0","1"]), \
        #(["0","0","1"],["0","0","1"]), \
        #(["1","1","0"],["1","0","0"]), \
        #(["1","0","1"],["0","1","0"]), \
        #(["0","1","1"],["1","0","1"]), \
        (["1","1","1"],["1","1","0"])]

        train_dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=train_data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])

        model = pylfit.models.WDMVLP(features=train_dataset.features, targets=train_dataset.targets)
        model.compile(algorithm="gula")
        model.fit(dataset=train_dataset)

        self.assertEqual(round(accuracy_score(model=model, dataset=test_dataset),2), 0.64)

        train_data = [ \
        (["0","0","0"],["0","0","1"]), \
        (["0","0","0"],["1","0","0"]), \
        (["1","0","0"],["0","0","0"]), \
        (["0","1","0"],["1","0","1"]), \
        #(["0","0","1"],["0","0","1"]), \
        #(["1","1","0"],["1","0","0"]), \
        #(["1","0","1"],["0","1","0"]), \
        #(["0","1","1"],["1","0","1"]), \
        (["1","1","1"],["1","1","0"])]

        train_dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=train_data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])

        model = pylfit.models.WDMVLP(features=train_dataset.features, targets=train_dataset.targets)
        model.compile(algorithm="gula")
        model.fit(dataset=train_dataset)

        self.assertEqual(round(accuracy_score(model=model, dataset=test_dataset),2), 0.84)

        train_data = [ \
        (["0","0","0"],["0","0","1"]), \
        (["0","0","0"],["1","0","0"]), \
        (["1","0","0"],["0","0","0"]), \
        (["0","1","0"],["1","0","1"]), \
        (["0","0","1"],["0","0","1"]), \
        #(["1","1","0"],["1","0","0"]), \
        (["1","0","1"],["0","1","0"]), \
        #(["0","1","1"],["1","0","1"]), \
        (["1","1","1"],["1","1","0"])]

        train_dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=train_data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])

        model = pylfit.models.WDMVLP(features=train_dataset.features, targets=train_dataset.targets)
        model.compile(algorithm="gula")
        model.fit(dataset=train_dataset)

        self.assertEqual(round(accuracy_score(model=model, dataset=test_dataset),2), 0.91)

        # random tests
        for i in range(self._nb_random_tests):
            nb_features = random.randint(1,self._nb_features)
            nb_targets = random.randint(1,self._nb_targets)
            max_feature_values = random.randint(1,self._nb_values)
            max_target_values = random.randint(1,self._nb_values)
            nb_transitions = random.randint(2,self._nb_transitions)

            dataset = random_DiscreteStateTransitionsDataset(nb_transitions, nb_features, nb_targets, max_feature_values, max_target_values)

            # Empty program
            model = WDMVLP(dataset.features, dataset.targets)
            model.compile(algorithm="gula")
            self.assertEqual(accuracy_score(model=model, dataset=dataset), 0.5)

            # Empty rule program
            model = WDMVLP(dataset.features, dataset.targets)
            model.compile(algorithm="gula")
            model.fit(DiscreteStateTransitionsDataset([], dataset.features, dataset.targets))
            self.assertEqual(accuracy_score(model=model, dataset=dataset), 0.5)

            # Train != test
            train_data = dataset.data[0:int(0.5*len(dataset.data))]
            test_data = dataset.data[int(0.5*len(dataset.data)):]
            train_dataset = DiscreteStateTransitionsDataset(train_data, dataset.features, dataset.targets)
            test_dataset = DiscreteStateTransitionsDataset(test_data, dataset.features, dataset.targets)

            model = WDMVLP(train_dataset.features, train_dataset.targets)
            model.compile(algorithm="gula")
            model.fit(dataset=train_dataset)

            # Train = Test -> 100 accuracy
            self.assertEqual(accuracy_score(model=model, dataset=train_dataset), 1.0)

            grouped_transitions = {tuple(s1) : set(tuple(s2_) for s1_,s2_ in test_dataset.data if tuple(s1) == tuple(s1_)) for s1,s2 in test_dataset.data}

            # expected output
            expected = {}
            count = 0
            for s1, successors in grouped_transitions.items():
                count += 1
                occurs = {}
                for var in range(len(test_dataset.targets)):
                    for val in range(len(test_dataset.targets[var][1])):
                        occurs[(var,val)] = 0.0
                        for s2 in successors:
                            if s2[var] == test_dataset.targets[var][1][val]:
                                occurs[(var,val)] = 1.0
                                break
                expected[s1] = occurs

            # predictions
            predicted = {}
            count = 0
            for s1, successors in grouped_transitions.items():
                count += 1
                occurs = {}
                prediction = model.predict([list(s1)])[s1]
                for var_id, (var,vals) in enumerate(test_dataset.targets):
                    for val_id, val in enumerate(test_dataset.targets[var_id][1]):
                        occurs[(var_id,val_id)] = prediction[var][val][0]

                predicted[s1] = occurs

            # compute average accuracy
            global_error = 0
            for s1, actual in expected.items():
                state_error = 0
                for var in range(len(test_dataset.targets)):
                    for val in range(len(test_dataset.targets[var][1])):
                        forecast = predicted[s1]
                        state_error += abs(actual[(var,val)] - forecast[(var,val)])

                global_error += state_error / len(actual.items())

            global_error = global_error / len(expected.items())

            accuracy = 1.0 - global_error

            self.assertEqual(accuracy_score(model=model,dataset=test_dataset), accuracy)

            # Exception
            self.assertRaises(ValueError, accuracy_score, model, DiscreteStateTransitionsDataset([],dataset.features, dataset.targets))

    def test_accuracy_score_from_predictions(self):
        print(">> pylfit.postprocessing.accuracy_score_from_predictions(predictions, dataset)")

        # unit test
        test_data = [ \
        (["0","0","0"],["0","0","1"]), \
        (["0","0","0"],["1","0","0"]), \
        (["1","0","0"],["0","0","0"]), \
        (["0","1","0"],["1","0","1"]), \
        (["0","0","1"],["0","0","1"]), \
        (["1","1","0"],["1","0","0"]), \
        #(["1","0","1"],["0","1","0"]), \
        (["0","1","1"],["1","0","1"]), \
        (["1","1","1"],["1","1","0"])]

        test_dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=test_data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])

        train_data = [ \
        (["0","0","0"],["0","0","1"]), \
        (["0","0","0"],["1","0","0"]), \
        #(["1","0","0"],["0","0","0"]), \
        #(["0","1","0"],["1","0","1"]), \
        #(["0","0","1"],["0","0","1"]), \
        #(["1","1","0"],["1","0","0"]), \
        #(["1","0","1"],["0","1","0"]), \
        #(["0","1","1"],["1","0","1"]), \
        (["1","1","1"],["1","1","0"])]

        train_dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=train_data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])

        model = pylfit.models.WDMVLP(features=train_dataset.features, targets=train_dataset.targets)
        model.compile(algorithm="gula")
        model.fit(dataset=train_dataset)

        init_states = [list(s) for s in set(tuple(s1) for s1,s2 in test_dataset.data)]
        predictions = model.predict(init_states)
        predictions = {s1: {variable: {value: item[variable][value][0] for value in values} for (variable, values) in model.targets} for s1, item in predictions.items()}

        self.assertEqual(round(accuracy_score_from_predictions(predictions=predictions, dataset=test_dataset),2), 0.64)

        train_data = [ \
        (["0","0","0"],["0","0","1"]), \
        (["0","0","0"],["1","0","0"]), \
        (["1","0","0"],["0","0","0"]), \
        (["0","1","0"],["1","0","1"]), \
        #(["0","0","1"],["0","0","1"]), \
        #(["1","1","0"],["1","0","0"]), \
        #(["1","0","1"],["0","1","0"]), \
        #(["0","1","1"],["1","0","1"]), \
        (["1","1","1"],["1","1","0"])]

        train_dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=train_data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])

        model = pylfit.models.WDMVLP(features=train_dataset.features, targets=train_dataset.targets)
        model.compile(algorithm="gula")
        model.fit(dataset=train_dataset)

        init_states = [list(s) for s in set(tuple(s1) for s1,s2 in test_dataset.data)]
        predictions = model.predict(init_states)
        predictions = {s1: {variable: {value: item[variable][value][0] for value in values} for (variable, values) in model.targets} for s1, item in predictions.items()}

        self.assertEqual(round(accuracy_score_from_predictions(predictions=predictions, dataset=test_dataset),2), 0.84)

        train_data = [ \
        (["0","0","0"],["0","0","1"]), \
        (["0","0","0"],["1","0","0"]), \
        (["1","0","0"],["0","0","0"]), \
        (["0","1","0"],["1","0","1"]), \
        (["0","0","1"],["0","0","1"]), \
        #(["1","1","0"],["1","0","0"]), \
        (["1","0","1"],["0","1","0"]), \
        #(["0","1","1"],["1","0","1"]), \
        (["1","1","1"],["1","1","0"])]

        train_dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=train_data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])

        model = pylfit.models.WDMVLP(features=train_dataset.features, targets=train_dataset.targets)
        model.compile(algorithm="gula")
        model.fit(dataset=train_dataset)

        init_states = [list(s) for s in set(tuple(s1) for s1,s2 in test_dataset.data)]
        predictions = model.predict(init_states)
        predictions = {s1: {variable: {value: item[variable][value][0] for value in values} for (variable, values) in model.targets} for s1, item in predictions.items()}

        self.assertEqual(round(accuracy_score_from_predictions(predictions=predictions, dataset=test_dataset),2), 0.91)

        # random tests
        for i in range(self._nb_random_tests):
            nb_features = random.randint(1,self._nb_features)
            nb_targets = random.randint(1,self._nb_targets)
            max_feature_values = random.randint(1,self._nb_values)
            max_target_values = random.randint(1,self._nb_values)
            nb_transitions = random.randint(2,self._nb_transitions)

            dataset = random_DiscreteStateTransitionsDataset(nb_transitions, nb_features, nb_targets, max_feature_values, max_target_values)

            # Empty program
            model = WDMVLP(dataset.features, dataset.targets)
            model.compile(algorithm="gula")

            init_states = [list(s) for s in set(tuple(s1) for s1,s2 in dataset.data)]
            predictions = model.predict(init_states)
            predictions = {s1: {variable: {value: item[variable][value][0] for value in values} for (variable, values) in model.targets} for s1, item in predictions.items()}

            self.assertEqual(round(accuracy_score_from_predictions(predictions=predictions, dataset=dataset),2), 0.5)

            # Empty rule program
            model = WDMVLP(dataset.features, dataset.targets)
            model.compile(algorithm="gula")
            model.fit(DiscreteStateTransitionsDataset([], dataset.features, dataset.targets))
            self.assertEqual(accuracy_score(model=model, dataset=dataset), 0.5)

            # Train != test
            train_data = dataset.data[0:int(0.5*len(dataset.data))]
            test_data = dataset.data[int(0.5*len(dataset.data)):]
            train_dataset = DiscreteStateTransitionsDataset(train_data, dataset.features, dataset.targets)
            test_dataset = DiscreteStateTransitionsDataset(test_data, dataset.features, dataset.targets)

            model = WDMVLP(train_dataset.features, train_dataset.targets)
            model.compile(algorithm="gula")
            model.fit(dataset=train_dataset)

            # Train = Test -> 100 accuracy
            init_states = [list(s) for s in set(tuple(s1) for s1,s2 in train_dataset.data)]
            predictions = model.predict(init_states)
            predictions = {s1: {variable: {value: item[variable][value][0] for value in values} for (variable, values) in model.targets} for s1, item in predictions.items()}

            self.assertEqual(round(accuracy_score_from_predictions(predictions=predictions, dataset=train_dataset),2), 1.0)

            grouped_transitions = {tuple(s1) : set(tuple(s2_) for s1_,s2_ in test_dataset.data if tuple(s1) == tuple(s1_)) for s1,s2 in test_dataset.data}

            # expected output
            expected = {}
            count = 0
            for s1, successors in grouped_transitions.items():
                count += 1
                occurs = {}
                for var in range(len(test_dataset.targets)):
                    for val in range(len(test_dataset.targets[var][1])):
                        occurs[(var,val)] = 0.0
                        for s2 in successors:
                            if s2[var] == test_dataset.targets[var][1][val]:
                                occurs[(var,val)] = 1.0
                                break
                expected[s1] = occurs

            # predictions
            predicted = {}
            count = 0
            for s1, successors in grouped_transitions.items():
                count += 1
                occurs = {}
                prediction = model.predict([list(s1)])[s1]
                for var_id, (var,vals) in enumerate(test_dataset.targets):
                    for val_id, val in enumerate(test_dataset.targets[var_id][1]):
                        occurs[(var_id,val_id)] = prediction[var][val][0]

                predicted[s1] = occurs

            # compute average accuracy
            global_error = 0
            for s1, actual in expected.items():
                state_error = 0
                for var in range(len(test_dataset.targets)):
                    for val in range(len(test_dataset.targets[var][1])):
                        forecast = predicted[s1]
                        state_error += abs(actual[(var,val)] - forecast[(var,val)])

                global_error += state_error / len(actual.items())

            global_error = global_error / len(expected.items())

            accuracy = 1.0 - global_error

            init_states = [list(s) for s in set(tuple(s1) for s1,s2 in test_dataset.data)]
            predictions = model.predict(init_states)
            predictions = {s1: {variable: {value: item[variable][value][0] for value in values} for (variable, values) in model.targets} for s1, item in predictions.items()}

            self.assertEqual(accuracy_score_from_predictions(predictions=predictions, dataset=test_dataset), accuracy)

            # Exception

             # empty dataset
            self.assertRaises(ValueError, accuracy_score_from_predictions, predictions, DiscreteStateTransitionsDataset([],dataset.features, dataset.targets))

            # Missing init state in dataset
            remove_s1, s2 = random.choice(test_dataset.data)
            self.assertRaises(ValueError, accuracy_score_from_predictions, predictions, DiscreteStateTransitionsDataset([(s1,s2) for (s1,s2) in test_dataset.data if list(s1) != list(remove_s1)], dataset.features, dataset.targets))

            # Missing init state in predictions
            remove_s1 = random.choice(list(predictions.keys()))
            predictions_ = predictions.copy()
            predictions_.pop(remove_s1, None)
            self.assertRaises(ValueError, accuracy_score_from_predictions, predictions_, test_dataset)

            # Bad target domain
            test_dataset_ = test_dataset.copy()
            test_dataset_.targets = [("a",["0"])]
            self.assertRaises(ValueError, accuracy_score_from_predictions, predictions, test_dataset_)


    def test_explanation_score(self):
        print(">> pylfit.postprocessing.explanation_score(model, expected_model, feature_states, verbose=False)")

        # unit test
        data = [ \
        (["0","0","0"],["0","0","1"]), \
        (["0","0","0"],["1","0","0"]), \
        (["1","0","0"],["0","0","0"]), \
        (["0","1","0"],["1","0","1"]), \
        (["0","0","1"],["0","0","1"]), \
        (["1","1","0"],["1","0","0"]), \
        #(["1","0","1"],["0","1","0"]), \
        (["0","1","1"],["1","0","1"]), \
        (["1","1","1"],["1","1","0"])]

        dataset_perfect = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])
        optimal_model = pylfit.models.WDMVLP(features=dataset_perfect.features, targets=dataset_perfect.targets)
        optimal_model.compile(algorithm="gula") # model.compile(algorithm="pride")
        optimal_model.fit(dataset=dataset_perfect)

        train_data = [ \
        (["0","0","0"],["0","0","1"]), \
        (["0","0","0"],["1","0","0"]), \
        #(["1","0","0"],["0","0","0"]), \
        #(["0","1","0"],["1","0","1"]), \
        #(["0","0","1"],["0","0","1"]), \
        #(["1","1","0"],["1","0","0"]), \
        #(["1","0","1"],["0","1","0"]), \
        #(["0","1","1"],["1","0","1"]), \
        (["1","1","1"],["1","1","0"])]

        train_dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=train_data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])
        model = pylfit.models.WDMVLP(features=train_dataset.features, targets=train_dataset.targets)
        model.compile(algorithm="gula")
        model.fit(dataset=train_dataset)

        self.assertEqual(round(explanation_score(model=model, expected_model=optimal_model, dataset=dataset_perfect),2), 0.28)

        train_data = [ \
        (["0","0","0"],["0","0","1"]), \
        (["0","0","0"],["1","0","0"]), \
        (["1","0","0"],["0","0","0"]), \
        (["0","1","0"],["1","0","1"]), \
        #(["0","0","1"],["0","0","1"]), \
        #(["1","1","0"],["1","0","0"]), \
        #(["1","0","1"],["0","1","0"]), \
        #(["0","1","1"],["1","0","1"]), \
        (["1","1","1"],["1","1","0"])]

        train_dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=train_data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])
        model = pylfit.models.WDMVLP(features=train_dataset.features, targets=train_dataset.targets)
        model.compile(algorithm="gula")
        model.fit(dataset=train_dataset)

        self.assertEqual(round(explanation_score(model=model, expected_model=optimal_model, dataset=dataset_perfect),2), 0.87)

        train_data = [ \
        (["0","0","0"],["0","0","1"]), \
        (["0","0","0"],["1","0","0"]), \
        (["1","0","0"],["0","0","0"]), \
        (["0","1","0"],["1","0","1"]), \
        (["0","0","1"],["0","0","1"]), \
        (["1","1","0"],["1","0","0"]), \
        (["1","0","1"],["0","1","0"]), \
        #(["0","1","1"],["1","0","1"]), \
        (["1","1","1"],["1","1","0"])]

        train_dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=train_data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])
        model = pylfit.models.WDMVLP(features=train_dataset.features, targets=train_dataset.targets)
        model.compile(algorithm="gula")
        model.fit(dataset=train_dataset)

        self.assertEqual(round(explanation_score(model=model, expected_model=optimal_model, dataset=dataset_perfect),2), 0.98)

        # random tests
        for i in range(self._nb_random_tests):
            nb_features = random.randint(1,self._nb_features)
            nb_targets = random.randint(1,self._nb_targets)
            max_feature_values = random.randint(1,self._nb_values)
            max_target_values = random.randint(1,self._nb_values)
            nb_transitions = random.randint(2,self._nb_transitions)

            dataset = random_DiscreteStateTransitionsDataset(nb_transitions, nb_features, nb_targets, max_feature_values, max_target_values)

            optimal_model = WDMVLP(dataset.features, dataset.targets)
            optimal_model.compile(algorithm="gula")
            optimal_model.fit(dataset=dataset)

            # Empty program
            model = WDMVLP(dataset.features, dataset.targets)
            model.compile(algorithm="gula")
            self.assertEqual(explanation_score(model=model, expected_model=optimal_model, dataset=dataset), 0.0)

            # Empty rule program
            model = WDMVLP(dataset.features, dataset.targets)
            model.compile(algorithm="gula")
            model.fit(DiscreteStateTransitionsDataset([], dataset.features, dataset.targets))
            self.assertEqual(explanation_score(model=model, expected_model=optimal_model, dataset=dataset), 0.0)

            # Train != test
            train_data = dataset.data[0:int(0.5*len(dataset.data))]
            test_data = dataset.data[int(0.5*len(dataset.data)):]
            train_dataset = DiscreteStateTransitionsDataset(train_data, dataset.features, dataset.targets)
            test_dataset = DiscreteStateTransitionsDataset(test_data, dataset.features, dataset.targets)

            model = WDMVLP(train_dataset.features, train_dataset.targets)
            model.compile(algorithm="gula")
            model.fit(dataset=train_dataset)

            # model = optimal -> 100 accuracy
            self.assertEqual(explanation_score(model=model, expected_model=model, dataset=train_dataset), 1.0)

            # Exception
            self.assertRaises(ValueError, explanation_score, model, optimal_model, DiscreteStateTransitionsDataset([], dataset.features, dataset.targets))

            # train != test
            grouped_transitions = {tuple(s1) : set(tuple(s2_) for s1_,s2_ in test_dataset.data if tuple(s1) == tuple(s1_)) for s1,s2 in test_dataset.data}

            # expected output: kinda one-hot encoding of values occurences
            expected = {}
            count = 0
            for s1, successors in grouped_transitions.items():
                count += 1
                occurs = {}
                for var in range(len(test_dataset.targets)):
                    for val in range(len(test_dataset.targets[var][1])):
                        occurs[(var,val)] = 0.0
                        for s2 in successors:
                            if s2[var] == test_dataset.targets[var][1][val]:
                                occurs[(var,val)] = 1.0
                                break
                expected[s1] = occurs

            sum_explanation_score = 0.0
            prediction = model.predict(feature_states=[list(s1) for s1 in expected], raw_rules=True)
            for feature_state, actual in expected.items():
                #eprint("Feature state: ", feature_state)
                #eprint(">> prediction: ",prediction[feature_state])

                sum_score = 0.0
                nb_targets = 0
                for var_id, (variable, values) in enumerate(model.targets):
                    #eprint(" "+variable+": ")
                    for val_id, (value, (proba, (w1, r1), (w2, r2))) in enumerate(prediction[feature_state][variable].items()):
                        #eprint(" "+value+" "+str(round(proba*100.0,2))+"%")

                        # No decision or bad prediction implies wrong explanation
                        if proba == 0.5 or (proba > 0.5 and actual[(var_id,val_id)] == 0.0) or (proba < 0.5 and actual[(var_id,val_id)] == 1.0):
                            score = 0.0
                            sum_score += score
                            nb_targets += 1
                            continue

                        encoded_feature_state = pylfit.algorithms.GULA.encode_state(feature_state, model.features)

                        # Predicted likely
                        if proba > 0.5:
                            expected_rules = [r for (w,r) in optimal_model.rules \
                            if r.head_variable == var_id and r.head_value == val_id and r.matches(encoded_feature_state)]
                            explanation_rule = r1

                        # Predicted unlikely
                        if proba < 0.5:
                            expected_rules = [r for (w,r) in optimal_model.unlikeliness_rules \
                            if r.head_variable == var_id and r.head_value == val_id and r.matches(encoded_feature_state)]
                            explanation_rule = r2

                        min_distance = len(model.features)
                        nearest_expected = None
                        for r in expected_rules:
                            distance = pylfit.postprocessing.hamming_distance(explanation_rule,r)
                            if distance <= min_distance:
                                min_distance = distance
                                nearest_expected = r

                        score = 1.0 - (min_distance / len(model.features))

                        #eprint(explanation_type + " explanation evaluation")
                        #eprint("Explanation rule: " + explanation)
                        #eprint("Explanation score: ", end='')
                        #eprint(str(round(score, 2)) + " (nearest expected " + explanation_type + " rule: " + nearest_expected.logic_form(model.features, model.targets) + " distance: " + str(min_distance) + ")")
                        sum_score += score
                        nb_targets += 1
                sum_explanation_score += sum_score / nb_targets

            expected_score = sum_explanation_score / len(expected)

            self.assertEqual(explanation_score(model=model, expected_model=optimal_model, dataset=test_dataset), expected_score)

            # Exception

            # Bad targets
            test_dataset_ = test_dataset.copy()
            test_dataset_.targets = [("a",["0"])]
            self.assertRaises(ValueError, explanation_score, model, optimal_model, test_dataset_)

    def test_explanation_score_from_predictions(self):
        print(">> pylfit.postprocessing.explanation_score_from_predictions(predictions, expected_model, dataset)")

        # unit test
        data = [ \
        (["0","0","0"],["0","0","1"]), \
        (["0","0","0"],["1","0","0"]), \
        (["1","0","0"],["0","0","0"]), \
        (["0","1","0"],["1","0","1"]), \
        (["0","0","1"],["0","0","1"]), \
        (["1","1","0"],["1","0","0"]), \
        #(["1","0","1"],["0","1","0"]), \
        (["0","1","1"],["1","0","1"]), \
        (["1","1","1"],["1","1","0"])]

        dataset_perfect = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])
        optimal_model = pylfit.models.WDMVLP(features=dataset_perfect.features, targets=dataset_perfect.targets)
        optimal_model.compile(algorithm="gula") # model.compile(algorithm="pride")
        optimal_model.fit(dataset=dataset_perfect)

        train_data = [ \
        (["0","0","0"],["0","0","1"]), \
        (["0","0","0"],["1","0","0"]), \
        #(["1","0","0"],["0","0","0"]), \
        #(["0","1","0"],["1","0","1"]), \
        #(["0","0","1"],["0","0","1"]), \
        #(["1","1","0"],["1","0","0"]), \
        #(["1","0","1"],["0","1","0"]), \
        #(["0","1","1"],["1","0","1"]), \
        (["1","1","1"],["1","1","0"])]

        train_dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=train_data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])
        model = pylfit.models.WDMVLP(features=train_dataset.features, targets=train_dataset.targets)
        model.compile(algorithm="gula")
        model.fit(dataset=train_dataset)

        init_states = [list(s) for s in set(tuple(s1) for s1,s2 in dataset_perfect.data)]
        predictions = model.predict(feature_states=init_states, raw_rules=True)

        self.assertEqual(round(explanation_score_from_predictions(predictions=predictions, expected_model=optimal_model, dataset=dataset_perfect),2), 0.28)

        train_data = [ \
        (["0","0","0"],["0","0","1"]), \
        (["0","0","0"],["1","0","0"]), \
        (["1","0","0"],["0","0","0"]), \
        (["0","1","0"],["1","0","1"]), \
        #(["0","0","1"],["0","0","1"]), \
        #(["1","1","0"],["1","0","0"]), \
        #(["1","0","1"],["0","1","0"]), \
        #(["0","1","1"],["1","0","1"]), \
        (["1","1","1"],["1","1","0"])]

        train_dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=train_data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])
        model = pylfit.models.WDMVLP(features=train_dataset.features, targets=train_dataset.targets)
        model.compile(algorithm="gula")
        model.fit(dataset=train_dataset)

        init_states = [list(s) for s in set(tuple(s1) for s1,s2 in dataset_perfect.data)]
        predictions = model.predict(feature_states=init_states, raw_rules=True)

        self.assertEqual(round(explanation_score_from_predictions(predictions=predictions, expected_model=optimal_model, dataset=dataset_perfect),2), 0.87)

        train_data = [ \
        (["0","0","0"],["0","0","1"]), \
        (["0","0","0"],["1","0","0"]), \
        (["1","0","0"],["0","0","0"]), \
        (["0","1","0"],["1","0","1"]), \
        (["0","0","1"],["0","0","1"]), \
        (["1","1","0"],["1","0","0"]), \
        (["1","0","1"],["0","1","0"]), \
        #(["0","1","1"],["1","0","1"]), \
        (["1","1","1"],["1","1","0"])]

        train_dataset = pylfit.preprocessing.discrete_state_transitions_dataset_from_array(data=train_data, feature_names=["p_t_1","q_t_1","r_t_1"], target_names=["p_t","q_t","r_t"])
        model = pylfit.models.WDMVLP(features=train_dataset.features, targets=train_dataset.targets)
        model.compile(algorithm="gula")
        model.fit(dataset=train_dataset)

        init_states = [list(s) for s in set(tuple(s1) for s1,s2 in dataset_perfect.data)]
        predictions = model.predict(feature_states=init_states, raw_rules=True)

        self.assertEqual(round(explanation_score_from_predictions(predictions=predictions, expected_model=optimal_model, dataset=dataset_perfect),2), 0.98)

        # None explanation
        predictions = {tuple(s1): {variable: {value: (proba, \
        (int(proba*100), None),\
        (100 - int(proba*100), None) )\
        for val_id, value in enumerate(values) for proba in [round(random.uniform(0.0,1.0),2)]}\
        for var_id, (variable, values) in enumerate(dataset_perfect.targets)}\
        for s1 in init_states}

        self.assertEqual(explanation_score_from_predictions(predictions=predictions, expected_model=optimal_model, dataset=dataset_perfect), 0.0)

        # random tests
        for i in range(self._nb_random_tests):
            nb_features = random.randint(1,self._nb_features)
            nb_targets = random.randint(1,self._nb_targets)
            max_feature_values = random.randint(1,self._nb_values)
            max_target_values = random.randint(1,self._nb_values)
            nb_transitions = random.randint(2,self._nb_transitions)

            dataset = random_DiscreteStateTransitionsDataset(nb_transitions, nb_features, nb_targets, max_feature_values, max_target_values)

            optimal_model = WDMVLP(dataset.features, dataset.targets)
            optimal_model.compile(algorithm="gula")
            optimal_model.fit(dataset=dataset)

            # Empty program
            model = WDMVLP(dataset.features, dataset.targets)
            model.compile(algorithm="gula")
            init_states = [list(s) for s in set(tuple(s1) for s1,s2 in dataset.data)]
            predictions = model.predict(feature_states=init_states, raw_rules=True)

            self.assertEqual(round(explanation_score_from_predictions(predictions=predictions, expected_model=optimal_model, dataset=dataset),2), 0.0)

            # Empty rule program
            model = WDMVLP(dataset.features, dataset.targets)
            model.compile(algorithm="gula")
            model.fit(DiscreteStateTransitionsDataset([], dataset.features, dataset.targets))

            init_states = [list(s) for s in set(tuple(s1) for s1,s2 in dataset.data)]
            predictions = model.predict(feature_states=init_states, raw_rules=True)

            self.assertEqual(round(explanation_score_from_predictions(predictions=predictions, expected_model=optimal_model, dataset=dataset),2), 0.0)

            # Train != test
            train_data = dataset.data[0:int(0.5*len(dataset.data))]
            test_data = dataset.data[int(0.5*len(dataset.data)):]
            train_dataset = DiscreteStateTransitionsDataset(train_data, dataset.features, dataset.targets)
            test_dataset = DiscreteStateTransitionsDataset(test_data, dataset.features, dataset.targets)

            model = WDMVLP(train_dataset.features, train_dataset.targets)
            model.compile(algorithm="gula")
            model.fit(dataset=train_dataset)

            # model = optimal -> 100 accuracy
            init_states = [list(s) for s in set(tuple(s1) for s1,s2 in train_dataset.data)]
            predictions = model.predict(feature_states=init_states, raw_rules=True)

            self.assertEqual(explanation_score_from_predictions(predictions=predictions, expected_model=model, dataset=train_dataset), 1.0)

            # Exception

            # Empty dataset
            self.assertRaises(ValueError, explanation_score_from_predictions, predictions, optimal_model, DiscreteStateTransitionsDataset([], dataset.features, dataset.targets))

            init_states = [list(s) for s in set(tuple(s1) for s1,s2 in test_dataset.data)]
            predictions = model.predict(feature_states=init_states, raw_rules=True)

            # Missing init state in dataset
            remove_s1, s2 = random.choice(test_dataset.data)
            self.assertRaises(ValueError, explanation_score_from_predictions, predictions, optimal_model, DiscreteStateTransitionsDataset([(s1,s2) for (s1,s2) in test_dataset.data if list(s1) != list(remove_s1)], dataset.features, dataset.targets))

            # Missing init state in predictions
            remove_s1 = random.choice(list(predictions.keys()))
            predictions_ = predictions.copy()
            predictions_.pop(remove_s1, None)
            self.assertRaises(ValueError, explanation_score_from_predictions, predictions_, optimal_model, test_dataset)

            # Bad target domain
            test_dataset_ = test_dataset.copy()
            test_dataset_.targets = [("a",["0"])]
            self.assertRaises(ValueError, explanation_score_from_predictions, predictions, optimal_model, test_dataset_)

            # train != test
            grouped_transitions = {tuple(s1) : set(tuple(s2_) for s1_,s2_ in test_dataset.data if tuple(s1) == tuple(s1_)) for s1,s2 in test_dataset.data}

            # expected output: kinda one-hot encoding of values occurences
            expected = {}
            count = 0
            for s1, successors in grouped_transitions.items():
                count += 1
                occurs = {}
                for var in range(len(test_dataset.targets)):
                    for val in range(len(test_dataset.targets[var][1])):
                        occurs[(var,val)] = 0.0
                        for s2 in successors:
                            if s2[var] == test_dataset.targets[var][1][val]:
                                occurs[(var,val)] = 1.0
                                break
                expected[s1] = occurs

            sum_explanation_score = 0.0
            prediction = model.predict(feature_states=[s1 for s1 in expected], raw_rules=True)
            for feature_state, actual in expected.items():
                #eprint("Feature state: ", feature_state)
                #eprint(">> prediction: ",prediction[feature_state])

                sum_score = 0.0
                nb_targets = 0
                for var_id, (variable, values) in enumerate(model.targets):
                    #eprint(" "+variable+": ")
                    for val_id, (value, (proba, (w1, r1), (w2, r2))) in enumerate(prediction[feature_state][variable].items()):
                        #eprint(" "+value+" "+str(round(proba*100.0,2))+"%")

                        # No decision or bad prediction implies wrong explanation
                        if proba == 0.5 or (proba > 0.5 and actual[(var_id,val_id)] == 0.0) or (proba < 0.5 and actual[(var_id,val_id)] == 1.0):
                            score = 0.0
                            sum_score += score
                            nb_targets += 1
                            continue

                        encoded_feature_state = pylfit.algorithms.GULA.encode_state(feature_state, model.features)

                        # Predicted likely
                        if proba > 0.5:
                            expected_rules = [r for (w,r) in optimal_model.rules \
                            if r.head_variable == var_id and r.head_value == val_id and r.matches(encoded_feature_state)]
                            explanation_rule = r1

                        # Predicted unlikely
                        if proba < 0.5:
                            expected_rules = [r for (w,r) in optimal_model.unlikeliness_rules \
                            if r.head_variable == var_id and r.head_value == val_id and r.matches(encoded_feature_state)]
                            explanation_rule = r2

                        min_distance = len(model.features)
                        nearest_expected = None
                        for r in expected_rules:
                            distance = pylfit.postprocessing.hamming_distance(explanation_rule,r)
                            if distance <= min_distance:
                                min_distance = distance
                                nearest_expected = r

                        score = 1.0 - (min_distance / len(model.features))

                        #eprint(explanation_type + " explanation evaluation")
                        #eprint("Explanation rule: " + explanation)
                        #eprint("Explanation score: ", end='')
                        #eprint(str(round(score, 2)) + " (nearest expected " + explanation_type + " rule: " + nearest_expected.logic_form(model.features, model.targets) + " distance: " + str(min_distance) + ")")
                        sum_score += score
                        nb_targets += 1
                sum_explanation_score += sum_score / nb_targets

            expected_score = sum_explanation_score / len(expected)

            self.assertEqual(explanation_score_from_predictions(predictions=prediction, expected_model=optimal_model, dataset=test_dataset), expected_score)


'''
@desc: main
'''
if __name__ == '__main__':

    unittest.main()
