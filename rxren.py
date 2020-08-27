from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score
import copy
from common_functions import perturbator, create_model, model_trainer
import sys

# Global variable
TOLERANCE = 0.01
MODEL_NAME = 'rxren_model.h5'

data, meta = arff.loadarff('datasets-UCI/UCI/diabetes.arff')
label_col = 'class'
data = pd.DataFrame(data)
data = data.dropna()
le = LabelEncoder()
for item in range(len(meta.names())):
    item_name = meta.names()[item]
    item_type = meta.types()[item]
    if item_type == 'nominal':
        data[item_name] = le.fit_transform(data[item_name].tolist())

n_classes = 2
hidden_neurons = 3

# Functions
def input_delete(insignificant_index, inDf, inWeight=None):
    """
    Delete the variable of the input vector corresponding the insignificant input neurons and, if required, the
    corresponding weights of the neural network
    :param inDf:
    :param inDf:
    :param inWeight:
    :return: the trimmed weights and input vector
    """
    outDf = copy.deepcopy(inDf)
    outDf = np.delete(outDf, insignificant_index, 1)
    outWeight = None
    if inWeight is not None:
        outWeight = copy.deepcopy(inWeight)
        outWeight[0] = np.delete(outWeight[0], insignificant_index, 0)
    return outDf, outWeight


def model_pruned_prediction(inputX, w):
    new_m = create_model(inputX, n_classes, hidden_neurons)
    new_m.set_weights(w)
    results = new_m.predict(inputX)
    results = np.argmax(results, axis=1)
    return results


def network_pruning(w, cX, cy, test_x, test_y, accuracy):
    """
    This function removes the insignificant input neurons
    :param w: model's weights
    :param t: theta
    :param cX: correctX
    :param cy: correcty
    :return: pruned weights
    """
    insignificant_neurons = []
    missclassified_list = []
    error_list = []
    new_acc = 0
    for i in range(w[0].shape[0]):
        newCx, new_w = input_delete(i, cX, inWeight=w)
        res = model_pruned_prediction(newCx, new_w)
        misclassified = [i for i in range(len(cy)) if res[i] != cy[i]]
        missclassified_list.append(misclassified)
        err = len(misclassified)
        error_list.append(err)
    pruning = True
    theta = min(error_list)
    while pruning:
        insignificant_neurons = [i for i, e in enumerate(error_list) if e <= theta]
        new_test_x, new_w = input_delete(insignificant_neurons, test_x, inWeight=w)
        new_res = model_pruned_prediction(new_test_x, new_w)
        new_acc = accuracy_score(new_res, test_y)
        if new_acc >= accuracy - TOLERANCE:
            new_error_list = [e for i, e in enumerate(error_list) if e > theta]
            # Leaving at least one significant neuron
            if len(new_error_list) > 1:
                theta = min(new_error_list)
            else:
                pruning = False
        else:
            pruning = False
    return missclassified_list, sorted(insignificant_neurons), new_acc, theta


def rule_limits_calculator(c_x, c_y, missclassified_list, significant_neurons, error, alpha=0.5):
    c_tot = np.column_stack((c_x, c_y))
    grouped_miss_class = []
    for i in significant_neurons:
        miss_class = c_tot[missclassified_list[i]]
        # Splitting the misclassified input values according to their output classes
        grouped_miss_class = grouped_miss_class + [{'neuron': i, 'class': k,
                               'limits': [min(miss_class[:, i][miss_class[:, -1] == k]),
                                max(miss_class[:, i][miss_class[:, -1] == k])]}
                                for k in np.unique(miss_class[:, -1])
                                if len(miss_class[:, i][miss_class[:, -1] == k]) > (error * alpha)]
    return grouped_miss_class

def rule_combiner(ruleset):
    rule_dict = {'class': [], 'neuron': [], 'limits': [], 'position': []}
    for i, rule in enumerate(ruleset):
        if rule['class'] not in rule_dict['class']:
            rule_dict['class'] += [rule['class']]
            rule_dict['neuron'].append([rule['neuron']])
            rule_dict['limits'].append([rule['limits']])
            rule_dict['position'].append([i])
        else:
            ix = rule_dict['class'].index(rule['class'])
            rule_dict['neuron'][ix].append(rule['neuron'])
            rule_dict['limits'][ix].append(rule['limits'])
            rule_dict['position'][ix].append(i)
    return rule_dict

def rule_pruning(train_x, train_y, ruleset):
    """
    Pruning the rules
    :param train_x:
    :param train_y:
    :param ruleset:
    :return:
    """
    rule_dict = rule_combiner(ruleset)
    for i, v in enumerate(rule_dict['class']):
        predicted_y = np.empty(train_y.shape)
        predicted_y[:] = np.NaN
        limits = rule_dict['limits'][i]
        neuron = rule_dict['neuron'][i]
        position = np.array(rule_dict['position'][i])
        single_accuracies = []
        for k, z in enumerate(neuron):
            predicted_y_single = np.empty(train_y.shape)
            predicted_y_single[:] = np.NaN
            minimum = limits[k][0]
            maximum = limits[k][1]
            predicted_y[(train_x[:, z] >= minimum) * (train_x[:, z] <= maximum)] = int(v)
            predicted_y_single[(train_x[:, z] >= minimum) * (train_x[:, z] <= maximum)] = int(v)
            # It is necessary to remove the nan values otherwise it is not possible to calculate accuracy
            predicted_y_single[np.isnan(predicted_y_single)] = n_classes + 10
            single_accuracies += [accuracy_score(predicted_y_single, train_y)]
        predicted_y[np.isnan(predicted_y)] = n_classes + 10
        overall_accuracy = accuracy_score(predicted_y, train_y)
        ixs = position[np.array(single_accuracies >= overall_accuracy, dtype=bool)]
        for item in ixs:
            ruleset.pop(item)
    return ruleset

def rule_evaluator(x, y, rule_list, class_list):
    predicted_y = np.empty(y.shape)
    predicted_y[:] = np.NaN
    class_list = class_list.tolist()
    for rule in rule_list:
        if rule['class'] in class_list and len(class_list) > 0:
            class_list.remove(rule['class'])
        col = rule['neuron']
        minimum = rule['limits'][0]
        maximum = rule['limits'][1]
        predicted_y[(x[:, col] >= minimum) * (x[:, col] <= maximum)] = int(rule['class'])
    if len(class_list) == 1:
        predicted_y[np.isnan(predicted_y)] = class_list[0]
    elif len(class_list) == 0:
        # Just in case the rules do not cover the entire dataset
        predicted_y[np.isnan(predicted_y)] = n_classes + 10
    else:
        print("It is not possible to identify default class")
    rule_accuracy = accuracy_score(predicted_y, y)
    # Calculate min and max of mismatched instances
    mismatched = [index for index, elem in enumerate(y) if elem != predicted_y[index]]
    ret_rules = []
    for rule in rule_list:
        col = rule['neuron']
        mismatched_values = x[mismatched, col]
        rule['new_limits'] = [min(mismatched_values), max(mismatched_values)]
        ret_rules.append(rule)
    return rule_accuracy, ret_rules


# Main code

# Separating independent variables from the target one
X = data.drop(columns=[label_col]).to_numpy()
y = le.fit_transform(data[label_col].tolist())


ix = [i for i in range(len(X))]
train_index = resample(ix, replace=True, n_samples=int(len(X)*0.5), random_state=0)
val_index = [x for x in ix if x not in train_index]
X_train, X_test = X[train_index], X[val_index]
y_train, y_test = y[train_index], y[val_index]

# define model
model = create_model(X, n_classes, hidden_neurons)

model_train = False
if model_train:
    model_trainer(X_train, to_categorical(y_train, num_classes=n_classes),
                  X_test, to_categorical(y_test, num_classes=n_classes), model, MODEL_NAME)

model = load_model(MODEL_NAME)
weights = np.array(model.get_weights())
results = model.predict(X_train)
results = np.argmax(results, axis=1)
correctX = X_train[[results[i] == y_train[i] for i in range(len(y_train))]]
correcty = y_train[[results[i] == y_train[i] for i in range(len(y_train))]]

acc = accuracy_score(results, y_train)
print("Accuracy of original model on the train dataset: ", acc)
test_pred = np.argmax(model.predict(X_test), axis=1)
test_acc = accuracy_score(test_pred, y_test)
print("Accuracy of original model on the test dataset: ", test_acc)

miss_list, ins_index, new_accuracy, err = network_pruning(weights, correctX, correcty, X_test, y_test, test_acc)
significant_index = [i for i in range(weights[0].shape[0]) if i not in ins_index]
print("Accuracy of pruned network", new_accuracy)


rule_limits = rule_limits_calculator(correctX, correcty, miss_list, significant_index, err, alpha=0.5)
if len(rule_limits) > 1:
    rule_limits = rule_pruning(X_train, y_train, rule_limits)

rule_simplifier = True
old_rule_acc = new_accuracy
final_rules = []
while rule_simplifier:
    rule_acc, final_rules = rule_evaluator(X_test, y_test, rule_limits, np.unique(y))
    if rule_acc > old_rule_acc:
        old_rule_acc = rule_acc
    else:
        rule_simplifier = False
    final_rules[0]['limits'] = final_rules[0].pop('new_limits')

print(final_rules)
predicted_labels = np.argmax(model.predict(X_test), axis=1)

num_test_examples = X_test.shape[0]
perturbed_data = perturbator(X_test)
rule_labels = np.empty(num_test_examples)
rule_labels[:] = np.nan
perturbed_labels = np.empty(num_test_examples)
perturbed_labels[:] = np.nan
for rule in final_rules:
    neuron = X_test[:, rule['neuron']]
    indexes = np.where((neuron >= rule['limits'][0]) & (neuron <= rule['limits'][1]))
    rule_labels[indexes] = rule['class']
    p_neuron = perturbed_data[:, rule['neuron']]
    indexes = np.where((p_neuron >= rule['limits'][0]) & (p_neuron <= rule['limits'][1]))
    perturbed_labels[indexes] = rule['class']

rule_labels[np.where(np.isnan(rule_labels))] = 1
perturbed_labels[np.where(np.isnan(perturbed_labels))] = 1

complete = 0
correct = 0
fidel = 0
rob = 0
for i in range(0, num_test_examples):
    fidel += (rule_labels[i] == predicted_labels[i])
    correct += (rule_labels[i] == y_test[i])
    rob += (predicted_labels[i] == perturbed_labels[i])
    complete += (rule_labels[i] == n_classes + 10)

fidelity = fidel / num_test_examples
print("Fidelity of the ruleset is: " + str(fidelity))
completeness = 1 - complete / num_test_examples
print("Completeness of the ruleset is: " + str(completeness))
correctness = correct / num_test_examples
print("Correctness of the ruleset is: " + str(correctness))
robustness = rob / num_test_examples
print("Robustness of the ruleset is: " + str(robustness))
print("Number of rules : " + str(len(final_rules)))
combined_rules = rule_combiner(final_rules)
avg_length = sum([len(item) for item in combined_rules['neuron']]) / len(combined_rules['class'])
print("Average rule length: " + str(avg_length))
