import pandas as pd
from keras.models import load_model
from keras.utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adam, Nadam
import numpy as np
from sklearn.metrics import accuracy_score
import copy
from common_functions import perturbator, model_train, dataset_uploader, create_model, rule_metrics_calculator
import sys


# Functions
def input_delete(insignificant_index, in_df, in_weight=None):
    """
    Delete the variable of the input vector corresponding the insignificant input neurons and, if required, the
    corresponding weights of the neural network
    :param insignificant_index:
    :param in_df:
    :param in_weight:
    :return: the trimmed weights and input vector
    """
    out_df = copy.deepcopy(in_df)
    out_df = np.delete(out_df, insignificant_index, 1)
    out_weight = None
    if in_weight is not None:
        out_weight = copy.deepcopy(in_weight)
        out_weight[0] = np.delete(out_weight[0], insignificant_index, 0)
    return out_df, out_weight


def model_pruned_prediction(insignificant_index, in_df, in_item=None, in_weight=None):
    """
    Calculate the output classes predicted by the pruned model.
    :param input_x: model's weights
    :param w: theta
    :param in_item: correctX
    :return: numpy array with the output classes predicted by the pruned model
    """
    input_x, w = input_delete(insignificant_index, in_df, in_weight=in_weight)
    if in_item is None:
        new_m = create_model(input_x, n_classes, hidden_neurons)
    else:
        new_m = create_model(input_x, in_item['classes'], in_item['neurons'], eval(in_item['optimizer']),
                             in_item['init_mode'], in_item['activation'], in_item['dropout_rate'],
                             eval(in_item['weight_constraint']))
    new_m.set_weights(w)
    ret = new_m.predict(input_x)
    ret = np.argmax(ret, axis=1)
    return ret


def network_pruning(w, correct_x, correct_y, test_x, test_y, accuracy, in_item=None):
    """
    Remove the insignificant input neurons of the input model, based on the weight w.
    :param w: model's weights
    :param correct_x: set of correctly classified instances (independent variables)
    :param correct_y: set of correctly classified instances (dependent variable)
    :param test_x: test dataset (independent variables)
    :param test_y: test dataset (dependent variable)
    :param accuracy: accuracy
    :param in_item: in_item
    :return: miss-classified instances, pruned weights, accuracy of pruned model,
    """
    insignificant_neurons = []
    miss_classified_list = []
    error_list = []
    new_acc = 0
    for i in range(w[0].shape[0]):
        res = model_pruned_prediction(i, correct_x, in_item, in_weight=w)
        misclassified = [i for i in range(len(correct_y)) if res[i] != correct_y[i]]
        miss_classified_list.append(misclassified)
        err_length = len(misclassified)
        error_list.append(err_length)
    # In case the pruned network correctly predicts all the test inputs, the original network cannot be pruned
    # and its accuracy must be set equal to the accuracy of the original network
    if sum(error_list) > 0:
        pruning = True
    else:
        pruning = False
        new_acc = accuracy
    theta = min(error_list)
    while pruning:
        insignificant_neurons = [i for i, e in enumerate(error_list) if e <= theta]
        new_res = model_pruned_prediction(insignificant_neurons, test_x, in_item, in_weight=w)
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
    return miss_classified_list, sorted(insignificant_neurons), new_acc, theta


def rule_limits_calculator(c_x, c_y, miss_classified_list, significant_neurons, error, alpha=0.5):
    c_tot = np.column_stack((c_x, c_y))
    grouped_miss_class = []
    for i in significant_neurons:
        miss_class = c_tot[miss_classified_list[i]]
        # Splitting the misclassified input values according to their output classes
        grouped_miss_class = grouped_miss_class + [{'neuron': i, 'class': k,
                                                    'limits': [min(miss_class[:, i][miss_class[:, -1] == k]),
                                                               max(miss_class[:, i][miss_class[:, -1] == k])]}
                                                   for k in np.unique(miss_class[:, -1])
                                                   if len(miss_class[:, i][miss_class[:, -1] == k]) > (error * alpha)]
    return grouped_miss_class


def rule_combiner(rule_set):
    rule_dict = {'class': [], 'neuron': [], 'limits': [], 'position': []}
    for i, r in enumerate(rule_set):
        if r['class'] not in rule_dict['class']:
            rule_dict['class'] += [r['class']]
            rule_dict['neuron'].append([r['neuron']])
            rule_dict['limits'].append([r['limits']])
            rule_dict['position'].append([i])
        else:
            ix = rule_dict['class'].index(r['class'])
            rule_dict['neuron'][ix].append(r['neuron'])
            rule_dict['limits'][ix].append(r['limits'])
            rule_dict['position'][ix].append(i)
    return rule_dict


def rule_pruning(train_x, train_y, rule_set):
    """
    Pruning the rules
    :param train_x:
    :param train_y:
    :param rule_set:
    :return:
    """
    rule_dict = rule_combiner(rule_set)
    for i, v in enumerate(rule_dict['class']):
        predicted_y = np.empty(train_y.shape)
        predicted_y[:] = np.NaN
        limits = rule_dict['limits'][i]
        rule_neuron = rule_dict['neuron'][i]
        position = np.array(rule_dict['position'][i])
        single_accuracies = []
        for k, z in enumerate(rule_neuron):
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
        for ix_index in ixs:
            rule_set.pop(ix_index)
    return rule_set


def rule_evaluator(x, y, rule_list, class_list):
    predicted_y = np.empty(y.shape)
    predicted_y[:] = np.NaN
    class_list = class_list.tolist()
    for r in rule_list:
        if r['class'] in class_list and len(class_list) > 0:
            class_list.remove(r['class'])
        col = r['neuron']
        minimum = r['limits'][0]
        maximum = r['limits'][1]
        predicted_y[(x[:, col] >= minimum) * (x[:, col] <= maximum)] = int(r['class'])
    if len(class_list) == 1:
        predicted_y[np.isnan(predicted_y)] = class_list[0]
    elif len(class_list) == 0:
        # Just in case the rules do not cover the entire dataset
        predicted_y[np.isnan(predicted_y)] = n_classes + 10
    else:
        # In case the rules do not consider more than 1 output class, the algorithm assigns as default one the
        # most frequent class not considered by the rules
        new_y = [i for i in y if i not in class_list]
        predicted_y[np.isnan(predicted_y)] = max(set(new_y), key=new_y.count)
        print("It is not possible to identify default class")
    rule_accuracy = accuracy_score(predicted_y, y)
    # Calculate min and max of mismatched instances
    mismatched = [index for index, elem in enumerate(y) if elem != predicted_y[index]]
    ret_rules = []
    for r in rule_list:
        col = r['neuron']
        mismatched_values = x[mismatched, col]
        r['new_limits'] = [min(mismatched_values), max(mismatched_values)]
        ret_rules.append(r)
    return rule_accuracy, ret_rules


# Main code
# Global variable
TOLERANCE = 0.01

# Loading dataset
parameters = pd.read_csv('datasets-UCI/Used_data/summary.csv')
dataset_par = parameters.iloc[8]
print('--------------------------------------------------')
print(dataset_par['dataset'])
print('--------------------------------------------------')


MODEL_NAME = 'trained_model_' + dataset_par['dataset'] + '.h5'
X_train, X_test, y_train, y_test, _, _ = dataset_uploader(dataset_par, train_split=dataset_par['split'])
y = np.concatenate((y_train, y_test), axis=0)
X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
n_classes = dataset_par['classes']

model = create_model(X_train, dataset_par['classes'], dataset_par['neurons'], eval(dataset_par['optimizer']),
                     dataset_par['init_mode'], dataset_par['activation'], dataset_par['dropout_rate'],
                     weight_constraint=eval(dataset_par['weight_constraint'])
                     )
model_train(X_train, to_categorical(y_train, num_classes=n_classes),
            X_test, to_categorical(y_test, num_classes=n_classes), model, MODEL_NAME,
            n_epochs=dataset_par['epochs'], batch_size=dataset_par['batch_size']
            )

# model = load_model(MODEL_NAME)
weights = np.array(model.get_weights())
results = model.predict(X_train)
results = np.argmax(results, axis=1)
correctX = X_train[[results[i] == y_train[i] for i in range(len(y_train))]]
print('Number of correctly classified examples', correctX.shape)
correcty = y_train[[results[i] == y_train[i] for i in range(len(y_train))]]
acc = accuracy_score(results, y_train)
print("Accuracy of original model on the train dataset: ", acc)
test_pred = np.argmax(model.predict(X_test), axis=1)
test_acc = accuracy_score(test_pred, y_test)
print("Accuracy of original model on the test dataset: ", test_acc)

miss_list, ins_index, new_accuracy, err = network_pruning(weights, correctX, correcty, X_test, y_test, test_acc,
                                                          dataset_par
                                                          )
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
    p_indexes = np.where((p_neuron >= rule['limits'][0]) & (p_neuron <= rule['limits'][1]))
    perturbed_labels[p_indexes] = rule['class']

perturbed_labels[np.where(np.isnan(perturbed_labels))] = n_classes + 10

completeness = sum(~np.isnan(rule_labels)) / num_test_examples
combined_rules = rule_combiner(final_rules)
avg_length = sum([len(item) for item in combined_rules['neuron']]) / len(combined_rules['class'])

rule_metrics_calculator(num_test_examples, y_test, rule_labels, predicted_labels, perturbed_labels, len(final_rules),
                        completeness, avg_length)