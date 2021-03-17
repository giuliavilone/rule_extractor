from keras.models import load_model
from keras.utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adam, Nadam, RMSprop
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import copy
from common_functions import perturbator, rule_metrics_calculator, rule_elicitation
import dictlib
from sklearn.model_selection import train_test_split
from rxren_rxncn_functions import rule_pruning, ruleset_accuracy, input_delete
from rxren_rxncn_functions import model_pruned_prediction, prediction_reshape, rule_formatter, rule_sorter
import sys


# Functions
def prediction_classifier(orig_y, predict_y, comparison="misclassified"):
    ret = {}
    out_classes = np.unique(orig_y)
    for cls in out_classes:
        if comparison == 'misclassified':
            ret[cls] = [i for i in range(len(orig_y)) if predict_y[i] != orig_y[i] and orig_y[i] == cls]
        else:
            ret[cls] = [i for i in range(len(orig_y)) if predict_y[i] == orig_y[i] and orig_y[i] == cls]
    return ret


def network_pruning(w, correct_x, correct_y, test_x, test_y, accuracy, columns, in_item=None):
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
    temp_w = copy.deepcopy(w)
    temp_x = copy.deepcopy(correct_x)
    temp_test_x = copy.deepcopy(test_x)
    significant_cols = copy.deepcopy(columns)
    theta = 0
    pruning = True
    while pruning:
        error_list = []
        miss_classified_dict = {}
        for i in range(temp_w[0].shape[0]):
            res = model_pruned_prediction(i, temp_x, in_item, in_weight=temp_w)
            misclassified = prediction_classifier(correct_y, res)
            miss_classified_dict[significant_cols[i]] = misclassified
            error_list.append(sum([len(value) for key, value in misclassified.items()]))
        # In case the pruned network correctly predicts all the test inputs, the original network cannot be pruned
        # and its accuracy must be set equal to the accuracy of the original network
        if sum(error_list) > 0:
            theta = min(error_list)
            insignificant_neurons_temp = [i for i, e in enumerate(error_list) if e == theta]
            new_res = model_pruned_prediction(insignificant_neurons_temp, temp_test_x, in_item, in_weight=temp_w)
            new_acc_temp = accuracy_score(new_res, test_y)
            if new_acc_temp >= accuracy:
                significant_cols = {i: significant_cols[i] for i in significant_cols
                                    if i not in insignificant_neurons_temp}
                significant_cols = {i: v for i, v in enumerate(significant_cols.values())}
                temp_x, temp_w = input_delete(insignificant_neurons_temp, temp_x, in_weight=temp_w)
                temp_test_x, _ = input_delete(insignificant_neurons_temp, temp_test_x)
            else:
                pruning = False
        else:
            pruning = False
    return miss_classified_dict, temp_x, temp_w, theta, significant_cols


def correct_examples_finder(correct_x, correct_y, in_item, significant_cols, in_weight=None):
    """
    This function finds the examples correctly classified by the pruned network for each significant input neuron. The
    paper is not clear on how to do that, so my interpretation is that these are the examples correctly classified
    when the pruned network contains only the significant neuron under analysis.
    :param correct_x: correctX
    :param correct_y: correctX
    :param in_item: correctX
    :param significant_cols: list of significant columns
    :param in_weight: correctX
    :return: dictionary with the indexes of the neurons as keys and the list of the examples correctly classified for
    each of them
    """
    ret = {}
    for item in range(correct_x.shape[1]):
        to_be_removed = [i for i in range(correct_x.shape[1]) if i != item]
        res = model_pruned_prediction(to_be_removed, correct_x, in_item, in_weight=in_weight)
        ret[significant_cols[item]] = prediction_classifier(correct_y, res, comparison="correct")
    return ret


def combine_dict_list(dict_1, dict_2):
    ret = {}
    for key, value in dict_1.items():
        temp_ret = dictlib.union_setadd(dict_1[key], dict_2[key])
        for key2, value2 in temp_ret.items():
            temp_ret[key2] = list(set(value2))
        ret[key] = temp_ret
    return ret


def rule_limits_calculator(c_x, c_y, classified_dict, significant_cols, alpha=0.1):
    c_tot = np.column_stack((c_x, c_y))
    grouped_miss_class = {k: [] for k in np.unique(c_y)}
    for i in range(c_x.shape[1]):
        mp = sum([len(value) for key, value in classified_dict[significant_cols[i]].items()])
        ucm_class = [len(c_tot[classified_dict[significant_cols[i]][k]]) for k in np.unique(c_y)]
        if min(ucm_class) > (mp * alpha):
            for k in np.unique(c_y):
                limit_data = c_tot[classified_dict[significant_cols[i]][k]]
                # Splitting the misclassified input values according to their output classes
                grouped_miss_class[k] += [{'columns': i, 'limits': [min(limit_data[:, i]), max(limit_data[:, i])]}]
    # Eliminate those classes that have empty lists
    grouped_miss_class = {k: v for k, v in grouped_miss_class.items() if len(v) > 0}
    return grouped_miss_class


def rule_evaluator(x, y, rule_list, orig_acc, class_list):
    ret = copy.deepcopy(rule_list)
    rule_accuracy = copy.deepcopy(orig_acc)
    predicted_y = np.empty(x.shape[0])
    predicted_y[:] = np.NaN
    for rule in rule_list:
        predicted_y, _ = rule_elicitation(x, predicted_y, rule)
    predicted_y[np.isnan(predicted_y)] = len(class_list) + 10
    for rule_number in range(len(rule_list)):
        rule = rule_list[rule_number]
        ixs = np.where(predicted_y == rule['class'])[0].tolist()
        if len(ixs) > 0:
            for pos in range(len(rule['columns'])):
                new_min = min(x[rule['columns'][pos]].iloc[ixs])
                new_max = max(x[rule['columns'][pos]].iloc[ixs])
                ret[rule_number]['limits'][pos] = [new_min, new_max]
                new_acc = ruleset_accuracy(x, y, ret[rule_number], len(np.unique(y)))
                if new_acc < orig_acc[rule['class']]:
                    ret[rule_number]['limits'][pos] = rule_list[rule_number]['limits'][pos]
                else:
                    rule_accuracy[rule['class']] = new_acc
    return rule_accuracy, ret


def rxncn_run(X_train, X_test, y_train, y_test, dataset_par, model):
    # Alpha is set equal to the percentage of input instances belonging to the least-represented class in the dataset
    alpha = 0.1
    n_class = dataset_par['classes']
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33)

    column_lst = X_train.columns.tolist()
    column_dict = {i: column_lst[i] for i in range(len(column_lst))}

    y = np.concatenate((y_train, y_test), axis=0)
    X_train, X_test, X_val = X_train.to_numpy(), X_test.to_numpy(), X_val.to_numpy()

    weights = np.array(model.get_weights())
    results = model.predict_classes(X_train)

    # This will be used for calculating the final metrics
    predicted_labels = prediction_reshape(model.predict(X_test))

    correctX = X_train[[results[i] == y_train[i] for i in range(len(y_train))]]
    print('Number of correctly classified examples', correctX.shape)
    correcty = y_train[[results[i] == y_train[i] for i in range(len(y_train))]]
    acc = accuracy_score(results, y_train)
    print("Accuracy of original model on the train dataset: ", acc)
    test_pred = prediction_reshape(model.predict(X_val))
    test_acc = accuracy_score(test_pred, y_val)
    print("Accuracy of original model on the validation dataset: ", test_acc)

    miss_dict, pruned_x, pruned_w, err, sig_cols = network_pruning(weights, correctX, correcty, X_val, y_val,
                                                                   test_acc, column_dict, in_item=dataset_par)

    corr_dict = correct_examples_finder(pruned_x, correcty, dataset_par, sig_cols, in_weight=pruned_w)
    final_dict = combine_dict_list(miss_dict, corr_dict)

    rule_limits = rule_limits_calculator(pruned_x, correcty, final_dict, sig_cols, alpha=alpha)
    rule_limits = rule_formatter(rule_limits)

    if len(rule_limits) > 0:
        insignificant_neurons = [key for key, value in column_dict.items() if value not in list(sig_cols.values())]
        X_test, _ = input_delete(insignificant_neurons, X_test)
        X_train, _ = input_delete(insignificant_neurons, X_train)
        X_val, _ = input_delete(insignificant_neurons, X_val)

        rule_limits, rule_accuracy = rule_pruning(X_val, y_val, rule_limits, n_class)

        final_rules = rule_sorter(rule_limits, X_test, sig_cols)

        y_val_predicted = model_pruned_prediction([], X_val, dataset_par, in_weight=pruned_w)
        X_val = pd.DataFrame(X_val, columns=sig_cols.values())
        rule_simplifier = True
        while rule_simplifier:
            new_rule_acc, final_rules = rule_evaluator(X_val, y_val_predicted, final_rules, rule_accuracy, np.unique(y))
            if sum(new_rule_acc.values()) > sum(rule_accuracy.values()):
                rule_accuracy = new_rule_acc
            else:
                rule_simplifier = False

        X_test = pd.DataFrame(X_test, columns=sig_cols.values())

        return rule_metrics_calculator(X_test, y_test, predicted_labels, final_rules, n_class)
    else:
        return np.zeros(8).tolist()
