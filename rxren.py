import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import copy
from common_functions import rule_elicitation, rule_metrics_calculator, attack_definer
from common_functions import save_list, create_empty_file
from rxren_rxncn_functions import rule_pruning, ruleset_accuracy, rule_sorter, input_delete
from rxren_rxncn_functions import model_pruned_prediction, rule_formatter
from scipy.stats import mode

# Global variables
TOLERANCE = 0.01


# Functions
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
    new_acc = accuracy
    for i in range(w[0].shape[0]):
        res = model_pruned_prediction(i, correct_x, in_item, in_weight=w)
        misclassified = [i for i in range(len(correct_y)) if res[i] != correct_y[i]]
        miss_classified_list.append(misclassified)
        error_list.append(len(misclassified))
    # In case the pruned network correctly predicts all the test inputs, the original network cannot be pruned
    # and its accuracy must be set equal to the accuracy of the original network
    if sum(error_list) > 0:
        pruning = True
    else:
        pruning = False

    theta = min(error_list)
    max_error = max(error_list)
    while pruning:
        insignificant_neurons = [i for i, e in enumerate(error_list) if e <= theta]
        new_res = model_pruned_prediction(insignificant_neurons, test_x, in_item, in_weight=w)
        new_acc = accuracy_score(new_res, test_y)
        if new_acc >= accuracy - TOLERANCE:
            new_error_list = [e for i, e in enumerate(error_list) if e > theta]
            # Leaving at least one significant neuron and avoid going up to the max error in case more than two
            # neurons share the same high error
            if len(new_error_list) > 1 and min(new_error_list) < max_error:
                theta = min(new_error_list)
            else:
                pruning = False
        else:
            pruning = False
    return miss_classified_list, sorted(insignificant_neurons), new_acc, error_list


def rule_limits_calculator(c_x, c_y, miss_classified_list, significant_neurons, error, alpha=0.1):
    c_tot = np.column_stack((c_x, c_y))
    grouped_miss_class = []
    for i in significant_neurons:
        miss_class = c_tot[miss_classified_list[i]]
        # Splitting the misclassified input values according to their output classes
        grouped_miss_class = grouped_miss_class + [{'columns': i, 'class': k,
                                                    'limits': [min(miss_class[:, i][miss_class[:, -1] == k]),
                                                               max(miss_class[:, i][miss_class[:, -1] == k])]}
                                                   for k in np.unique(miss_class[:, -1])
                                                   if len(miss_class[:, i][miss_class[:, -1] == k]) > (error[i] * alpha)
                                                   ]
    # If no rules can be created, the rule set must assign the majority class
    if len(grouped_miss_class) == 0:
        grouped_miss_class += [{'columns': 0, 'class': mode(c_y)[0][0], 'limits': [min(c_x[:, 0]), max(c_x[:, 0])]}]
    return grouped_miss_class


def rule_evaluator(x, y, rule_list, orig_acc, class_list):
    ret_rules = copy.deepcopy(rule_list)
    rule_accuracy = copy.deepcopy(orig_acc)
    predicted_y = np.empty(x.shape[0])
    predicted_y[:] = np.NaN
    for rule in rule_list:
        predicted_y, _ = rule_elicitation(x, rule)
    predicted_y[np.isnan(predicted_y)] = len(class_list) + 10
    # Calculate min and max of mismatched instances
    mismatched = [index for index, elem in enumerate(y) if elem != predicted_y[index]]
    for rule_number in range(len(rule_list)):
        rule = rule_list[rule_number]
        ixs = np.where(y == rule['class'])[0].tolist()
        ixs = [ix for ix in ixs if ix in mismatched]
        if len(ixs) > 0:
            for pos in range(len(rule['columns'])):
                new_min = min(x[ixs, rule['columns'][pos]])
                new_max = max(x[ixs, rule['columns'][pos]])
                ret_rules[rule_number]['limits'][pos] = [new_min, new_max]
                new_acc = ruleset_accuracy(x, y, ret_rules[rule_number], len(np.unique(y)))
                if new_acc < rule_accuracy[rule['class']]:
                    ret_rules[rule_number]['limits'][pos] = rule_list[rule_number]['limits'][pos]
                else:
                    rule_accuracy[rule['class']] = new_acc
    return rule_accuracy, ret_rules


def rule_combiner(rule_set):
    ret = {}
    for i, r in enumerate(rule_set):
        if r['class'] not in ret.keys():
            ret[r['class']] = [{'columns': r['columns'], 'limits': r['limits']}]
        else:
            ret[r['class']].append({'columns': r['columns'], 'limits': r['limits']})
    return ret


def rxren_run(X_train, X_test, y_train, y_test, dataset_par, model, save_graph):
    y = np.concatenate((y_train, y_test), axis=0)
    column_lst = X_train.columns.tolist()
    column_dict = {i: column_lst[i] for i in range(len(column_lst))}

    X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
    n_class = dataset_par['classes']
    # This will be used for calculating the final metrics
    predicted_labels = np.argmax(model.predict(X_test), axis=1)

    # model = load_model(MODEL_NAME)
    weights = np.array(model.get_weights())
    results = np.argmax(model.predict(X_train), axis=1)

    correctX = X_train[[results[i] == y_train[i] for i in range(len(y_train))]]
    print('Number of correctly classified examples', correctX.shape)
    correcty = y_train[[results[i] == y_train[i] for i in range(len(y_train))]]
    acc = accuracy_score(results, y_train)
    print("Accuracy of original model on the train dataset: ", acc)
    test_pred = np.argmax(model.predict(X_test), axis=1)
    test_acc = accuracy_score(test_pred, y_test)
    print("Accuracy of original model on the test dataset: ", test_acc)

    miss_list, ins_index, new_accuracy, err = network_pruning(weights, correctX, correcty, X_test, y_test,
                                                              test_acc, in_item=dataset_par
                                                              )

    significant_index = [i for i in range(weights[0].shape[0]) if i not in ins_index]
    significant_columns = {i: v for i, v in column_dict.items() if i in significant_index}

    print("Accuracy of pruned network", new_accuracy)
    rule_limits = rule_limits_calculator(correctX, correcty, miss_list, significant_index, err, alpha=0.5)
    rule_limits = rule_formatter(rule_combiner(rule_limits))

    rule_limits, rule_acc = rule_pruning(X_test, y_test, rule_limits, n_class)

    y_test_predicted = np.argmax(model.predict(X_test), axis=1)
    rule_simplifier = True
    while rule_simplifier:
        new_rule_acc, rule_limits = rule_evaluator(X_test, y_test_predicted, rule_limits, rule_acc, np.unique(y))
        if sum(new_rule_acc.values()) > sum(rule_acc.values()):
            rule_acc = new_rule_acc
        else:
            rule_simplifier = False

    final_rules = rule_sorter(rule_limits, X_test, significant_columns)

    X_test, _ = input_delete(ins_index, X_test)
    X_test = pd.DataFrame(X_test, columns=significant_columns.values())
    metrics = rule_metrics_calculator(X_test, y_test, predicted_labels, final_rules, n_class)
    if save_graph:
        attack_list, final_rules = attack_definer(X_test, final_rules)
        create_empty_file('RxREN_' + dataset_par['dataset'] + "_attack_list")
        save_list(attack_list, 'RxREN_' + dataset_par['dataset'] + "_attack_list")
        create_empty_file('RxREN_' + dataset_par['dataset'] + "_final_rules")
        save_list(final_rules, 'RxREN_' + dataset_par['dataset'] + "_final_rules")

    return metrics
