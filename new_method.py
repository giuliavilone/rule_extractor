import pandas as pd
import numpy as np
from keras.models import load_model
from rxren_rxncn_functions import input_delete, model_pruned_prediction
from common_functions import save_list, create_empty_file, attack_definer, rule_metrics_calculator
from refne import synthetic_data_generator
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from itertools import cycle, islice
from sklearn.metrics import accuracy_score
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import combinations
from dataset_split import dataset_splitter
import sys


# These are to remove some of the tensorflow warnings. The code works in any case
# import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# This is to avoid showing a Pandas warning when creating a new column in a dataframe by assigning it the values
# from a list
pd.options.mode.chained_assignment = None


def prediction_classifier(orig_y, predict_y, comparison="misclassified"):
    """
    Return the dictionary containing, for each output class, the list of the indexes of the instances that are
    wrongly or correctly classified, depending on the comparison type.
    :param orig_y: list/array of the original labels
    :param predict_y: list/array of the labels predicted by a model
    :param comparison: type of comparison. The default is misclassified, meaning that the function will return the
    indexes of the instances that are misclassified by the model. Otherwise, the function returns the instances that
    are correctly classified.
    :return: dictionary containing for each output class (each class is a key) the list of the wrongly/correctly
    classified instances
    """
    ret = {}
    out_classes = np.unique(orig_y)
    for cls in out_classes:
        if comparison == 'misclassified':
            ret[cls] = [i for i in range(len(orig_y)) if predict_y[i] != orig_y[i] and orig_y[i] == cls]
        else:
            ret[cls] = [i for i in range(len(orig_y)) if predict_y[i] == orig_y[i] and orig_y[i] == cls]
    return ret


def network_pruning(w, correct_x, correct_y, in_item=None):
    """
    Remove the insignificant input neurons of the input model, based on the weight w.
    :param w: model's weights
    :param correct_x: set of correctly classified instances (independent variables)
    :param correct_y: set of correctly classified instances (dependent variable)
    :param in_item: in_item
    :return: miss-classified instances, pruned weights, accuracy of pruned model,
    """
    temp_w = copy.deepcopy(w)
    temp_x = copy.deepcopy(correct_x.to_numpy())
    significant_cols = correct_x.columns.tolist()
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
        if min(error_list) == 0:
            insignificant_cols_tmp = [i for i, e in enumerate(error_list) if e == 0]
            significant_cols = [significant_cols[i] for i in range(len(significant_cols)) if i not in insignificant_cols_tmp]
            temp_x, temp_w = input_delete(insignificant_cols_tmp, temp_x, in_weight=temp_w)
        else:
            pruning = False
    return significant_cols


def remove_column(df, column_tbm, in_weight=None):
    """
    Remove the columns not listed in column_tbm from the input dataframe and, if not none, from the model's weights
    :param df: input pandas dataframe
    :param column_tbm: list of columns to be maintained in the dataframe
    :param in_weight: array of model's weights
    :param column_tbm: list of columns to be maintained in the input dataframe and the array of weights
    :return: the input dataframe (df) devoided of the columns to be deleted
    """
    all_cols = df.columns.tolist()
    columns_tbd = [all_cols[i] for i, col in enumerate(all_cols) if col not in column_tbm]
    df = df.drop(columns_tbd, axis=1)
    if in_weight is not None:
        weight_tdb = [i for i, col in enumerate(all_cols) if col not in column_tbm]
        in_weight[0] = np.delete(in_weight[0], weight_tdb, 0)
    return df, in_weight


def cluster_plots(in_df, clusters, label_col):
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']), int(max(clusters) + 1))))
    features = in_df.columns.tolist()
    features.remove(label_col)
    x = in_df.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components, columns=['PCA1', 'PCA2'])
    plt.scatter(principal_df['PCA1'], principal_df['PCA2'], color=colors[clusters])
    plt.show()


def rule_creator(in_df_list, original_x, label_col, *args):
    """
    Define the limits of each cluster contained in the input list of clusters (in_df_list) to determine the
    antecedents of each rule
    :param in_df_list: list of clusters
    :param original_x: original dataset
    :param label_col: name of the column containing the output labels of each cluster
    :param args: list of the columns that are not deemed relevant and must be dropped to create the rule
    :return: list of rules. Each rule is a dictionary where 'class' is the conclusion of the rule (the output label),
    'columns' is the list of input variables that are relevant to reach the conclusion and 'limits' is the list of
    the ranges (one range per each relevant variable) that determine the subset of the input space where the rule is
    valid.
    """
    rule_set = []
    for df in in_df_list:
        rule = {'class': df[label_col].unique()[0]}
        df = df.drop(label_col, axis=1)
        df = df.drop(list(args), axis=1)
        rule['columns'] = df.columns.tolist()
        rule['limits'] = [[df[col].min(), df[col].max()] for col in df.columns.tolist()]
        rule['samples'] = rule_elicitation(original_x, rule)
        rule_set.append(rule)
    return rule_set


def rule_elicitation(x, in_rule):
    """
    Calculate the accuracy score of the input rule and the indexes of the instances that fire the input rule. The
    accuracy score is calculated over the instances that are affected by the rule.
    :param x: dataframe containing the independent variables of the input instances
    :param in_rule: rule to be evaluated
    :return: the accuracy score of the rule and the indexes of the instances that fire the rule
    """
    indexes = []
    for item in range(len(in_rule['columns'])):
        minimum = in_rule['limits'][item][0]
        maximum = in_rule['limits'][item][1]
        column = in_rule['columns'][item]
        item_indexes = np.where(np.logical_and(x[column] >= minimum, x[column] <= maximum))[0]
        indexes.append(list(item_indexes))
    intersect_indexes = list(set.intersection(*[set(lst) for lst in indexes]))
    return intersect_indexes


def rule_set_evaluator(original_y, rule_set, rule_area_only=False):
    """
    Evaluates a set of rules by eliciting each of them and calculate the overall accuracy. The rules are first
    sorted by the number of instances that they cover (in reverse order) so that the bigger rules do not cancel out the
    smaller one in case of overlapping rules.
    :param original_y:
    :param rule_set:
    :param rule_area_only:
    :return:
    """
    predicted_y = []
    rule_indexes = []
    for rule in rule_set:
        indexes = rule['samples']
        rule_indexes += indexes
        predicted_y.append((len(indexes), indexes, rule['class']))
    predicted_y = sorted(predicted_y, reverse=True, key=lambda tup: tup[0])
    ret_labels = np.empty(len(original_y))
    ret_labels[:] = np.nan
    for t in predicted_y:
        ret_labels[t[1]] = t[2]
    empty_index = list(np.where(np.isnan(ret_labels))[0])
    if rule_area_only:
        rule_indexes = list(set(rule_indexes))
        predicted_labels = ret_labels[rule_indexes]
        predicted_labels[np.where(np.isnan(predicted_labels))] = len(np.unique(original_y)) + 10
        accuracy = accuracy_score(predicted_labels.round(), original_y[rule_indexes])
    else:
        if len(empty_index) > 0:
            ret_labels[np.where(np.isnan(ret_labels))] = len(np.unique(original_y)) + 10
        accuracy = accuracy_score(ret_labels.round(), original_y)
    return accuracy, ret_labels, empty_index


def rule_extractor(original_data, original_label, in_df, label_col, minimum_acc, rule_set, number_clusters=2,
                   linkage='complete', min_sample=300):
    """
    Return the ruleset automatically extracted to mimic the logic of a machine-learned model.
    :param original_data:
    :param original_label:
    :param in_df:
    :param label_col:
    :param minimum_acc:
    :param rule_set:
    :param number_clusters:
    :param linkage:
    :param min_sample:
    :return: a list of dictionaries where each dictionary is a rule
    """
    groups = in_df.groupby(label_col, as_index=False)
    for key, group in groups:
        if len(group) > min_sample:
            clustering = AgglomerativeClustering(n_clusters=number_clusters, linkage=linkage).fit(group.to_numpy())
            group['clusters'] = clustering.labels_
            ans = [pd.DataFrame(x) for _, x in group.groupby('clusters', as_index=False)]
            new_rules = rule_creator(ans, original_data, label_col, 'clusters')
            new_ruleset_accuracy, _, _ = rule_set_evaluator(original_label, new_rules, rule_area_only=True)
            print("New ruleset accuracy: ", new_ruleset_accuracy)
            if new_ruleset_accuracy > minimum_acc:
                best_acc = new_ruleset_accuracy
                best_rule = []
                for r in range(1, number_clusters+1):
                    combos = combinations(new_rules, r)
                    for combo in combos:
                        combo = list(combo)
                        combo_accuracy, _, _ = rule_set_evaluator(original_label, combo, rule_area_only=True)
                        if combo_accuracy >= best_acc:
                            best_rule = combo
                rule_set = rule_set + best_rule
            else:
                for rule in new_rules:
                    rule_acc, _, _ = rule_set_evaluator(original_label, [rule], rule_area_only=True)
                    if rule_acc < minimum_acc:
                        rule_indexes = rule_elicitation(in_df, rule)
                        new_x = in_df[in_df.index.isin(rule_indexes)]
                        rule_set = rule_extractor(original_data, original_label, new_x, label_col, minimum_acc,
                                                  rule_set, number_clusters=number_clusters)
                    else:
                        rule_set.append(rule)
        else:
            group = group.drop(label_col, axis=1)
            rule = {'class': key, 'columns': group.columns.tolist(),
                    'limits': [[group[col].min(), group[col].max()] for col in group.columns.tolist()]
                    }
            rule['samples'] = rule_elicitation(original_data, rule)
            rule_set.append(rule)
    return rule_set


def find_min_position(array):
    """
    Return the position of the minimum positive number (>0) of each row of the input array
    :param array:
    :return: list of the position of the minimum value of each row of input array
    """
    min_list = []
    for row in array:
        min_elem = min(row[row > 0])
        min_list.append(list(np.where(row == min_elem)[0])[0])
    return min_list


def complete_rule(in_df, original_y, rule_list, label_col):
    new_df = copy.deepcopy(in_df)
    new_df[label_col] = original_y
    _, _, empty_index = rule_set_evaluator(original_y, rule_list)
    if len(empty_index) > 0:
        uncovered_instances = new_df.iloc[empty_index]
        min_dist = np.zeros((len(uncovered_instances), len(rule_list)))
        for rule_number in range(len(rule_list)):
            rule = rule_list[rule_number]
            class_indexes = np.where(uncovered_instances[label_col] == rule['class'])[0]
            for column_number in range(len(rule['columns'])):
                element = uncovered_instances[rule['columns'][column_number]]
                range_min = rule['limits'][column_number][0]
                range_max = rule['limits'][column_number][1]
                min_dist[:, rule_number] += [max(range_min - v, 0) + max(v - range_max, 0)
                                             if i in class_indexes else 0 for i, v in enumerate(element)]
        min_dist_per_instance = find_min_position(min_dist)
        min_rules = set(min_dist_per_instance)
        for m_rule in min_rules:
            indices = [i for i, x in enumerate(min_dist_per_instance) if x == m_rule]
            rule_instances = uncovered_instances.iloc[indices]
            rule = rule_list[rule_number]
            temp_rule = copy.deepcopy(rule)
            for cn in range(len(rule['columns'])):
                temp_rule['limits'][cn][0] = min(min(rule_instances[rule['columns'][cn]]), rule['limits'][cn][0])
                temp_rule['limits'][cn][1] = max(max(rule_instances[rule['columns'][cn]]), rule['limits'][cn][1])
            temp_rule['samples'] += list(rule_instances.index.values)
            rule_list[m_rule] = temp_rule
    return rule_list


def rule_pruning(rule_set, original_accuracy, original_y):
    new_rules = copy.deepcopy(rule_set)
    ret = []
    item = 0
    while item < len(new_rules):
        rule = new_rules.pop(item)
        new_accuracy, _, _ = rule_set_evaluator(original_y, new_rules)
        if new_accuracy < original_accuracy:
            ret.append(rule)
            new_rules = [rule] + new_rules
            item += 1
    return ret


def number_split_finder(y, max_row):
    """
    Determine the number of subsets (or splits) depending on the number of instances per each input class and the total
    number of instances. The number of subsets must be a divider of all these numbers and the resulting subsets must
    not contain more instances than a maximum number defined by the user.
    :param y: list/array of output labels
    :param max_row: maximum number of instance per subset
    :return: the number of subsets (or splits)
    """
    data_length = list(np.bincount(y))
    find_split = True
    split = 2
    while find_split:
        division = [np.mod(i, split) for i in data_length]
        if sum(division) == 0 and np.round(len(y) / split) < max_row:
            return split
        else:
            split += 1


def ruleset_definer(original_x, original_y, dataset_par, weights, out_column, minimum_acc, max_row, min_row,
                    split_x=None):
    if split_x is None:
        xSynth = synthetic_data_generator(original_x, min(original_x.shape[0] * 2, max_row-original_x.shape[0]))
        X = xSynth.append(original_x, ignore_index=True)
    else:
        # xSynth = synthetic_data_generator(split_x, min(split_x.shape[0] * 2, max_row - split_x.shape[0]))
        X = split_x

    y = model_pruned_prediction([], X, dataset_par, in_weight=weights)
    X[out_column] = y

    rules = rule_extractor(original_x, original_y, X, out_column, minimum_acc, [], number_clusters=4,
                           min_sample=min_row)
    rule_accuracy, _, _ = rule_set_evaluator(original_y, rules)
    final_rules = rule_pruning(rules, rule_accuracy, original_y)
    return final_rules


def new_rule_extractor(X_train, X_test, y_train, y_test, dataset_par, save_graph):
    MINIMUM_ACC = 0.7
    MAX_ROW = 30000
    LABEL_COL = dataset_par['output_name']
    n_class = dataset_par['classes']
    MIN_ROW = dataset_par['minimum_row']
    X_train = pd.concat([X_train, X_test], ignore_index=True)
    model = load_model('trained_models/trained_model_' + dataset_par['dataset'] + '_'
                       + str(dataset_par['best_model']) + '.h5'
                       )

    results = np.argmax(model.predict(X_train), axis=1)
    weights = np.array(model.get_weights())
    significant_features = network_pruning(weights, X_train, results, in_item=dataset_par)

    X_train, weights = remove_column(X_train, significant_features, in_weight=weights)
    X_test, _ = remove_column(X_test, significant_features)
    results = model_pruned_prediction([], X_train, dataset_par, in_weight=weights)
    print("Model pruned")

    if len(X_train) > MAX_ROW:
        overall_rules = []
        n_splits = number_split_finder(results, MAX_ROW)
        print("Splitting the dataset")
        splits, split_labels = dataset_splitter(X_train, results, n_splits, n_class, distance_name='euclidean')
        for split in splits:
            print("----------------- Working on a split ---------------------")
            split_X = pd.DataFrame(split, columns=X_train.columns.tolist())
            split_rules = ruleset_definer(X_train, results, dataset_par, weights, LABEL_COL, MINIMUM_ACC, MAX_ROW,
                                          MIN_ROW, split_x=split_X)
            overall_rules += split_rules
        rule_accuracy, _, _ = rule_set_evaluator(results, overall_rules)
        overall_rules = rule_pruning(overall_rules, rule_accuracy, results)
    else:
        overall_rules = ruleset_definer(X_train, results, dataset_par, weights, LABEL_COL, MINIMUM_ACC, MAX_ROW,
                                        MIN_ROW)

    final_rules = complete_rule(X_train, results, overall_rules, LABEL_COL)
    rule_accuracy, rule_prediction, _ = rule_set_evaluator(results, final_rules)
    # print(final_rules)
    # cluster_plots(X, y, LABEL_COL)
    # cluster_plots(X, rule_prediction.round().astype(int), LABEL_COL)

    metrics = rule_metrics_calculator(X_train, np.concatenate([y_train, y_test], axis=0), results, final_rules, n_class)
    if save_graph:
        attack_list, final_rules = attack_definer(X_train, final_rules)
        create_empty_file('NEW_METHOD_' + dataset_par['dataset'] + "_attack_list")
        save_list(attack_list, 'NEW_METHOD_' + dataset_par['dataset'] + "_attack_list")
        create_empty_file('NEW_METHOD_' + dataset_par['dataset'] + "_final_rules")
        save_list(final_rules, 'NEW_METHOD_' + dataset_par['dataset'] + "_final_rules")

    return metrics
