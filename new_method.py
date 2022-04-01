import pandas as pd
import numpy as np
from keras.models import load_model
from rxren_rxncn_functions import input_delete, model_pruned_prediction
from common_functions import save_list, create_empty_file, attack_definer, rule_metrics_calculator
# from refne import synthetic_data_generator
# from sklearn.cluster import AgglomerativeClustering,
from sklearn.cluster import KMeans, OPTICS
import matplotlib.pyplot as plt
from itertools import cycle, islice
from sklearn.metrics import accuracy_score
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dataset_split import number_split_finder, dataset_splitter_new  # dataset_splitter,


# These are to remove some tensorflow warnings. The code works in any case
# import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# This is to avoid showing a Pandas warning when creating a new column in a dataframe by assigning it the values
# from a list
pd.options.mode.chained_assignment = None


def nan_replacer(in_array, n_classes):
    in_array[np.where(np.isnan(in_array))] = n_classes + 10
    return in_array


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
        # In case the pruned network correctly predicts all the test inputs, the original network can be pruned
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


def rule_assembler(df, original_x, conclusion):
    """
    Given the input dataframe with the data to be used to create the rule and the rule's conclusion,
    the function generates the rule's dictionary and assigns the indexes of the samples (in the original dataset)
    that are covered by the new rule.
    :param df: pandas dataframe containing the data that will be covered by the new rule
    :param original_x: pandas dataframe containing the entire dataset (either training, valuation or both)
    :param conclusion: the name of the output class that must be assigned to the rule's conclusion
    :return:
    """
    rule = {'class': conclusion,
            'columns': df.columns.tolist(),
            'limits': [[df[col].min(), df[col].max()] for col in df.columns.tolist()]}
    rule['samples'] = rule_elicitation(original_x, rule)
    return rule


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
        conclusion = df[label_col].unique()[0]
        df = df.drop(label_col, axis=1)
        if args:
            df = df.drop(list(args), axis=1)
        rule = rule_assembler(df, original_x, conclusion)
        rule_set.append(rule)
    return rule_set


def rule_elicitation(x, in_rule):
    """
    Return the list of the indexes of the sample that fire that input rule.
    :param x: dataframe containing the independent variables of the input instances
    :param in_rule: rule to be elicited
    :return: set of samples that fire the input rule
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
        predicted_labels = nan_replacer(predicted_labels, len(np.unique(original_y)))
        accuracy = accuracy_score(predicted_labels.round(), original_y[rule_indexes])
    else:
        if len(empty_index) > 0:
            ret_labels = nan_replacer(ret_labels, len(np.unique(original_y)))
        accuracy = accuracy_score(ret_labels.round(), original_y)
    return accuracy, ret_labels, empty_index


def elbow_method(in_df, threshold=0.05):
    """
    Returning the best number of clusters by using the Elbow Method
    Ref: https://en.wikipedia.org/wiki/Elbow_method_(clustering)
    Code: https://predictivehacks.com/k-means-elbow-method-code-for-python/
    :param in_df:
    :param threshold:
    :return: n_clusters
    """
    baseline = KMeans(n_clusters=1).fit(in_df)
    distances = [baseline.inertia_]
    adding_another_cluster = True
    n_clusters = 1
    while adding_another_cluster:
        n_clusters += 1
        new_distance = KMeans(n_clusters=n_clusters).fit(in_df)
        distances.append(new_distance.inertia_)
        perc_distance = round((distances[-2]-distances[-1])/distances[0], 2)
        if perc_distance <= threshold:
            adding_another_cluster = False
    return n_clusters


def iterative_clustering(data, index_list, min_sample, xi_value, min_cluster_number):
    """
    Applied the OPTICS clustering algorithm iteratively until just a few samples are considered as outliers and not
    placed in any cluster (meaning that their label is -1).
    :return:
    """
    clustering = OPTICS(min_samples=min_sample, xi=xi_value).fit(data.loc[index_list].to_numpy())
    # Increasing the labels of the clusters in case the OPTICS algorithm was already applied and some data are
    # already assigned to clusters
    new_clusters = [i + min_cluster_number if i != -1 else i for i in clustering.labels_.tolist()]
    data.loc[index_list, ['clusters']] = new_clusters
    unique_labels, labels_frequency = np.unique(new_clusters, return_counts=True)
    # Retrieving the index of the -1 element (if it exists)
    idx = np.where(unique_labels == -1)[0]
    # If the label -1 exists, check that its frequency is greater than min_sample * 2. If this is the case, the
    # OPTICS algorithm can be applied again
    if len(idx) > 0 and labels_frequency[idx[0]] >= min_sample * 2:
        min_new_label = np.max(unique_labels) + 1
        new_index_list = data[data['clusters'] == -1].index.tolist()
        data = iterative_clustering(data, new_index_list, min_sample, xi_value, min_new_label)
    return data


def rule_extractor(original_data, original_label, in_df, label_col, minimum_acc, rule_set, min_sample=10):
    """
    Return the ruleset automatically extracted to mimic the logic of a machine-learned model.
    :param original_data:
    :param original_label:
    :param in_df:
    :param label_col:
    :param minimum_acc:
    :param rule_set:
    :param min_sample:
    :return: a list of dictionaries where each dictionary is a rule
    """
    best_accuracy = minimum_acc
    groups = in_df.groupby(label_col, as_index=False)
    for key, group in groups:
        print('I am working on a group with length: ', len(group))
        if len(group) > min_sample * 2:  # To allow the OPTICS to have enough samples to create at least 2 clusters
            # The OPTICS parameter xi is set equal to 0 to minimize the number of outliers
            group['clusters'] = -2
            group = iterative_clustering(group, group.index.tolist(), min_sample, 0, 0)
            # Removing outliers (instances with cluster label = -1). They might be covered by other rules or dealt with
            # at the end when the ruleset will be completed
            group = group[group['clusters'] != -1]
            ans = [pd.DataFrame(x) for _, x in group.groupby('clusters', as_index=False)]
            new_rules = rule_creator(ans, original_data, label_col, 'clusters')
            # Removing the antecedents that have limits spanning the entire range of values
            new_rules = antecedent_pruning(new_rules, original_data)
            # Removing the rules that do not improve the accuracy
            new_rules_accuracy, _, _ = rule_set_evaluator(original_label, new_rules, rule_area_only=True)
            new_rules = rule_pruning(new_rules, new_rules_accuracy, original_label)
            # Calculating the accuracy of the entire ruleset that includes also the new rules
            new_ruleset_accuracy, _, _ = rule_set_evaluator(original_label, rule_set + new_rules, rule_area_only=True)
            print("New ruleset accuracy: ", new_rules_accuracy)
            if new_ruleset_accuracy > best_accuracy:
                rule_set = rule_set + new_rules
                best_accuracy = new_ruleset_accuracy
            else:
                for rule in new_rules:
                    rule_acc, _, _ = rule_set_evaluator(original_label, rule_set + [rule], rule_area_only=True)
                    if rule_acc < best_accuracy:
                        rule_indexes = rule_elicitation(in_df, rule)
                        new_x = in_df[in_df.index.isin(rule_indexes)]
                        rule_set = rule_extractor(original_data, original_label, new_x, label_col, minimum_acc,
                                                  rule_set)
                    else:
                        rule_set.append(rule)
                        best_accuracy = rule_acc
        else:
            group = group.drop(label_col, axis=1)
            rule = rule_assembler(group, original_data, key)
            rule = antecedent_pruning([rule], original_data)
            rule_set = rule_set + rule
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


def minimum_distance(df, ruleset, label_col):
    """
    Given an input dataframe and a list of rules, this function calculate the minimum distance between each instance
    of the dataset and the rule limits for each column of the input dataset. For examples, let's assume that the rule
    on variable A has a range of [5,10] and the dataframe contains 3 instances that have the following values for
    variable A: 1, 7, 12. Their distances to the rule will be 4 (5-1), 0 (the second instance falls within the rule's
    range) and 2 (12-10).
    :param df: pandas dataframe
    :param ruleset: list of rules
    :param label_col: name of the output column
    :return: numpy array with the distance
    """
    ret = np.zeros((len(df), len(ruleset)))
    for rule_number in range(len(ruleset)):
        rule = ruleset[rule_number]
        indexes = np.where(df[label_col] == rule['class'])[0]
        for column_number in range(len(rule['columns'])):
            element = df[rule['columns'][column_number]]
            col_min = rule['limits'][column_number][0]
            col_max = rule['limits'][column_number][1]
            ret[:, rule_number] += [max(col_min - v, 0) + max(v - col_max, 0) if i in indexes
                                    else 0 for i, v in enumerate(element)]
    return ret


def complete_rule(in_df, original_y, rule_list, label_col):
    new_df = copy.deepcopy(in_df)
    new_df[label_col] = original_y
    _, _, empty_index = rule_set_evaluator(original_y, rule_list)
    if len(empty_index) > 0:
        uncovered_instances = new_df.iloc[empty_index]
        min_dist = minimum_distance(uncovered_instances, rule_list, label_col)
        min_dist_per_instance = find_min_position(min_dist)
        min_rules = set(min_dist_per_instance)
        for m_rule in min_rules:
            indices = [i for i, x in enumerate(min_dist_per_instance) if x == m_rule]
            rule_instances = uncovered_instances.iloc[indices]
            rule = rule_list[m_rule]
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


def ruleset_definer(original_x, predicted_y, dataset_par, weights, out_column, minimum_acc, max_row, min_row,
                    split_x=None, split_y=None):
    if split_x is None:
        #xSynth = synthetic_data_generator(original_x,
        #                                  min(original_x.shape[0] * 2, max_row-max(np.bincount(predicted_y)))
        #                                  )
        #y = model_pruned_prediction([], xSynth, dataset_par, in_weight=weights)
        # X = xSynth.append(original_x, ignore_index=True)
        X = original_x
        # predicted_y = np.append(predicted_y, y)
        X[out_column] = predicted_y
    else:
        # xSynth = synthetic_data_generator(split_x, min(split_x.shape[0] * 2, max_row - split_x.shape[0]))
        X = split_x
        X[out_column] = split_y

    rules = rule_extractor(original_x, predicted_y, X, out_column, minimum_acc, [], min_sample=min_row)
    rule_accuracy, _, _ = rule_set_evaluator(predicted_y, rules)
    final_rules = rule_pruning(rules, rule_accuracy, predicted_y)
    return final_rules


def item_remover(in_list, remove_index_list):
    """
    Remove the elements in the remove_index_list from the in_list and return the shortened list
    :param in_list:
    :param remove_index_list:
    :return:
    """
    return [val for ind, val in enumerate(in_list) if ind not in remove_index_list]


def antecedent_pruning(ruleset, original_x):
    """
    Remove the antecedents whose limits cover the entire range of the original dataset.
    :param ruleset:
    :param original_x:
    :return:
    """
    limits = [[original_x[col].min(), original_x[col].max()] for col in original_x.columns.tolist()]
    for rule in ruleset:
        remove_indices = []
        for i, v in enumerate(rule['limits']):
            if v == limits[i]:
                remove_indices.append(i)
        rule['limits'] = item_remover(rule['limits'], remove_indices)
        rule['columns'] = item_remover(rule['columns'], remove_indices)
    return ruleset


def cluster_rule_extractor(x_train, x_test, y_train, y_test, dataset_par, save_graph):
    MINIMUM_ACC = 0.8
    MAX_ROW = 25000
    LABEL_COL = dataset_par['output_name']
    n_class = dataset_par['classes']
    try:
        MIN_ROW = dataset_par['minimum_row']
    except:
        MIN_ROW = 10

    X_train = pd.concat([x_train, x_test], ignore_index=True)
    model = load_model('trained_models/trained_model_' + dataset_par['dataset'] + '_'
                       + str(dataset_par['best_model']) + '.h5'
                       )

    results = np.argmax(model.predict(X_train), axis=1)
    weights = np.array(model.get_weights())
    significant_features = network_pruning(weights, X_train, results, in_item=dataset_par)

    X_train, weights = remove_column(X_train, significant_features, in_weight=weights)
    X_test, _ = remove_column(x_test, significant_features)
    results = model_pruned_prediction([], X_train, dataset_par, in_weight=weights)
    print("Model pruned")

    # The idea is to split the dataset only if the occurrence of one class exceeds the max number of rows allowed.
    if max(np.bincount(results)) >= MAX_ROW:
        overall_rules = []
        n_splits = number_split_finder(results, MAX_ROW)
        print('number of splits per class: ', n_splits)
        print("Splitting the dataset")
        # splits, split_labels = dataset_splitter(X_train, results, n_splits, n_class, distance_name='euclidean')
        splits, split_labels = dataset_splitter_new(X_train, results, n_splits)
        for i in range(len(splits)):
            print("----------------- Working on a split ---------------------")
            split_X = pd.DataFrame(splits[i], columns=X_train.columns.tolist())
            split_rules = ruleset_definer(X_train, results, dataset_par, weights, LABEL_COL, MINIMUM_ACC,
                                          MAX_ROW, MIN_ROW, split_x=split_X, split_y=split_labels[i])
            overall_rules += split_rules
        rule_accuracy, _, _ = rule_set_evaluator(results, overall_rules)
        overall_rules = rule_pruning(overall_rules, rule_accuracy, results)
    else:
        overall_rules = ruleset_definer(X_train, results, dataset_par, weights, LABEL_COL, MINIMUM_ACC, MAX_ROW,
                                        MIN_ROW)

    final_rules = complete_rule(X_train, results, overall_rules, LABEL_COL)
    rule_accuracy, rule_prediction, _ = rule_set_evaluator(results, final_rules)
    final_rules = antecedent_pruning(final_rules, X_train)
    # print(final_rules)
    # cluster_plots(X, y, LABEL_COL)
    # cluster_plots(X, rule_prediction.round().astype(int), LABEL_COL)

    metrics = rule_metrics_calculator(X_train, np.concatenate([y_train, y_test], axis=0), results, final_rules, n_class)
    if save_graph:
        attack_list, final_rules = attack_definer(final_rules)
        create_empty_file('NEW_METHOD_' + dataset_par['dataset'] + "_attack_list")
        save_list(attack_list, 'NEW_METHOD_' + dataset_par['dataset'] + "_attack_list")
        create_empty_file('NEW_METHOD_' + dataset_par['dataset'] + "_final_rules")
        save_list(final_rules, 'NEW_METHOD_' + dataset_par['dataset'] + "_final_rules")

    return metrics
