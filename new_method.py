import pandas as pd
import numpy as np
from rxren_rxncn_functions import model_pruned_prediction
from common_functions import save_list, create_empty_file, attack_definer, rule_metrics_calculator
from refne import synthetic_data_generator
# from sklearn.cluster import AgglomerativeClustering,
from sklearn.cluster import KMeans, OPTICS
import matplotlib.pyplot as plt
from itertools import cycle, islice
from sklearn.metrics import accuracy_score
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import mode
from dataset_split import number_split_finder, dataset_splitter_new  # dataset_splitter,
import multiprocessing
CPU_COUNT = multiprocessing.cpu_count()

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
    wrongly or correctly classified, depending on the comparison type
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
    that are covered by the new rule
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
    rule_application_matrix = []
    for df in in_df_list:
        conclusion = df[label_col].unique()[0]
        df = df.drop(label_col, axis=1)
        if args:
            df = df.drop(list(args), axis=1)
        rule = rule_assembler(df, original_x, conclusion)
        rule_application_matrix.append([1 if x in rule['samples'] else 0 for x in range(len(original_x))])
        rule_set.append(rule)
    return rule_set, rule_application_matrix


def rule_elicitation(x, in_rule):
    """
    Return the list of the indexes of the sample that fire that input rule
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


def attack_matrix_definer(rule_set, list_of_attacks):
    ret = np.zeros((len(rule_set), len(rule_set)))
    for item in list_of_attacks:
        m_row = int(item['source_index'])
        m_col = int(item['target_index'])
        if m_row < len(rule_set) and m_col < len(rule_set):
            ret[m_row, m_col] = float(item['weight'])
    return ret


def ruleset_evaluator(original_y, rule_set, rule_application_matrix, list_of_attacks):
    """
    Evaluates a set of rules by eliciting each of them and calculate the overall accuracy. The rules are first
    sorted by the number of instances that they cover (in reverse order) so that the bigger rules do not cancel out the
    smaller one in case of overlapping rules.
    :param original_y:
    :param rule_set:
    :param rule_application_matrix:
    :param list_of_attacks:
    :return:
    """
    attack_matrix = attack_matrix_definer(rule_set, list_of_attacks)
    # Calculate the sum of the attacks received by each active rule from the other active rules
    attack_application = np.matmul(rule_application_matrix, attack_matrix)
    # Replacing the 0s corresponding to the non-active rules with a high value
    max_attack = np.amax(attack_application) + 10
    attack_application = np.where(rule_application_matrix == 1, attack_application, max_attack)
    min_attacks = np.amin(attack_application, axis=1)
    predicted_y = np.empty(len(original_y))
    predicted_y[:] = np.nan
    conclusions = [r['class'] for r in rule_set]
    for idx in range(len(predicted_y)):
        min_attack = min_attacks[idx]
        if min_attack < max_attack:
            min_index = list(np.where(attack_application[idx] == min_attack)[0])
            # To be improved by considering the case when the mode does not exist
            predicted_y[idx] = mode([conclusions[i] for i in min_index])[0][0]
    empty_index = list(np.where(np.isnan(predicted_y))[0])
    if len(empty_index) > 0:
        predicted_y = nan_replacer(predicted_y, len(np.unique(original_y))+10)
    predicted_y = predicted_y.astype(np.int)
    accuracy = accuracy_score(predicted_y, original_y)
    return accuracy, empty_index


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
        perc_distance = round((distances[-2] - distances[-1]) / distances[0], 2)
        if perc_distance <= threshold:
            adding_another_cluster = False
    return n_clusters


def iterative_clustering(data, index_list, min_sample, min_cluster_number, xi_value=0.05, distance_type='minkowski'):
    """
    Applied the OPTICS clustering algorithm iteratively until just a few samples are considered as outliers and not
    placed in any cluster (meaning that their label is -1).
    :return:
    """
    print("I am extracting the clusters")
    clustering = OPTICS(min_samples=min_sample, xi=xi_value, n_jobs=CPU_COUNT, metric=distance_type
                        ).fit(data.loc[index_list].to_numpy(dtype=object))
    # Increasing the labels of the clusters in case the OPTICS algorithm was already applied and some data are
    # already assigned to clusters
    new_clusters = [i + min_cluster_number if i != -1 else i for i in clustering.labels_.tolist()]
    data.loc[index_list, ['clusters']] = new_clusters
    unique_labels, labels_frequency = np.unique(new_clusters, return_counts=True)
    # Retrieving the index of the -1 element (if it exists)
    idx = np.where(unique_labels == -1)[0]
    # If the label -1 exists, check that its frequency is greater than min_sample * 2. If this is the case, the
    # OPTICS algorithm can be applied again
    if len(idx) > 0:
        print("I still have ", labels_frequency[idx[0]], " instances not classified.")
        if labels_frequency[idx[0]] >= min_sample * 2:
            min_new_label = np.max(unique_labels) + 1
            new_index_list = data[data['clusters'] == -1].index.tolist()
            data = iterative_clustering(data, new_index_list, min_sample, min_new_label, xi_value=xi_value)
    return data


def rule_extractor(original_data, in_df, label_col, rule_set, min_sample=20):
    """
    Return the ruleset automatically extracted to mimic the logic of a machine-learned model.
    :param original_data:
    :param in_df:
    :param label_col:
    :param rule_set:
    :param min_sample:
    :return: a list of dictionaries where each dictionary is a rule
    """
    groups = in_df.groupby(by=label_col, as_index=False)
    total_rule_application_matrix = []
    for key, group in groups:
        print("Working on class: ", key)
        # To allow the OPTICS to have enough samples to create at least 2 clusters
        if len(group) > min_sample * 2:
            print("I am working on a group of length ", len(group))
            group['clusters'] = -2
            group = iterative_clustering(group, group.index.tolist(), min_sample, 0)
            # Removing outliers (instances with cluster label = -1). They might be covered by other rules or dealt with
            # at the end when the ruleset will be completed
            group = group[group['clusters'] != -1]
            ans = [pd.DataFrame(x) for _, x in group.groupby('clusters', as_index=False)]
            print("I am in the rule_creator now")
            new_rules, rule_application_matrix = rule_creator(ans, original_data, label_col, 'clusters')
            # Removing the antecedents that have limits spanning the entire range of values
            print("I am pruning the antecedents of the rules")
            new_rules = antecedent_pruning(new_rules, original_data)
            # Removing the rules that do not improve the accuracy
            rule_set = rule_set + new_rules
            total_rule_application_matrix += rule_application_matrix
    total_rule_application_matrix = np.array(total_rule_application_matrix).transpose()
    return rule_set, total_rule_application_matrix


def find_min_position(array):
    """
    Return the position of the minimum positive number (>0) of each row of the input array
    :param array: numpy array
    :return: list of the position of the minimum value of each row of input array
    """
    min_list = []
    for row in array:
        if max(row) > 0:
            min_elem = min(row[row > 0])
        else:
            min_elem = 0
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


def complete_rule(in_df, original_y, rule_list, label_col, app_matrix, attack_list):
    new_df = copy.deepcopy(in_df)
    new_df[label_col] = original_y
    _, empty_index = ruleset_evaluator(original_y, rule_list, app_matrix, attack_list)
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


def rule_counter(removed_rule, rule_set, attack_list):
    for drs in rule_set:
        if drs['rule_index'] > removed_rule['rule_index']:
            drs['rule_index'] = drs['rule_index'] - 1
    for dal in attack_list:
        if dal['source_index'] > removed_rule['rule_index']:
            dal['source_index'] = dal['source_index'] - 1
        if dal['target_index'] > removed_rule['rule_index']:
            dal['target_index'] = dal['target_index'] - 1
    return rule_set, attack_list


def rule_pruning(rule_set, original_accuracy, original_y, app_matrix, attack_list):
    for item in reversed(range(len(rule_set))):
        new_rules = copy.deepcopy(rule_set)
        new_app_matrix = copy.deepcopy(app_matrix)
        new_attack_list = copy.deepcopy(attack_list)
        removed_rule = new_rules.pop(item)
        new_app_matrix = np.delete(new_app_matrix, item, axis=1)
        new_rules, new_attack_list = rule_counter(removed_rule, new_rules, new_attack_list)
        new_accuracy, _ = ruleset_evaluator(original_y, new_rules, new_app_matrix, new_attack_list)
        if new_accuracy >= original_accuracy:
            rule_set = new_rules
            app_matrix = new_app_matrix
            original_accuracy = new_accuracy
            attack_list = new_attack_list
    return rule_set, app_matrix


def ruleset_definer(original_x, predicted_y, dataset_par, weights, out_column, max_row, min_row,
                    disc_var, cont_var, add_synthetic_data=False, split_x=None, split_y=None):
    if split_x is None:
        if add_synthetic_data:
            x_synth = synthetic_data_generator(original_x,
                                               min(original_x.shape[0] * 2, max_row - max(np.bincount(predicted_y))),
                                               cont_var, disc_var
                                               )
            y = model_pruned_prediction([], x_synth, dataset_par, in_weight=weights)
            x = x_synth.append(original_x, ignore_index=True)
            predicted_y = np.append(predicted_y, y)
        else:
            x = original_x
        x[out_column] = predicted_y
    else:
        x = split_x
        x[out_column] = split_y

    # x = x.iloc[:1000]
    # original_x = original_x.iloc[:1000]
    # out_column = out_column[:1000]
    # predicted_y = predicted_y[:1000]
    rules, app_matrix = rule_extractor(original_x, x, out_column, [], min_sample=min_row)
    attack_list, rules = attack_definer(rules)
    print("I am calculating the accuracy of the new ruleset")
    rule_accuracy, _ = ruleset_evaluator(predicted_y, rules, app_matrix, attack_list)
    rule_pruning_var = True
    if rule_pruning_var:
        print("I am pruning the rules")
        final_rules, app_matrix = rule_pruning(rules, rule_accuracy, predicted_y, app_matrix, attack_list)
    else:
        final_rules = rules
    final_rules = [{k: v for k, v in d.items() if k != 'rule_number'} for d in final_rules]
    print("I am done, leaving the rule_definer function")
    return final_rules, app_matrix


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
    out_ruleset = []
    for rule in ruleset:
        remove_indices = []
        for i, v in enumerate(rule['limits']):
            if v == limits[i]:
                remove_indices.append(i)
        rule['limits'] = item_remover(rule['limits'], remove_indices)
        rule['columns'] = item_remover(rule['columns'], remove_indices)
        # Append only those rules that have at least one antecedent that has not been removed
        if len(rule['limits']) > 0:
            out_ruleset.append(rule)
    return out_ruleset


def cluster_rule_extractor(x_train, x_test, y_train, y_test, dataset_par, save_graph, disc_attributes, cont_attributes,
                           model):
    max_row = 100000
    label_col = dataset_par['output_name']
    n_class = dataset_par['classes']
    try:
        min_row = dataset_par['minimum_row']
    except Exception as ex:
        print(ex)
        min_row = 10
    print("The minimum number of rows is: ", min_row)

    x_train = pd.concat([x_train, x_test], ignore_index=True)

    results = np.argmax(model.predict(x_train), axis=1)
    weights = np.array(model.get_weights())

    # The idea is to split the dataset only if the occurrence of one class exceeds the max number of rows allowed.

    overall_rules, application_matrix = ruleset_definer(x_train, results, dataset_par, weights, label_col,
                                                        max_row, min_row, disc_attributes, cont_attributes)

    attack_list, overall_rules = attack_definer(overall_rules)
    final_rules = complete_rule(x_train, results, overall_rules, label_col, application_matrix, attack_list)
    final_rules = antecedent_pruning(final_rules, x_train)
    fidelity, _ = ruleset_evaluator(results, final_rules, application_matrix, attack_list)
    accuracy, _ = ruleset_evaluator(np.concatenate([y_train, y_test], axis=0), final_rules, application_matrix, attack_list)
    print('These are the ruleset accuracy and fidelity: ', accuracy, ', ', fidelity)
    # print(final_rules)
    # cluster_plots(X, y, label_col)
    # cluster_plots(X, rule_prediction.round().astype(int), label_col)
    print("I am calculating the metrics")
    metrics = rule_metrics_calculator(x_train, np.concatenate([y_train, y_test], axis=0), results, final_rules, n_class,
                                      new_method={'accuracy': accuracy, 'fidelity': fidelity})
    if save_graph:
        attack_list, final_rules = attack_definer(final_rules)
        create_empty_file('NEW_METHOD_' + dataset_par['dataset'] + "_attack_list")
        save_list(attack_list, 'NEW_METHOD_' + dataset_par['dataset'] + "_attack_list")
        create_empty_file('NEW_METHOD_' + dataset_par['dataset'] + "_final_rules")
        save_list(final_rules, 'NEW_METHOD_' + dataset_par['dataset'] + "_final_rules")
        # Must create the file with the list of the output classes

    return metrics
