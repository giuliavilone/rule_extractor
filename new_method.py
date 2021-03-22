import pandas as pd
import numpy as np
from keras.models import load_model
from rxren_rxncn_functions import input_delete, model_pruned_prediction
from common_functions import dataset_uploader, attack_definer
from refne import synthetic_data_generator
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from itertools import cycle, islice
from sklearn.metrics import accuracy_score
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mysql_queries import mysql_queries_executor
import sys

# These are to remove some of the tensorflow warnings. The code works in any case
# import os
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# This is to avoid showing a Pandas warning when creating a new column in a dataframe by assigning it the values
# from a list
pd.options.mode.chained_assignment = None


def prediction_classifier(orig_y, predict_y, comparison="misclassified"):
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
            insignificant_neurons_temp = [i for i, e in enumerate(error_list) if e == 0]
            significant_cols = [significant_cols[i] for i in significant_cols if i not in insignificant_neurons_temp]
            temp_x, temp_w = input_delete(insignificant_neurons_temp, temp_x, in_weight=temp_w)
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
    :return:
    """
    for i, col in enumerate(df.columns.tolist()):
        if col not in column_tbm:
            df = df.drop(col, axis=1)
            if in_weight is not None:
                in_weight[0] = np.delete(in_weight[0], i, 0)
    return df, in_weight


def cluster_plots(in_df, clusters):
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']), int(max(clusters) + 1))))
    features = in_df.columns.tolist()
    features.remove(LABEL_COL)
    x = in_df.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components, columns=['PCA1', 'PCA2'])
    plt.scatter(principal_df['PCA1'], principal_df['PCA2'], color=colors[clusters])
    plt.show()


def rule_creator(in_df_list, *args):
    rule_set = []
    for df in in_df_list:
        rule = {'class': df['class'].unique()[0]}
        df = df.drop(list(args), axis=1)
        rule['columns'] = df.columns.tolist()
        rule['limits'] = [[df[col].min(), df[col].max()] for col in df.columns.tolist()]
        rule_set.append(rule)
    return rule_set


def rule_elicitation(x, in_rule):
    original_y = x[LABEL_COL].to_numpy()
    predicted_y = np.empty(len(y))
    indexes = []
    for item in range(len(in_rule['columns'])):
        minimum = in_rule['limits'][item][0]
        maximum = in_rule['limits'][item][1]
        column = in_rule['columns'][item]
        item_indexes = np.where(np.logical_and(x[column] >= minimum, x[column] <= maximum))[0]
        indexes.append(list(item_indexes))
    intersect_indexes = list(set.intersection(*[set(lst) for lst in indexes]))
    predicted_y[intersect_indexes] = in_rule['class']
    ret = accuracy_score(original_y[intersect_indexes], predicted_y[intersect_indexes])
    return ret, intersect_indexes


def rule_set_evaluator(x, original_y, rule_set):
    predicted_y = []
    for rule in rule_set:
        _, indexes = rule_elicitation(x, rule)
        predicted_y.append((len(indexes), indexes, rule['class']))
    predicted_y = sorted(predicted_y, reverse=True, key=lambda tup: tup[0])
    ret = np.empty(len(x))
    for t in predicted_y:
        ret[t[1]] = t[2]
    ret[np.where(np.isnan(ret))] = len(np.unique(y)) + 10
    accuracy = accuracy_score(ret.round(), original_y)
    return accuracy, ret


def rule_extractor(in_df, label_col, number_clusters=2, linkage='complete', min_sample=30):
    rule_set = []
    groups = in_df.groupby(label_col, as_index=False)
    for key, group in groups:
        if len(group) > min_sample:
            clustering = AgglomerativeClustering(n_clusters=number_clusters, linkage=linkage).fit(group.to_numpy())
            group['clusters'] = clustering.labels_
            ans = [pd.DataFrame(x) for _, x in group.groupby('clusters', as_index=False)]
            new_rules = rule_creator(ans, label_col, 'clusters')
            for rule in new_rules:
                rule_acc, indexes = rule_elicitation(in_df, rule)
                if rule_acc < MINIMUM_ACC:
                    new_x = in_df[in_df.index.isin(indexes)]
                    rule_set += rule_extractor(new_x, label_col)
                else:
                    rule_set.append(rule)
        else:
            group = group.drop(label_col, axis=1)
            rule = {'class': key, 'columns': group.columns.tolist(),
                    'limits': [[group[col].min(), group[col].max()] for col in group.columns.tolist()]
                    }
            rule_set.append(rule)
    return rule_set


def rule_pruning(rule_set, original_accuracy, x, original_y):
    new_rules = copy.deepcopy(rule_set)
    ret = []
    item = 0
    while item < len(new_rules):
        rule = new_rules.pop(item)
        new_accuracy, _ = rule_set_evaluator(x, original_y, new_rules)
        if new_accuracy < original_accuracy:
            ret.append(rule)
            new_rules = [rule] + new_rules
            item += 1
    return ret


MINIMUM_ACC = 0.7
LABEL_COL = 'class'
parameters = pd.read_csv('datasets-UCI/UCI_csv/summary.csv')
data_path = 'datasets-UCI/UCI_csv/'
dataset_par = parameters.iloc[0]
print(dataset_par['dataset'])
X_train, X_test, _, _, labels, _, _ = dataset_uploader(dataset_par, data_path, apply_smothe=False)
print(labels)

X_train = pd.concat([X_train[0], X_test[0]], ignore_index=True)
model = load_model('trained_models/trained_model_' + dataset_par['dataset'] + '_'
                   + str(dataset_par['best_model']) + '.h5'
                   )

results = model.predict_classes(X_train)
weights = np.array(model.get_weights())
significant_features = network_pruning(weights, X_train, results, in_item=dataset_par)
print(significant_features)

X_train, weights = remove_column(X_train, significant_features, in_weight=weights)

xSynth = synthetic_data_generator(X_train, X_train.shape[0] * 4)

X = xSynth.append(X_train, ignore_index=True)
y = np.argmax(model.predict(X), axis=1)
X[LABEL_COL] = y
rules = rule_extractor(X, LABEL_COL, number_clusters=2)

print('------------------------------- Final rules -------------------------------')
print(len(rules))
rule_accuracy, _ = rule_set_evaluator(X, y, rules)
print('Original accuracy: ', rule_accuracy)
final_rules = rule_pruning(rules, rule_accuracy, X, y)
print('------------------------------- Final rules -------------------------------')
print(len(final_rules))
print(final_rules)
rule_accuracy, rule_prediction = rule_set_evaluator(X, y, final_rules)
print('New original accuracy: ', rule_accuracy)
# cluster_plots(X, y)
# cluster_plots(X, rule_prediction.round().astype(int))


attack_list, final_rules = attack_definer(X_train, final_rules)

feature_set_name = 'New_method_' + dataset_par['dataset'] + "_featureset"
graph_name = 'New_method_' + dataset_par['dataset'] + "_graph"
mysql_queries_executor(ruleset=final_rules, attacks=attack_list, conclusions=labels,
                       feature_set_name=feature_set_name, graph_name=graph_name)
