import pandas as pd
import copy
import numpy as np
from keras.models import load_model
from rxren_rxncn_functions import input_delete, model_pruned_prediction
from common_functions import dataset_uploader
from refne import synthetic_data_generator
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from itertools import cycle, islice
from sklearn.metrics import accuracy_score
from mpl_toolkits.mplot3d import Axes3D
import sys

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
    :param test_x: test dataset (independent variables)
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
    plt.scatter(in_df['sepalwidth'], in_df['petallength'], color=colors[clusters])
    plt.show()
    plt.scatter(in_df['sepallength'], in_df['petalwidth'], color=colors[clusters])
    plt.show()
    plt.scatter(in_df['sepalwidth'], in_df['petalwidth'], color=colors[clusters])
    plt.show()
    plt.scatter(in_df['petallength'], in_df['petalwidth'], color=colors[clusters])
    plt.show()


def rule_creator(in_df_list, *args):
    ruleset = []
    for df in in_df_list:
        rule = {'class': df['class'].unique()[0]}
        df = df.drop(list(args), axis=1)
        rule['columns'] = df.columns.tolist()
        rule['limits'] = [(df[col].min(), df[col].max()) for col in df.columns.tolist()]
        ruleset.append(rule)
    return ruleset


def rule_elicitation(x, in_rule):
    orig_y = x[LABEL_COL].to_numpy()
    pred_y = np.empty(len(y))
    indexes = []
    for item in range(len(in_rule['columns'])):
        minimum = in_rule['limits'][item][0]
        maximum = in_rule['limits'][item][1]
        clmn = in_rule['columns'][item]
        item_indexes = np.where(np.logical_and(x[clmn] >= minimum, x[clmn] <= maximum))[0]
        indexes.append(list(item_indexes))
    intersect_indexes = list(set.intersection(*[set(lst) for lst in indexes]))
    pred_y[intersect_indexes] = in_rule['class']
    ret = accuracy_score(orig_y[intersect_indexes], pred_y[intersect_indexes])
    return ret, intersect_indexes


def rule_extractor(in_df, label_col, number_clusters=2):
    ruleset = []
    groups = in_df.groupby(label_col, as_index=False)
    for key, group in groups:
        if len(group) > 1:
            clustering = AgglomerativeClustering(n_clusters=number_clusters, linkage='complete').fit(group.to_numpy())
            group['clusters'] = clustering.labels_
            ans = [pd.DataFrame(x) for _, x in group.groupby('clusters', as_index=False)]
            new_rules = rule_creator(ans, label_col, 'clusters')
            for rule in new_rules:
                rule_acc, indexes = rule_elicitation(in_df, rule)
                if rule_acc < MINIMUM_ACC:
                    new_x = in_df[in_df.index.isin(indexes)]
                    ruleset += rule_extractor(new_x, label_col)
                else:
                    ruleset.append(rule)
        else:
            group = group.drop(label_col, axis=1)
            rule = {'class': key, 'columns': group.columns.tolist(),
                    'limits': [(group[col].min(), group[col].max()) for col in group.columns.tolist()]
                    }
            ruleset.append(rule)
    return ruleset


MINIMUM_ACC = 0.7
LABEL_COL = 'class'
parameters = pd.read_csv('datasets-UCI/UCI_csv/summary.csv')
data_path = 'datasets-UCI/UCI_csv/'
dataset_par = parameters.iloc[0]
print(dataset_par['dataset'])
X_train, X_test, y_train, y_test, _, _ = dataset_uploader(dataset_par, data_path, apply_smothe=False)

X_train, X_test, y_train, y_test = X_train[0], X_test[0], y_train[0], y_test[0]
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


