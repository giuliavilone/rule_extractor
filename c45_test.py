import pandas as pd
from keras.utils import to_categorical
from keras.models import load_model
from keras.optimizers import SGD, Adagrad, Adam, Nadam
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz, plot_tree
from common_functions import perturbator, create_model, model_train, ensemble_predictions, dataset_uploader
from common_functions import rule_metrics_calculator
import sys


# Functions
def load_all_models(n_models):
    """
    This function returns the list of the trained models
    :param n_models:
    :return: list of trained models
    """
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = 'c45_model_' + str(i) + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


def synthetic_data_generator(indf, n_samples):
    """
    Given an input dataframe, the function returns a new dataframe containing random numbers
    generated within the value ranges of the input attributes.
    :param indf:
    :param n_samples: integer number of samples to be generated
    :return: outdf: of synthetic data
    """
    outdf = pd.DataFrame()
    for column in indf.columns.tolist():
        outdf[column] = np.random.choice(np.unique(indf[column]).tolist(), n_samples)
    return outdf


def print_decision_tree(tree, feature_names=None, offset_unit='    '):
    """Plots textual representation of rules of a decision tree
    tree: scikit-learn representation of tree
    feature_names: list of feature names. They are set to f1,f2,f3,... if not specified
    offset_unit: a string of offset of the conditional block"""

    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    value = tree.tree_.value
    if feature_names is None:
        features = ['f%d'%i for i in tree.tree_.feature]
    else:
        features = [feature_names[i] for i in tree.tree_.feature]

    def recurse(left, right, threshold, features, node, depth=0):
            offset = offset_unit*depth
            if threshold[node] != -2:
                    print(offset+"if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
                    if left[node] != -1:
                            recurse (left, right, threshold, features, left[node], depth+1)
                    print(offset+"} else {")
                    if right[node] != -1:
                            recurse (left, right, threshold, features, right[node], depth+1)
                    print(offset+"}")
            else:
                    print(offset+"class: " + str(np.argmax(value[node])))

    recurse(left, right, threshold, features, 0, 0)


def get_node_depths(tree):
    """
    Get the node depths of the decision tree

    d = DecisionTreeClassifier()
    d.fit([[1,2,3],[4,5,6],[7,8,9]], [1,2,3])
    get_node_depths(d.tree_)
    array([0, 1, 1, 2, 2])
    """
    def get_node_depths_(current_node, current_depth, l, r, depths):
        if l[current_node] != -1 and r[current_node] != -1:
            get_node_depths_(l[current_node], current_depth + 1, l, r, depths)
            get_node_depths_(r[current_node], current_depth + 1, l, r, depths)
        else:
            depths += [current_depth]

    depths = []
    get_node_depths_(0, 0, tree.children_left, tree.children_right, depths)
    return np.array(depths)


def run_c45_pane_old(X_train, X_test, y_train, y_test, dataset_par):
    n_members = 10
    for member in range(n_members):
        # define model
        model = create_model(X_train, dataset_par['classes'], dataset_par['neurons'], eval(dataset_par['optimizer']),
                             dataset_par['init_mode'], dataset_par['activation'], dataset_par['dropout_rate']
                             )
        model_train(X_train, to_categorical(y_train, num_classes=dataset_par['classes']),
                    X_test, to_categorical(y_test, num_classes=dataset_par['classes']), model,
                    'c45_model_' + str(member) + '.h5', n_epochs=dataset_par['epochs'],
                    batch_size=dataset_par['batch_size'])

    X = pd.concat([X_train, X_test], ignore_index=True)
    y = np.concatenate((y_train, y_test))

    # load all models
    members = load_all_models(n_members)
    print('Loaded %d models' % len(members))

    ensemble_res = ensemble_predictions(members, X)
    print(accuracy_score(ensemble_res[0], y))

    # Same process of REFNE
    synth_samples = X.shape[0] * 2
    xSynth = synthetic_data_generator(X.to_numpy(), synth_samples)
    ySynth = ensemble_predictions(members, xSynth)
    # Concatenate the synthetic array and the array with the labels predicted by the ensemble
    xTot = np.concatenate((X, xSynth), axis=0)
    yTot = np.transpose(np.concatenate([ensemble_res, ySynth], axis=1))
    return xTot, yTot


def create_tree(in_df, model):
    synth_samples = in_df.shape[0] * 2
    x_synth = synthetic_data_generator(in_df, synth_samples)
    y_synth = np.argmax(model.predict(x_synth), axis=1)
    model_res = np.argmax(model.predict(in_df), axis=1)
    x_tot = np.concatenate((in_df, x_synth), axis=0)
    y_tot = np.transpose(np.concatenate([model_res, y_synth], axis=0))
    clf = DecisionTreeClassifier()
    clf = clf.fit(x_tot, y_tot)
    return x_tot, y_tot, clf


def metric_calculator(rule_class, original_class, model_class, perturbed_class, rule_depths, n_classes, data_len):
    correct = 0
    fidel = 0
    rob = 0
    for i in range(0, data_len):
        fidel += (rule_class[i] == model_class[i])
        correct += (rule_class[i] == original_class[i])
        rob += (rule_class[i] == perturbed_class[i])

    correctness = correct / data_len
    print("Correctness of the ruleset is: " + str(correctness))
    fidelity = fidel / data_len
    print("Fidelity of the ruleset is: " + str(fidelity))
    robustness = rob / data_len
    print("Robustness of the ruleset is: " + str(robustness))
    rule_n = len(rule_depths)
    print("Number of rules : " + str(rule_n))
    avg_length = np.average(rule_depths)
    print("Average rule length: " + str(avg_length))
    labels_considered = set(rule_class)
    labels_considered.discard(n_classes + 10)
    class_fraction = len(set(labels_considered)) / n_classes
    print("Fraction of classes: " + str(class_fraction))
    return [1.0, correctness, fidelity, robustness, rule_n, avg_length, 0, class_fraction]


def run_c45_pane(X_train, X_test, y_test, dataset_par, model, labels):
    print(labels)
    x_tot, y_tot, clf = create_tree(X_train, model)
    rules = export_text(clf)

    # Showing the rules
    # print_decision_tree(clf)

    predicted_labels = clf.predict(X_test)
    model_test_labels = np.argmax(model.predict(X_test), axis=1)

    perturbed_data = perturbator(X_test)
    perturbed_labels = clf.predict(perturbed_data)

    num_test_examples = X_test.shape[0]
    depths = get_node_depths(clf.tree_)

    return metric_calculator(predicted_labels, y_test, model_test_labels, perturbed_labels, depths, len(labels),
                             num_test_examples)
