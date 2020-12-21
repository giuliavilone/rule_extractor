import pandas as pd
from keras.utils import to_categorical
from keras.models import load_model
from keras.optimizers import SGD, Adagrad, Adam, Nadam
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from common_functions import perturbator, create_model, model_train, ensemble_predictions, dataset_uploader
from common_functions import rule_metrics_calculator

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


def synthetic_data_generator(inarr, n_samples):
    """
    Given an input numpy array, the function returns a new array containing random numbers
    generated within the value ranges of the input attributes.
    :param indf:
    :param n_samples: integer number of samples to be generated
    :return: outdf: of synthetic data
    """
    outshape = (n_samples, inarr.shape[1])
    outarr = np.zeros(outshape)
    for column in range(inarr.shape[1]):
        minvalue = min(inarr[:, column])
        maxvalue = max(inarr[:, column])
        outarr[:, column] = np.round(np.random.uniform(minvalue, maxvalue, n_samples), 1)
    return outarr


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


def run_c45_pane(x_tot, y_tot, x_test):
    clf = DecisionTreeClassifier()
    clf = clf.fit(x_tot, y_tot)

    rules = export_text(clf)
    string_data = export_graphviz(clf, out_file=None)

    # Showing the rules
    print_decision_tree(clf)
    # print(rules)
    # print(string_data)

    predicted_labels = clf.predict(x_test)
    if original_study:
        model_test_labels = ensemble_predictions(members, x_test)[0]
    else:
        model_test_labels = np.argmax(model.predict(x_test), axis=1)

    perturbed_data = perturbator(x_test)
    perturbed_labels = clf.predict(perturbed_data)

    num_test_examples = x_test.shape[0]
    completeness = len(predicted_labels) / num_test_examples
    depths = get_node_depths(clf.tree_)
    overlap = 0

    ret = rule_metrics_calculator(num_test_examples, y_test, predicted_labels, model_test_labels, perturbed_labels,
                                  clf.get_n_leaves(), completeness, np.mean(depths), overlap, dataset_par['classes'])
    return ret


# Main code
parameters = pd.read_csv('datasets-UCI/Used_data/summary_new.csv')
# Global variables
original_study = False
if original_study:
    n_members = 10
    for member in range(n_members):
        X_train, X_test, y_train, y_test, _, _ = dataset_uploader(dataset_par, apply_smothe=False)
        X_train = X_train[0]
        X_test = X_test[0]
        y_train = y_train[0]
        y_test = y_test[0]
        # define model
        model = create_model(X_train, dataset_par['classes'], dataset_par['neurons'], eval(dataset_par['optimizer']),
                             dataset_par['init_mode'], dataset_par['activation'], dataset_par['dropout_rate'],
                             weight_constraint=eval(dataset_par['weight_constraint'])
                             )
        model_train(X_train, to_categorical(y_train, num_classes=dataset_par['classes']),
                    X_test, to_categorical(y_test, num_classes=dataset_par['classes']), model,
                    'c45_model_'+str(member)+'.h5', n_epochs=dataset_par['epochs'],
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
else:
    metric_list = []
    for ix in range(len(parameters)):
        dataset_par = parameters.iloc[ix]
        X_train_list, X_test_list, y_train_list, y_test_list, _, _ = dataset_uploader(dataset_par, apply_smothe=False)
        print('--------------------------------------------------')
        print(dataset_par['dataset'])
        print('--------------------------------------------------')
        X_train = X_train_list[ix]
        X_test = X_test_list[ix]
        y_train = y_train_list[ix]
        y_test = y_test_list[ix]
        X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
        model = load_model('trained_model_' + dataset_par['dataset'] + '_' + str(ix) + '.h5')
        synth_samples = X_train.shape[0] * 2
        xSynth = synthetic_data_generator(X_train, synth_samples)
        ySynth = np.argmax(model.predict(xSynth), axis=1)
        model_res = np.argmax(model.predict(X_train), axis=1)
        xTot = np.concatenate((X_train, xSynth), axis=0)
        yTot = np.transpose(np.concatenate([model_res, ySynth], axis=0))
        metric_list.append(run_c45_pane(xTot, yTot, X_test))

    pd.DataFrame(metric_list, columns=['complete', 'correctness', 'fidelity', 'robustness', 'rule_n', 'avg_length',
                                       'overlap', 'class_fraction']
                 ).to_csv('c45_metrics.csv')
# This uses the CART algorithm (see https://scikit-learn.org/stable/modules/tree.html)

