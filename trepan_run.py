import pandas as pd
from keras.utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adam, Nadam
import numpy as np
from common_functions import create_model, model_train, perturbator, dataset_uploader, rule_metrics_calculator
from trepan import Tree, Oracle
from keras.models import load_model
import sys

# Main code
parameters = pd.read_csv('datasets-UCI/Used_data/summary.csv')
dataset_par = parameters.iloc[11]
print('--------------------------------------------------')
print(dataset_par['dataset'])
print('--------------------------------------------------')
original_study = False
if original_study:
    X_train, X_test, y_train, y_test, discrete_list, _ = dataset_uploader(dataset_par)
    X_train = X_train[0]
    X_test = X_test[0]
    y_train = y_train[0]
    y_test = y_test[0]
    X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
    n_class = 2
    model = create_model(X_train, dataset_par['classes'], dataset_par['neurons'], eval(dataset_par['optimizer']),
                         dataset_par['init_mode'], dataset_par['activation'], dataset_par['dropout_rate'],
                         weight_constraint=eval(dataset_par['weight_constraint'])
                         )
    model_train(X_train, to_categorical(y_train, num_classes=dataset_par['classes']),
                X_test, to_categorical(y_test, num_classes=dataset_par['classes']), model,
                'trepan_model.h5', n_epochs=dataset_par['epochs'], batch_size=dataset_par['batch_size'])
else:
    X_train_list, X_test_list, y_train_list, y_test_list, discrete_list, _ = dataset_uploader(dataset_par,
                                                                                              apply_smothe=False)
    metric_list = []
    for ix in range(len(X_train_list)):
        X_train = X_train_list[ix]
        X_test = X_test_list[ix]
        y_train = y_train_list[ix]
        y_test = y_test_list[ix]
        # X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
        model = load_model('trained_model_' + dataset_par['dataset'] + '_' + str(ix) + '.h5')
        n_class = dataset_par['classes']

        oracle = Oracle(model, n_class, X_train, discrete_list)
        tree_obj = Tree(oracle)

        # build tree with TREPAN
        root = tree_obj.build_tree()
        tree_obj.assign_levels(root, 0)

        # tree_obj.print_tree_levels(root)
        final_rules = tree_obj.leaf_values(root)
        # print(final_rules)
        tree_obj.print_tree_rule(root)

        # calculate metrics
        num_test_examples = X_test.shape[0]

        predi_tree = []
        predi_torch = np.argmax(model.predict(X_test), axis=1)
        perturbed_data = perturbator(X_test)
        perturbed_labels = np.argmax(model.predict(perturbed_data), axis=1)

        rule_labels = []
        for i in range(0, num_test_examples):
            instance = X_test[i, :]
            instance_label = tree_obj.predict(instance, root)
            rule_labels.append(instance_label)
            predi_tree.append(instance_label)

        completeness = len(predi_tree) / num_test_examples
        avg_length = sum(final_rules) / len(final_rules)
        overlap = 0

        metric_list = rule_metrics_calculator(num_test_examples, y_test, rule_labels, predi_torch, perturbed_labels,
                                              len(final_rules), completeness, avg_length, overlap,
                                              dataset_par['classes'])
