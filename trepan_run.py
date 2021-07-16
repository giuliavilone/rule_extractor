from keras.optimizers import SGD, Adagrad, Adam, Nadam
import numpy as np
from common_functions import perturbator, rule_metrics_calculator
from trepan import Tree, Oracle


def run_trepan(X_train, X_test, y_train, y_test, discrete_list, dataset_par, model):
    X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
    n_class = dataset_par['classes']

    oracle = Oracle(model, n_class, X_train, discrete_list)
    tree_obj = Tree(oracle)

    # build tree with TREPAN
    root = tree_obj.build_tree()
    tree_obj.assign_levels(root, 0)

    # tree_obj.print_tree_levels(root)
    final_rules = tree_obj.leaf_values(root)
    print(final_rules)
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

    return rule_metrics_calculator(X_test, y_test, predi_torch, final_rules, dataset_par['classes'])

