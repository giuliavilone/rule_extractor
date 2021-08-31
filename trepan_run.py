from keras.optimizers import SGD, Adagrad, Adam, Nadam
import numpy as np
from common_functions import perturbator, rule_metrics_calculator, rule_write
from trepan import Tree, Oracle
from sklearn.metrics import accuracy_score


def run_trepan(X_train, X_test, y_train, y_test, discrete_list, dataset_par, model):
    X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
    n_class = dataset_par['classes']

    oracle = Oracle(model, n_class, X_train, discrete_list)
    tree_obj = Tree(oracle)

    # build tree with TREPAN
    root = tree_obj.build_tree()
    tree_obj.assign_levels(root, 0)

    # tree_obj.print_tree_levels(root)
    final_labels = tree_obj.leaf_values(root)
    print(final_labels)
    tree_obj.print_tree_rule(root)
    final_rules = tree_obj.rule_list(root)
    print(final_rules)

    # calculate metrics
    num_test_examples = X_test.shape[0]

    predi_torch = np.argmax(model.predict(X_test), axis=1)
    perturbed_data = perturbator(X_test)

    rule_labels = []
    perturbed_labels = []
    for i in range(0, num_test_examples):
        instance = X_test[i, :]
        instance_label = tree_obj.predict(instance, root)
        rule_labels.append(instance_label)
        perturbed_instance = perturbed_data[i, :]
        perturbed_labels.append(tree_obj.predict(perturbed_instance, root))

    rule_write('TREPAN_', final_rules, dataset_par)
    correctness = accuracy_score(y_test, rule_labels)
    fidelity = accuracy_score(predi_torch, rule_labels)
    robustness = accuracy_score(rule_labels, perturbed_labels)
    rule_n = len(final_rules)
    avg_length = 0
    for item in final_rules:
        avg_length += sum([len(d['n']) for d in item])
    avg_length = avg_length / rule_n
    class_fraction = len(set(final_labels)) / n_class
    print("Completeness of the ruleset is: " + str(1))
    print("Correctness of the ruleset is: " + str(correctness))
    print("Fidelity of the ruleset is: " + str(fidelity))
    print("Robustness of the ruleset is: " + str(robustness))
    print("Number of rules : " + str(rule_n))
    print("Average rule length: " + str(avg_length))
    print("Fraction overlap: " + str(0))
    print("Fraction of classes: " + str(class_fraction))
    return [1, correctness, fidelity, robustness, rule_n, avg_length, 0, class_fraction]

