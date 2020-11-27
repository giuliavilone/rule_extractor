import pandas as pd
from keras.utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adam, Nadam
import numpy as np
from common_functions import create_model, model_train, perturbator, dataset_uploader, rule_metrics_calculator
from trepan import Tree, Oracle
from keras.models import load_model
import sys


def vote_db_modifier(indf):
    """
    Modify the vote database by replacing yes/no answers with boolean
    :type indf: Pandas dataframe
    """
    indf.replace(b'y', 1, inplace=True)
    indf.replace(b'n', 0, inplace=True)
    indf.replace(b'?', 0, inplace=True)
    return indf


# Main code
parameters = pd.read_csv('datasets-UCI/Used_data/summary.csv')
dataset_par = parameters.iloc[11]
print('--------------------------------------------------')
print(dataset_par['dataset'])
print('--------------------------------------------------')
original_study = True
if original_study:
    X_train, X_test, y_train, y_test, discrete_list, _ = dataset_uploader(dataset_par, train_split=dataset_par['split'])
    # Translating the name of the discrete columns into their position number
    discrete_list = [i for i, v in enumerate(X_train.columns) if v in discrete_list]
    X_train, X_test = X_train.to_numpy(), X_test.to_numpy()

    n_cross_val = 2
    n_class = 2
    n_nodes = 3
    model = create_model(X_train, dataset_par['classes'], dataset_par['neurons'], eval(dataset_par['optimizer']),
                         dataset_par['init_mode'], dataset_par['activation'], dataset_par['dropout_rate'],
                         weight_constraint=eval(dataset_par['weight_constraint'])
                         )
    model_train(X_train, to_categorical(y_train, num_classes=dataset_par['classes']),
                X_test, to_categorical(y_test, num_classes=dataset_par['classes']), model,
                'trepan_model.h5', n_epochs=dataset_par['epochs'], batch_size=dataset_par['batch_size'])
else:
    X_train, X_test, y_train, y_test, _, _ = dataset_uploader(dataset_par)
    X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
    model = load_model('trained_model_' + dataset_par['dataset'] + '.h5')
    n_class = dataset_par['classes']


oracle = Oracle(model, n_class, X_train, discrete_list)
tree_obj = Tree(oracle)

# build tree with TREPAN
root = tree_obj.build_tree()
tree_obj.assign_levels(root, 0)

tree_obj.print_tree_levels(root)
final_rules = tree_obj.leaf_values(root)
print(final_rules)
tree_obj.print_tree_rule(root)
sys.exit()

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

rule_metrics_calculator(num_test_examples, y_test, rule_labels, predi_torch, perturbed_labels,
                        len(final_rules), completeness, avg_length)
