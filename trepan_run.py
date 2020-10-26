import pandas as pd
from scipy.io import arff
#from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np
from common_functions import create_model, model_train, perturbator, dataset_uploader, rule_metrics_calculator
from trepan import Tree, Oracle
from sklearn.utils import resample
from keras.models import load_model
import sys



###########################################

def vote_db_modifier(indf):
    """
    Modify the vote database by replacing yes/no answers with boolean
    :type indf: Pandas dataframe
    """
    indf.replace(b'y', 1, inplace=True)
    indf.replace(b'n', 0, inplace=True)
    indf.replace(b'?', 0, inplace=True)
    return indf


original_study = False
if original_study:
    # X = data.drop(columns=['physician-fee-freeze', label_col])
    data, meta = arff.loadarff('datasets-UCI/UCI/hepatitis.arff')
    label_col = 'Class'
    data = pd.DataFrame(data)
    data = data.dropna().reset_index(drop=True)

    le = LabelEncoder()
    for item in range(len(meta.names())):
        item_name = meta.names()[item]
        item_type = meta.types()[item]
        if item_type == 'nominal':
            data[item_name] = le.fit_transform(data[item_name].tolist())

    n_cross_val = 2
    n_class = 2
    n_nodes = 3
    X = data.drop(columns=[label_col])
    le = LabelEncoder()
    y = le.fit_transform(data[label_col].tolist())

    # Replacing yes/no answers with 1/0
    X = vote_db_modifier(X)
    X = X.to_numpy()
    # Create the object to perform cross validation
    #skf = StratifiedKFold(n_splits=n_cross_val, random_state=7, shuffle=True)

    ix = [i for i in range(len(X))]
    train_index = resample(ix, replace=True, n_samples=int(len(X)*0.5), random_state=0)
    val_index = [x for x in ix if x not in train_index]
    X_train, X_test = X[train_index], X[val_index]
    y_train, y_test = y[train_index], y[val_index]
    # X_train = X_train.to_numpy()
    # X_test = X_test.to_numpy()
    y_train = to_categorical(y_train, num_classes=n_class)
    y_test = to_categorical(y_test, num_classes=n_class)
    model = create_model(X_train, n_class, n_nodes)
    model = model_train(X_train, y_train, X_test, y_test, model, 'trepan_model.h5')
    y_test = np.argmax(y_test, axis=0)
else:
    parameters = pd.read_csv('datasets-UCI/Used_data/summary.csv')
    dataset = parameters.iloc[3]
    print('--------------------------------------------------')
    print(dataset['dataset'])
    print('--------------------------------------------------')
    X_train, X_test, y_train, y_test, _, _ = dataset_uploader(dataset)
    X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
    model = load_model('trained_model_' + dataset['dataset'] + '.h5')
    n_class = dataset['classes']

oracle = Oracle(model, n_class, X_train)
tree_obj = Tree(oracle)

# build tree with TREPAN
root = tree_obj.build_tree()
tree_obj.assign_levels(root, 0)

tree_obj.print_tree_levels(root)
final_rules = tree_obj.leaf_values(root)

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
