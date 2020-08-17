import pandas as pd
from scipy.io import arff
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import numpy as np
from common_functions import create_model, model_trainer, perturbator
from trepan import Tree, Oracle

data = arff.loadarff('datasets-UCI/UCI/vote.arff')
data = pd.DataFrame(data[0])
n_cross_val = 10
n_class = 2
n_nodes = 4

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


X = data.drop(columns=['physician-fee-freeze', 'Class'])
le = LabelEncoder()
y = le.fit_transform(data['Class'].tolist())

# Replacing yes/no answers with 1/0
X = vote_db_modifier(X)

# Create the object to perform cross validation
skf = StratifiedKFold(n_splits=n_cross_val, random_state=7, shuffle=True)


fold_var = 1
for train_index, val_index in skf.split(X,y):
    print('Working on model number ' + str(fold_var))
    X_train, X_test = X[X.index.isin(train_index)], X[X.index.isin(val_index)]
    y_train, y_test = y[train_index], y[val_index]
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = to_categorical(y_train, num_classes=n_class)
    y_test = to_categorical(y_test, num_classes=n_class)
    model = create_model(X_train, n_class, n_nodes)
    model = model_trainer(X_train, y_train, X_test, y_test, model, 'trepan_model.h5')
    oracle = Oracle(model, n_class, X_train)
    tree_obj = Tree(oracle)

    # build tree with TREPAN
    MIN_EXAMPLES_PER_NODE = 30
    MAX_NODES = 200
    root = tree_obj.build_tree()
    tree_obj.assign_levels(root, 0)
    tree_obj.print_tree_levels(root)
    # calculate metrics
    num_test_examples = X_test.shape[0]
    correct = 0
    fidel = 0
    rob = 0
    predi_tree = []
    predi_torch = np.argmax(model.predict(X_test), axis=1)
    perturbed_data = perturbator(X_test)
    perturbed_labels = np.argmax(model.predict(perturbed_data), axis=1)
    y_test = np.argmax(y_test, axis=1)
    for i in range(0, num_test_examples):
        instance = X_test[i, :]
        instance_label = tree_obj.predict(instance, root)
        predi_tree.append(instance_label)
        fidel += (instance_label == predi_torch[i])
        correct += (instance_label == y_test[i])
        perturbed_instance = perturbed_data[i, :]
        perturbed_instance_label = tree_obj.predict(perturbed_instance, root)
        rob += (instance_label == perturbed_instance_label)

    fidelity = fidel / num_test_examples
    print("Fidelity of the ruleset is : " + str(fidelity))
    completeness = len(predi_tree) / num_test_examples
    print("Completeness of the ruleset is : " + str(completeness))
    correctness = correct / num_test_examples
    print("Correctness of the ruleset is : " + str(correctness))
    robustness = rob / num_test_examples
    print("Robustness of the ruleset is : " + str(robustness))

    fold_var += 1