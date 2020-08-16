from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_text, export
import sys

# Global variables
n_members = 10


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
        filename = 'c45_model_' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)
    # combining the members via plurality voting
    voted_yhats = np.argmax(yhats, axis=2)
    results = mode(voted_yhats, axis=0)[0]
    return results


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


def perturbator(indf, mu=0, sigma=0.1):
    """
    Add white noise to input dataset
    :type indf: Pandas dataframe
    """
    noise = np.random.normal(mu, sigma, indf.shape)
    return indf + noise

# Main code
data = arff.loadarff('datasets-UCI/UCI/iris.arff')
data = pd.DataFrame(data[0])

# Separating independent variables from the target one
X = data.drop(columns=['class']).to_numpy()
le = LabelEncoder()
y = le.fit_transform(data['class'].tolist())

# define model
model = Sequential()
model.add(Dense(20, input_dim=X.shape[-1], activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

fold_var = 1
for _ in range(n_members):
    # select indexes
    ix = [i for i in range(len(X))]
    train_index = resample(ix, replace=True, n_samples=int(len(X)*0.7))
    val_index = [x for x in ix if x not in train_index]
    X_train, X_test = X[train_index], X[val_index]
    y_train, y_test = y[train_index], y[val_index]
    checkpointer = ModelCheckpoint(filepath='c45_model_'+str(fold_var)+'.h5',
                                  save_weights_only=False,
                                  monitor='loss',
                                  save_best_only=True,
                                  verbose=1)
    history = model.fit(X_train, to_categorical(y_train, num_classes=3),
                        validation_data=(X_test, to_categorical(y_test, num_classes=3)),
                        epochs=50,
                        callbacks=[checkpointer])

    fold_var += 1

# load all models
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

ensemble_res = ensemble_predictions(members, X)
print(accuracy_score(ensemble_res[0], y))

# Same process of REFNE
synth_samples = X.shape[0] * 2
xSynth = synthetic_data_generator(X, synth_samples)
ySynth = ensemble_predictions(members, xSynth)

# Concatenate the synthetic array and the array with the labels predicted by the ensemble
xTot = np.concatenate((X, xSynth), axis=0)
yTot = np.transpose(np.concatenate([ensemble_res, ySynth], axis=1))

# This uses the CART algorithm (see https://scikit-learn.org/stable/modules/tree.html)
clf = DecisionTreeClassifier()
clf = clf.fit(xTot, yTot)
rules = export_text(clf)

# Showing the rules
print_decision_tree(clf)
print(rules)

predicted_labels = clf.predict(X_test)
model_test_labels = ensemble_predictions(members, X_test)[0]

perturbed_data = perturbator(X_test)
perturbed_labels = clf.predict(perturbed_data)

num_test_examples = X_test.shape[0]
correct = 0
fidel = 0
rob = 0

for i in range(0, num_test_examples):
    fidel += (predicted_labels[i] == model_test_labels[i])
    correct += (predicted_labels[i] == y_test[i])
    rob += (predicted_labels[i] == perturbed_labels[i])

fidelity = fidel / num_test_examples
print("Fidelity of the ruleset is : " + str(fidelity))
completeness = len(predicted_labels) / num_test_examples
print("Completeness of the ruleset is : " + str(completeness))
correctness = correct / num_test_examples
print("Correctness of the ruleset is : " + str(correctness))
robustness = rob / num_test_examples
print("Robustness of the ruleset is : " + str(robustness))