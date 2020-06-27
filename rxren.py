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
from sklearn.metrics import accuracy_score
import copy
import sys
from itertools import groupby

# Global variable
TOLERANCE = 0.01
MODEL_NAME = 'rxren_model.h5'


# Functions
def input_delete(insignificant_index, inDf, inWeight=None):
    """
    Delete the variable of the input vector corresponding the insignificant input neurons and, if required, the
    corresponding weights of the neural network
    :param inDf:
    :param inDf:
    :param inWeight:
    :return: the trimmed weights and input vector
    """
    outDf = copy.deepcopy(inDf)
    outDf = np.delete(outDf, insignificant_index, 1)
    outWeight = None
    if inWeight is not None:
        outWeight = copy.deepcopy(inWeight)
        outWeight[0] = np.delete(outWeight[0], insignificant_index, 0)
    return outDf, outWeight

def model_pruned_prediction(inputX,w):
    new_m = model_builder(w[0].shape[0])
    new_m.set_weights(w)
    results = new_m.predict(inputX)
    results = np.argmax(results, axis=1)
    return results

def network_pruning(w, cX, cy, train_x, train_y, accuracy):
    """
    This function removes the insignificant input neurons
    :param w: model's weights
    :param t: theta
    :param cX: correctX
    :param cy: correcty
    :return: pruned weights
    """
    insignificant_neurons = []
    missclassified_list = []
    error_list = []
    new_acc = 0
    for i in range(w[0].shape[0]):
        newCx, new_w = input_delete(i, cX, inWeight=w)
        res = model_pruned_prediction(newCx, new_w)
        misclassified = [i for i in range(len(cy)) if res[i] != cy[i]]
        missclassified_list.append(misclassified)
        err = len(misclassified)
        error_list.append(err)
    pruning = True
    theta = min(error_list)
    while pruning:
        insignificant_neurons = insignificant_neurons + [i for i, e in enumerate(error_list) if e == theta]
        new_train_x, new_w = input_delete(insignificant_neurons, train_x, inWeight=w)
        new_res = model_pruned_prediction(new_train_x, new_w)
        new_acc = accuracy_score(new_res, train_y)
        if new_acc >= accuracy - TOLERANCE:
            new_error_list = [e for i, e in enumerate(error_list) if e > theta]
            # Leaving at least two significant neurons
            if len(new_error_list) > 2:
                theta = min(new_error_list)
            else:
                pruning = False
        else:
            pruning = False
    return missclassified_list, sorted(insignificant_neurons), new_acc, theta


def model_builder(input_shape):
    mod = Sequential()
    mod.add(Dense(3, input_dim=input_shape, activation='relu'))
    mod.add(Dense(2, activation='softmax'))
    mod.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return mod

def rule_limits_calculator(c_x, c_y, missclassified_list, significant_neurons, error, alpha=0.5):
    c_tot = np.column_stack((c_x, c_y))
    grouped_miss_class = []
    for i in significant_neurons:
        miss_class = c_tot[missclassified_list[i]]
        # Splitting the misclassified input values according to their output classes
        grouped_miss_class = grouped_miss_class + [{'neuron': i, 'class': k,
                               'limits': [min(miss_class[:, i][miss_class[:, -1] == k]),
                                max(miss_class[:, i][miss_class[:, -1] == k])]}
                                for k in np.unique(miss_class[:, -1])
                                if len(miss_class[:, i][miss_class[:, -1] == k]) > (error * alpha)]
    return grouped_miss_class


# Main code
data = arff.loadarff('datasets-UCI/UCI/diabetes.arff')
data = pd.DataFrame(data[0])

# Separating independent variables from the target one
X = data.drop(columns=['class']).to_numpy()
le = LabelEncoder()
y = le.fit_transform(data['class'].tolist())

# define model
model = model_builder(X.shape[-1])

checkpointer = ModelCheckpoint(filepath=MODEL_NAME,
                                  save_weights_only=False,
                                  monitor='loss',
                                  save_best_only=True,
                                  verbose=1)

ix = [i for i in range(len(X))]
train_index = resample(ix, replace=True, n_samples=int(len(X)*0.7), random_state=0)
val_index = [x for x in ix if x not in train_index]
X_train, X_test = X[train_index], X[val_index]
y_train, y_test = y[train_index], y[val_index]

model_train = False
if model_train:
    history = model.fit(X_train, to_categorical(y_train, num_classes=2),
                            validation_data=(X_test, to_categorical(y_test, num_classes=2)),
                            epochs=500,
                            callbacks=[checkpointer])

model = load_model(MODEL_NAME)
weights = np.array(model.get_weights())
results = model.predict(X)
results = np.argmax(results, axis=1)
correctX = X[[results[i] == y[i] for i in range(len(y))]]
correcty = y[[results[i] == y[i] for i in range(len(y))]]

new_res = model.predict(X_train)
new_res = np.argmax(new_res, axis=1)
acc = accuracy_score(new_res, y_train)
print(acc)

miss_list, ins_index, new_accuracy, err = network_pruning(weights, correctX, correcty, X_train, y_train, acc)
significant_index = [i for i in range(weights[0].shape[0]) if i not in ins_index]
print(ins_index)
print(new_accuracy)
print(err)
rule_limits = rule_limits_calculator(correctX, correcty, miss_list, significant_index, err, alpha=0.5)
print(rule_limits)

