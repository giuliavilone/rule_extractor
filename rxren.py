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

# Global variable
TOLERANCE = 0.01
MODEL_NAME = 'rxren_model.h5'


# Functions
def network_pruning(m, w, cX, cy):
    """
    This function removes the insignificant input neurons
    :param m: model
    :param w: model's weights
    :param t: theta
    :param cX: correctX
    :param cy: correcty
    :return: pruned weights
    """
    retw = copy.deepcopy(w)
    retcX = copy.deepcopy(cX)
    # Theta is initialised to be equal to the length of the correct cases as the assumption is that the removal of a
    # node will lead to all the cases to be misclassified. This number will be reduced in the for loop to the minimum
    # number of errors
    theta = len(cy)
    insignificant_neurons = []
    for i in range(w[0].shape[0]):
        if not np.array_equal(w[0][i], np.zeros(3)):  # Avoiding working on already insignificant nodes
            new_w = copy.deepcopy(w)
            new_w[0][i] = np.zeros(3)
            m.set_weights(new_w)
            res = m.predict(cX)
            res = np.argmax(res, axis=1)
            misclassified = cX[[res[i] != cy[i] for i in range(len(cy))]]
            misclassified = misclassified[:, i]
            err = len(misclassified)
            if err <= theta:
                theta = err
                insignificant_neurons.append(i)
    for item in reversed(insignificant_neurons):
        retw[0] = np.delete(retw[0], item, 0)
        retcX = np.delete(retcX, item, 1)
    return retw, retcX, insignificant_neurons

def model_builder(input_shape):
    model = Sequential()
    model.add(Dense(3, input_dim=input_shape, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Main code
data = arff.loadarff('datasets-UCI/UCI/diabetes.arff')
data = pd.DataFrame(data[0])

# Separating independent variables from the target one
X = data.drop(columns=['class']).to_numpy()
le = LabelEncoder()
y = le.fit_transform(data['class'].tolist())

# define model
model = Sequential()
model.add(Dense(3, input_dim=X.shape[-1], activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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

# Accuracy on training dataset
pruning = True
while pruning:
    new_weights, new_correctX, ins_index = network_pruning(model, weights, correctX, correcty)
    if new_weights[0].shape[0] > 0:
        model = model_builder(new_weights[0].shape[0])
        model.set_weights(new_weights)
        X_train = np.delete(X_train, ins_index, 1)
        new_res = model.predict(X_train)
        new_res = np.argmax(new_res, axis=1)
        new_acc = accuracy_score(new_res, y_train)
        if new_acc >= acc - TOLERANCE:
            weights = new_weights
            correctX = new_correctX
        else:
            pruning = False
    else:
        pruning = False
