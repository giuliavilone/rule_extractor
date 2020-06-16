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
train_index = resample(ix, replace=True, n_samples=int(len(X)*0.7))
val_index = [x for x in ix if x not in train_index]
X_train, X_test = X[train_index], X[val_index]
y_train, y_test = y[train_index], y[val_index]

model_train = False
if model_train:
    history = model.fit(X_train, to_categorical(y_train, num_classes=2),
                            validation_data=(X_test, to_categorical(y_test, num_classes=2)),
                            epochs=800,
                            callbacks=[checkpointer])

model = load_model(MODEL_NAME)
results = model.predict(X)
results = np.argmax(results, axis=1)
correctX = X[[results[i] == y[i] for i in range(len(y))]]
correcty = y[[results[i] == y[i] for i in range(len(y))]]

# Accuracy on training dataset
train_res = model.predict(X_train)
train_res = np.argmax(train_res, axis=1)
acc = accuracy_score(train_res, y_train)
print(acc)
weights = np.array(model.get_weights())
insignificant_neuron = []
for i in range(weights[0].shape[0]):
    print('Working on neuron ' + str(i))
    new_weights = copy.deepcopy(weights)
    new_weights[0][i] = np.zeros(3)
    model.set_weights(new_weights)
    new_results = model.predict(correctX)
    new_results = np.argmax(new_results, axis=1)
    misclassified = correctX[[new_results[i] != correcty[i] for i in range(len(correcty))]]
    misclassified = misclassified[:,i]
    print(len(misclassified))
    if len(misclassified) > 0:
        print(max(misclassified))
        print(min(misclassified))
    # Accuracy on training dataset
    res = model.predict(X_train)
    res = np.argmax(res, axis=1)
    new_acc = accuracy_score(res, y_train)
    print(new_acc)
    print('--------------------------------------------------')
    if new_acc >= acc - TOLERANCE:
        insignificant_neuron.append(i)

print(insignificant_neuron)

