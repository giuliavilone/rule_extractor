# Use scikit-learn to grid search the number of neurons
from scipy.io import arff
import pandas as pd
import numpy
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from sklearn.preprocessing import LabelEncoder
import sys


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
le = LabelEncoder()
load_arff = False
if load_arff:
    dataset, meta = arff.loadarff('datasets-UCI/UCI/waveform-5000.arff')
    label_col = 'class'
    dataset = pd.DataFrame(dataset)
    dataset = dataset.dropna()

    for item in range(len(meta.names())):
        item_name = meta.names()[item]
        item_type = meta.types()[item]
        if item_type == 'nominal':
            dataset[item_name] = le.fit_transform(dataset[item_name].tolist())
    # split into input (X) and output (Y) variables
    X = dataset.drop(columns=[label_col]).to_numpy()
    Y = le.fit_transform(dataset[label_col].tolist())
else:
    dataset = pd.read_csv('datasets-UCI/new_datasets/poker_hand.csv')
    col_types = dataset.dtypes
    for index, value in col_types.items():
        if value in ('object', 'bool'):
            dataset[index] = le.fit_transform(dataset[index].tolist())
    label_col = 'class'
    # split into input (X) and output (Y) variables
    # dataset = dataset.drop(columns=['sequence_name'])
    X = dataset.drop(columns=[label_col]).to_numpy()
    Y = le.fit_transform(dataset[label_col].tolist())

INPUT_DIM = X.shape[1]
OUT_CLASS = len(set(Y))
print(X.shape)
print(set(Y))
print(len(set(Y)))

# -------------- Tuning the optimizer -----------------#
# Function to create model, required for KerasClassifier
def create_model(neurons=20, optimizer ='adam'):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=INPUT_DIM, activation='sigmoid'))
    model.add(Dense(OUT_CLASS, activation="softmax"))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10000, verbose=0)
# define the grid search parameters
optimizer = ['Adam', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam', 'SGD', 'RMSprop']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
grid_result = grid.fit(X, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

best_optimizer = grid_result.best_params_['optimizer']


# -------------- Optimizing weight initialization -----------------#
def create_model2(neurons=20, optimizer=best_optimizer, init_mode='uniform'):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=INPUT_DIM, activation='sigmoid', kernel_initializer=init_mode))
    model.add(Dense(OUT_CLASS, kernel_initializer=init_mode, activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# create model
model = KerasClassifier(build_fn=create_model2, epochs=100, batch_size=10000, verbose=0)
# define the grid search parameters
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
grid_result = grid.fit(X, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

best_init_mode = grid_result.best_params_['init_mode']


# -------------- Optimizing neuron activation function -----------------#
def create_model3(neurons=20, optimizer=best_optimizer, init_mode=best_init_mode, activation='relu'):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=INPUT_DIM, activation=activation, kernel_initializer=init_mode))
    model.add(Dense(OUT_CLASS, kernel_initializer=init_mode, activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# create model
model = KerasClassifier(build_fn=create_model3, epochs=100, batch_size=10000, verbose=0)
# define the grid search parameters
activation = ['relu', 'softmax', 'softplus', 'softsign', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
grid_result = grid.fit(X, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

best_activation = grid_result.best_params_['activation']


# -------------- Tuning dropout regularization -----------------#
def create_model4(neurons=20, optimizer=best_optimizer, init_mode=best_init_mode, activation=best_activation,
                  dropout_rate=0.0, weight_constraint=0):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=INPUT_DIM, activation=activation, kernel_initializer=init_mode,
                    kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(OUT_CLASS, kernel_initializer=init_mode, activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# create model
model = KerasClassifier(build_fn=create_model4, epochs=100, batch_size=10000, verbose=0)
# define the grid search parameters
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
grid_result = grid.fit(X, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

best_dropout_rate = grid_result.best_params_['dropout_rate']
best_weight_constraint = grid_result.best_params_['weight_constraint']

# -------------- Optimizing batch size and epochs -----------------#
def create_model5(neurons=20, optimizer=best_optimizer, init_mode=best_init_mode, activation=best_activation,
                  dropout_rate=best_dropout_rate, weight_constraint=best_weight_constraint):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=INPUT_DIM, activation=activation, kernel_initializer=init_mode,
                    kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(OUT_CLASS, activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# create model
model = KerasClassifier(build_fn=create_model5, verbose=0)
# define the grid search parameters
batch_size = [5000, 10000]
epochs = [250, 500]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
grid_result = grid.fit(X, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

best_batch_size = grid_result.best_params_['batch_size']
best_epochs = grid_result.best_params_['epochs']


# -------------- Optimizing number of hidden neurons -----------------#
# create model
model = KerasClassifier(build_fn=create_model5, epochs=best_epochs, batch_size=best_batch_size, verbose=0)
# define the grid search parameters
neurons = [20, 30, 40, 50]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
