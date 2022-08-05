# Use scikit-learn to grid search the number of neurons
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from scikeras.wrappers import KerasClassifier
from common_functions import DatasetUploader
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


parameters = pd.read_csv('datasets/summary.csv')
dataset_par = parameters.iloc[14]
label_col = dataset_par['output_name']
print('--------------------------------------------------')
print(dataset_par['dataset'])
print('--------------------------------------------------')
dataset = DatasetUploader(dataset_par['dataset'], 'datasets/',
                          cross_split=2,
                          target_var=label_col,
                          apply_smote=True,
                          smote_sampling_strategy='not minority',
                          data_normalization=True)

X, _, Y, _ = dataset.stratified_k_fold(best_split=0)
X, Y = X[0], Y[0]

INPUT_DIM = X.shape[1]
OUT_CLASS = len(set(Y))
print(X.shape)
print(set(Y))
print(len(set(Y)))


best_optimizer = dataset_par['optimizer']
best_init_mode = dataset_par['init_mode']
best_activation = dataset_par['activation']
best_dropout_rate = dataset_par['dropout_rate']
best_batch_size = dataset_par['batch_size']
best_neurons = dataset_par['neurons']
best_epochs = 50
initial_neurons = 100
initial_batch_size = 10


# -------------- Tuning the optimizer -----------------#
# Function to create model, required for KerasClassifier
def create_model(neurons=initial_neurons, optimizer ='adam', dropout_rate=0.0):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=INPUT_DIM, activation='sigmoid'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation='sigmoid'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(OUT_CLASS, activation="softmax"))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


if pd.isna(best_optimizer):
    # create model
    model = KerasClassifier(model=create_model, epochs=50, batch_size=initial_batch_size, verbose=0)
    # define the grid search parameters
    # optimizer = ['Adam', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam', 'SGD', 'RMSprop']
    optimizer = ['Adam', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']
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
def create_model1(neurons=initial_neurons, optimizer=best_optimizer, init='uniform', dropout_rate=0.0):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=INPUT_DIM, activation='sigmoid', kernel_initializer=init))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation='sigmoid', kernel_initializer=init))
    model.add(Dropout(dropout_rate))
    model.add(Dense(OUT_CLASS, kernel_initializer=init, activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


if pd.isna(best_init_mode):
    # create model
    model = KerasClassifier(model=create_model1, epochs=50, batch_size=initial_batch_size, verbose=0)
    # define the grid search parameters
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal',
                 'he_uniform']
    param_grid = dict(model__init=init_mode)
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
def create_model2(neurons=initial_neurons, optimizer=best_optimizer, init=best_init_mode, activation='relu',
                  dropout_rate=0.0):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=INPUT_DIM, activation=activation, kernel_initializer=init))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation=activation, kernel_initializer=init))
    model.add(Dropout(dropout_rate))
    model.add(Dense(OUT_CLASS, kernel_initializer=init, activation='softmax'))
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


if pd.isna(best_activation):
    # create model
    model = KerasClassifier(model=create_model2, epochs=50, batch_size=initial_batch_size, verbose=0)
    # define the grid search parameters
    activation = ['relu', 'softmax', 'softplus', 'softsign', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    param_grid = dict(model__activation=activation)
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
# create model
def create_model3(neurons=initial_neurons, optimizer=best_optimizer, init=best_init_mode,
                  activation=best_activation, dropout_rate=0.0):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=INPUT_DIM, activation=activation,
                    kernel_initializer=init
                    )
              )
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation=activation,
                    kernel_initializer=init
                    )
              )
    model.add(Dropout(dropout_rate))
    model.add(Dense(OUT_CLASS, kernel_initializer=init,
                    activation='softmax'
                    )
              )
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy']
                  )
    return model


if pd.isna(best_dropout_rate):
    model = KerasClassifier(model=create_model3, epochs=50, batch_size=500, verbose=0)
    # define the grid search parameters
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    print(dropout_rate)
    param_grid = dict(model__dropout_rate=dropout_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
    grid_result = grid.fit(X, Y)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    best_dropout_rate = grid_result.best_params_['model__dropout_rate']


# -------------- Optimizing batch size and epochs -----------------#
def create_model4(neurons=initial_neurons, optimizer=best_optimizer,
                  init=best_init_mode,
                  activation=best_activation,
                  dropout_rate=best_dropout_rate):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=INPUT_DIM, activation=activation,
                    kernel_initializer=init
                    )
    )
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation=activation,
                    kernel_initializer=init
                    )
    )
    model.add(Dropout(dropout_rate))
    model.add(Dense(OUT_CLASS, kernel_initializer=init,
                    activation='softmax'
                    )
    )
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy']
                  )
    return model


if pd.isna(best_batch_size):
    # create model
    model = KerasClassifier(model=create_model4, verbose=0)
    # define the grid search parameters
    # batch_size = [1000, 5000, 10000]
    # epochs = [250, 500, 1000]
    batch_size = [8, 16, 32]
    epochs = [100]
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
else:
    # When uploaded from the csv file, this number is considered as a float, so it must be converted into an integer
    best_batch_size = int(best_batch_size)

# -------------- Optimizing number of hidden neurons -----------------#
# create model
if pd.isna(best_neurons):
    model = KerasClassifier(model=create_model4, epochs=best_epochs,
                            batch_size=best_batch_size, verbose=0)
    # define the grid search parameters
    # neurons = [2**x for x in range(1, 10) if 2**x < X.shape[1] * 2]
    neurons = [256, 512]
    param_grid = dict(model__neurons=neurons)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
    grid_result = grid.fit(X, Y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    best_neurons = grid_result.best_params_['neurons']


out_list = [[dataset_par['dataset'], best_optimizer, best_init_mode, best_activation, best_dropout_rate,
             best_batch_size, best_neurons
             ]]
out_df = pd.DataFrame(out_list, columns=['dataset', 'optimizer', 'init_mode', 'activation', 'dropout_rate',
                                         'batch_size', 'neurons']
                      )
save_name = str(dataset_par['dataset']) + ".csv"
path = F"datasets/grid_search_output/{save_name}"

out_df.to_csv(path)
