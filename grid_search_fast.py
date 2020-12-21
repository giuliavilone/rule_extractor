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
from imblearn.over_sampling import SMOTE



# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
def dataset_uploader(item, target_var='class', remove_columns=True):
    le = LabelEncoder()
    file_name = item['dataset']
    feat_to_be_deleted = {'bank': ['euribor3m', 'emp.var.rate'],
                          'cover_type': ['Wilderness_Area1', 'Aspect', 'Hillshade_9am', 'Hor_Dist_Hydrology'],
                          'letter_recognition': ['y-box', 'high', 'width'],
                          'online_shoppers_intention': ['BounceRates', 'ProductRelated', 'Inform_Duration'],
                          'avila': ['F10'],
                          'credit_card_default': ['BILL_AMT6', 'BILL_AMT5', 'BILL_AMT4', 'BILL_AMT3', 'BILL_AMT2',
                                                  'PAY_6', 'PAY_5', 'PAY_4', 'PAY_3', 'PAY_2'],
                          'eeg_eye_states': ['P7', 'F8', 'T8', 'P8', 'FC5'],
                          'skin_nonskin': ['B'],
                          'htru': ['mean_dm_snr_curve', 'kurtosis_dm_snr_curve', 'skewness_profile', 'mean_profile'],
                          'occupancy': ['HumidityRatio', 'Temperature'],
                          'shuttle': ['S7', 'S8', 'S9']
                          }
    dataset = pd.read_csv('datasets-UCI/new_datasets/' + item['dataset'] + '.csv')
    dataset = dataset.dropna().reset_index(drop=True)
    if remove_columns:
        if file_name in feat_to_be_deleted.keys():
            columns_to_be_deleted = [item for item in dataset.columns.tolist() if item in feat_to_be_deleted[file_name]]
            dataset = dataset.drop(columns=columns_to_be_deleted)
    col_types = dataset.dtypes
    out_disc_temp = []
    for index, value in col_types.items():
        if value in ['object', 'bool']:
            if index != target_var:
                dataset = pd.get_dummies(dataset, columns=[index])
                out_disc_temp.append(index)
    # The number of the discrete features must take into account the new dummy columns
    out_disc = []
    for col in out_disc_temp:
        out_disc += [i for i, v in enumerate(dataset.columns) if v.find(col + '_') > -1]
    out_cont = [i for i, v in enumerate(dataset.columns) if i not in out_disc]

    # Separating independent variables from the target one
    y_ret = le.fit_transform(dataset[target_var].tolist())
    X_ret = dataset.drop(columns=[target_var])
    # X_ret, y_ret = SMOTE().fit_sample(X_ret, y_ret)
    return X_ret, y_ret


parameters = pd.read_csv('datasets-UCI/new_datasets/summary_new2.csv')
dataset_par = parameters.iloc[2]
print('--------------------------------------------------')
print(dataset_par['dataset'])
print('--------------------------------------------------')
label_col = 'class'

X, Y = dataset_uploader(dataset_par)

INPUT_DIM = X.shape[1]
OUT_CLASS = len(set(Y))
print(X.shape)
print(set(Y))
print(len(set(Y)))

best_optimizer = dataset_par['optimizer']
best_init_mode = dataset_par['init_mode']
best_activation = dataset_par['activation']


# -------------- Tuning dropout regularization -----------------#
# create model
def create_model1(neurons=20, optimizer=best_optimizer, init_mode=best_init_mode,
                  activation=best_activation, dropout_rate=0.0):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=INPUT_DIM, activation=activation,
                    kernel_initializer=init_mode
                    )
    )
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation=activation,
                    kernel_initializer=init_mode
                    )
    )
    model.add(Dropout(dropout_rate))
    model.add(Dense(OUT_CLASS, kernel_initializer=init_mode,
                    activation='softmax'
                    )
    )
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy']
                  )
    return model


model = KerasClassifier(build_fn=create_model1, epochs=50, batch_size=500, verbose=0)
# define the grid search parameters
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
print(dropout_rate)
param_grid = dict(dropout_rate=dropout_rate)
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


# -------------- Optimizing batch size and epochs -----------------#
def create_model2(neurons=20, optimizer=best_optimizer,
                  init_mode=best_init_mode,
                  activation=best_activation,
                  dropout_rate=best_dropout_rate):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=INPUT_DIM, activation=activation,
                    kernel_initializer=init_mode
                    )
    )
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation=activation,
                    kernel_initializer=init_mode
                    )
    )
    model.add(Dropout(dropout_rate))
    model.add(Dense(OUT_CLASS, kernel_initializer=init_mode,
                    activation='softmax'
                    )
    )
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy']
                  )
    return model


# create model
model = KerasClassifier(build_fn=create_model2, verbose=0)
# define the grid search parameters
# batch_size = [1000, 5000, 10000]
# epochs = [250, 500, 1000]
batch_size = [16, 32, 64]
epochs = [50]
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
model = KerasClassifier(build_fn=create_model2, epochs=best_epochs,
                        batch_size=best_batch_size, verbose=0)
# define the grid search parameters
neurons = [2**x for x in range(1,10) if 2**x < X.shape[1]]
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


out_list = [[dataset_par['dataset'], best_dropout_rate, best_batch_size,
            grid_result.best_params_['neurons']
            ]]
out_df = pd.DataFrame(out_list, columns = ['dataset', 'dropout_rate',
                                           'batch_size', 'neurons'])
save_name = str(dataset_par['dataset']) + ".csv"
path = F"datasets-UCI/{save_name}"

out_df.to_csv(path)
