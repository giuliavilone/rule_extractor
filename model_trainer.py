from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE


def create_model(train_x, n_classes, neurons, optimizer='Adam', init_mode='glorot_uniform',
                 activation='sigmoid', dropout_rate=0.0, loss='categorical_crossentropy',
                 out_activation='softmax'
                 ):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=train_x.shape[1], activation=activation,
                    kernel_initializer=init_mode
                    )
              )
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation=activation, kernel_initializer=init_mode
                    )
              )
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_classes, activation=out_activation))
    # Compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def model_train(train_x, train_y, test_x, test_y, model, model_name, n_classes, n_epochs=1000, batch_size=10):
    check_pointer = ModelCheckpoint(filepath=model_name, save_weights_only=False, monitor='accuracy',
                                    save_best_only=True,verbose=1
                                    )
    early_stop = EarlyStopping(monitor='accuracy', patience=5)
    history = model.fit(train_x, to_categorical(train_y, num_classes=n_classes),
                        validation_data=(test_x, to_categorical(test_y, num_classes=n_classes)),
                        batch_size=batch_size, epochs=n_epochs, callbacks=[check_pointer, early_stop]
                        )
    return model, history


def model_creator(item, target_var='class', cross_split=5, remove_columns=True):
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
    le = LabelEncoder()
    dataset = pd.read_csv('datasets-UCI/new_rules/' + item['dataset'] + '.csv')
    dataset = dataset.dropna().reset_index(drop=True)
    if remove_columns and item['dataset'] in feat_to_be_deleted.keys():
        columns_to_be_deleted = [col for col in dataset.columns.tolist() if col in feat_to_be_deleted[item['dataset']]]
        dataset = dataset.drop(columns=columns_to_be_deleted)
    col_types = dataset.dtypes
    for index, value in col_types.items():
        if value in ['object', 'bool']:
            if index != target_var:
                dataset = pd.get_dummies(dataset, columns=[index])

    # Separating independent variables from the target one
    y = le.fit_transform(dataset[target_var].tolist())
    X = dataset.drop(columns=[target_var])
    cv = StratifiedKFold(n_splits=cross_split)
    ix = 1
    ret = []
    for train_idx, test_idx, in cv.split(X, y):
        X_train, y_train = X[X.index.isin(train_idx)], y[train_idx]
        # X_train, y_train = SMOTE().fit_resample(X_train, y_train)
        X_test, y_test = X[X.index.isin(test_idx)], y[test_idx]
        model = create_model(X_train, item['classes'], item['neurons'], item['optimizer'], item['init_mode'],
                             item['activation'], item['dropout_rate']
                             )
        m, h = model_train(X_train, y_train, X_test, y_test, model,
                           'trained_model_' + item['dataset'] + '_' + str(ix) + '.h5',
                           int(item['classes']), batch_size=int(item['batch_size'])
                           )

        hist_index = h.history['accuracy'].index(max(h.history['accuracy']))
        ret.append([ix, h.history['accuracy'][hist_index], h.history['val_accuracy'][hist_index]])
        ix += 1

    return ret


parameters = pd.read_csv('datasets-UCI/new_rules/summary.csv')
dataset_par = parameters.iloc[2]
print('--------------------------------------------------')
print(dataset_par['dataset'])
print('--------------------------------------------------')

out_list = model_creator(dataset_par, target_var=dataset_par['output_name'])
out_list = pd.DataFrame(out_list, columns=['model_number', 'accuracy', 'val_accuracy'])
out_list.to_csv('accuracy_' + dataset_par['dataset'] + '.csv')
