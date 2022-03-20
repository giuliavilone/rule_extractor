from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from common_functions import dataset_uploader
from matplotlib import pyplot
import copy


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
                                    save_best_only=True, verbose=1
                                    )
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5)
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
    dataset = pd.read_csv('datasets/' + item['dataset'] + '.csv')
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
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
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


def model_permutation_importance(train_x, train_y, test_x, test_y, model_par):
    """
    Perform the permutation importance on input model
    :param train_x: Pandas dataframe containing the training dataset
    :param train_y: list containing the original labels of the input instances in the training dataset
    :param test_x: Pandas dataframe containing the evaluation dataset
    :param test_y: list containing the original labels of the input instances in the evaluation dataset
    :param model_par: parameters of the model
    :return:
    """
    wrapped_model = KerasClassifier(build_fn=lambda: create_model(train_x, model_par['classes'], model_par['neurons'],
                                                                  model_par['optimizer'], model_par['init_mode'],
                                                                  model_par['activation'], model_par['dropout_rate']),
                                    epochs=100,
                                    batch_size=10
                                    )
    history = wrapped_model.fit(train_x,
                                train_y,
                                validation_data=(test_x, to_categorical(test_y, num_classes=model_par['classes'])
                                                 ),
                                verbose=0
                                )
    # Calculate the maximum prediction accuracy that the network can obtain on the test valuation dataset
    accuracy = max(history.history['val_accuracy'])
    results = permutation_importance(wrapped_model, train_x, train_y, scoring='accuracy')
    return results, accuracy


def importance_dictionary(in_df, importance_obj, plot_importance_scores=False):
    """
    Return a dictionary where the keys are the columns of the input Pandas dataframe and the value their importance
    scores
    :param in_df: Pandas dataframe
    :param importance_obj: object of the importance scores of the variables of in_df
    :param plot_importance_scores: boolean variable. If true, the function returns the barchart of the importance scores
    :return: dictionary of the importance scores of the variables of in_df
    """
    importance = importance_obj.importances_mean
    # plot feature importance
    if plot_importance_scores:
        pyplot.bar([x for x in range(len(importance))], importance)
        pyplot.show()
    ret = {}
    cols = in_df.columns.tolist()
    for i, v in enumerate(importance):
        ret[cols[i]] = round(v, 10)
        # print(f'Feature: {columns[i]}, Score: {v}')
    return ret


def variable_remover(train_df, test_df, importance_scores_dict):
    """
    Remove the variable with the minimum importance score from the train and test dataset and the dictionary with the
    importance scores
    :param train_df: Pandas dataframe with the training dataset
    :param test_df: Pandas dataframe with the evaluation dataset
    :param importance_scores_dict: dictionary containing the importance scores
    :return: two Pandas dataframes with the training and evaluation datasets and the dictionary with the
    importance scores
    """
    min_importance_variable = min(importance_scores_dict, key=importance_scores_dict.get)
    print("Minimum importance: ", min_importance_variable)
    importance_scores_dict.pop(min_importance_variable, None)
    train_df.drop(min_importance_variable, axis=1, inplace=True)
    test_df.drop(min_importance_variable, axis=1, inplace=True)
    return train_df, test_df, importance_scores_dict


def variable_selector(train_df, test_df, train_y, test_y, original_accuracy, importance_scores_dict, model_par):
    """
    Remove the not relevant variables of the input Pandas dataframe
    :param train_df: Pandas dataframe
    :param test_df: Pandas dataframe
    :param train_y: list
    :param test_y: list
    :param original_accuracy: prediction accuracy of the model trained on all variables
    :param model_par:
    :param importance_scores_dict: dictionary containing the importance scores of the input variables
    :return: two Pandas dataframes with the training and evaluation datasets containing only the relevant variables
    """
    new_train_df = copy.deepcopy(train_df)
    new_test_df = copy.deepcopy(test_df)
    print("This is the length of the dictionary: ", len(importance_scores_dict))
    print("This is the shape of the training dataset: ", new_train_df.shape)
    if len(importance_scores_dict) > 0:
        new_train_df, new_test_df, importance_scores_dict = variable_remover(new_train_df, new_test_df,
                                                                             importance_scores_dict)
        print(new_train_df.head())
        _, new_accuracy = model_permutation_importance(new_train_df, train_y, new_test_df, test_y, model_par)
        print("The original accuracy of the network is: ", original_accuracy)
        print("The accuracy of the pruned network is: ", new_accuracy)
        if new_accuracy >= original_accuracy:
            train_df = new_train_df
            test_df = new_test_df
        variable_selector(train_df, test_df, train_y, test_y, original_accuracy, importance_scores_dict, model_par)
    else:
        return train_df, test_df


parameters = pd.read_csv('datasets-UCI/new_rules/summary.csv')
parameters = parameters.iloc[0]
print('--------------------------------------------------')
print(parameters['dataset'])
print('--------------------------------------------------')
data_path = 'datasets-UCI/new_rules/'

x_train, x_test, y_train, y_test, _, _, _ = dataset_uploader(parameters['dataset'], data_path,
                                                             target_var=parameters['output_name'],
                                                             cross_split=5, apply_smothe=False,
                                                             data_normalization=False
                                                             )
x_train, x_test, y_train, y_test = x_train[0], x_test[0], y_train[0], y_test[0]
columns = x_train.columns.tolist()
perm_res, network_accuracy = model_permutation_importance(x_train, y_train, x_test, y_test, parameters)
# get minimum importance
importance_dict = importance_dictionary(x_test, perm_res)
x_train, x_test = variable_selector(x_train, x_test, y_train, y_test, network_accuracy, importance_dict, parameters)



# out_list = model_creator(parameters, target_var=dataset_par['output_name'])
# out_list = pd.DataFrame(out_list, columns=['model_number', 'accuracy', 'val_accuracy'])
# out_list.to_csv('accuracy_' + dataset_par['dataset'] + '.csv')
