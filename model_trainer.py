from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils.np_utils import to_categorical
# from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier
import pandas as pd
from sklearn.inspection import permutation_importance
from common_functions import DatasetUploader, relevant_column_selector, create_empty_file, save_list
from matplotlib import pyplot
import copy
import multiprocessing
CPU_COUNT = multiprocessing.cpu_count()


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


def model_creator(item, data, relevant_variable=None):
    train_x, test_x, train_y, test_y = data.stratified_k_fold()
    ret = []
    for idx in range(len(train_x)):
        train_x_idx, test_x_idx, train_y_idx, test_y_idx = train_x[idx], test_x[idx], train_y[idx], test_y[idx]
        if relevant_variable is not None:
            train_x_idx, _ = relevant_column_selector(train_x_idx, relevant_variable)
            test_x_idx, _ = relevant_column_selector(test_x_idx, relevant_variable)
        model = create_model(train_x_idx, int(item['classes']), item['neurons'], item['optimizer'], item['init_mode'],
                             item['activation'], item['dropout_rate']
                             )
        m, h = model_train(train_x_idx, train_y_idx, test_x_idx, test_y_idx, model,
                           'trained_model_' + item['dataset'] + '_' + str(idx) + '.h5',
                           int(item['classes']), batch_size=int(item['batch_size'])
                           )

        hist_index = h.history['accuracy'].index(max(h.history['accuracy']))
        ret.append([idx, h.history['accuracy'][hist_index], h.history['val_accuracy'][hist_index]])

    return ret


def model_permutation_importance(train_x, train_y, test_x, test_y, model_par, epochs=50, batch_size=1000):
    """
    Perform the permutation importance on input model
    :param train_x: Pandas dataframe containing the training dataset
    :param train_y: list containing the original labels of the input instances in the training dataset
    :param test_x: Pandas dataframe containing the evaluation dataset
    :param test_y: list containing the original labels of the input instances in the evaluation dataset
    :param model_par: parameters of the model
    :param epochs: number of epochs for training the model
    :param batch_size: number of samples processed before the model is updated
    :return:
    """
    num_out_classes = int(model_par['classes'])
    wrapped_model = KerasClassifier(build_fn=lambda: create_model(train_x, num_out_classes, model_par['neurons'],
                                                                  model_par['optimizer'], model_par['init_mode'],
                                                                  model_par['activation'], model_par['dropout_rate']),
                                    epochs=epochs,
                                    batch_size=batch_size
                                    )
    other_args = {"validation_data": (test_x, to_categorical(test_y, num_classes=num_out_classes)),
                  "verbose": 1
                  }
    history = wrapped_model.fit(train_x,
                                to_categorical(train_y, num_classes=num_out_classes),
                                **other_args
                                )
    # Calculate the maximum prediction accuracy that the network can obtain on the test valuation dataset.
    # When the final model will be trained, the fit function will save the model with the highest prediction accuracy
    # on the evaluation dataset.
    accuracy = max(history.history_['val_accuracy'])
    results = permutation_importance(wrapped_model, train_x, to_categorical(train_y, num_classes=num_out_classes),
                                     scoring='accuracy', n_jobs=CPU_COUNT
                                     )
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


def variable_selector(train_df, test_df, train_y, test_y, original_accuracy, importance_scores_dict, model_par,
                      reduction_threshold=0.99):
    """
    Remove the not relevant variables of the input Pandas dataframe
    :param train_df: Pandas dataframe
    :param test_df: Pandas dataframe
    :param train_y: list of the original classes of the instances in the training dataset
    :param test_y: list of the original classes of the instances in the evaluation dataset
    :param original_accuracy: prediction accuracy of the model trained on all variables
    :param model_par: Pandas dataframe with the parameters of the neural network to be trained
    :param importance_scores_dict: dictionary containing the importance scores of the input variables
    :param reduction_threshold: percentage in the reduction of the original accuracy that can be tolerated to remove one
                                variable
    :return: two Pandas dataframes with the training and evaluation datasets containing only the relevant variables
    """
    new_train_df = copy.deepcopy(train_df)
    new_test_df = copy.deepcopy(test_df)
    print("This is the length of the dictionary: ", len(importance_scores_dict))
    print("This is the shape of the training dataset: ", new_train_df.shape)
    if len(importance_scores_dict) > 0 and new_train_df.shape[1] > 2:
        new_train_df, new_test_df, importance_scores_dict = variable_remover(new_train_df, new_test_df,
                                                                             importance_scores_dict)
        # print(new_train_df.head())
        _, new_accuracy = model_permutation_importance(new_train_df, train_y, new_test_df, test_y, model_par)
        print("The original accuracy of the network is: ", original_accuracy)
        print("The original accuracy reduced by ", reduction_threshold * 100, "% is: ",
              original_accuracy * reduction_threshold)
        print("The accuracy of the pruned network is: ", new_accuracy)
        if new_accuracy >= original_accuracy * reduction_threshold:
            train_df = new_train_df
            test_df = new_test_df
        train_df, test_df = variable_selector(train_df, test_df, train_y, test_y, original_accuracy,
                                              importance_scores_dict, model_par)
    return train_df, test_df


parameters_list = pd.read_csv('datasets/summary.csv')
data_list = [9, 10, 12, 13, 16, 17, 19]
for d in data_list:
    parameters = parameters_list.iloc[d]
    print('--------------------------------------------------')
    print(parameters['dataset'])
    print('--------------------------------------------------')
    data_path = 'datasets/'

    dataset = DatasetUploader(parameters['dataset'],
                              data_path,
                              target_var=parameters['output_name'],
                              cross_split=5,
                              )
    x_train_list, x_test_list, y_train_list, y_test_list = dataset.stratified_k_fold()

    best_accuracy = 0
    relevant_columns = []
    for ix in range(len(x_train_list)):
        x_train, x_test, y_train, y_test = x_train_list[ix], x_test_list[ix], y_train_list[ix], y_test_list[ix]
        columns = x_train.columns.tolist()
        perm_res, network_accuracy = model_permutation_importance(x_train, y_train, x_test, y_test, parameters)
        print("The best accuracy reached is: ", network_accuracy)
        if network_accuracy > best_accuracy:
            best_accuracy = network_accuracy
            # get minimum importance
            importance_dict = importance_dictionary(x_test, perm_res)
            df_train, df_test = variable_selector(x_train, x_test, y_train, y_test, network_accuracy, importance_dict,
                                                  parameters)
            relevant_columns = df_train.columns.tolist()

    print(relevant_columns)
    create_empty_file(parameters['dataset'] + '_relevant_columns')
    save_list(relevant_columns, parameters['dataset'] + '_relevant_columns')

    out_list = model_creator(parameters, dataset, relevant_variable=relevant_columns)
    out_list = pd.DataFrame(out_list, columns=['model_number', 'accuracy', 'val_accuracy'])
    out_list.to_csv('accuracy_' + parameters['dataset'] + '.csv')
