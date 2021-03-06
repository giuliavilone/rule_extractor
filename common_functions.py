from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from scipy.stats import mode
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import itertools
import pickle
import os
from sklearn.metrics import accuracy_score
import sys


def vote_db_modifier(in_df):
    """
    Modify the vote database by replacing yes/no answers with boolean
    :param in_df: Pandas dataframe
    """
    in_df.replace(b'y', 1, inplace=True)
    in_df.replace(b'n', 0, inplace=True)
    in_df.replace(b'?', 0, inplace=True)
    return in_df


def create_model(train_x, n_classes, neurons, optimizer='Adam', init_mode='glorot_uniform', activation='sigmoid',
                 dropout_rate=0.0, loss='categorical_crossentropy', out_activation='softmax'
                 ):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=train_x.shape[1], activation=activation, kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation=activation, kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_classes, activation=out_activation))
    # Compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def model_train(train_x, train_y, test_x, test_y, model, model_name, n_epochs=100, batch_size=10):
    check_pointer = ModelCheckpoint(filepath=model_name,
                                    save_weights_only=False,
                                    monitor='accuracy',
                                    save_best_only=True,
                                    verbose=1)
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=batch_size,
                        epochs=n_epochs, callbacks=[check_pointer]
                        )
    return model, history


def perturbator(in_df, mu=0, sigma=0.1):
    """
    Add white noise to input dataset
    :param in_df: Pandas dataframe
    :param mu: mean of the normally distributed white noise
    :param sigma: standard deviation of the normally distributed white noise
    """
    noise = np.random.normal(mu, sigma, in_df.shape)
    return in_df + noise


def ensemble_predictions(members, test_x):
    # make an ensemble prediction for multi-class classification
    # make predictions
    y_hats = [model.predict(test_x) for model in members]
    y_hats = np.array(y_hats)
    # combining the members via plurality voting
    voted_y_hats = np.argmax(y_hats, axis=2)
    results = mode(voted_y_hats, axis=0)[0]
    return results


def dataset_uploader(item, path, target_var='class', cross_split=5, apply_smothe=True, remove_columns=True):
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
    dataset = pd.read_csv(path + item['dataset'] + '.csv')
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
    independent_columns = [item for item in dataset.columns.tolist() if item != target_var]
    out_disc = []
    for col in out_disc_temp:
        out_disc += [i for i, v in enumerate(independent_columns) if v.find(col + '_') > -1]
    out_cont = [i for i, v in enumerate(independent_columns) if i not in out_disc]

    # Separating independent variables from the target one

    y = le.fit_transform(dataset[target_var].tolist())
    labels = list(le.fit(dataset[target_var].tolist()).classes_)
    x = dataset.drop(columns=[target_var])
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - train_split))
    x_train_list, x_test_list, y_train_list, y_test_list = [], [], [], []
    cv = StratifiedKFold(n_splits=cross_split)
    for train_idx, test_idx, in cv.split(x, y):
        x_train, y_train = x[x.index.isin(train_idx)], y[train_idx]
        if apply_smothe:
            x_train, y_train = SMOTE().fit_sample(x_train, y_train)
        x_test, y_test = x[x.index.isin(test_idx)], y[test_idx]
        x_train_list.append(x_train)
        x_test_list.append(x_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
    return x_train_list, x_test_list, y_train_list, y_test_list, labels, out_disc, out_cont


def rule_elicitation(x, in_rule):
    """
    Calculate the accuracy score of the input rule and the indexes of the instances that fire the input rule. The
    accuracy score is calculated over the instances that are affected by the rule.
    :param x: dataframe containing the independent variables of the input instances
    :param in_rule: rule to be evaluated
    :return: the accuracy score of the rule and the indexes of the instances that fire the rule
    """
    indexes = []
    for item in range(len(in_rule['columns'])):
        minimum = in_rule['limits'][item][0]
        maximum = in_rule['limits'][item][1]
        column = in_rule['columns'][item]
        item_indexes = np.where(np.logical_and(x[column] >= minimum, x[column] <= maximum))[0]
        indexes.append(list(item_indexes))
    intersect_indexes = list(set.intersection(*[set(lst) for lst in indexes]))
    return intersect_indexes


def rule_set_evaluator(x, rule_set):
    """
    Evaluates a set of rules by eliciting each of them and calculate the overall accuracy. The rules are first
    sorted by the number of instances that they cover (in reverse order) so that the bigger rules do not cancel out the
    smaller one in case of overlapping rules.
    :param x:
    :param rule_set:
    :return:
    """
    predicted_y = []
    rule_indexes = []
    for rule in rule_set:
        indexes = rule_elicitation(x, rule)
        rule_indexes.append(indexes)
        predicted_y.append((len(indexes), indexes, rule['class']))
    predicted_y = sorted(predicted_y, reverse=True, key=lambda tup: tup[0])
    ret_labels = np.empty(len(x))
    ret_labels[:] = np.nan
    for t in predicted_y:
        ret_labels[t[1]] = t[2]
    return ret_labels


def overlap_calculator(in_df, rules):
    """
    Calculate the overlap area
    :param in_df:
    :param rules:
    :return: iny
    """
    overlap = np.zeros(in_df.shape[0])
    for rule in rules:
        rule_overlap = np.zeros(in_df.shape[0])
        indexes = rule_elicitation(in_df, rule)
        rule_overlap[indexes] = 1
        overlap += rule_overlap
    overlap = len([1 for i in overlap if i > 1]) / in_df.shape[0]
    return overlap


def rule_metrics_calculator(in_df, y_test, model_labels, final_rules, n_classes):
    """
    Calculate the correctness, fidelity, robustness and number of rules. The completeness, average length and number
    of rules are calculated in a different way for each rule extractor and passed as inputs
    :return: list of metrics
    """
    rule_n = len(final_rules)
    perturbed_data = perturbator(in_df)
    rule_labels = rule_set_evaluator(in_df, final_rules)
    perturbed_labels = rule_set_evaluator(perturbed_data, final_rules)
    overlap = overlap_calculator(in_df, final_rules)

    y_test = y_test.tolist()
    if len(final_rules) > 0:
        avg_length = sum([len(item['columns']) for item in final_rules]) / rule_n
    else:
        avg_length = 0
    complete = sum(~np.isnan(rule_labels)) / in_df.shape[0]
    rule_labels[np.where(np.isnan(rule_labels))] = n_classes + 10
    perturbed_labels[np.where(np.isnan(perturbed_labels))] = n_classes + 10

    print("Completeness of the ruleset is: " + str(complete))
    correctness = accuracy_score(y_test, rule_labels)
    print("Correctness of the ruleset is: " + str(correctness))
    fidelity = accuracy_score(model_labels, rule_labels)
    print("Fidelity of the ruleset is: " + str(fidelity))
    robustness = accuracy_score(rule_labels, perturbed_labels)
    print("Robustness of the ruleset is: " + str(robustness))
    print("Number of rules : " + str(rule_n))
    print("Average rule length: " + str(avg_length))
    print("Fraction overlap: " + str(overlap))
    labels_considered = set(rule_labels)
    labels_considered.discard(n_classes + 10)
    class_fraction = len(set(labels_considered)) / n_classes
    print("Fraction of classes: " + str(class_fraction))
    return [complete, correctness, fidelity, robustness, rule_n, avg_length, overlap, class_fraction]


def rule_merger(ruleset, cover_list):
    """
    This function checks if there are rules with the same class and list of columns. If this is the case, these rules
    are merged together and will be treated as 'OR' logical disjunction
    :param ruleset:
    :param cover_list:
    :return:
    """
    ix1 = 0
    while ix1 < len(ruleset):
        ix2 = ix1 + 1
        while ix2 < len(ruleset):
            rule_a, rule_b = ruleset[ix1], ruleset[ix2]
            if rule_a['class'] == rule_b['class'] and rule_a['columns'] == rule_b['columns']:
                if type(rule_a['limits'][-1][0]) is not list:
                    rule_a['limits'] = [rule_a['limits']] + [rule_b['limits']]
                else:
                    rule_a['limits'].append(rule_b['limits'])
                ruleset.pop(ix2)
                cover_list[ix1]['rule_cover'] += cover_list[ix2]['rule_cover']
                cover_list[ix1]['rule_cover'][cover_list[ix1]['rule_cover'] > 1] = 1
                cover_list.pop(ix2)
            else:
                ix2 += 1
        ix1 += 1
    return ruleset, cover_list


def attack_definer(in_df, final_rules, merge_rules=False):
    ret = []
    total_cover = []
    for rule_number in range(len(final_rules)):
        rule = final_rules[rule_number]
        rule_cover = rule_elicitation(in_df, rule)
        rule_cover = {'rule_number': "R"+str(rule_number), 'rule_cover': rule_cover, 'rule_class': rule['class']}
        total_cover.append(rule_cover)
        rule['rule_number'] = "R"+str(rule_number)
    if merge_rules:
        final_rules, total_cover = rule_merger(final_rules, total_cover)
    for pair in itertools.combinations(total_cover, 2):
        a, b = pair
        if a['rule_class'] != b['rule_class']:
            a_index = np.where(a['rule_cover'] == 1)[0]
            b_index = np.where(b['rule_cover'] == 1)[0]
            comparison_index = np.intersect1d(a_index, b_index)
            if np.array_equal(a_index, b_index) and len(comparison_index) > 0:
                ret.append({"source": a['rule_number'], "target": b['rule_number'], "type": "rebuttal"})
                ret.append({"source": b['rule_number'], "target": a['rule_number'], "type": "rebuttal"})
            else:
                if np.array_equal(a_index, comparison_index):
                    ret.append({"source": a['rule_number'], "target": b['rule_number'], "type": "undercut"})
                elif np.array_equal(b_index, comparison_index):
                    ret.append({"source": b['rule_number'], "target": a['rule_number'], "type": "undercut"})
                elif len(comparison_index) > 0:
                    ret.append({"source": a['rule_number'], "target": b['rule_number'], "type": "rebuttal"})
                    ret.append({"source": b['rule_number'], "target": a['rule_number'], "type": "rebuttal"})
    return ret, final_rules


def create_empty_file(filename):
    if not os.path.exists(filename + '.txt'):
        open(filename + '.txt', 'w').close()


def save_list(in_list, filename):
    with open(filename + '.txt', 'wb') as fp:
        pickle.dump(in_list, fp)


def load_list(filename, path):
    with open(path + filename + '.txt', 'rb') as fp:
        return pickle.load(fp)
