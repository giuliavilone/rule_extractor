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
    independent_columns = [item for item in dataset.columns.tolist() if item != 'class']
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


def rule_elicitation(in_df, iny, rules):
    """
    Apply the input rules to the list of labels iny according to the rule conditions on in_df
    :param in_df:
    :param iny:
    :param rules:
    :return: iny
    """
    indexes = []
    over_y = np.zeros(len(iny))
    for r in range(len(rules['columns'])):
        x = in_df[rules['columns'][r]]
        if len(rules['limits']) > 0:
            ix = list(np.where((x >= rules['limits'][r][0]) & (x <= rules['limits'][r][1]))[0])
            indexes.append(list(ix))
            over_y[ix] = 1
    intersect_indexes = list(set.intersection(*[set(lst) for lst in indexes]))
    iny[intersect_indexes] = rules['class']
    return iny, over_y


def rule_metrics_calculator(in_df, y_test, model_labels, final_rules, n_classes):
    """
    Calculate the correctness, fidelity, robustness and number of rules. The completeness, average length and number
    of rules are calculated in a different way for each rule extractor and passed as inputs
    :return: list of metrics
    """
    rule_n = len(final_rules)
    num_test_examples = in_df.shape[0]
    perturbed_data = perturbator(in_df)
    rule_labels = np.empty(num_test_examples)
    rule_labels[:] = np.nan
    perturbed_labels = np.empty(num_test_examples)
    perturbed_labels[:] = np.nan
    overlap = np.zeros(num_test_examples)

    for rule in final_rules:
        columns = in_df[rule['columns']]
        rule_labels, rule_overlap = rule_elicitation(columns, rule_labels, rule)
        overlap += rule_overlap
        p_neuron = perturbed_data[rule['columns']]
        perturbed_labels, _ = rule_elicitation(p_neuron, rule_labels, rule)

    y_test = y_test.tolist()
    if len(final_rules) > 0:
        avg_length = sum([len(item['columns']) for item in final_rules]) / rule_n
    else:
        avg_length = 0
    complete = sum(~np.isnan(rule_labels)) / num_test_examples
    rule_labels[np.where(np.isnan(rule_labels))] = n_classes + 10
    perturbed_labels[np.where(np.isnan(perturbed_labels))] = n_classes + 10
    correct = 0
    fidel = 0
    rob = 0
    for i in range(0, num_test_examples):
        fidel += (rule_labels[i] == model_labels[i])
        correct += (rule_labels[i] == y_test[i])
        rob += (rule_labels[i] == perturbed_labels[i])

    print("Completeness of the ruleset is: " + str(complete))
    correctness = correct / num_test_examples
    print("Correctness of the ruleset is: " + str(correctness))
    fidelity = fidel / num_test_examples
    print("Fidelity of the ruleset is: " + str(fidelity))
    robustness = rob / num_test_examples
    print("Robustness of the ruleset is: " + str(robustness))
    print("Number of rules : " + str(rule_n))
    print("Average rule length: " + str(avg_length))
    overlap = sum(overlap) / (rule_n * num_test_examples)
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
    rule_labels = np.zeros(in_df.shape[0])
    total_cover = []
    for rule_number in range(len(final_rules)):
        rule = final_rules[rule_number]
        columns = in_df[rule['columns']]
        _, rule_cover = rule_elicitation(columns, rule_labels, rule)
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
