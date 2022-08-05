# import copy
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from scipy.stats import mode
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
import itertools
import pickle
import os
from os.path import exists
from sklearn.metrics import accuracy_score


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
    """
    Create a deep neural network with the following characteristics: 1) the input layer, densely connected, coupled with
    a dropout layer, 2) a densely-connected hidden layer coupled with a dropout layer, and 3) the output layer, also
    densely connected
    :param train_x: Pandas dataset containing the training dataset
    :param n_classes: number of output classes
    :param neurons: number of neurons of the hidden layer
    :param optimizer: optimising function
    :param init_mode: layer weight initializer
    :param activation: activation function of input and hidden layers
    :param dropout_rate: dropout rate to be used in the dropout layers
    :param loss: loss function
    :param out_activation: activation function of output layer
    :return: compiled model object
    """
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
    """
    Train a compiled model and return it
    :param train_x: Pandas dataset containing the training dataset
    :param train_y: list of output classes as recorded in the training dataset
    :param test_x: Pandas dataset containing the evaluation dataset
    :param test_y: list of output classes as recorded in the evaluation dataset
    :param model: compiled model object
    :param model_name: path and filename where to store the trained model
    :param n_epochs: number of training epochs
    :param batch_size: number of samples to be propagated through the network.
    :return: trained model, training history object
    """
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
    ret = in_df + noise
    # The rules are extracted on a specific input space whose limits must be preserved, so the addition of the
    # white noise must keep the input instances between these limits
    for col in ret.columns:
        print("Perturbing column: ", col)
        ret[col] = np.minimum(ret[col], max(in_df[col])).tolist()
        ret[col] = np.maximum(ret[col], min(in_df[col])).tolist()
    return ret


def ensemble_predictions(members, test_x):
    # return the predictions of an ensemble model for multi-class classification
    # make predictions
    y_hats = [model.predict(test_x) for model in members]
    y_hats = np.array(y_hats)
    # combining the members via plurality voting
    voted_y_hats = np.argmax(y_hats, axis=2)
    results = mode(voted_y_hats, axis=0)[0]
    return results


def data_file(file_name, path, remove_columns=True):
    """
    Upload csv files containing the original data and remove columns that are not suitable as predictors
    :param file_name: name of the csv file to be uploaded
    :param path: path to the csv file to be uploaded
    :param remove_columns: boolean variable. If true, the columns listed in the variable "feat_to_be_deleted" are
    deleted
    :return: Pandas dataframe containing the data retrieved from the csv file
    """
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
                          'shuttle': ['S7', 'S8', 'S9'],
                          'adult': ['education']  # Because there is the field education-num which represents the same
                                                  # information but in numbers
                          }
    dataset = pd.read_csv(path + file_name + '.csv')
    dataset = dataset.dropna().reset_index(drop=True)
    if remove_columns:
        if file_name in feat_to_be_deleted.keys():
            columns_to_be_deleted = [item for item in dataset.columns.tolist() if item in feat_to_be_deleted[file_name]]
            dataset = dataset.drop(columns=columns_to_be_deleted)
    return dataset


def column_type_finder(file_name, dataset, target_var, path='relevant_columns/'):
    """
    Define the type of data contained in the input Pandas dataframe and remove the irrelevant columns
    :param file_name: name of the dataset under analysis
    :param dataset: Pandas dataframe
    :param target_var: name of the dependent variable (to be predicted by a model)
    :param path:
    :return: the input dataset, the names of the columns containing categorical variable, the position of the columns
    containing discrete and continuous variables
    """
    col_types = dataset.dtypes
    discrete_column_names = []
    for index, value in col_types.items():
        if value in ['object', 'bool']:
            if index != target_var:
                dataset = pd.get_dummies(dataset, columns=[index])
                discrete_column_names.append(index)
    # Removing the not relevant columns
    relevant_column = load_list(file_name + '_relevant_columns', path)
    if len(relevant_column) > 0:
        relevant_column.append(target_var)
        dataset, _ = relevant_column_selector(dataset, relevant_column)
    # The number of the discrete features must take into account the new dummy columns
    independent_columns = [item for item in dataset.columns.tolist() if item != target_var]
    out_disc = []
    for col in discrete_column_names:
        out_disc += [i for i, v in enumerate(independent_columns) if v.find(col + '_') > -1]
    out_cont = [i for i, v in enumerate(independent_columns) if i not in out_disc]
    return dataset, discrete_column_names, out_disc, out_cont


def relevant_column_selector(in_df, relevant_column_list, in_weight=None):
    """
    Remove from the input Pandas dataframe the columns that are not included in the list of the relevant ones.
    If in_weight is not none, the function removes also the model's weights corresponding to the not relevant variables
    :param in_df: Pandas dataframe
    :param relevant_column_list: list of relevant columns
    :param in_weight: array of model's weights
    :return: the input Pandas dataframe containing only the relevant columns
    """
    columns_to_be_deleted = [item for item in in_df.columns.tolist() if item not in relevant_column_list]
    in_df = in_df.drop(columns=columns_to_be_deleted)
    if in_weight is not None:
        weight_tdb = [i for i, col in enumerate(in_df.columns.tolist()) if col in relevant_column_list]
        in_weight[0] = np.delete(in_weight[0], weight_tdb, 0)
    return in_df, in_weight


def dataset_uploader(file_name,
                     path,
                     target_var='class',
                     cross_split=5,
                     apply_smote=True,
                     remove_columns=True,
                     ):
    """
    Upload a dataset from a csv file into a Pandas dataframe and applies, if requested, the SMOTE algorithm to
    oversample the minority class(es). The input dataset is split into a list of training and evaluation datasets with
    the Stratified K Fold algorithm
    :param file_name: name of the csv file to be uploaded
    :param path: path to the csv file to be uploaded
    :param target_var: name of the dependent variable (to be predicted by a model)
    :param cross_split: number of folds for the Stratified K Fold algorithm
    :param apply_smote: boolean variable. If true, the SMOTE oversampling algorithm is applied
    :param remove_columns: boolean variable. If true, the columns listed in the variable "feat_to_be_deleted" are
    deleted (see function data_file)
    function
    :return: list of Pandas datasets containing the training and evaluations datasets, a separate list of the output
    variable of the training and evaluation datasets, list of the output class labels, list of the discrete and
    continuous variables
    """
    le = LabelEncoder()
    # Remove the columns that were deemed irrelevant from the correlation analysis
    dataset = data_file(file_name, path, remove_columns=remove_columns)
    # Remove the columns that were deemed irrelevant from the feature
    dataset, _, out_disc, out_cont = column_type_finder(file_name, dataset, target_var)

    # Separating independent variables from the target one
    y = le.fit_transform(dataset[target_var].tolist())
    labels = list(le.fit(dataset[target_var].tolist()).classes_)
    x = dataset.drop(columns=[target_var])
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - train_split))
    x_train_list, x_test_list, y_train_list, y_test_list = [], [], [], []
    cv = StratifiedKFold(n_splits=cross_split)
    for train_idx, test_idx, in cv.split(x, y):
        x_train, y_train = x[x.index.isin(train_idx)], y[train_idx]
        if apply_smote:
            x_train, y_train = SMOTE().fit_resample(x_train, y_train)
        x_test, y_test = x[x.index.isin(test_idx)], y[test_idx]
        x_train_list.append(x_train)
        x_test_list.append(x_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
    return x_train_list, x_test_list, y_train_list, y_test_list, labels, out_disc, out_cont


class DatasetUploader:
    """
    This objects allows to upload a dataset from a csv file into a Pandas dataframe and applies, if requested, the
    following data manipulation techniques:
    1) the SMOTE algorithm to oversample the minority class(es);
    2) split the input dataset into a list of training and  evaluation datasets with the Stratified K Fold algorithm;
    3) normalise the input data.

    Parameters
    ----------
    file_name: name of the csv file to be uploaded
    path: path to the csv file to be uploaded
    target_var: name of the dependent variable (to be predicted by a model)
    cross_split: number of folds for the Stratified K Fold algorithm
    apply_smote: boolean variable. If true, the SMOTE oversampling algorithm is applied
    remove_columns: boolean variable. If true, the columns listed in the variable "feat_to_be_deleted" are
    deleted (see function data_file)
    seed:

    Attributes
    ----------
    dataset: the whole dataset (with descriptive and target variables) but without the columns to be removed
    y: a list containing the original target variable
    labels: a list with the unique values of the target variable
    x: a panda dataframe with the descriptive variables

    """

    def __init__(self,
                 file_name,
                 path,
                 target_var='class',
                 cross_split=5,
                 apply_smote=True,
                 smote_sampling_strategy='auto',
                 remove_columns=True,
                 data_normalization=True,
                 seed=1983
                 ):
        # Parameters
        self.file_name = file_name
        self.path = path
        self.target_var = target_var
        self.cross_split = cross_split
        self.apply_smote = apply_smote
        self.smote_sampling_strategy = smote_sampling_strategy
        self.remove_columns = remove_columns
        self.data_normalization = data_normalization
        self.seed = seed
        # Attributes
        self.discrete_column_names = []
        self.continuous_columns_indexes = []
        self.discrete_columns_indexes = []
        self.cv = StratifiedKFold(n_splits=cross_split)
        self.std_scale = MinMaxScaler()
        self.dataset, self.labels, self.x, self.y = self.data_uploader()
        # This contains the list of the original names of the discrete columns

    def data_scaler(self, in_df):
        """
        Normalise the input dataset with the MinMaxScaler sklearn function
        :return: pandas dataframe with normalised data
        """
        self.std_scale.fit(in_df)
        x_norm = self.std_scale.transform(in_df)
        x_norm = pd.DataFrame(x_norm, index=in_df.index, columns=in_df.columns)
        return x_norm

    def column_types(self, in_df):
        col_types = in_df.dtypes
        for index, value in col_types.items():
            if value in ['object', 'bool']:
                if index != self.target_var:
                    self.discrete_column_names.append(index)
                    # The drop_first option is set to True to avoid the issue of multicollinearity. 
                    # See this blog for more info:
                    # https://towardsdatascience.com/introduction-to-data-preprocessing-in-machine-learning-a9fa83a5dc9d
                    in_df = pd.get_dummies(in_df, columns=[index], drop_first=True)
        return in_df

    def irrelevant_column_remover(self, ind_df, path='relevant_columns/'):
        relevant_column = load_list(self.file_name + '_relevant_columns', path)
        if len(relevant_column) > 0:
            relevant_column.append(self.target_var)
            ind_df, _ = relevant_column_selector(ind_df, relevant_column)
        return ind_df

    def label_encoder(self, in_df):
        le = LabelEncoder()
        y = le.fit_transform(in_df[self.target_var].tolist())
        labels = list(le.fit(in_df[self.target_var].tolist()).classes_)
        x = in_df.drop(columns=[self.target_var])
        return labels, x, y

    def data_uploader(self):
        dataset = data_file(self.file_name, self.path, remove_columns=self.remove_columns)
        dataset = self.column_types(dataset)
        dataset = self.irrelevant_column_remover(dataset)
        # Separating independent variables from the target one
        labels, x, y = self.label_encoder(dataset)
        if self.data_normalization:
            x = self.data_scaler(x)
        return dataset, labels, x, y

    def column_type_counter(self):
        """This function returns the lists of the indexes of the continuous and discrete columns"""
        independent_columns = [item for item in self.dataset.columns.tolist() if item != self.target_var]
        out_disc = []
        for col in self.discrete_column_names:
            self.discrete_columns_indexes += [i for i, v in enumerate(independent_columns) if v.find(col + '_') > -1]
        self.continuous_columns_indexes += [i for i, v in enumerate(independent_columns) if i not in out_disc]

    def stratified_k_fold(self, best_split=None):
        """
        The input parameter best_split can be used after the model training to get the split related to the best model
        """
        x_train_list, x_test_list, y_train_list, y_test_list = [], [], [], []
        for train_idx, test_idx, in self.cv.split(self.x, self.y):
            x_train, y_train = self.x[self.x.index.isin(train_idx)], self.y[train_idx]
            if self.apply_smote:
                x_train, y_train = SMOTE(sampling_strategy=self.smote_sampling_strategy).fit_resample(x_train, y_train)
            x_test, y_test = self.x[self.x.index.isin(test_idx)], self.y[test_idx]
            x_train_list.append(x_train)
            x_test_list.append(x_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)
        if type(best_split) == int:
            return [x_train_list[best_split]], [x_test_list[best_split]], \
                   [y_train_list[best_split]], [y_test_list[best_split]]
        else:
            return x_train_list, x_test_list, y_train_list, y_test_list


def rule_elicitation(x, in_rule):
    """
    Return the list of the indexes of the samples in x that fire the input rule
    :param x: dataframe containing the independent variables of the input instances
    :param in_rule: rule to be evaluated
    :return: list of indexes of the firing samples
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
    smaller one in case of overlapping rules
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
    Calculate the overlap areas between rules
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


def rule_metrics_calculator(in_df, y_test, model_labels, final_rules, n_classes, new_method=None):
    """
    Calculate the correctness, fidelity, robustness and number of rules. The completeness, average length and number
    of rules are calculated in a different way for each rule extractor and passed as inputs
    :param in_df: Pandas dataframe
    :param y_test: list of the original output classes of the samples contained in in_df
    :param model_labels: list of the predicted output classes of the samples contained in in_df
    :param final_rules: list of the rules contained in the final ruleset
    :param n_classes: number of output classes
    :param new_method
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
    if new_method is None:
        correctness = accuracy_score(y_test, rule_labels)
    else:
        correctness = new_method['accuracy']
    print("Correctness of the ruleset is: " + str(correctness))
    if new_method is None:
        fidelity = accuracy_score(model_labels, rule_labels)
    else:
        fidelity = new_method['fidelity']
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


def rule_merger(ruleset):
    """
    This function checks if there are rules with the same class and list of columns. If this is the case, these rules
    are merged together and will be treated as 'OR' logical disjunction. The cover sets of these rules are joined
    together
    :param ruleset: list of rules
    :return: modified ruleset with joint rules, updated list of the cover sets of the modified ruleset
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
                ruleset[ix1]['samples'] += ruleset[ix2]['samples']
                ruleset[ix1]['samples'][ruleset[ix1]['samples'] > 1] = 1
                ruleset.pop(ix2)
            else:
                ix2 += 1
        ix1 += 1
    return ruleset


def attack_weight_calculator(rule_a_sample, rule_b_sample, intersection):
    """
    Calculated the normalised weights of a rebuttal attack (undercut attacks have weight equal to 1 by default) as such:
    1) Calculate the number of samples supporting each rule
    2) Calculate the cardinality of the intersection set and divide it by the number of supporting samples of each
    attacked rule. The idea is that the strength of the attack depends on the number of samples that are responsible for
    it in proportion to the total number of supporting samples. For example, let’s take two attacking rules, A and B,
    where A has 3 supporting samples and B has 4 and 2 samples are in the intersection. Then, Rule A has 2 out of 3
    samples responsible for the attack (meaning that the attack from B has strength 0.667) whereas Rule B has 2 out
    of 4 attacking samples (so the strength of the attack from A is 0.5)
    :param rule_a_sample: list of supporting samples of first rule
    :param rule_b_sample: list of supporting samples of second rule
    :param intersection: list of sample in the intersection set between the two rules
    :return: list of rules' weights
    """
    ret = {'rule_a': float(float(len(rule_a_sample) / len(intersection))),
           'rule_b': float(float(len(rule_b_sample) / len(intersection)))}
    return ret


def attack_definer(final_rules, merge_rules=False, inconsistency_budget=0.55):
    """
    Define the attack between conflicting rules
    :param final_rules: list of rules
    :param merge_rules: boolean variable. If true, two rules with the same conclusion and same list of variables in
    their antecedents will be joined together (see rule_merger function)
    :param inconsistency_budget:
    :param set_rule_name:
    :return: list of attackers, modified ruleset with merged rules
    """
    ret = []
    for rule_number in range(len(final_rules)):
        rule = final_rules[rule_number]
        # rule_cover = rule_elicitation(in_df, rule)
        rule['rule_number'] = "Rule "+str(rule_number)
        rule['rule_index'] = rule_number
    if merge_rules:
        final_rules = rule_merger(final_rules)
    for pair in itertools.combinations(final_rules, 2):
        a, b = pair
        if a['class'] != b['class']:
            a_index = a['samples']
            b_index = b['samples']
            comparison_index = np.intersect1d(a_index, b_index)
            # Excluding the case the index lists are empty
            if len(comparison_index) > 0:
                if np.array_equal(a_index, b_index):
                    # In this case a_index, b_index and comparison_index contain the same list of samples, so the
                    # weights are equal to 1
                    ret.append({"source": a['rule_number'], "target": b['rule_number'], "type": "rebuttal",
                                "weight": 1, "source_index": a['rule_index'], "target_index": b['rule_index']})
                    ret.append({"source": b['rule_number'], "target": a['rule_number'], "type": "rebuttal",
                                "weight": 1, "source_index": b['rule_index'], "target_index": a['rule_index']})
                else:
                    # The weights of the undercut attacks is set equal to 1 as the undercutting rule always wins
                    if np.array_equal(a_index, comparison_index):  # Rule a fully included in rule b
                        ret.append({"source": a['rule_number'], "target": b['rule_number'], "type": "undercut",
                                    "weight": 1, "source_index": a['rule_index'], "target_index": b['rule_index']})
                    elif np.array_equal(b_index, comparison_index):  # Rule b fully included in rule a
                        ret.append({"source": b['rule_number'], "target": a['rule_number'], "type": "undercut",
                                    "weight": 1, "source_index": b['rule_index'], "target_index": a['rule_index']})
                    else:
                        attack_weights = attack_weight_calculator(a_index, b_index, comparison_index)
                        if attack_weights['rule_b'] >= inconsistency_budget:
                            ret.append({"source": a['rule_number'], "target": b['rule_number'], "type": "rebuttal",
                                        "weight": attack_weights['rule_b'], "source_index": a['rule_index'],
                                        "target_index": b['rule_index']})
                        elif attack_weights['rule_a'] >= inconsistency_budget:
                            ret.append({"source": b['rule_number'], "target": a['rule_number'], "type": "rebuttal",
                                        "weight": attack_weights['rule_a'], "source_index": b['rule_index'],
                                        "target_index": a['rule_index']})
    return ret, final_rules


def create_empty_file(filename):
    if not os.path.exists(filename + '.txt'):
        open(filename + '.txt', 'w').close()


def save_list(in_list, filename):
    with open(filename + '.txt', 'wb') as fp:
        pickle.dump(in_list, fp)


def load_list(filename, path):
    if exists(path + filename + '.txt'):
        with open(path + filename + '.txt', 'rb') as fp:
            return pickle.load(fp)
    else:
        return []


def rule_write(method_name, final_rules, dataset_par, path='final_rules/'):
    # Write ruleset into files
    create_empty_file(path + method_name + dataset_par['dataset'] + "_final_rules")
    save_list(final_rules, path + method_name + dataset_par['dataset'] + "_final_rules")
