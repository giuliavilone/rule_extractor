import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import load_model
from keras.optimizers import SGD, Adagrad, Adam, Nadam
import numpy as np
from common_functions import perturbator, create_model, model_train, ensemble_predictions, dataset_uploader
from common_functions import rule_metrics_calculator
from collections import Counter
import random
import copy
import sys


# Functions
def load_all_models(n_models):
    """
    This function returns the list of the trained models
    :param n_models:
    :return: list of trained models
    """
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = 'refne_model_' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


def synthetic_data_generator_old(indf, n_samples, discrete=[]):
    """
    Given an input dataframe, the function returns a new dataframe containing random numbers
    generated within the value ranges of the input attributes.
    :param indf:
    :param n_samples: integer number of samples to be generated
    :param discrete
    :return: outdf: of synthetic data
    """
    outdf = pd.DataFrame()
    for column in indf.columns.tolist():
        if column in discrete:
            outdf[column] = np.random.choice(np.unique(indf[column]).tolist(), n_samples)
        else:
            minvalue = indf[column].min()
            maxvalue = indf[column].max()
            outdf[column] = np.round(np.random.uniform(minvalue, maxvalue, n_samples), 4)
    return outdf


def synthetic_data_generator(indf, n_samples):
    """
    Given an input dataframe, the function returns a new dataframe containing random numbers
    generated within the value ranges of the input attributes.
    :param indf:
    :param n_samples: integer number of samples to be generated
    :param discrete
    :return: outdf: of synthetic data
    """
    outdf = pd.DataFrame()
    for column in indf.columns.tolist():
        outdf[column] = np.random.choice(np.unique(indf[column]).tolist(), n_samples)
    return outdf


def chimerge(data, attr, label):
    """
    Code copied from https://gist.github.com/alanzchen/17d0c4a45d59b79052b1cd07f531689e
    :param data:
    :param attr:
    :param label:
    :return:
    """
    distinct_vals = sorted(set(data[attr]))  # Sort the distinct values
    labels = sorted(set(data[label]))  # Get all possible labels
    empty_count = {l: 0 for l in labels}  # A helper function for padding the Counter()
    # Initialize the intervals for each attribute
    intervals = [[distinct_vals[i], distinct_vals[i]] for i in range(len(distinct_vals))]
    more_merges = True
    while more_merges:  # While loop
        chi = []
        for i in range(len(intervals) - 1):
            lab0 = sorted(set(data[label][data[attr].between(intervals[i][0], intervals[i][1])]))
            lab1 = sorted(set(data[label][data[attr].between(intervals[i + 1][0], intervals[i + 1][1])]))
            if len(lab0) + len(lab1) > 2 or lab0 != lab1:
            # if lab0 != lab1:
                chi.append(1000000.0)
                continue
            else:
                # Calculate the Chi2 value
                obs0 = data[data[attr].between(intervals[i][0], intervals[i][1])]
                obs1 = data[data[attr].between(intervals[i + 1][0], intervals[i + 1][1])]
                total = len(obs0) + len(obs1)
                count_0 = np.array([v for i, v in {**empty_count, **Counter(obs0[label])}.items()])
                count_1 = np.array([v for i, v in {**empty_count, **Counter(obs1[label])}.items()])
                count_total = count_0 + count_1
                expected_0 = count_total * sum(count_0) / total
                expected_1 = count_total * sum(count_1) / total
                chi_ = (count_0 - expected_0) ** 2 / expected_0 + (count_1 - expected_1) ** 2 / expected_1
                chi_ = np.nan_to_num(chi_)  # Deal with the zero counts
                chi.append(sum(chi_))  # Finally do the summation for Chi2
        min_chi = min(chi)  # Find the minimal Chi2 for current iteration
        if min_chi == 1000000.0:
            break
        min_chi_index = [i for i, v in enumerate(chi) if v == min_chi]
        new_intervals = []  # Prepare for the merged new data array
        skip = False
        done = False
        for j in range(len(intervals)):
            if skip:
                skip = False
                done = False
                continue
            elif j in min_chi_index and not done:  # Merge the intervals
                t = intervals[j] + intervals[j + 1]
                new_intervals.append([min(t), max(t)])
                skip = True
                done = True
            else:
                new_intervals.append(intervals[j])
        intervals = new_intervals
        # Let's keep at least two intervals
        if len(intervals) < 3:
            more_merges = False
    return intervals


def discretizer(indf, intervals):
    """
    The function takes a continuous attribute of a dataframe and the list of the intervals, then it performs the
    discretization of the attribute
    :param indf:
    :param intervals:
    :return: indf: with discrete values
    """
    indf = np.array(indf)
    for i in range(len(intervals)):
        minVal = intervals[i][0]
        maxVal = intervals[i][1]
        indf[(indf >= minVal) & (indf <= maxVal)] = minVal
    return indf.tolist()


def select_random_item(int_list, ex_item_list):
    ex_list = copy.deepcopy(int_list)
    for i in ex_item_list:
        ex_list.remove(i)
    new_item = random.choice(ex_list)
    return new_item


def rule_evaluator(df, rule_columns, new_rule, ruleset, out_var, fidelity=0, origin=False):
    """
    Evaluate the fidelity of the new rule
    :param rule:
    :return: boolean (true, false)
    """
    n_samples = len(df)
    all_columns = df.columns.values.tolist()
    # Creating synthetics data within the limits set by the rule under evaluation
    eval_df = df[rule_columns]
    indexes = []
    for col in rule_columns:
        all_columns.remove(col)
        ix = np.where(eval_df[col] == new_rule[col])[0]
        indexes = [x for x in ix if x not in indexes]
    eval_df = eval_df.iloc[indexes]
    if len(eval_df) == 0:
        return False
    eval_df = synthetic_data_generator(eval_df, n_samples)
    # Adding synthetics data on the columns that are not considered by the rule under evaluation
    all_columns.remove(out_var)
    other_df = df[all_columns]
    other_df = synthetic_data_generator(other_df, n_samples)
    tot_df = pd.concat([eval_df, other_df], axis=1).reset_index(drop=True)
    # Removing the instances that are covered by the existing rules
    if len(ruleset) > 0:
        for eval_rule in ruleset:
            for s in range(len(eval_rule['neuron'])):
                x = tot_df[eval_rule['neuron'][s]]
                ix = np.where((x < eval_rule['limits'][s][0]) | (x > eval_rule['limits'][s][1]))[0]
                tot_df = tot_df.iloc[ix]
        extra_samples = n_samples - len(tot_df)
        if extra_samples > 0:
            extra_df = synthetic_data_generator(tot_df, extra_samples)
            tot_df = pd.concat([tot_df, extra_df], axis=0)
    # Evaluating the fidelity of the rule under evaluation
    if origin:
        tot_df[out_var] = ensemble_predictions(members, tot_df)[0]
    else:
        tot_df[out_var] = np.argmax(model.predict(tot_df), axis=1)
    agreement = len(tot_df[tot_df[out_var] == new_rule['max_class']]) / len(tot_df)
    return agreement > fidelity


def rule_maker(df, intervals, col, target_var):
    """
    Creates the IF-THEN rules
    :param df: input dataframe
    :param intervals: list of intervals to build the rules
    :param col: list of attributes to be analysed
    :param target_var: name of dependent variable
    :return: rules
    :return: outdf
    """
    outdf = copy.deepcopy(df)
    # Col contains the number of the df columns to be analysed. Here we need to get their names
    df_columns = outdf.columns.tolist()
    df_columns.remove(target_var)
    col = [v for i, v in enumerate(df_columns) if i in col]
    # Randomly shuffling the attributes to be analysed
    random.shuffle(col)
    ret = []
    for item in col:
        attr_list = outdf[[item, target_var]].groupby(item).agg(unique_class=(target_var, 'nunique'),
                                                                max_class=(target_var, 'max')
                                                                ).reset_index(drop=False)
        new_rules = attr_list[attr_list['unique_class'] == 1]
        if (len(new_rules) == 0) and (len(col) > 1):
            item1 = select_random_item(col, [item])
            attr_list = outdf[[item, item1, target_var]].groupby([item, item1]).agg(
                unique_class=(target_var, 'nunique'),
                max_class=(target_var, 'max')
                ).reset_index(drop=False)
            new_rules = attr_list[attr_list['unique_class'] == 1]
            if (len(new_rules) == 0) and (len(col) > 2):
                item2 = select_random_item(col, [item, item1])
                attr_list = outdf[[item, item1, item2, target_var]].groupby([item, item1, item2]).agg(
                    unique_class=(target_var, 'nunique'),
                    max_class=(target_var, 'max')
                ).reset_index(drop=False)
                new_rules = attr_list[attr_list['unique_class'] == 1]
        new_col = new_rules.columns.values.tolist()
        new_col.remove('unique_class')
        new_col.remove('max_class')
        print(new_rules.shape)
        if len(new_rules) > 0:
            for index, row in new_rules.iterrows():
                # Evaluate the fidelity of the new rule
                evaluation = rule_evaluator(outdf, new_col, row, ret, target_var, fidelity=0.2)
                if evaluation:
                    new_dict = {'neuron': new_col}
                    new_dict['class'] = row['max_class'].tolist()
                    rule_intervals = []
                    for c in new_col:
                        interv = intervals[c]
                        rule_int = [item for item in interv if item[0] == row[c]]
                        x = outdf[c]
                        ix = np.where((x < rule_int[0][0]) | (x > rule_int[0][1]))[0]
                        outdf = outdf.iloc[ix]
                        rule_intervals += rule_int
                    new_dict['limits'] = rule_intervals
                    ret.append(new_dict)
                if len(df) == 0:
                    break
    return ret, outdf


def rule_applier(indf, iny, over_y, rules):
    """
    Apply the input rules to the list of labels iny according to the rule conditions on indf
    :param indf:
    :param iny:
    :param over_y:
    :param rules:
    :return: iny
    """
    indexes = []
    for r in range(len(rules['neuron'])):
        x = indf[rules['neuron'][r]]
        ix = np.where((x >= rules['limits'][r][0]) & (x <= rules['limits'][r][1]))[0]
        indexes = [x for x in ix if x not in indexes]

    over_y += [x for x in indexes if not np.isnan(iny[x])]

    iny[indexes] = rule['class']
    return iny, over_y


# Main code
parameters = pd.read_csv('datasets-UCI/Used_data/summary_new.csv')
dataset_par = parameters.iloc[0]
print('--------------------------------------------------')
print(dataset_par['dataset'])
print('--------------------------------------------------')

label_col = 'class'
original_study = False
if original_study:
    X_train, X_test, y_train, y_test, discrete_attributes, continuous_attributes = dataset_uploader(dataset_par)
    X_train, X_test, y_train, y_test, _, _ = dataset_uploader(dataset_par)
    X_train = X_train[0]
    X_test = X_test[0]
    y_train = y_train[0]
    y_test = y_test[0]
    X = pd.concat([X_train, X_test], ignore_index=True)
    y = np.concatenate((y_train, y_test))

    n_members = 2

    # Extracting the classes to be predicted
    classes = np.unique(y).tolist()

    # Create the object to perform cross validation
    skf = StratifiedKFold(n_splits=n_members, shuffle=True)

    # define model
    model = create_model(X, dataset_par['classes'], dataset_par['neurons'], eval(dataset_par['optimizer']),
                         dataset_par['init_mode'], dataset_par['activation'], dataset_par['dropout_rate'],
                         weight_constraint=eval(dataset_par['weight_constraint'])
                         )

    # Training the model on the 5 cross validation datasets
    fold_var = 1
    for train_index, val_index in skf.split(X, y):
        X_train, X_test = X[X.index.isin(train_index)], X[X.index.isin(val_index)]
        y_train, y_test = y[train_index], y[val_index]
        model_train(X_train, to_categorical(y_train, num_classes=dataset_par['classes']),
                    X_test, to_categorical(y_test, num_classes=dataset_par['classes']), model,
                    'refne_model_' + str(fold_var) + '.h5',
                    n_epochs=dataset_par['epochs'], batch_size=dataset_par['batch_size'])

        fold_var += 1


    # load all models
    members = load_all_models(n_members)
    print('Loaded %d models' % len(members))

    ensemble_res = ensemble_predictions(members, X)
    print(accuracy_score(ensemble_res[0], y))

    # According to the paper, it is enough to generate a new dataset which is twice the training set to obtain
    # very accurate rules
    synth_samples = X.shape[0] * 2
    xSynth = synthetic_data_generator(X, synth_samples)
    ySynth = ensemble_predictions(members, xSynth)[0]
else:
    X_train_list, X_test_list, y_train_list, y_test_list, discrete_attributes, continuous_attributes = dataset_uploader(dataset_par, apply_smothe=False)
    metric_list = []
    for ix in range(len(X_train_list)):
        X_train = X_train_list[ix]
        X_test = X_test_list[ix]
        y_train = y_train_list[ix]
        y_test = y_test_list[ix]
        model = load_model('trained_models/trained_model_' + dataset_par['dataset'] + '_' + str(ix) + '.h5')
        synth_samples = X_train.shape[0] * 2
        xSynth = synthetic_data_generator(X_train, synth_samples)
        ySynth = np.argmax(model.predict(xSynth), axis=1)
        classes = np.unique(np.concatenate((y_train, y_test), axis=0)).tolist()

        # Discretising the continuous attributes
        attr_list = xSynth.columns.tolist()
        xSynth[label_col] = ySynth
        interv_dict = {}
        for attr in attr_list:
            if attr in continuous_attributes:
                interv = chimerge(data=xSynth, attr=attr, label=label_col)
                xSynth[attr] = discretizer(xSynth[attr], interv)
                interv_dict[attr] = interv
            else:
                unique_values = np.unique(xSynth[attr]).tolist()
                interv_dict[attr] = [list(a) for a in zip(unique_values, unique_values)]

        final_rules = []
        if len(discrete_attributes) > 0:
            out_rule, discreteSynth = rule_maker(xSynth, interv_dict, discrete_attributes, label_col)
            final_rules += out_rule
            if len(final_rules) == 0 or len(discreteSynth) > 0:
                out_rule, discreteSynth = rule_maker(discreteSynth, interv_dict, continuous_attributes, label_col)
                final_rules += out_rule
        else:
            out_rule, contSynth = rule_maker(xSynth, interv_dict, continuous_attributes, label_col)
            final_rules += out_rule

        print(final_rules)

        # Calculation of metrics
        predicted_labels = np.argmax(model.predict(X_test), axis=1)

        num_test_examples = X_test.shape[0]
        perturbed_data = perturbator(X_test)
        rule_labels = np.empty(num_test_examples)
        rule_labels[:] = np.nan
        perturbed_labels = np.empty(num_test_examples)
        perturbed_labels[:] = np.nan
        overlap = []

        for rule in final_rules:
            neuron = X_test[rule['neuron']]
            rule_labels, overlap = rule_applier(neuron, rule_labels, overlap, rule)
            p_neuron = perturbed_data[rule['neuron']]
            perturbed_labels, overlap = rule_applier(p_neuron, rule_labels, overlap, rule)

        y_test = y_test.tolist()

        avg_length = sum([len(item['neuron']) for item in final_rules]) / len(final_rules)
        completeness = sum(~np.isnan(rule_labels)) / num_test_examples
        rule_labels[np.where(np.isnan(rule_labels))] = max(classes) + 10
        perturbed_labels[np.where(np.isnan(perturbed_labels))] = max(classes) + 10
        overlap = len(set(overlap)) / len(X_test)

        metric_list.append(rule_metrics_calculator(num_test_examples, y_test, rule_labels, predicted_labels,
                                                   perturbed_labels, len(final_rules), completeness, avg_length,
                                                   overlap, dataset_par['classes'])
                           )

    pd.DataFrame(metric_list, columns=['complete', 'correctness', 'fidelity', 'robustness', 'rule_n', 'avg_length',
                                       'overlap', 'class_fraction']
                 ).to_csv('refne_metrics_' + dataset_par['dataset'] + '.csv')
