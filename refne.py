import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import load_model
from keras.optimizers import SGD, Adagrad, Adam, Nadam
import numpy as np
from common_functions import perturbator, ensemble_predictions, dataset_uploader
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
    :return: outdf: of synthetic data
    """
    outdf = pd.DataFrame()
    for column in indf.columns.tolist():
        outdf[column] = np.random.choice(np.unique(indf[column]).tolist(), n_samples)
    return outdf


def interval_definer(data, attr, label, remove_single_values = False):
    """
    According to the paper, this function should do the ChiMerge Discretization algorithm, but the authors say that
    it continues as long as there are no instances belonging to different classes assigned to the same interval. This
    means that the intervals must contain instances belonging to the same class, but this is equivalent to find all
    the intervals of instances belonging to the same class and merge them. The ChiMerge calculations are superfluous.
    :param data:
    :param attr:
    :param label:
    :return:
    """
    out_df = copy.deepcopy(data[[attr, label]])
    out_df = out_df.sort_values(by=[attr], ignore_index=True)
    changes = out_df[out_df[label].diff() != 0].index.tolist()
    values = out_df[attr].tolist()
    intervals = [[values[changes[i]], values[changes[i+1]-1]] for i in range(len(changes) - 1)]
    if remove_single_values:
        to_be_deleted = []
        for ix in range(len(intervals)-1):
            if intervals[ix][0] == intervals[ix][1] and intervals[ix][1] == intervals[ix+1][0]:
                to_be_deleted.append(ix)
        to_be_deleted.reverse()
        for item in to_be_deleted:
            intervals.pop(item)

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
        min_val = intervals[i][0]
        max_val = intervals[i][1]
        indf[(indf >= min_val) & (indf < max_val)] = min_val
    return indf.tolist()


def select_random_item(int_list, ex_item_list):
    ex_list = copy.deepcopy(int_list)
    for i in ex_item_list:
        ex_list.remove(i)
    new_item = random.choice(ex_list)
    return new_item


def rule_evaluator(df, rule_columns, new_rule_set, ruleset, out_var, fidelity=0):
    """
    Evaluate the fidelity of the new rule
    :param new_rule_set:
    :return: boolean (true, false)
    """
    n_samples = len(df)
    all_columns = df.columns.values.tolist()
    # Creating synthetics data within the limits set by the rule under evaluation
    eval_df = df[rule_columns]
    eval_df = eval_df.merge(new_rule_set[rule_columns], on=rule_columns)
    eval_df = synthetic_data_generator(eval_df, n_samples)
    # Adding synthetics data on the columns that are not considered by the rule under evaluation
    all_columns.remove(out_var)
    all_columns = [i for i in all_columns if i not in rule_columns]
    other_df = df[all_columns]
    tot_df = pd.concat([eval_df, other_df], axis=1).reset_index(drop=True)
    tot_df[out_var] = np.argmax(model.predict(tot_df), axis=1)
    tot_df = tot_df.merge(new_rule_set, on=rule_columns)
    tot_df = tot_df.drop(columns=['unique_class'])
    tot_df = tot_df.drop(columns=all_columns)
    tot_df['same'] = [1 if tot_df['class'].iloc[i] == tot_df['max_class'].iloc[i] else 0 for i in range(len(tot_df))]
    group_df = tot_df.groupby(rule_columns).agg(total=('same', 'sum'),
                                                frequency=('same', 'count')).reset_index(drop=False)
    group_df['fidelity'] = group_df['total'] / group_df['frequency']
    while True:
        out_df = group_df[group_df['fidelity'] >= fidelity]
        if len(out_df) == 0:
            fidelity = max(fidelity - 0.1, 0)
        else:
            break
    new_rule_set = new_rule_set.merge(out_df[rule_columns], on=rule_columns)
    return new_rule_set


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
    # Randomly shuffling the attributes to be analysed
    random.shuffle(col)
    ret = []
    col_index = 0
    while True:
        item = col[col_index]
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
        if len(new_rules) > 0:
            new_rules = rule_evaluator(outdf, new_col, new_rules, ret, target_var, fidelity=0.7)
            for index, row in new_rules.iterrows():
                # Evaluate the fidelity of the new rule
                new_dict = {'neuron': new_col}
                new_dict['class'] = row['max_class'].tolist()
                rule_intervals = []
                for c in new_col:
                    interv = intervals[c]
                    rule_int = [item for item in interv if item[0] == row[c]]
                    if len(rule_int) > 0:
                        x = outdf[c]
                        ix = np.where((x < rule_int[0][0]) | (x > rule_int[0][1]))[0]
                        outdf = outdf.iloc[ix]
                        rule_intervals += rule_int
                new_dict['limits'] = rule_intervals
                ret.append(new_dict)
            if len(ret) > 0:
                break
        else:
            col_index += 1
            if col_index <= len(col):
                continue
            else:
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
        if len(rules['limits']) > 0:
            ix = np.where((x >= rules['limits'][r][0]) & (x <= rules['limits'][r][1]))[0]
            indexes = [x for x in ix if x not in indexes]

    over_y += [x for x in indexes if not np.isnan(iny[x])]

    iny[indexes] = rule['class']
    return iny, over_y


def column_translator(in_df, target_var, col_number_list):
  # col_number_list contains the number of the df columns to be analysed.
  # Here we need to get their names
  df_columns = in_df.columns.tolist()
  if target_var in df_columns:
    df_columns.remove(target_var)
  ret = [v for i, v in enumerate(df_columns) if i in col_number_list]
  return ret

# Main code
parameters = pd.read_csv('datasets-UCI/Used_data/summary_new.csv')
dataset_par = parameters.iloc[8]
print('--------------------------------------------------')
print(dataset_par['dataset'])
print('--------------------------------------------------')
label_col = 'class'

X_train_list, X_test_list, y_train_list, y_test_list, discrete_attributes, continuous_attributes = dataset_uploader(dataset_par,
                                                                                                                    'datasets-UCI/new_datasets/',
                                                                                                                    apply_smothe=False)
discrete_attributes = column_translator(X_train_list[0], label_col, discrete_attributes)
continuous_attributes = column_translator(X_train_list[0], label_col, continuous_attributes)
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
            print(attr)
            interv = interval_definer(data=xSynth, attr=attr, label=label_col)
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
