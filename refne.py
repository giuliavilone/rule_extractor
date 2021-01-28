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
import itertools


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


def interval_definer(data, attr, label):
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
    out_df = out_df.sort_values(by=[attr, label], ignore_index=True)
    out_df.drop_duplicates(inplace=True, ignore_index=True)
    changes = out_df[out_df[label].diff() != 0].index.tolist()
    values = out_df[attr].tolist()
    cvalues = sorted(list(set([values[changes[i]] for i in range(len(changes))])))
    uvalues = sorted(list(set(values)))
    intervals = [[cvalues[i], uvalues[uvalues.index(cvalues[i+1])-1]]for i in range(len(cvalues) - 1)]
    intervals += [[values[changes[-1]], max(values)]]

    # It can happen that there are no changes at the extremes of the value ranges, so these values might not be
    # included in the list of intervals. Hence, it must be checked that the intervals cover the entire range,
    # otherwise they must be extended by adding the min and/or max values
    if intervals[-1][1] < max(values):
        intervals.append([intervals[-1][1], max(values)])
    if intervals[0][0] > min(values):
        intervals.append([min(values), intervals[0][0]])
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
        indf[(indf >= min_val) & (indf <= max_val)] = min_val
    return indf.tolist()


def select_random_item(int_list, ex_item_list):
    ex_list = copy.deepcopy(int_list)
    for i in ex_item_list:
        ex_list.remove(i)
    new_item = random.choice(ex_list)
    return new_item


def rule_evaluator(df, rule_columns, new_rule_set, out_var, model, fidelity=0.0):
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
    out_df = group_df[group_df['fidelity'] >= fidelity]
    new_rule_set = new_rule_set.merge(out_df[rule_columns], on=rule_columns)
    return new_rule_set


def column_combos(categorical_var=None, continuous_var=None):
    """
    This function returns the combinations of 1, 2 and 3 categorical, continuous and categorical & continuous input
    variables to be analysed by the rule_maker function to generate the rules
    :return:
    """
    out_list = []
    if categorical_var is not None:
        random.shuffle(categorical_var)
        for col_number in range(min(3, len(categorical_var))):
            out_list.append(list(itertools.combinations(categorical_var, col_number + 1)))
        if continuous_var is not None:
            random.shuffle(continuous_var)
            all_columns = categorical_var + continuous_var
            tmp_list = []
            for cont_col_number in range(min(3, len(all_columns))):
                tmp_list.append(list(itertools.combinations(all_columns, cont_col_number + 1)))
            tmp_list = [item for item in tmp_list if item not in out_list]
            out_list += tmp_list
    elif continuous_var is not None:
        random.shuffle(continuous_var)
        for col_number in range(min(3, len(continuous_var))):
            out_list.append(list(itertools.combinations(continuous_var, col_number + 1)))

    return out_list


def rule_maker(df, intervals, combo_list, target_var, model):
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
    ret = []
    combo_number = 0
    while len(ret) == 0 and combo_number < len(combo_list) and len(outdf) > 0:
        combos = combo_list[combo_number]
        for combo in combos:
            attr_list = outdf[list(combo) + [target_var]].groupby(list(combo)).agg(unique_class=(target_var, 'nunique'),
                                                                                   max_class=(target_var, max)
                                                                                   ).reset_index(drop=False)
            new_rules = attr_list[attr_list['unique_class'] == 1]
            if len(new_rules) > 0:
                new_col = new_rules.columns.values.tolist()
                new_col.remove('unique_class')
                new_col.remove('max_class')
                new_rules = rule_evaluator(outdf, new_col, new_rules, target_var, model, fidelity=0.5)
                for index, row in new_rules.iterrows():
                    # Evaluate the fidelity of the new rule
                    new_dict = {'neuron': new_col, 'class': row['max_class'].tolist()}
                    rule_intervals = []
                    for c in new_col:
                        interv = intervals[c]
                        rule_int = [item for item in interv if item[0] == row[c]]
                        x = outdf[c]
                        # Selecting the indexes of the instances that are not covered by the rules
                        ix = np.where((x < rule_int[0][0]) | (x > rule_int[0][1]))[0]
                        outdf = outdf.iloc[ix]
                        rule_intervals += rule_int
                    new_dict['limits'] = rule_intervals
                    ret.append(new_dict)
        combo_number += 1
    return ret


def rule_applier(indf, iny, rules):
    """
    Apply the input rules to the list of labels iny according to the rule conditions on indf
    :param indf:
    :param iny:
    :param over_y:
    :param rules:
    :return: iny
    """
    indexes = []
    over_y = np.zeros(len(iny))
    for r in range(len(rules['neuron'])):
        x = indf[rules['neuron'][r]]
        if len(rules['limits']) > 0:
            ix = list(np.where((x >= rules['limits'][r][0]) & (x <= rules['limits'][r][1]))[0])
            indexes += ix

    indexes = list(set(indexes))
    over_y[indexes] = 1
    iny[indexes] = rules['class']
    return iny, over_y


def column_translator(in_df, target_var, col_number_list):
  # col_number_list contains the number of the df columns to be analysed.
  # Here we need to get their names
  df_columns = in_df.columns.tolist()
  if target_var in df_columns:
    df_columns.remove(target_var)
  ret = [v for i, v in enumerate(df_columns) if i in col_number_list]
  return ret


def refne_run(X_train, X_test, y_train, y_test, discrete_attributes, continuous_attributes, label_col, dataset_par,
              model):

    discrete_attributes = column_translator(X_train, label_col, discrete_attributes)
    continuous_attributes = column_translator(X_train, label_col, continuous_attributes)
    all_column_combos = column_combos(categorical_var=discrete_attributes, continuous_var=continuous_attributes)
    synth_samples = X_train.shape[0] * 2
    xSynth = synthetic_data_generator(X_train, synth_samples)
    ySynth = np.argmax(model.predict(xSynth), axis=1)
    n_class = dataset_par['classes']

    # Discretising the continuous attributes
    attr_list = xSynth.columns.tolist()
    xSynth[label_col] = ySynth
    interv_dict = {}
    for attr in attr_list:
        if attr in continuous_attributes:
            interv = interval_definer(data=xSynth, attr=attr, label=label_col)
            xSynth[attr] = discretizer(xSynth[attr], interv)
            interv_dict[attr] = interv
        else:
            unique_values = np.unique(xSynth[attr]).tolist()
            interv_dict[attr] = [list(a) for a in zip(unique_values, unique_values)]

    final_rules = rule_maker(xSynth, interv_dict, all_column_combos, label_col, model)

    #print(final_rules)
    # Calculation of metrics
    predicted_labels = np.argmax(model.predict(X_test), axis=1)

    num_test_examples = X_test.shape[0]
    perturbed_data = perturbator(X_test)
    rule_labels = np.empty(num_test_examples)
    rule_labels[:] = np.nan
    perturbed_labels = np.empty(num_test_examples)
    perturbed_labels[:] = np.nan
    overlap = np.zeros(num_test_examples)

    for rule in final_rules:
        neuron = X_test[rule['neuron']]
        rule_labels, rule_overlap = rule_applier(neuron, rule_labels, rule)
        overlap += rule_overlap
        p_neuron = perturbed_data[rule['neuron']]
        perturbed_labels, _ = rule_applier(p_neuron, rule_labels, rule)

    y_test = y_test.tolist()
    if len(final_rules) > 0:
        avg_length = sum([len(item['neuron']) for item in final_rules]) / len(final_rules)
    else:
        avg_length = 0
    completeness = sum(~np.isnan(rule_labels)) / num_test_examples
    rule_labels[np.where(np.isnan(rule_labels))] = n_class + 10
    perturbed_labels[np.where(np.isnan(perturbed_labels))] = n_class + 10

    return rule_metrics_calculator(num_test_examples, y_test, rule_labels, predicted_labels,
                                   perturbed_labels, len(final_rules), completeness, avg_length,
                                   overlap, dataset_par['classes']
                                   )



