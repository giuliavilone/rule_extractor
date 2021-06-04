import pandas as pd
from keras.models import load_model
import numpy as np
from common_functions import rule_metrics_calculator, attack_definer
import random
import copy
import itertools
from mysql_queries import mysql_queries_executor
from common_functions import save_list, create_empty_file


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
    cvalues = sorted(list(set(out_df[out_df.index.isin(changes)][attr])))
    all_values = sorted(list(set(out_df[attr])))
    intervals = [[cvalues[i], all_values[all_values.index(cvalues[i+1])-1]]for i in range(len(cvalues) - 1)]
    values = out_df[attr].tolist()
    intervals += [[values[changes[-1]], max(values)]]

    # It can happen that there are no changes at the extremes of the value ranges, so these values might not be
    # included in the list of intervals. Hence, it must be checked that the intervals cover the entire range,
    # otherwise they must be extended by adding the min and/or max values
    if intervals[-1][1] < max(all_values):
        intervals.append([intervals[-1][1], max(all_values)])
    if intervals[0][0] > min(all_values):
        intervals.append([min(all_values), intervals[0][0]])
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


def rule_evaluator(df, rule_columns, new_rule_set, out_var, model, n_samples, fidelity=1):
    """
    Evaluate the fidelity of the new rule
    :param df: input dataframe containing the training samples
    :param rule_columns: list of the variables used by the input rule to be evaluated
    :param new_rule_set: set of rules to be evaluated
    :param out_var: name of the target variable
    :param model: trained model
    :param n_samples: number of new samples to be randomly generated
    :param fidelity: fidelity threshold
    :return: the rule set containing only the rules with fidelity greater than the input threshold
    """
    # Creating synthetics data. As the input df does not contain the instances covered by the previous rules,
    # the new instances are not covered by old rules
    tot_df = synthetic_data_generator(df.drop(columns=[out_var]), n_samples)
    tot_df[out_var] = np.argmax(model.predict(tot_df), axis=1)
    tot_df = tot_df.merge(new_rule_set, on=rule_columns)
    tot_df = tot_df.drop(columns=['unique_class'])
    tot_df['same'] = [1 if tot_df[out_var].iloc[i] == tot_df['max_class'].iloc[i] else 0 for i in range(len(tot_df))]
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


def instance_remover(rule, df, column):
    df = pd.merge(df, rule[column], how='outer', indicator=True)
    df = df.loc[df._merge == 'left_only']
    df = df.drop('_merge', axis=1)
    df = df.reset_index(drop=True)
    return df


def rule_maker(df, intervals, combo_list, target_var, model):
    """
    Creates the IF-THEN rules
    :param df: input dataframe
    :param x_train: training dataset
    :param intervals: list of intervals to build the rules
    :param combo_list: list of attributes to be analysed
    :param target_var: name of dependent variable
    :param model: trained model
    :return: rules
    """
    outdf = copy.deepcopy(df)
    # Randomly shuffling the attributes to be analysed
    ret = []
    combo_number = 0
    while combo_number < len(combo_list) and len(outdf) > 0:
        combos = combo_list[combo_number]
        i = 0
        for combo in combos:
            i += 1
            attr_list = outdf[list(combo) + [target_var]].groupby(list(combo)).agg(unique_class=(target_var, 'nunique'),
                                                                                   max_class=(target_var, max)
                                                                                   ).reset_index(drop=False)
            new_rules = attr_list[attr_list['unique_class'] == 1]
            if len(new_rules) > 0:
                new_col = new_rules.columns.values.tolist()
                new_col.remove('unique_class')
                new_col.remove('max_class')
                new_rules = rule_evaluator(outdf, new_col, new_rules, target_var, model, len(outdf), fidelity=1)
                if len(new_rules) > 0:
                    outdf = instance_remover(new_rules, outdf, new_col)
                    for index, row in new_rules.iterrows():
                        # Evaluate the fidelity of the new rule
                        new_dict = {'columns': new_col, 'class': row['max_class'].tolist()}
                        rule_intervals = []
                        for c in new_col:
                            interv = intervals[c]
                            rule_int = [item for item in interv if item[0] == row[c]]
                            rule_intervals += rule_int
                        new_dict['limits'] = rule_intervals
                        ret.append(new_dict)
        combo_number += 1
    return ret


def column_translator(in_df, target_var, col_number_list):
    # col_number_list contains the number of the df columns to be analysed.
    # Here we need to get their names
    df_columns = in_df.columns.tolist()
    if target_var in df_columns:
        df_columns.remove(target_var)
    ret = [v for i, v in enumerate(df_columns) if i in col_number_list]
    return ret


def refne_run(X_train, X_test, y_test, discrete_attributes, continuous_attributes, dataset_par, model, save_graph):

    label_col = dataset_par['output_name']
    discrete_attributes = column_translator(X_train, label_col, discrete_attributes)
    continuous_attributes = column_translator(X_train, label_col, continuous_attributes)
    all_column_combos = column_combos(categorical_var=discrete_attributes, continuous_var=continuous_attributes)
    synth_samples = X_train.shape[0] * 2
    xSynth = synthetic_data_generator(X_train, synth_samples)
    xSynth = xSynth.append(X_train, ignore_index=True)
    ySynth = np.argmax(model.predict(xSynth), axis=1)
    n_class = dataset_par['classes']

    # Discretizing the continuous attributes
    attr_list = xSynth.columns.tolist()
    xSynth[label_col] = ySynth

    interv_dict = {}
    for attr in attr_list:
        if attr in continuous_attributes:
            interv = interval_definer(data=xSynth, attr=attr, label=label_col)
            xSynth[attr] = discretizer(xSynth[attr], interv)
            X_train[attr] = discretizer(X_train[attr], interv)
            interv_dict[attr] = interv
        else:
            unique_values = np.unique(xSynth[attr]).tolist()
            interv_dict[attr] = [list(a) for a in zip(unique_values, unique_values)]

    final_rules = rule_maker(xSynth, interv_dict, all_column_combos, label_col, model)

    # Calculation of metrics
    predicted_labels = np.argmax(model.predict(X_test), axis=1)
    metrics = rule_metrics_calculator(X_test, y_test, predicted_labels, final_rules, n_class)
    if save_graph:
        attack_list, final_rules = attack_definer(X_test, final_rules, merge_rules=True)
        create_empty_file('REFNE_' + dataset_par['dataset'] + "_attack_list")
        save_list(attack_list, 'REFNE_' + dataset_par['dataset'] + "_attack_list")
        create_empty_file('REFNE_' + dataset_par['dataset'] + "_final_rules")
        save_list(final_rules, 'REFNE_' + dataset_par['dataset'] + "_final_rules")
        # feature_set_name = 'REFNE_' + dataset_par['dataset'] + "_featureset"
        # graph_name = 'REFNE_' + dataset_par['dataset'] + "_graph"
        # mysql_queries_executor(ruleset=final_rules, attacks=attack_list, conclusions=labels,
        #                        feature_set_name=feature_set_name, graph_name=graph_name)

    return metrics



