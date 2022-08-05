from scipy.stats import friedmanchisquare
import pandas as pd
from scipy.stats import rankdata, entropy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from common_functions import load_list


def method_ranking(df):
    ret_list = []
    for label, row in df.iteritems():
        if (label == 'Method') or (label == 'Dataset'):
            row = row.tolist()
            row.insert(0, label)
            ret_list.append(row)
        else:
            row = row.to_numpy()
            ranks = rankdata(row).tolist()
            ranks.insert(0, label)
            ret_list.append(ranks)
    ret_df = pd.DataFrame(ret_list, columns=['item', 'rank1', 'rank2', 'rank3', 'rank4', 'rank5'])
    return ret_df


def decreasing_ranking(df, column_list):
    for column in column_list:
        max_column = max(df[column])
        if max_column > 0:
            df[column] = [1 - i/max_column for i in df[column].tolist()]
    return df


def metric_table_formatter(method_groups_list):
    for ix, im in method_groups_list:
        print('--------------------- Working on dataset: ', ix, '---------------------')
        for c in ['Completeness', 'Correctness', 'Fidelity', 'Robustness', 'Number of rules', 'Average rule length',
                  'Fraction overlap', 'Fraction of classes']:
            print(c)
            out_string = ''
            for i, v in im[c].items():
                out_string = out_string + str(round(v, 4)) + ' & '
            print(out_string)


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


data = pd.read_csv('metrics.csv')
method_groups = data.groupby(['Dataset'], as_index=False)

method_list = []
out_list = []
out_df = pd.DataFrame()
friedmann_test = False
if friedmann_test:
    for index, item in method_groups:
        print(index)
        grouped_item = item.groupby('Method', as_index=False).mean()
        grouped_item['Dataset'] = index
        grouped_item = decreasing_ranking(grouped_item, ['Number of rules', 'Average rule length', 'Fraction overlap'])
        out_df = out_df.append(method_ranking(grouped_item))
        grouped_item = grouped_item.drop(['Method', 'Dataset'], axis=1)
        grouped_item = grouped_item.to_numpy()
        data1, data2, data3, data4, data5 = grouped_item
        stat, p = friedmanchisquare(data1, data2, data3, data4, data5)
        out_list.append([index, stat, p])
        print('Friedman statistics=%.3f, p=%.3f' % (stat, p))
    pd.DataFrame(out_list, columns=['dataset', 'stat', 'p-value']).to_csv('Friedman_test.csv')
    out_df.to_csv('all_ranks.csv')

    # Overall Friedmann test
    data = data.drop(['Dataset'], axis=1)
    grouped_item = data.groupby('Method', as_index=False)
    corrected_ranks = []
    for index, group in grouped_item:
        group = decreasing_ranking(group, ['Number of rules', 'Average rule length', 'Fraction overlap'])
        group = group.drop(['Method'], axis=1)
        corrected_ranks.append(group)
    data1, data2, data3, data4, data5 = corrected_ranks
    stat, p = friedmanchisquare(data1, data2, data3, data4, data5)
    print('Overall Friedman statistics=%.3f, p=%.3f' % (stat, p))

info_entropy = True
if info_entropy:
    data = data.drop(['Dataset', 'Method'], axis=1)
    data = normalize(data)
    x_entropy = []
    y_entropy = []
    for col in data.columns.tolist():
        print(col)
        x_entropy.append(col)
        distr = data[col].tolist()
        distr2 = [x if x > 0 else 0.0001 for x in distr]
        e = entropy(np.ones(len(distr2)), distr2)
        print("Entropy: ", e)
        distr3 = [x for x in distr if x > 0]
        e = entropy(np.ones(len(distr3)), distr3)
        print("Entropy: ", e)
        y_entropy.append(round(e, 2))
    sns.set(font_scale=1.5)
    ax = sns.barplot(x=x_entropy, y=y_entropy, palette="flare")
    ax.bar_label(ax.containers[0])
    plt.show()

draw_boxplot = False
if draw_boxplot:
    rank_data = pd.read_csv('all_ranks2.csv')
    rank_data = rank_data[rank_data['Metric'] == 'Total']
    # rank_data.boxplot(column=['C45Rule-PANE', 'REFNE', 'RxNCM', 'RxREN', 'TREPAN'])
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    sns.boxplot(data=rank_data[['C45Rule-PANE', 'REFNE', 'RxNCM', 'RxREN', 'TREPAN']], orient="h", palette="Set2")
    plt.show()

show_rules = False
if show_rules:
    rule = load_list('RxNCM_avila_final_rules', 'final_rules/')
    print(rule)

    rule2 = load_list('RxNCM_chess_final_rules', 'final_rules/')
    print(rule2)
