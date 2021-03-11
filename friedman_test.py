from scipy.stats import friedmanchisquare
import pandas as pd
from scipy.stats import rankdata


def method_ranking(df):
    ret_list = []
    for label, col in df.iteritems():
        if (label == 'Method') or (label == 'Dataset'):
            col = col.tolist()
            col.insert(0, label)
            ret_list.append(col)
        else:
            col = col.to_numpy()
            ranks = rankdata(col).tolist()
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


data = pd.read_csv('metrics.csv')
method_groups = data.groupby(['Dataset'], as_index=False)

method_list = []
out_list = []
out_df = pd.DataFrame()
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
