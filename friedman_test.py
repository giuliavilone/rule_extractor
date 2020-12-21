from scipy.stats import friedmanchisquare
import pandas as pd
from scipy.stats import rankdata


def method_ranking(df, dataset):
    outlist = []
    for label, col in df.iteritems():
        if (label == 'Method') or (label == 'Dataset'):
            col = col.tolist()
            col.insert(0, label)
            outlist.append(col)
        else:
            col = col.to_numpy()
            ranks = rankdata(col).tolist()
            #if (label == 'n_rules') or (label == 'avg_length'):
            #    ranks = list(map(lambda x: 5-x, ranks))
            ranks.insert(0, label)
            outlist.append(ranks)
        outdf = pd.DataFrame(outlist, columns=['item', 'rank1', 'rank2', 'rank3', 'rank4', 'rank5'])
        outdf.to_csv('ranks_'+dataset+'.csv')


data = pd.read_csv('metrics.csv')
method_groups = data.groupby(['Dataset'], as_index=False)


method_list = []
out_list = []
out_df = pd.DataFrame()
for index, item in method_groups:
    print(index)
    grouped_item = item.groupby('Method', as_index=False).mean()
    grouped_item['Dataset'] = index
    out_df = out_df.append(grouped_item)
    # method_ranking(grouped_item, index)
    grouped_item = grouped_item.drop(['Method', 'Dataset'], axis=1)
    grouped_item = grouped_item.to_numpy()
    data1, data2, data3, data4, data5 = grouped_item
    stat, p = friedmanchisquare(data1, data2, data3, data4, data5)
    out_list.append([index, stat, p])
    print('Friedman statistics=%.3f, p=%.3f' % (stat, p))
pd.DataFrame(out_list, columns=['dataset', 'stat', 'p-value']).to_csv('Friedman_test.csv')
out_df.to_csv('Mean_metrics_values.csv')
