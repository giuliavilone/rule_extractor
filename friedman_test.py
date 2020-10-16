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
        outdf = pd.DataFrame(outlist, columns=['item', 'rank1', 'rank2', 'rank3', 'rank4'])
        outdf.to_csv('ranks_'+dataset+'.csv')


data = pd.read_csv('metrics.csv')
method_groups = data.groupby(['Dataset'], as_index=False)


method_list = []
for index, item in method_groups:
    print(index)
    method_ranking(item, index)
    item = item.drop(['Method', 'Dataset'], axis=1)
    item = item.to_numpy()
    data1, data2, data3, data4 = item
    stat, p = friedmanchisquare(data1, data2, data3, data4)
    print('Friedman statistics=%.3f, p=%.3f' % (stat, p))

