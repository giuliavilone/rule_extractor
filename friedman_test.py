from scipy.stats import friedmanchisquare
import pandas as pd

data = pd.read_csv('metrics.csv')

method_groups = data.groupby(['Method'], as_index=False)

method_list = []
for index, item in method_groups:
    item = item.drop(['Method', 'Dataset'], axis=1)
    item = item.to_numpy()
    method_list.append(item)

data1, data2, data3, data4 = method_list

stat, p = friedmanchisquare(data1, data2, data3, data4)
print('Statistics=%.3f, p=%.3f' % (stat, p))

