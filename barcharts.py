import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

by_method = False
if by_method:
    data = pd.read_csv('dataplot.csv')
    groups = data.groupby(['Metrics'], as_index=False)
    for index, item in groups:
        ax = item.plot.bar(x='Method', rot=0)
        ax.set_title(index, fontsize=18)
        ax.legend(loc="best", fontsize=12)
        ax.get_legend().remove()
        ax.plot()
        plt.show()
    ranks = pd.read_csv('all_ranks.csv')
    ax = ranks.plot.bar(x='Method', rot=0)
    ax.set_title('Method ranks', fontsize=18)
    ax.legend(loc="best", fontsize=12, bbox_to_anchor=(1, 1))
    ax.plot()
    plt.show()
else:
    data = pd.read_csv('dataplot2.csv')
    groups = data.groupby(['Metrics'], as_index=False)
    for index, item in groups:
        ax = item.plot.bar(x='Database', rot=18)
        ax.set_title(index, fontsize=18)
        ax.legend(loc="best", fontsize=12)
        #ax.get_legend().remove()
        ax.plot()
        plt.show()
    ranks = pd.read_csv('all_ranks2.csv')
    ax = ranks.plot.bar(x='Dataset', rot=18, fontsize=14)
    ax.set_title('Total ranks', fontsize=18)
    ax.set_ylabel('Sum of ranks', fontsize=14)
    ax.legend(loc="best", fontsize=14, bbox_to_anchor=(1, 1))
    ax.plot()
    plt.show()
