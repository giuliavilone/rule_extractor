import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys
matplotlib.use('TkAgg')
plt.rcParams["figure.figsize"] = (20, 5)

by_method = False
if by_method:
    data = pd.read_csv('dataplot.csv')
    groups = data.groupby(['Metric'], as_index=False)
    for index, item in groups:
        ax = item.plot.bar(x='Method', rot=0, fontsize=20, width=0.8)
        if index == 'Number of rules':
            ax.set_yscale('log')
        ax.set_title(index, fontsize=24)
        ax.legend(loc="best", fontsize=18, bbox_to_anchor=(1, 1), ncol=2)
        #ax.legend.remove()
        ax.plot()
        plt.tight_layout()
        plt.show()
    ranks = pd.read_csv('all_ranks2.csv')
    ranks = ranks[ranks['Metric'] == 'Total']
    ranks = ranks.set_index('Dataset')
    ranks = ranks.T
    ranks = ranks.reset_index()
    ranks = ranks.rename(columns={'index': 'Method'})
    ranks = ranks.drop(ranks[ranks['Method'] == 'Metric'].index)
    ax = ranks.plot.bar(x='Method', rot=0, fontsize=20, width=0.8)
    ax.set_title('Total ranks', fontsize=24)
    ax.legend(loc="best", fontsize=18, bbox_to_anchor=(1, 1), ncol=2)
    ax.plot()
    plt.tight_layout()
    plt.show()
else:
    data = pd.read_csv('dataplot2.csv')
    groups = data.groupby(['Metric'], as_index=False)
    for index, item in groups:
        ax = item.plot.bar(x='Dataset', rot=20, fontsize=18, width=0.7)
        if index == 'Number of rules':
            ax.set_yscale('log')
        ax.set_title(index, fontsize=24)
        ax.legend(loc="best", fontsize=18, bbox_to_anchor=(1, 1))
        # ax.get_legend().remove()
        ax.plot()
        plt.tight_layout()
        plt.xticks(ha="right")
        plt.show()
    ranks = pd.read_csv('all_ranks2.csv')
    ranks = ranks[ranks['Metric'] == 'Total']
    ax = ranks.plot.bar(x='Dataset', rot=20, fontsize=18)
    ax.set_title('Total ranks', fontsize=25)
    ax.set_ylabel('Sum of ranks', fontsize=24)
    ax.legend(loc="best", fontsize=18, bbox_to_anchor=(1, 1))
    ax.plot()
    plt.tight_layout()
    plt.xticks(ha="right")
    plt.show()


print_scatterplot = False
if print_scatterplot:
    # Scatterplot of input datasets
    colors = ['#808080', '#000000', '#FF0000', '#008000', '#00FFFF', '#F4D03F', '#0000FF', '#21618C', '#FF00FF',
              '#800080', '#34495E', '#8B0000', '#EA9C01', '#00FF2E', '#E74C3C']
    # Set your custom color palette
    #sns.set_palette(sns.color_palette(colors))

    # Create dataframe
    df = pd.DataFrame({
        'x': [2, 12, 2, 18, 3, 7, 2, 2, 2, 26, 2, 2, 11, 7, 2],
        'y': [48842, 20867, 45207, 28056, 67557, 581012, 30000, 14980, 17898, 20000, 12417, 12330, 164860, 58000, 245057],
        'Dataset': ['Adult', 'Avila', 'Bank marketing', 'Chess', 'Connect 4', 'Cover type', 'Credit card default',
                  'EEG eye state', 'HTRU', 'Letter recognition', 'Occupancy', 'Online shopper intention',
                  'Person activity', 'Shuttle', 'Skin']
    })

    p1 = sns.lmplot(data=df, x="x", y="y", fit_reg=False, scatter_kws={'s': 400})

    for line in range(0, df.shape[0]):
        if df.Dataset[line] == 'Occupancy':
            plt.gca().text(df.x[line] + 0.6, df.y[line]-1000, df.y[line], horizontalalignment='left', size='medium',
                           color='black')
        elif df.Dataset[line] == 'Adult':
            plt.gca().text(df.x[line] + 0.6, df.y[line]+100, df.y[line], horizontalalignment='left', size='medium',
                           color='black')
        else:
            plt.gca().text(df.x[line]+0.6, df.y[line], df.y[line], horizontalalignment='left', size='medium',
                           color='black')

    plt.xlabel('Number of output classes', fontsize=16)
    plt.ylabel('Number of instances', fontsize=16)
    plt.yscale('log')
    plt.yticks([])
    plt.legend(labelspacing=1)
    plt.show()

print_dataset_barchart = False
if print_dataset_barchart:
    # Stacked barplot of input datasets
    df = pd.DataFrame({
        'continuous': [6, 10, 11, 3, 0, 10, 20, 14, 8, 16, 5, 14, 3, 9, 3],
        'categorical': [8, 0, 9, 3, 42, 44, 3, 0, 0, 0, 0, 3, 1, 0, 0],
        'total': [14, 10, 20, 6, 42, 54, 23, 14, 8, 16, 5, 17, 4, 9, 3],
        'Dataset': ['Adult', 'Avila', 'Bank marketing', 'Chess', 'Connect 4', 'Cover type', 'Credit card default',
                  'EEG eye state', 'HTRU', 'Letter recognition', 'Occupancy', 'Online shopper intention',
                  'Person activity', 'Shuttle', 'Skin']
    })

    sns.barplot(x=df.Dataset, y=df.total, color='#E74C3C')

    #Plot 2 - overlay - "bottom" series
    bottom_plot = sns.barplot(x=df.Dataset, y=df.continuous, color='#21618C')

    topbar = plt.Rectangle((0, 0), 1, 1, fc='#E74C3C', edgecolor='white')
    bottombar = plt.Rectangle((0, 0), 1, 1, fc='#21618C',  edgecolor='white')
    l = plt.legend([bottombar, topbar], ['Continuous features', 'Categorical features'], loc=1, ncol=1, prop={'size': 20})
    l.draw_frame(False)

    #Optional code - Make plot look nicer
    sns.despine(left=True)
    bottom_plot.set_ylabel("Number of features")

    #Set fonts to consistent 16pt size
    for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
                 bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
        item.set_fontsize(16)

    plt.xticks(rotation='vertical', fontsize=24)
    plt.show()
