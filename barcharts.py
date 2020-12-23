import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
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
    ax.set_title('Total ranks', fontsize=18)
    ax.legend(loc="best", fontsize=12, bbox_to_anchor=(1, 1))
    ax.plot()
    plt.show()
# else:
    data = pd.read_csv('dataplot2.csv')
    groups = data.groupby(['Metrics'], as_index=False)
    for index, item in groups:
        ax = item.plot.bar(x='Dataset', rot=18)
        ax.set_title(index, fontsize=18)
        ax.legend(loc="best", fontsize=12)
        ax.get_legend().remove()
        ax.plot()
        plt.show()
    ranks = pd.read_csv('all_ranks2.csv')
    ax = ranks.plot.bar(x='Dataset', rot=18, fontsize=14)
    ax.set_title('Total ranks', fontsize=18)
    ax.set_ylabel('Sum of ranks', fontsize=14)
    ax.legend(loc="best", fontsize=14, bbox_to_anchor=(1, 1))
    ax.plot()
    plt.show()

# Scatterplot of input datasets
colors = ['#808080', '#000000', '#FF0000', '#008000', '#00FFFF', '#F4D03F', '#0000FF', '#21618C', '#FF00FF',
          '#800080', '#34495E', '#8B0000', '#EA9C01', '#00FF2E', '#E74C3C']
# Set your custom color palette
sns.set_palette(sns.color_palette(colors))

# Create dataframe
df = pd.DataFrame({
    'x': [2, 12, 2, 18, 3, 7, 2, 2, 2, 26, 2, 2, 11, 7, 2],
    'y': [48842, 20867, 45207, 28056, 67557, 581012, 30000, 14980, 17898, 20000, 12417, 12330, 164860, 58000, 245057],
    'Dataset': ['Adult', 'Avila', 'Bank marketing', 'Chess', 'Connect 4', 'Cover type', 'Credit card default',
              'EEG eye state', 'HTRU', 'Letter recognition', 'Occupancy', 'Online shopper intention',
              'Person activity', 'Shuttle', 'Skin']
})

p1 = sns.lmplot(data=df, x="x", y="y", hue='Dataset', fit_reg=False, scatter_kws={'s': 250})

for line in range(0, df.shape[0]):
    if df.Dataset[line] == 'Occupancy':
        plt.gca().text(df.x[line] + 0.6, df.y[line]-1000, df.y[line], horizontalalignment='left', size='small',
                       color='black')
    elif df.Dataset[line] == 'Adult':
        plt.gca().text(df.x[line] + 0.6, df.y[line]+100, df.y[line], horizontalalignment='left', size='small',
                       color='black')
    else:
        plt.gca().text(df.x[line]+0.6, df.y[line], df.y[line], horizontalalignment='left', size='small',
                       color='black')

plt.xlabel('Number of output classes')
plt.ylabel('Number of instances')
plt.yscale('log')
plt.yticks([])

plt.show()

# Stacked barplot of input datasets
df = pd.DataFrame({
    'continuous': [6, 10, 11, 3, 0, 10, 20, 14, 8, 16, 5, 14, 3, 9, 3],
    'categorical': [8, 0, 9, 2, 42, 44, 3, 0, 0, 0, 0, 3, 1, 0, 0],
    'total': [14, 10, 20, 5, 42, 54, 23, 14, 8, 16, 5, 17, 4, 9, 3],
    'Dataset': ['Adult', 'Avila', 'Bank marketing', 'Chess', 'Connect 4', 'Cover type', 'Credit card default',
              'EEG eye state', 'HTRU', 'Letter recognition', 'Occupancy', 'Online shopper intention',
              'Person activity', 'Shuttle', 'Skin']
})

sns.barplot(x=df.Dataset, y=df.total, color="red")

#Plot 2 - overlay - "bottom" series
bottom_plot = sns.barplot(x=df.Dataset, y=df.continuous, color="#0000A3")

topbar = plt.Rectangle((0, 0), 1, 1, fc="red", edgecolor='white')
bottombar = plt.Rectangle((0, 0), 1, 1, fc='#0000A3',  edgecolor='white')
l = plt.legend([bottombar, topbar], ['Continuous features', 'Categorical features'], loc=1, ncol=2, prop={'size': 16})
l.draw_frame(False)

#Optional code - Make plot look nicer
sns.despine(left=True)
bottom_plot.set_ylabel("Number of features")

#Set fonts to consistent 16pt size
for item in ([bottom_plot.xaxis.label, bottom_plot.yaxis.label] +
             bottom_plot.get_xticklabels() + bottom_plot.get_yticklabels()):
    item.set_fontsize(16)

plt.xticks(rotation='vertical')
plt.show()
