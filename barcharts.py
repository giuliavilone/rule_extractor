import tkinter
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

data = pd.read_csv('dataplot.csv')
groups = data.groupby(['Metrics'], as_index=False)
for index, item in groups:
    ax = item.plot.bar(x='Method', rot=0, title=index)
    ax.plot()
    plt.show()
