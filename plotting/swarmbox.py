# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica Neue'
matplotlib.rcParams['font.weight'] = 'light'
import matplotlib.pyplot as plt

import seaborn as sns

#### plotting
def stars(p):
    if p < 0.0001:
        return "****"
    elif (p < 0.001):
        return "***"
    elif (p < 0.01):
        return "**"
    elif (p < 0.05):
        return "*"
    else:
        return "ns"

def swarmbox(x=None, y=None, hue=None, data=None, order=None, hue_order=None,
                dodge=False, orient=None, color=None, palette=None,
                size=5, edgecolor="gray", linewidth=0, ax=None, **kwargs):
    # default parameters
    defs = {
                'ps':   3,          # pointsize for swarmplot
                'pc':   '#666666',  # pointcolor for swarmplot
                'w':    .35,        # boxwidth for boxplot
                'lw':   0.0,        # linewidth for boxplot
                'sat':  1.,         # saturation for boxplot
    }

    # axis dimensions


    # actual plotting using seaborn functions
    # boxplot
    ax = sns.boxplot(x=x, y=y, hue=hue, data=data, order=order, hue_order=hue_order,
                        orient=orient, color=color, palette=palette, saturation=defs['sat'],
                        width=defs['w'], linewidth=defs['lw'], ax=ax, boxprops=dict(lw=0.0), showfliers=False, **kwargs)
    # swarmplot
    ax = sns.swarmplot(x=x, y=y, hue=hue, data=data, order=order, hue_order=hue_order,
                        dodge=dodge, orient=orient, color=defs['pc'], palette=palette, size=3, ax=ax, **kwargs)
    # median lines
    medians = data.groupby(['state', 'class'])['value'].median()
    dx = 0.3
    for pos, median in enumerate(medians):
        ax.hlines(median, pos-dx, pos+dx, lw=1.5, zorder=10)
    return ax

if __name__ == "__main__":
    import pandas as pd

    ### Example Data
    classes = ['A','B','C','B','C']
    states = [0,0,0,1,1]
    metacats = ['A0', 'B0', 'C0', 'B1', 'C1']
    listdfs = []
    means = [0.1, 0.1, 0.3, 1.5, 2.]
    stds = [0.1, 0.15, 0.25, 0.7, 1.]
    siz = 20
    for ix, aclass in enumerate(classes):
        y = np.clip(stds[ix]*np.random.randn(siz)+means[ix], 0, None)
        dfdict = {}
        dfdict['metacat'] = [metacats[ix]]*siz
        dfdict['class'] = [aclass]*siz
        dfdict['value'] = y
        dfdict['state'] = [states[ix]]*siz
        listdfs.append(pd.DataFrame(dfdict))
    df = pd.concat(listdfs)
    #print(df)

    f, ax = plt.subplots(1)
    ax = swarmbox(x='metacat', y='value', data=df, ax=ax)
    ax.set_title('Careful, this is artificial data!')
    plt.show()
