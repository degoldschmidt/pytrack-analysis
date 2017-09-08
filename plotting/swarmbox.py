# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica Neue'
matplotlib.rcParams['text.latex.unicode'] = True
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
                dodge=False, orient=None, color=None, palette=None, table=False,
                size=5, edgecolor="gray", linewidth=0, ax=None, **kwargs):
    # default parameters
    defs = {
                'ps':   3,          # pointsize for swarmplot
                'pc':   '#666666',  # pointcolor for swarmplot
                'w':    .35,        # boxwidth for boxplot
                'lw':   0.0,        # linewidth for boxplot
                'sat':  1.,         # saturation for boxplot
                'mlw':  0.3,        # width for median lines
    }

    # multiconditions
    if type(x) is str:
        sorted_x = x
    elif type(x) is list:
        if len(x) > 1:
            table=True
            all_combins = data[x].apply(tuple, axis=1)
            unique_ones = np.unique(all_combins)
            data['metacat'] = [-1]*len(data.index)
            for ix, cat in enumerate(unique_ones):
                data.loc[all_combins==cat, 'metacat'] = ix
            sorted_x = 'metacat'
        else:
            sorted_x = x

    # axis dimensions
    #ax.set_ylim([-2.,max_dur + 2.]) # this is needed for swarmplot to work!!!

    # actual plotting using seaborn functions
    # boxplot
    ax = sns.boxplot(x=sorted_x, y=y, hue=hue, data=data, order=order, hue_order=hue_order,
                        orient=orient, color=color, palette=palette, saturation=defs['sat'],
                        width=defs['w'], linewidth=defs['lw'], ax=ax, boxprops=dict(lw=0.0), showfliers=False, **kwargs)
    # swarmplot
    ax = sns.swarmplot(x=sorted_x, y=y, hue=hue, data=data, order=order, hue_order=hue_order,
                        dodge=dodge, orient=orient, color=defs['pc'], palette=palette, size=3, ax=ax, **kwargs)
    # median lines
    medians = data.groupby(sorted_x)[y].median()
    dx = defs['mlw']
    for pos, median in enumerate(medians):
        ax.hlines(median, pos-dx, pos+dx, lw=1.5, zorder=10)

    ## figure aesthetics
    #ax.set_yticks(np.arange(0, max_dur+1, div))
    sns.despine(ax=ax, bottom=True, trim=True)
    ax.get_xaxis().set_visible(False)

    # Add a table at the bottom of the axes
    ### requires UNICODE encoding
    if table:
        cells = []
        for ix, each_row in enumerate(x):
            if data[each_row].dtype == bool:
                #cells.append([ "⬤" if entry[ix] else "◯" for entry in unique_ones])
                cells.append([u"\u2B24" if entry[ix] else u"\u25EF" for entry in unique_ones])
            else:
                cells.append([str(entry[ix]) for entry in unique_ones])
        rows = x
        condition_table = plt.table(cellText=cells, cellLoc='center', rowLabels=rows, loc='bottom', fontsize=12, bbox=[0.00, -0.25, 1., 0.2])
        for k,v in condition_table._cells.items():
            this_text = condition_table._cells[k]._text.get_text()
            if this_text == u"\u2B24" or this_text == u"\u25EF":
                print(this_text) #DejaVu Sans
                condition_table._cells[k]._text.set_fontname("DejaVu Sans")
                condition_table._cells[k]._text.set_fontsize(36)
        # Adjust layout to make room for the table:
        plt.subplots_adjust(left=0.2, bottom=0.2)
    #plt.tight_layout()
    return ax

if __name__ == "__main__":
    import pandas as pd
    ### Generating fake example data
    categories = ['A','B','C','B','C']
    states = [False,False,False,True,True]
    listdfs = []
    means = [0.1, 0.1, 0.3, 1.5, 2.]
    stds = [0.1, 0.15, 0.25, 0.7, 1.]
    siz = 20
    for ix, acat in enumerate(categories):
        y = np.clip(stds[ix]*np.random.randn(siz)+means[ix], 0, None)
        dfdict = {}
        dfdict['category'] = [acat]*siz
        dfdict['value'] = y
        dfdict['state'] = [states[ix]]*siz
        listdfs.append(pd.DataFrame(dfdict))
    df = pd.concat(listdfs, ignore_index=True)

    # The actual magic
    f, ax = plt.subplots(1)
    ax = swarmbox(x=['state', 'category'], y='value', data=df, ax=ax)
    ax.set_title('Careful, this is artificial data!')
    #plt.show()
    f.savefig("here.pdf")
