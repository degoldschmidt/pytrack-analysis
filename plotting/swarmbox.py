# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import matplotlib
import platform
matplotlib.use('TKAgg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica Neue'
if platform.system() == "Darwin":
    matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
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
                size=5, edgecolor="gray", linewidth=0, colors=None, ax=None, **kwargs):
    # default parameters
    defs = {
                'ps':   2,          # pointsize for swarmplot (3)
                'pc':   '#666666',  # pointcolor for swarmplot
                'w':    .5,         # boxwidth for boxplot (0.35)
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
                        orient=orient, color=color, palette=colors, saturation=defs['sat'],
                        width=defs['w'], linewidth=defs['lw'], ax=ax, boxprops=dict(lw=0.0), showfliers=False, **kwargs)
    # swarmplot
    ax = sns.swarmplot(x=sorted_x, y=y, hue=hue, data=data, order=order, hue_order=hue_order,
                        dodge=dodge, orient=orient, color=defs['pc'], palette=palette, size=defs['ps'], ax=ax, **kwargs)
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
        condition_table = ax.table(cellText=cells, cellLoc='center', rowLabels=rows, rowLoc = 'right', loc='bottom', fontsize=12, bbox=[0.00, -0.25, 1., 0.2])
        for k,v in condition_table._cells.items():
            this_text = condition_table._cells[k]._text.get_text()
            if this_text == u"\u2B24" or this_text == u"\u25EF":
                condition_table._cells[k]._text.set_fontname("DejaVu Sans")
                condition_table._cells[k]._text.set_fontsize(36)
        # Adjust layout to make room for the table:
        plt.subplots_adjust(left=0.2, bottom=0.2, top=0.9, hspace=0.4, wspace=0.5)
    return ax

def generate_data(dim):
    ### Generating fake example data
    categories = ['A','C','A','B','C']
    states = [False,False,True,True,True]
    listdfs = []
    means = [0.1, 0.1, 0.3, 1.5, 2.]
    stds = [0.1, 0.15, 0.25, 0.7, 1.]
    siz = 20
    for ix, acat in enumerate(categories[:dim]):
        y = np.clip(stds[ix]*np.random.randn(siz)+means[ix], 0, None)
        dfdict = {}
        dfdict['category'] = [acat]*siz
        dfdict['value'] = y
        dfdict['state'] = [states[ix]]*siz
        listdfs.append(pd.DataFrame(dfdict))
    return pd.concat(listdfs, ignore_index=True)



if __name__ == "__main__":
    import pandas as pd
    ### some data
    colors = ['#dd1c77', '#f9c6dd', '#2ca25f', '#99d8b3', '#b9eec3']
    many_df = [generate_data(4) for i in range(6)]

    # The actual magic
    f, axes = plt.subplots(2,3, figsize=(8,6), dpi=300)
    print(f.get_size_inches())
    ix = 0
    for row in axes:
        for ax in row:
            ax = swarmbox(x=['state', 'category'], y='value', data=many_df[ix], colors=colors, ax=ax)
            ix += 1
    f.suptitle('Careful, this is artificial data!')
    #plt.show()
    f.savefig("here.pdf")
