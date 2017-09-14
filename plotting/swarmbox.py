# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import matplotlib
import sys
matplotlib.use('TKAgg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Helvetica Neue'
if sys.platform == "Darwin":
    matplotlib.rcParams['font.sans-serif'] = 'Helvetica'
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['font.weight'] = 'light'
import matplotlib.font_manager as fm
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

def merge(a,b):
    for ax, bx in zip(a,b):
        yield ax + bx

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
        sorted_order = order
    elif type(x) is list:
        if len(x) > 1:
            table=True
            all_combins = data[x].apply(tuple, axis=1)
            unique_ones = np.unique(all_combins)
            extra = []
            for each_tuple in unique_ones:
                number = 0.0
                if type(order) is list:
                    for lvl, orders in enumerate(order):
                        for index, item in enumerate(orders):
                            if each_tuple[lvl] == item:
                                number += index*10**(-lvl)
                    extra.append((number,))
            merged = list(merge(unique_ones, extra))
            merged = sorted(merged, key=lambda x: x[2])
            print("Swarmbox plot for multiple categories:", merged)
            data['metacat'] = [-1]*len(data.index)
            for ix, cat in enumerate(merged):
                data.loc[all_combins==cat[:-1], 'metacat'] = ix
            sorted_x = 'metacat'
            sorted_order = None
        else:
            sorted_x = x
            sorted_order = order

    # axis dimensions
    #ax.set_ylim([-2.,max_dur + 2.]) # this is needed for swarmplot to work!!!

    # actual plotting using seaborn functions
    # boxplot
    ax = sns.boxplot(x=sorted_x, y=y, hue=hue, data=data, order=sorted_order, hue_order=hue_order,
                        orient=orient, color=color, palette=colors, saturation=defs['sat'],
                        width=defs['w'], linewidth=defs['lw'], ax=ax, boxprops=dict(lw=0.0), showfliers=False, **kwargs)
    # swarmplot
    ax = sns.swarmplot(x=sorted_x, y=y, hue=hue, data=data, order=sorted_order, hue_order=hue_order,
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

    fontfile = "C:\\Windows\\Fonts\\Quicksand-Regular.ttf"
    print(sys.platform)
    if sys.platform == "darwin":
        fontfile = "/Users/degoldschmidt/Library/Fonts/Quicksand-Regular.ttf"
    print("Load font:", fontfile)
    textprop = fm.FontProperties(fname=fontfile)
    ax.set_ylabel(y, fontproperties=textprop, fontsize=12)
    for label in ax.get_yticklabels():
        label.set_fontproperties(textprop)

    # Add a table at the bottom of the axes
    ### requires UNICODE encoding
    if table:
        cells = []
        for ix, each_row in enumerate(x):
            if data[each_row].dtype == bool:
                #cells.append([ "⬤" if entry[ix] else "◯" for entry in unique_ones])
                cells.append([u"\u25CF" if entry[ix] else u"\u25CB" for entry in unique_ones])
            else:
                cells.append([str(entry[ix]) for entry in merged])
        rows = x
        xtrarows = [each.count("\n") for each in rows]
        nrows = len(rows)
        for each in xtrarows:
            nrows += each
        print("#rows:", nrows)
        condition_table = ax.table(cellText=cells, cellLoc='center', rowLabels=rows, rowLoc = 'right', loc='bottom', fontsize=12, bbox=[0.00, -0.35, 1., 0.3])
        for k,v in condition_table._cells.items():
            this_text = condition_table._cells[k]._text.get_text()
            if this_text == u"\u25CF" or this_text == u"\u25CB":
                condition_table._cells[k]._text.set_fontname("Arial Unicode MS")
                condition_table._cells[k]._text.set_fontsize(24)
            else:
                condition_table._cells[k]._text.set_fontproperties(textprop)
        table_props = condition_table.properties()
        table_cells = table_props['celld']
        for pos, cell in table_cells.items():
            cell.set_height(xtrarows[pos[0]]+1)
            cell.set_linewidth(0.5)
            cell.set_edgecolor('#424242')
        acells = table_props['child_artists']

        # Adjust layout to make room for the table:
        plt.subplots_adjust(top=0.9, bottom=0.05*nrows, hspace=0.15*nrows, wspace=1.)
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
