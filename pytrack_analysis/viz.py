import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import imageio
from matplotlib.patches import Circle, Ellipse
import matplotlib.font_manager as fm
import matplotlib.table as mpl_table
import matplotlib.text as mpl_text
import tkinter as tk
import warnings
import numpy as np
import math
from datetime import timedelta
import os, sys
import seaborn
import seaborn as sns; sns.set(color_codes=True)
sns.set_style('ticks')


colors = [  '#a6cee3',
            '#1f78b4',
            '#b2df8a',
            '#33a02c',
            '#fb9a99',
            '#e31a1c',
            '#fdbf6f',
            '#ff7f00',
            '#cab2d6',
            '#6a3d9a',
            '#ffff99',
            '#b15928']

spot_colors = {'yeast': '#ffc04c', 'sucrose': '#4c8bff'}

"""
Plot figs along program flow (VISUAL)
"""
def plot_along(f, ax):
    warnings.filterwarnings("ignore")
    mng = plt.get_current_fig_manager()
    ### works on Ubuntu??? >> did NOT working on windows
# mng.resize(*mng.window.maxsize())
    mng.window.state('zoomed') #works fine on Windows!
    f.show()
    try:
        f.canvas.start_event_loop(0)
    except tk.TclError:
        pass
    warnings.filterwarnings("default")


"""
Plotting trajectory in arenas
"""
def plot_fly(data, x=None, y=None, hx=None, hy=None, arena=None, spots=None, title=None):
    f, ax = plt.subplots()
    if arena is not None:
        arena_border = plt.Circle((0, 0), arena.rr, color='k', fill=False)
        ax.add_artist(arena_border)
        outer_arena_border = plt.Circle((0, 0), arena.ro, color='#aaaaaa', fill=False)
        ax.add_artist(outer_arena_border)
        ax.plot(0, 0, 'o', color='black', markersize=2)
    if spots is not None:
        for each_spot in spots:
            substr = each_spot.substrate
            spot = plt.Circle((each_spot.rx, each_spot.ry), each_spot.rr, color=spot_colors[substr], alpha=0.5)
            ax.add_artist(spot)
    ax.plot(data[x], data[y])
    ax.plot(data[hx], data[hy], 'r-')
    if arena is not None:
        ax.set_xlim([-1.1*arena.ro, 1.1*arena.ro])
        ax.set_ylim([-1.1*arena.ro, 1.1*arena.ro])
    ax.set_aspect("equal")
    return f, ax

"""
Plotting trajectory intervals in arenas
"""
def plot_interval(ax, data, x=None, y=None, flip=None, time=None, arena=None, spots=None, start=None, end=None, title=None, **kwargs):
    if arena is not None:
        arena_border = plt.Circle((0, 0), arena.rr, color='k', fill=False)
        ax.add_artist(arena_border)
        outer_arena_border = plt.Circle((0, 0), arena.ro, color='#aaaaaa', fill=False)
        ax.add_artist(outer_arena_border)
        ax.plot(0, 0, 'o', color='black', markersize=2)
    if spots is not None:
        for each_spot in spots:
            substr = each_spot.substrate
            spot = plt.Circle((each_spot.rx, each_spot.ry), each_spot.rr, color=spot_colors[substr], alpha=0.5)
            ax.add_artist(spot)

    ### arrays where head is flipped or not
    flips = np.array(data.loc[start:end, flip])
    x_flip = np.array(data.loc[start:end, x])[flips>0]
    y_flip = np.array(data.loc[start:end, y])[flips>0]
    x_noflip = np.array(data.loc[start:end, x])[flips==0]
    y_noflip = np.array(data.loc[start:end, y])[flips==0]

    if 's' in kwargs.keys():
        pointsize = kwargs['s']
    else:
        pointsize = 2
    ax.scatter(x_flip, y_flip, marker='.', c='g', edgecolor='none', zorder=5, s=pointsize)
    ax.scatter(x_noflip, y_noflip, marker='.', c='r', edgecolor='none', zorder=5, s=pointsize)
    if arena is not None:
        ax.set_xlim([-1.1*arena.ro, 1.1*arena.ro])
        ax.set_ylim([-1.1*arena.ro, 1.1*arena.ro])
    ax.set_aspect("equal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    tstr = timedelta(seconds=int(data.loc[start, time]))
    ax.set_title("{} s".format(tstr))
    ax.axis('off')
    return ax

def plot_intervals(n, data, x=None, y=None, flip=None, time=None, arena=None, spots=None, title=None, ncols=6):
    sc = 3
    f, axes = plt.subplots(math.ceil(n/ncols), ncols, figsize=(sc*ncols, sc*math.ceil(n/ncols)), dpi=600)
    for ir, ar in enumerate(axes):
        for ic, ax in enumerate(ar):
            this_index = ic + ir * ncols
            start = session_data.first_frame + i*(108000/n)
            end = start + 108000/n - 1
            ax = plot_interval(ax, data, x=x, y=y, flip=flip, time=time, start=start, end=end, arena=arena, spots=spots, title=title)
    return f, axes

"""
Plotting overlay
"""
def plot_overlay(datal, frame, x=None, y=None, hx=None, hy=None, arenas=None, scale=0, trace=0, video=None):
    vid = imageio.get_reader(video)
    f, ax = plt.subplots()
    image = vid.get_data(frame)
    ax.imshow(image)

    for ix,data in enumerate(datal):
        arena = arenas[ix]
        arena_border = plt.Circle((arena.x, arena.y), arena.r, color='#f96bde', fill=False)
        ax.add_artist(arena_border)
        outer_arena_border = plt.Circle((arena.x, arena.y), arena.outer, color='#fa1edd', fill=False)
        ax.add_artist(outer_arena_border)
        ax.plot(arena.x, arena.y, '+', color='#fa1edd', markersize=10)
        for each_spot in arena.spots:
            substr = each_spot.substrate
            spot = plt.Circle((each_spot.x + arena.x, each_spot.y + arena.y), each_spot.r, color=spot_colors[substr], alpha=0.5)
            ax.add_artist(spot)
        x0 = arena.x
        y0 = arena.y
        xtrace = np.array(data.loc[frame-trace:frame+1,x]) * scale + x0
        ytrace = -np.array(data.loc[frame-trace:frame+1,y]) * scale + y0
        xp = data.loc[frame,x] * scale + x0
        yp = -data.loc[frame,y] * scale + y0
        major = data.loc[frame,'major'] * scale
        minor = data.loc[frame,'minor'] * scale
        angle = data.loc[frame,'angle']
        ax.plot(xtrace, ytrace,'m-', alpha=0.5, lw=0.25)
        e = Ellipse((xp, yp), major, minor, angle=np.degrees(angle), color="#6bf9b5", alpha=0.25)
        ax.add_artist(e)
        if hx is not None:
            hxp = data.loc[frame,hx] * scale + x0
            hyp = -data.loc[frame,hy] * scale + y0
            head = Circle((hxp, hyp), 2, color="#6bf9b5")
            ax.add_artist(head)
    return f, ax

"""
Plotting timeseries
"""
def plot_ts(data, x=None, y=None, units=None):
    if type(y) is str:
        ys = [y]
    elif type(y) is list:
        ys = y
    if units is None:
        units = len(ys)*[""]
    nplots = len(ys)
    f, axs = plt.subplots(nplots, sharex=True)
    axs = np.array(axs)
    if y is not None:
        for i, ax in enumerate(axs.reshape(-1)):
            if x is None:
                ax.plot(data[ys[i]], color=colors[i])
            else:
                if x == 'frame':
                    ax.plot(data.index, data[ys[i]], color=colors[i])
                else:
                    ax.plot(data[x], data[ys[i]], color=colors[i])
                if i == nplots-1:
                    ax.set_xlabel(x)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ylabel = "{}\n[{}]".format(ys[i], units[i])
            ax.set_ylabel(ylabel, rotation=0, fontsize=11, labelpad=30)
    return f, axs

"""
Swarmbox plot
"""
def swarmbox(x=None, y=None, hue=None, data=None, order=None, hue_order=None, m_order=None, multi=False,
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

    # axis dimensions
    #ax.set_ylim([-2.,max_dur + 2.]) # this is needed for swarmplot to work!!!

    # actual plotting using seaborn functions
    # boxplot
    ax = sns.boxplot(x=x, y=y, hue=hue, data=data, order=order, hue_order=hue_order,
                        orient=orient, color=color, palette=palette, saturation=defs['sat'],
                        width=defs['w'], linewidth=defs['lw'], ax=ax, boxprops=dict(lw=0.0), showfliers=False, **kwargs)
    #ax = sns.boxplot(x=x, y=y, hue=hue, data=data, palette=my_pal, showfliers=False, boxprops=dict(lw=1))
    # swarmplot
    ax = sns.swarmplot(x=x, y=y, hue=hue, data=data, order=order, hue_order=hue_order, dodge=True,
                     orient=orient, color=defs['pc'], size=defs['ps'], ax=ax, **kwargs)
    # median lines
    medians = data.groupby(x)[y].median()
    #print(medians)
    dx = defs['mlw']
    new = m_order
    if new is not None:
        for pos, median in enumerate(medians):
            ax.hlines(median, new[pos]-dx, new[pos]+dx, lw=1.5, zorder=10)
    else:
        for pos, median in enumerate(medians):
            ax.hlines(median, pos-dx, pos+dx, lw=1.5, zorder=10)

    ## figure aesthetics
    #ax.set_yticks(np.arange(0, max_dur+1, div))
    sns.despine(ax=ax, bottom=True, trim=True)
    #ax.get_xaxis().set_visible(False)
    ax.tick_params('x', length=0, width=0, which='major')

    # Adjust layout to make room for the table:
    #plt.subplots_adjust(top=0.9, bottom=0.05*nrows, hspace=0.15*nrows, wspace=1.)
    return ax

def set_font(name, ax=None, VERBOSE=False):
    if ax is None:
        ax = plt.gca()
    for sites in sys.path:
        if os.path.isdir(sites):
            if "fonts" in [folder for folder in os.listdir(sites) if os.path.isdir(os.path.join(sites,folder))]:
                fontfile = os.path.join(sites, "fonts", name+".ttf")

    if os.path.exists(fontfile):
        if VERBOSE: print("Loading font:", fontfile)
        textprop = fm.FontProperties(fname=fontfile)
        ### find all text objects
        texts = ax.findobj(match=mpl_text.Text)
        for eachtext in texts:
            eachtext.set_fontproperties(textprop)
        ### find all table objects (these contain further text objects)
        tables = ax.findobj(match=mpl_table.Table)
        for eachtable in tables:
            for k, eachcell in eachtable._cells.items():
                this_text = eachcell._text.get_text()
                if this_text == u"\u25CF" or this_text == u"\u25CB":
                    eachcell._text.set_fontname("Arial Unicode MS")
                else:
                    eachcell._text.set_fontproperties(textprop)
    else:
        print(fontfile, os.getcwd())
        print("[ERROR]: selected font does not exist.")
        raise FileNotFoundError
    return ax
