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
from scipy.stats import ranksums


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
def plot_along(f, ax, fullscreen=True):
    warnings.filterwarnings("ignore")
    mng = plt.get_current_fig_manager()
    ### works on Ubuntu??? >> did NOT working on windows
# mng.resize(*mng.window.maxsize())
    if fullscreen:
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

def annotate(i,j,pval,X,Y, stars=True, pad=1.1, ax=None, avoid=0, dy=0, align='center', _y=None, only_tick=False, _h=None, _ht=None):
    if ax is None:
        ax = plt.gca()
    if align == 'center':
        x = (i+j)/2   # x coordinate is in the middle between both two x-values
    elif align == 'right':
        x = j
    elif align == 'left':
        x = i
    ylims = ax.get_ylim()
    maxyX = np.max(X)
    maxyY = np.max(Y)
    maxY = np.max([maxyX, maxyY])
    h = 0.02*(ylims[1]-ylims[0])         # some extra height for the annotation
    if _h is not None:
        h = _h
    if avoid == 0:
        y = pad*maxY  # y coordinate is a bit above data
    else:
        y = avoid+dy
    if _y is not None:
        y = _y

    if _ht is None:
        ht = 0.1
    else:
        ht = _ht
    if stars:
        nstars = 0
        if pval < 0.05:
            nstars += 1
        if pval < 0.01:
            nstars += 1
        if pval < 0.001:
            nstars += 1
        if pval < 0.0001:
            nstars += 1
        if nstars > 0:
            ax.text(x, (y+h), nstars*"*", ha='center', va='center') # write stars
        else:
            if align == 'center':
                ax.text(x, (y+h+ht), "ns", ha='center', va='baseline', fontsize=8) # write ns
            else:
                ax.text(x, (y+h+ht), "ns", ha='center', va='baseline', fontsize=8) # write ns
    else:
        ax.text((i+j)/2, (y+h+ht), "p = {}".format(pval), ha='center', va='bottom') # write out the p-value text
    if align == 'center':
        ax.plot([i, i, j, j], [y, y+h, y+h, y], lw=1, c="k")  # plots a line
    elif align == 'right':
        if only_tick:
            ax.plot([j, j], [y+h, y], lw=1, c="k")  # plots a line
        else:
            ax.plot([i, j, j], [y+h, y+h, y], lw=1, c="k")  # plots a line
    elif align == 'left':
        if only_tick:
            ax.plot([i, i], [y, y+h], lw=1, c="k")  # plots a line
        else:
            ax.plot([i, i, j], [y+h, y, y], lw=1, c="k")  # plots a line
    return ax, y


def get_indices(ax, left, right):
    for index, lbltxt in enumerate([label.get_text() for label in ax.get_xticklabels()]):
        if lbltxt == left:
            pos0 = index
        if lbltxt == right:
            pos1 = index
    return pos0, pos1

"""
Swarmbox plot
"""
def swarmbox(x=None, y=None, hue=None, data=None, order=None, hue_order=None, m_order=None, multi=False,
                dodge=False, orient=None, color=None, palette=None, table=False, compare=[],
                size=5, edgecolor="gray", linewidth=0, colors=None, ax=None, boxonly=False, **kwargs):
    # default parameters
    defs = {
                'ps':   1.5,          # pointsize for swarmplot (3)
                'pc':   '#666666',  # pointcolor for swarmplot
                'w':    .65,         # boxwidth for boxplot (0.35)
                'lw':   1,        # linewidth for boxplot
                'sat':  0.5,         # saturation for boxplot
                'mlw':  0.3,        # width for median lines
    }

    # axis dimensions
    #ax.set_ylim([-2.,max_dur + 2.]) # this is needed for swarmplot to work!!!

    # actual plotting using seaborn functions
    # boxplot
    ax = sns.boxplot(x=x, y=y, hue=hue, data=data, order=order, hue_order=hue_order,
                        orient=orient, color=color, palette=palette, saturation=defs['sat'],
                        linewidth=defs['lw'], ax=ax, boxprops=dict(lw=0.0), whiskerprops={'linewidth':0}, capprops={'linewidth':0}, showfliers=False, **kwargs)
    #ax = sns.boxplot(x=x, y=y, hue=hue, data=data, palette=my_pal, showfliers=False, boxprops=dict(lw=1))
    # swarmplot
    if not boxonly:
        sns.stripplot(x=x, y=y, hue=hue, data=data, jitter=1.0, size=defs['ps'], dodge=True, color=".3", alpha=.75, ax=ax)
        """
        ax = sns.swarmplot(x=x, y=y, hue=hue, data=data, order=order, hue_order=hue_order, dodge=True, palette=palette,linewidth=1,edgecolor='gray',
                     orient=orient, size=defs['ps'], ax=ax, **kwargs)
        """

    ## figure aesthetics
    ax.tick_params('x', length=0, width=0, which='major')
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
    sns.despine(ax=ax, bottom=True, trim=True)
    ax.legend_.remove()

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

def arena(arena, spots, spot_pal={'yeast': '#ffc04c', 'sucrose': '#4c8bff'}, ref='center', despine=True, mark_center=True, ax=None):
    if ax is None:
        ax = plt.gca()
    ### data about arenas
    sca = arena['scale']
    if ref == 'center':
        xa, ya, orad, irad = 0., 0., arena['outer']/sca, arena['radius']/sca
    elif ref == 'video':
        xa, ya, orad, irad = arena['x'], arena['y'], arena['outer'], arena['radius']
    ### plot arena geometry as gray circles
    ax.add_artist(plt.Circle((xa, ya), orad, color='#bbbbbb', lw=1, fill=False))
    ax.add_artist(plt.Circle((xa, ya), irad, color='#717171', lw=1, fill=False))
    if mark_center:
        ax.add_artist(plt.Circle((xa, ya), 0.01*orad, color='#ff0000'))
    ### plot food spots as colored circles
    for spot in spots:
        if ref == 'center':
            x, y, r, s = spot['x'], spot['y'], spot['r'], spot['substr']
        elif ref == 'video':
            x, y, r, s = xa+spot['x']*sca, ya+spot['y']*sca, spot['r']*sca, spot['substr']
        ax.add_artist(plt.Circle((x, y), r, color=spot_pal[s], alpha=0.5))
    ### despining
    if despine:
        ax.set_xlim([-orad*1.1, orad*1.1])
        ax.set_ylim([-orad*1.1, orad*1.1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for k, spine in ax.spines.items():
            spine.set_visible(False)
    return ax

def trajectory(xc='head_x', yc='head_y', xs='body_x', ys='body_y', data=None, hue='etho', color=None, no_hue=[], to_body=[], lw=0.4, body_lw=0.5, size=.75, palette=None, ax=None):
    if ax is None:
        ax = plt.gca()
    if palette is None:
        palette = {    -1: '#ff00fc',
                        0: '#ff00c7',
                        1: '#c97aaa',
                        2: '#000000',
                        3: '#30b050',
                        4: '#ffc04c',
                        5: '#4c8bff',
                        6: '#ff1100'}
    red_data = data.copy()
    for each in no_hue:
        red_data = red_data.ix[~(red_data[hue] == each)]
    X, Y = data[xc], data[yc]
    rX, rY = red_data[xc], red_data[yc]
    H = red_data[hue].apply(lambda x: palette[x])
    ax.plot(X, Y, c='#424242', zorder=1, lw=lw, alpha=0.5)
    for each in to_body:
        qdata = data.query('{} == {}'.format(hue, each))
        for i, row in qdata.iterrows():
            #if i%2==0:
            if i+1 in qdata.index:
                if color is None:
                    ax.plot([row['body_x'], row['head_x']], [row['body_y'], row['head_y']], lw=body_lw, c=palette[each])
                else:
                    ax.plot([row['body_x'], row['head_x']], [row['body_y'], row['head_y']], lw=body_lw, c=color)
    if color is None:
        ax.scatter(rX, rY, color=H, zorder=2, s=size, alpha=0.75, marker='.')
    else:
        ax.scatter(rX, rY, color=color, zorder=2, s=size, alpha=0.75, marker='.')
    return ax

def tseries(data, x=None, y=[], c=[], label=[], ax=None, lw=[], ls=[], bottom=True):
    if ax is None:
        ax = plt.gca()
    for iy,each_y in enumerate(y):
        ax.plot(data[x], data[each_y], color=c[iy], lw=lw[iy], ls=ls[iy])
    return ax

def ccdf(data, xy=None, c=None, order=None, palette=None, ax=None):
    """
    Plot the complementary cumulative distribution function
    (1-CDF(x)) based on the data on the axes object.
    Note that this way of computing and plotting the ccdf is not
    the best approach for a discrete variable, where many
    observations can have exactly same value!
    """
    if ax is None:
        ax = plt.gca()
    # Note that, here we use the convention for presenting an
    # empirical 1-CDF (ccdf) as discussed
    # a quick way of computing a ccdf (valid for continuous data):
    ### get all y dimensions
    if order is None:
        myorder = np.unique(data[c])
    else:
        myorder = order
    for each in myorder:
        sub_data = data.query('{} == "{}"'.format(c, each))
        sorted_vals = np.sort(np.unique(sub_data[xy]))
        ccdf = np.zeros(len(sorted_vals))
        n = float(len(sub_data))
        for i, val in enumerate(sorted_vals):
            ccdf[i] = np.sum(sub_data[xy] >= val)/n
        ax.plot(sorted_vals, ccdf, ".", color=palette[each], markersize=4)
    ax.set_yscale('log')
    ax.set_xscale('log')
    # faster (approximative) way:
    # sorted_vals = np.sort(data)
    # ccdf = np.linspace(1, 1./len(data), len(data))
    # ax.plot(sorted_vals, ccdf)
    return ax
