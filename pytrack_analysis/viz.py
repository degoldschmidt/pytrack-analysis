import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import imageio
from matplotlib.patches import Circle, Ellipse
import tkinter as tk
import warnings
import numpy as np
import math


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
def plot_intervals(n, data, x=None, y=None, hx=None, hy=None, arena=None, spots=None, title=None):
    f, axes = plt.subplots(math.ceil(n/4), 4, figsize=(2*4, 2*math.ceil(n/4)), dpi=300)
    for ir, ar in enumerate(axes):
        for ic, ax in enumerate(ar):
            this_index = ic + ir * 4
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
            first_frame = data.index[0]
            start = first_frame + this_index*(108000/n)
            end = start + 108000/n - 1
            ax.plot(data.loc[start:end, x], data.loc[start:end, y])
            ax.plot(data.loc[start:end, hx], data.loc[start:end, hy], 'r-')
            if arena is not None:
                ax.set_xlim([-1.1*arena.ro, 1.1*arena.ro])
                ax.set_ylim([-1.1*arena.ro, 1.1*arena.ro])
            ax.set_aspect("equal")
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
