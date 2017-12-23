import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import imageio
from matplotlib.patches import Circle, Ellipse
import tkinter as tk
import warnings
import numpy as np


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
    ax.plot(data[x], data[y])
    ax.plot(data[hx], data[hy], 'r-')
    ax.set_aspect('equal')
    return f, ax

"""
Plotting overlay
"""
def plot_overlay(datal, frame, x=None, y=None, arena=None, scale=0, trace=0, video=None):
    vid = imageio.get_reader(video)
    f, ax = plt.subplots()
    image = vid.get_data(frame)
    ax.imshow(image)
    for ix,data in enumerate(datal):
        x0 = arena[ix].x
        y0 = arena[ix].y
        xtrace = np.array(data.loc[frame-trace:frame+1,x]) * scale + x0
        ytrace = np.array(data.loc[frame-trace:frame+1,y]) * scale + y0
        xp = data.loc[frame,x] * scale + x0
        yp = data.loc[frame,y] * scale + y0
        major = data.loc[frame,'major'] * scale
        minor = data.loc[frame,'minor'] * scale
        angle = data.loc[frame,'angle']
        ax.plot(xtrace, ytrace,'m-', alpha=0.5, lw=0.25)
        e = Ellipse((xp, yp), major, minor, angle=np.degrees(angle), edgecolor="#6bf9b5", lw=1, facecolor='none', alpha=0.6)
        ax.add_artist(e)
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
