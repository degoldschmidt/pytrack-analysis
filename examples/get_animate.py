"""
===============
Demo Animation
===============
"""
from pytrack_analysis.profile import get_profile
import seaborn as sns
import numpy as np
import imageio
import pandas as pd
import os, sys

from pytrack_analysis.plot import set_font, swarmbox
from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Classifier
from pytrack_analysis import Multibench
import pytrack_analysis.plot as plot

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as anim
import argparse

ONLY_VIDEO = False
ONLY_TRAJ = False
NO_ANNOS = False

MAGENTA = '#ff00c7'
BLACK = '#000000'
WHITE = '#ffffff'
ROSE = '#c97aaa'
GREEN = '#30b050'
YEAST = '#ffc04c'
SUCROSE = '#4c8bff'
RED = '#ff1100'

palette = {    -1: RED,
                0: RED,
                1: WHITE,
                2: WHITE,
                3: WHITE,
                4: YEAST,
                5: SUCROSE,
                6: RED}
palette2 ={     0: WHITE,
                1: YEAST,
                2: SUCROSE}


def unflatten(l):
  newlist = []
  for each in l:
    if type(each) is list:
      for i in each:
        newlist.append(i)
    else:
      newlist.append(each)
  return newlist

def get_tseries_axes(grid, position, sharex=None, n=1, labels=[]):
    row=position[0]
    col=position[1]
    if sharex is None:
        ax = plt.subplot(grid[row, col:], sharex=sharex)
    else:
        ax = plt.subplot(grid[row, col:])
        ax.xaxis.set_visible(False)
    if n == 1:
        lines, = ax.plot([], [])
    elif n > 1:
        lines = []
        for i in range(n):
            line, = ax.plot([], [], label=labels[i])
            lines.append(line)
    return ax, lines


# animate time series
def animate(frame, *args):
    lines = args[0]
    xarray = args[1]
    yarrays = args[2]
    ax_ts = args[3]
    cols = args[4]
    play_line = args[5]
    xdata = np.array(xarray.loc[:frame+1])
    ydata = [None for i in range(6)]
    for j, each in enumerate(cols):
      if j < 2:
          ydata[j] = int(yarrays.loc[frame, each])
      else:
          ydata[j] = np.array(yarrays.loc[:frame+1, each])
    # update the data of both line objects
    for pl in play_line:
        pl.set_xdata(xdata[-1])
    for j,each_line in enumerate(lines):
        if j < 2:
            if j == 1:
                if ydata[j] > 3:
                    ax_ts[j].vlines(xarray.loc[frame], 0, 1, color=palette[ydata[j]])
            if j == 0:
                if ydata[j] > 0:
                    ax_ts[j].vlines(xarray.loc[frame], 0, 1, color=palette2[ydata[j]])
                if frame-1 in xarray.index:
                    if xarray.loc[frame] - xarray.loc[frame-1] > 0.1:
                        a = xarray.loc[frame]-0.0333
                        while a > xarray.loc[frame-1]:
                            if ydata[j] > 0:
                                ax_ts[j].vlines(a, 0, 1, color=palette2[ydata[j]])
                            a -= 0.0333
        else:
            each_line.set_data(xdata, ydata[j])
    return lines

# animate video
radius = int(8.543 * 12.5)
print(radius) ##100
def updatefig(i, *image):
    beh = {    -1: "unclassified",
                0: "resting",
                1: "micromovement",
                2: "walking",
                3: "sharp turn",
                4: "yeast micromovement",
                5: "sucrose micromovement",
                6: "jump/mistrack"}
    xpos, ypos = image[3][0][i-4970], image[3][1][i-4970]
    etho = image[3][2][i-4970]
    bxpos, bypos = image[3][3][i-4970], image[3][4][i-4970]
    y = int(359.3853 - 8.543 * (-1.))##image[3][1][i-6101]
    x = int(366.1242 + 8.543 * 15.)##image[3][0][i-6101]
    a, b = (x-radius), (y-radius)
    if i%100==0:
        print(i, xpos, ypos, etho)

    ### update plots
    if len(image[2].get_lines()) == 0:
        image[2].plot([xpos-a, bxpos-a], [ypos-b, bypos-b], color='#ff228c', ls='-', lw=1)
    image[0].set_array(image[1].get_data(i)[y-radius:y+radius, x-radius:x+radius])
    if NO_ANNOS:
        image[2].set_title("frame #{}".format(i))
    else:
        image[2].set_title("frame #{}".format(i))
        #image[2].set_title("frame #{}\n{}".format(i, beh[etho]))
    if not NO_ANNOS:
        if palette[etho] == '#ffffff':
            al = 1
            ms = 2
        else:
            al = 1
            ms = 3
        image[2].plot(xpos-a, ypos-b, color=palette[etho], marker='.', alpha=al, markersize=ms)
        image[2].get_lines()[0].set_data([xpos-a, bxpos-a], [ypos-b, bypos-b])
        image[2].get_lines()[0].set_color(palette[etho])
    else:
        image[2].plot(xpos-a, ypos-b, color='#ff228c', marker='.', markersize=4)
        image[2].get_lines()[0].set_data([xpos-a, bxpos-a], [ypos-b, bypos-b])
    return image[0],

def init_animation(data, time=None, cols=None, video=None, figsize=(10.666,6), interval=None, playback=1, meta=None):
    ts_colors = [   '#000000',
                    '#000000',
                    '#fb8072',
                    '#1f5fd5',
                    '#000000',
                    '#000000',]
    spot_colors = {'yeast': '#ffc04c', 'sucrose': '#4c8bff'}
    fig = plt.figure(figsize=figsize, dpi=180)
    #fig.set_size_inches(1920/300., 1080/300., True)
    ## gridspec
    N = len(cols)   # number of rows
    if interval is None:
        T = np.array(data.index)
    else:
        T = np.arange(interval[0],interval[1],dtype=np.int32)
    xarray = data.loc[T[0]:T[-1], time] - data.loc[T[0], time]
    ucols = unflatten(cols)
    print(ucols)
    yarrays = data.loc[T[0]:T[-1], ucols]
    gs = GridSpec(N, 2*N+1, height_ratios=[2,2,2,1,1])
    gs.update(wspace=2)
    ## video axis
    vid = imageio.get_reader(video)
    if ONLY_VIDEO:
        fig, ax_video = plt.subplots(figsize=figsize, dpi=180)
        #fig.set_size_inches(1920/300., 1080/300., True)
    else:
        ax_video = plt.subplot(gs[:,:5]) ### N == 5
    ax_video.set_aspect('equal')
    sns.despine(ax=ax_video, bottom=True, left=True)
    ax_video.get_xaxis().set_visible(False)
    ax_video.get_yaxis().set_visible(False)
    opos = ax_video.get_position()
    #if not ONLY_VIDEO:
      #ax_video.set_position([opos.x0-0.075, opos.y0, opos.width*1.25, opos.height*1.25])
    # initial frame
    scale = meta['arena']['scale']
    radius = int(scale * 12.5)
    x0, y0 = meta['arena']['x'], meta['arena']['y']
    x, y = int(x0 + scale * 15.), int(y0 - scale * (-1.))

    im = ax_video.imshow(vid.get_data(interval[0])[y-radius:y+radius, x-radius:x+radius], animated=True)
    if not ONLY_TRAJ:
        for ii,each in enumerate(meta['food_spots']):
            sx, sy = scale * each['x'] + x0 - (x-radius), -scale * each['y'] + y0 - (y-radius)
            ax_video.add_artist(plt.Circle((sx,sy), scale*1.5, color=spot_colors[each['substr']], lw=2.5, alpha=0.5, fill=False, zorder=100))
            if ii in [1, 3, 9]:
                ax_video.add_artist(plt.Circle((sx,sy), scale*2.5, color='#ffffff', ls='dashed', lw=1.5, alpha=0.5, fill=False, zorder=100))
            if ii == 1:
                ax_video.add_artist(plt.Circle((sx,sy), scale*5, color='#ffffff', ls='dotted', lw=1, alpha=0.5, fill=False, zorder=100))
            #ax_video.text(sx,sy, "{}".format(ii), zorder=100)

    ax_tseries, lines = None, None
    if not ONLY_VIDEO:
        ax_tseries, lines, play_line = [], [], []
        ax, l = get_tseries_axes(gs, [4, 5])
        play_line.append(ax.axvline(0, ymax=2, c='#ff0000', ls='-', lw=.75, zorder=100, clip_on=False))
        opos = ax.get_position()
        ax.set_position([opos.x0+0.025, opos.y0, opos.width*1.05, opos.height])
        l.set_color(ts_colors[0])
        ax.set_xlim(0, 1.1*np.max(xarray))
        if np.min(data[cols[-1]]) < 0:
            ax.set_ylim(1.1*np.min(data[cols[-1]]), 1.1*np.max(data[cols[-1]]))
        else:
            ax.set_ylim(0, 1.1*np.max(data[cols[-1]]))
        sns.despine(ax=ax, trim=True)
        ax.set_xlabel(time)
        ax.set_ylabel(cols[-1])
        ax_tseries.append(ax)
        lines.append(l)
        count = 0
        for i in range(1,5):
            if type(cols[i]) is list:
              ax, l = get_tseries_axes(gs, [4-i, 5], sharex=ax_tseries[0], n=2, labels=['head', 'body'])
            else:
              ax, l = get_tseries_axes(gs, [4-i, 5], sharex=ax_tseries[0])
            if i < 3:
                play_line.append(ax.axvline(0, ymax=3, c='#ff0000', ls='-', lw=.75, zorder=100, clip_on=False))
            else:
                play_line.append(ax.axvline(0, ymax=1, c='#ff0000', ls='-', lw=.75, zorder=100, clip_on=False))
            opos = ax.get_position()
            ax.set_position([opos.x0+0.025, opos.y0, opos.width*1.05, opos.height])
            if type(l) is list:
              for jj,each in enumerate(l):
                each.set_color(ts_colors[count])
                lines.append(each)
                ax.legend()
                count += 1
            else:
              l.set_color(ts_colors[count])
              lines.append(l)
              count += 1
            ax_tseries.append(ax)

        print(len(ax_tseries), len(lines))

        ### thresholds
        if not NO_ANNOS:
            print('Thresholds drawn')
            ax_tseries[4].hlines(2.5, 0, 30, colors="#aaaaaa", linestyles='dashed')
            ax_tseries[4].hlines(5, 0, 30, colors="#aaaaaa", linestyles='dashed')
            ax_tseries[3].hlines(2., 0, 30, colors="#aaaaaa", linestyles='dashed')
            #ax_tseries[3].hlines(4., 0, 30, colors="#aaaaaa", linestyles='dashed')
            ax_tseries[2].hlines(125., 0, 30, colors="#aaaaaa", linestyles='dashed')
            ax_tseries[2].hlines(-125., 0, 30, colors="#aaaaaa", linestyles='dashed')

    return fig, ax_video, ax_tseries, T, xarray, yarrays, im, vid, lines, play_line, ucols

def run_animation(fig, frames, xarray, yarrays, lines, play_line, im, vid, ax_video, ax_ts, pixelpos, factor=1, cols=None, outfile="out"):
    myinterval = 1000./(30*factor)
    print("Interval between frame: {}".format(myinterval))
    if not ONLY_VIDEO:
        ani_lines = anim.FuncAnimation(fig, animate, frames, blit=True, fargs=(lines, xarray, yarrays, ax_ts, cols, play_line), interval=myinterval)
    ani_image = anim.FuncAnimation(fig, updatefig, frames, blit=True, fargs=(im, vid, ax_video, pixelpos), interval=myinterval)
    #plt.tight_layout()
    print(fig.get_size_inches()*fig.dpi)
    if ONLY_VIDEO:
        #ani_image.save(outfile+'.gif', dpi=90, writer='imagemagick')
        ani_image.save(outfile+'.mp4', writer='ffmpeg', dpi=180)
    else:
        #ani_lines.save(outfile+'.gif', extra_anim=[ani_image], dpi=90, writer='imagemagick')
        ani_lines.save(outfile+'.mp4', extra_anim=[ani_image], writer='ffmpeg', dpi=180)

def respine(ax, interval, tickint, bottom):
    if interval is None:
        ax.set_ylim([0, 1/tickint])
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        sns.despine(ax=ax, left=True, bottom=bottom, trim=True)
    else:
        if type(tickint) is list:
            ax.set_yticks(tickint)
        else:
            ax.set_yticks(np.arange(interval[0], interval[1]+1,tickint))
        ax.set_ylim(interval)
        sns.despine(ax=ax, bottom=bottom, trim=True)
    return ax

def main():
    ### CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-ses', action="store", dest="session", type=int)
    parser.add_argument('-sf', action="store", dest="startfr", type=int)
    parser.add_argument('-ef', action="store", dest="endfr", type=int)
    SESSION = parser.parse_args().session
    START = parser.parse_args().startfr
    END = parser.parse_args().endfr
    OUT = '/home/degoldschmidt/post_tracking'
    DB = '/home/degoldschmidt/post_tracking/DIFF.yaml'

    ### input data
    _in, _in2 = 'kinematics', 'classifier'
    _out = 'plots'
    infolder = os.path.join(OUT, _in)
    infolder2 = os.path.join(OUT, _in2)
    outfolder = os.path.join(OUT, _out)
    _outfile = 'video'
    db = Experiment(DB)
    session = db.sessions[SESSION]
    meta = session.load_meta(VERBOSE=False)
    if os.name == 'posix':
        _file = meta['video']['file'].split('\\')[-1]
        video_file = os.path.join("/media/degoldschmidt/DATA_BACKUP/data/tracking/videos", _file)
        print("MacOSX:", video_file)
    else:
        video_file = meta['video']['file']
    first = meta['video']['first_frame']
    try:
        csv_file = os.path.join(infolder,  '{}_{}.csv'.format(session.name, _in))
        csv_file2 = os.path.join(infolder2,  '{}_{}.csv'.format(session.name, _in2))
        kinedf = pd.read_csv(csv_file, index_col='frame')
        ethodf = pd.read_csv(csv_file2, index_col='frame')
    except FileNotFoundError:
        pass

    kinedf['min_patch'] = kinedf.loc[:,['dpatch_{}'.format(i) for i in range(11)]].min(axis=1)
    df = pd.concat([kinedf[['elapsed_time', 'sm_head_speed', 'sm_body_speed', 'angular_speed', 'min_patch']], ethodf[['etho', 'visit']]], axis=1)
    data_cols = ['visit', 'etho', 'angular_speed', ['sm_head_speed', 'sm_body_speed'], 'min_patch']
    fig, ax_video, ax_tseries, frames, xarray, yarrays, im, vid, lines, play_line, ucols = init_animation(df, time='elapsed_time', cols=data_cols, video=video_file, figsize=(13,6), interval=[START,END], meta=meta) ##(10.6666666,6)
    if not ONLY_VIDEO:
        lines[0].set_color('#222222')
        ax_tseries[0].set_xlabel('Time [s]')
        ax_tseries[4].set_xlim([0,30])
        ax_tseries[3].set_xlim([0,30])
        ax_tseries[2].set_xlim([0,30])
        ax_tseries[1].set_xlim([0,30])
        ax_tseries[0].set_xlim([0,30])
        ax_tseries[4] = respine(ax_tseries[4], [0,10], 2.5, True)
        ax_tseries[3] = respine(ax_tseries[3], [0,20], [0,2,5,10,15], True)
        if NO_ANNOS:
            ax_tseries[2] = respine(ax_tseries[2], [-600,600], [-500,-125,0,125,500], False)
            ax_tseries[2].get_xaxis().set_visible(True)
            ax_tseries[2].set_xticks(np.arange(0,30+1,5))
            ax_tseries[2].set_xlabel('Time [s]')
        else:
            ax_tseries[2] = respine(ax_tseries[2], [-600,600], [-500,-125,0,125,500], True)
        ax_tseries[1] = respine(ax_tseries[1], None, 1, True)
        ax_tseries[0] = respine(ax_tseries[0], None, 0.5, False)
        ax_tseries[4].set_ylabel('Min. distance\nto patch [mm]')
        ax_tseries[3].set_ylabel('Linear\nspeed [mm/s]')
        ax_tseries[2].set_ylabel('Angular\nspeed [º/s]')
        ax_tseries[1].set_ylabel('Feeding\nevent', labelpad=40)
        ax_tseries[0].set_ylabel('Visit', labelpad=40)
        if NO_ANNOS:
          ax_tseries[1].set_visible(False)
          ax_tseries[0].set_visible(False)

    pixelpos = [np.zeros(frames.shape), np.zeros(frames.shape), np.zeros(frames.shape), np.zeros(frames.shape), np.zeros(frames.shape)]
    scale, x0, y0 = meta['arena']['scale'], meta['arena']['x'], meta['arena']['y']
    pixelpos[0] = (scale*np.array(kinedf['head_x']) + x0).astype(int)
    pixelpos[1] = (-scale*np.array(kinedf['head_y']) + y0).astype(int)
    pixelpos[2] = np.array(ethodf['etho'])
    pixelpos[3] = (scale*np.array(kinedf['body_x']) + x0).astype(int)
    pixelpos[4] = (-scale*np.array(kinedf['body_y']) + y0).astype(int)

    ### save animation to file
    _file = os.path.join(outfolder, "{}_{}".format(_outfile, session.name))
    print(_file)
    run_animation(fig, frames, xarray, yarrays, lines, play_line, im, vid, ax_video, ax_tseries, pixelpos, cols=ucols, outfile=_file)


if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
