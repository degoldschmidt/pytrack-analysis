"""
===============
Demo Animation
===============
"""
from pytrack_analysis.profile import get_profile
import numpy as np
import imageio
import pandas as pd
import os, sys
import os.path as op

from pytrack_analysis.plot import set_font, swarmbox
from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Classifier
from pytrack_analysis import Multibench
import pytrack_analysis.plot as plot
from pytrack_analysis.yamlio import read_yaml

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as anim
import seaborn as sns
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
LIGHTRED = '#fb8072'
FANCYBLUE = '#1f5fd5'

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
ts_colors = [   BLACK,
                BLACK,
                LIGHTRED,
                FANCYBLUE,
                BLACK,
                BLACK,
                BLACK,]
spot_colors = { 'yeast': YEAST,
                'sucrose': SUCROSE}

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

### 10.666, 6 ->  7.111, 4
def init_animation(data, video, pixelpos, config=None, figsize=(5, 4), dpi=90, playback=1, meta=None):
    f = plt.figure(figsize=figsize, dpi=dpi)
    ## gridspec
    cols = config['cols']
    N = len(cols)   # number of rows
    gs = GridSpec(N, 2*N, height_ratios=config['height_ratios'])
    gs.update(wspace=10)
    gs.update(hspace=.25)
    ## video axis
    if ONLY_VIDEO:
        f, axv = plt.subplots(figsize=figsize, dpi=dpi)
        #fig.set_size_inches(1920/300., 1080/300., True)
    else:
        axv = plt.subplot(gs[:,:N])
    axv.set_aspect('equal')
    sns.despine(ax=axv, bottom=True, left=True)
    axv.get_xaxis().set_visible(False)
    axv.get_yaxis().set_visible(False)

    """
    Initial image
    """
    print(pixelpos.head(3))
    xs, ys = pixelpos['body_x_px'], pixelpos['body_y_px']
    rangex, rangey = xs.max()-xs.min(), ys.max()-ys.min()
    w = int(round(1.25*max(rangex,rangey)/2))
    x0, y0 = int(round((xs.max()+xs.min())/2)), int(round((ys.max()+ys.min())/2))
    frame = video.get_data(data.index[0])[y0-w:y0+w, x0-w:x0+w]
    im = axv.imshow(frame, animated=True)
    config['video_dims'] = {'x': x0, 'y': y0, 'r': w}
    pixelpos[['body_x_px', 'head_x_px']] -= (x0-w)
    pixelpos[['body_y_px', 'head_y_px']] -= (y0-w)
    print(pixelpos.head(3))

    ## data
    xarray = np.array(data[config['time']])
    series = unflatten(cols)
    yarrays = np.array(data[series])

    ### draw food spots
    if not ONLY_TRAJ:
        for ii,each in enumerate(meta['food_spots']):
            ax, ay, scale = meta['arena']['x'], meta['arena']['y'], meta['arena']['scale']
            sx, sy = scale * each['x'] + ax - (x0-w), -scale * each['y'] + ay - (y0-w)
            axv.add_artist(plt.Circle((sx,sy), scale*1.5, color=spot_colors[each['substr']], lw=1.5, alpha=0.5, fill=False, zorder=100))
            if ii in config['visited_spots']:
                axv.add_artist(plt.Circle((sx,sy), scale*2.5, color='#ffffff', ls='dashed', lw=1., alpha=0.5, fill=False, zorder=100))
            #if ii in config['multiple_mmovs']:
                #axv.add_artist(plt.Circle((sx,sy), scale*5, color='#ffffff', ls='dotted', lw=.75, alpha=0.5, fill=False, zorder=100))
            #axv.text(sx,sy, "{:02d}".format(ii), zorder=100)

    ### time series
    axts, lines = None, None
    if not ONLY_VIDEO:
        axts, lines, play_line = [], [], []
        ax, l = get_tseries_axes(gs, [4, 5])
        play_line.append(ax.axvline(0, ymax=2, c='#ff0000', ls='-', lw=.75, zorder=100, clip_on=False))

        l.set_color('#000000')
        ax.set_xlim(0, 1.1*np.max(xarray))
        ax.set_ylabel(cols[0][0])
        axts.append(ax)
        lines.append(l)


        count = 0
        for i in range(1,5):
            if len(cols[i])>1:
                ax, l = get_tseries_axes(gs, [4-i, 5], sharex=axts[0], n=2, labels=['head', 'body'])
            else:
                ax, l = get_tseries_axes(gs, [4-i, 5], sharex=axts[0])
            play_line.append(ax.axvline(0, ymin=-4, ymax=1, c='#ff0000', ls='-', lw=.75, zorder=100, clip_on=False))
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
            axts.append(ax)
        ### AXES labelling etc
        if not ONLY_VIDEO:
            axts[0].set_xlabel('Time [s]')
            for lab, ax in zip(config['ylabels'], axts):
                ax.set_xlim([0,config['nframes']/30])
                lab['label'] = lab['label'].replace('\\n', '\n')
                ax.set_ylabel(lab['label'], labelpad=lab['pad'])
            axts[4] = respine(axts[4], [0,10], 5, True)
            axts[3] = respine(axts[3], [0,12], [0,2,5,10], True)
            if NO_ANNOS:
                axts[2] = respine(axts[2], [-400,400], [-250,-125,0,125,250], False)
                axts[2].get_xaxis().set_visible(True)
                axts[2].set_xticks(np.arange(0,30+1,5))
            else:
                axts[2] = respine(axts[2], [-300,300], [-250,-125,0,125,250], True)
            axts[1] = respine(axts[1], None, 1, True)
            axts[0] = respine(axts[0], None, 0.5, False)
            if NO_ANNOS:
              axts[1].set_visible(False)
              axts[0].set_visible(False)

        ### thresholds
        if not NO_ANNOS:
            for thr in config['thresholds']:
                ax = axts[thr['index']]
                ax.hlines(thr['value'], ax.get_xlim()[0], ax.get_xlim()[1], colors="#aaaaaa", linestyles='dashed')

    datarrays = {'x': xarray, 'y': yarrays}

    return f, im, lines, config, datarrays, play_line

def animate(frame, *args):
    if frame%150 == 0:
        print('Animated {} seconds.'.format(int(frame/30)))
    ### video inputs
    f = args[0]
    image = args[1]
    video = args[2]
    pos = args[3]
    conf = args[4]
    axvid = f.get_axes()[0]
    axts = f.get_axes()[1:]

    ### update video frame
    x0, y0, r = conf['video_dims']['x'], conf['video_dims']['y'], conf['video_dims']['r']
    sf = conf['start_frame']
    image.set_array(video.get_data(sf+frame)[y0-r:y0+r,x0-r:x0+r])
    ### update overlays
    x,y, hx,hy, etho = np.array(pos['body_x_px'])[frame], np.array(pos['body_y_px'])[frame], np.array(pos['head_x_px'])[frame], np.array(pos['head_y_px'])[frame], np.array(pos['etho'])[frame]
    if len(axvid.get_lines()) == 0:
        axvid.plot([hx, x], [hy, y], color=palette[etho], ls='-', lw=1)
    axvid.set_title("frame #{}".format(frame+sf))
    #image[2].set_title("frame #{}\n{}".format(i, beh[etho]))
    if not NO_ANNOS:
        if palette[etho] == '#ffffff':
            al = 1
            ms = 2
        else:
            al = 1
            ms = 3
        axvid.plot(hx, hy, color=palette[etho], marker='.', alpha=al, markersize=ms)
        axvid.get_lines()[0].set_data([hx, x], [hy, y])
        axvid.get_lines()[0].set_color(palette[etho])

    ### time series inputs
    lines = args[5]
    xarray = args[6]['x']
    yarrays = args[6]['y']
    play_line = args[7]
    cols = unflatten(conf['cols'])
    xdata = np.array(xarray[:frame+1])
    ydata = []
    for j, each in enumerate(cols):
        if j < 2:
            ydata.append(int(yarrays[frame, j]))
        else:
            ydata.append(np.array(yarrays[:frame+1, j]))
    # update the data of both line objects
    for pl in play_line:
        pl.set_xdata(xdata[-1])
    for j,each_line in enumerate(lines):
        if j < 2:
            if j == 1:
                if ydata[j] > 3:
                    axts[j].vlines(xarray[frame], 0, 1, color=palette[ydata[j]])
            if j == 0:
                if ydata[j] > 0:
                    axts[j].vlines(xarray[frame], 0, 1, color=palette2[ydata[j]])
                if xarray[frame] - xarray[frame-1] > 0.1:
                    a = xarray[frame]-0.0333
                    while a > xarray[frame-1]:
                        if ydata[j] > 0:
                            axts[j].vlines(a, 0, 1, color=palette2[ydata[j]])
                        a -= 0.0333
        else:
            each_line.set_data(xdata, ydata[j])
    return tuple(lines) + (image,)

def run_animation(_file, _f, _image, _video, _pixelpos, _lines, _datarrays, _play_line, config=None, playback=1, n=None):
    fps = 30
    dt = 1000./(fps*playback)
    if n is None:
        n = config['nframes']
    ani_image = anim.FuncAnimation(_f, animate, np.arange(1,n+1), blit=True, fargs=(_f, _image, _video, _pixelpos, config, _lines, _datarrays, _play_line), interval=dt)
    if _file.endswith('gif'):
        writer = 'imagemagick'
    else:
        writer = 'ffmpeg'
    print('Using writer ', writer)
    ani_image.save(_file, writer=writer, dpi=_f.dpi)

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

def get_data(parser):
    conf = {}
    SESSION = parser.parse_args().session
    START = parser.parse_args().startfr
    END = parser.parse_args().endfr
    BASE = parser.parse_args().folder#'/home/degoldschmidt/post_tracking'
    MEDIA = parser.parse_args().backup#'/media/degoldschmidt/DATA_BACKUP'
    DB = op.join(BASE, parser.parse_args().db)#'/home/degoldschmidt/post_tracking/DIFF.yaml'
    CONF_FILE = op.join(BASE, parser.parse_args().conf)
    conf = read_yaml(CONF_FILE)
    print('read yaml')
    print(conf)
    conf['time'] = 'elapsed_time'
    conf['cols'] = [['visit'], ['etho'], ['angular_speed'], ['sm_head_speed', 'sm_body_speed'], ['min_patch']]
    conf['start_frame'] = START
    conf['end_frame'] = END
    conf['nframes'] = END - START

    ### get data
    sufs = ['kinematics', 'classifier', 'plots'] ### last one is output
    folders = [op.join(BASE, _suf) for _suf in sufs]
    conf['folders'] = folders
    conf['outfile'] = 'video'
    db = Experiment(DB)
    session = db.sessions[SESSION]
    conf['session'] = session.name
    meta = session.load_meta(VERBOSE=False)
    conf['arena'] = meta['arena']
    print(conf['arena'])
    if os.name == 'posix':
        _file = meta['video']['file'].split('\\')[-1]
        video_file = op.join(MEDIA, "data","tracking","videos", _file)
        print("Loading video:", video_file, op.isfile(video_file))
    else:
        video_file = meta['video']['file']
    vid = imageio.get_reader(video_file)
    try:
        csvs = [op.join(_base,'{}_{}.csv'.format(session.name, _in)) for _base, _in in zip(folders[:-1], sufs[:-1])]
        dfs = [pd.read_csv(_file, index_col=['frame']) for _file in csvs]
        dfs[1] = dfs[1].drop(dfs[1].columns[[i for i in range(2)]], axis=1)
        df = pd.concat(dfs, axis=1)
        df = df.loc[START:END,:]
        print('Interval: [{}, {}]'.format(START, END))
    except FileNotFoundError:
        print('not found')
        pass
    ### get pixelpositions
    pixelpos = get_pixel_positions(df, meta)
    pixelpos.index = df.index

    conf['visited_spots'] = df.query('visit_index > 0')['visit_index'].unique()
    conf['multiple_mmovs'] = df.query('visit > 0 and etho < 4')['visit_index'].unique()
    ### get minimum distance to patch
    df['min_patch'] = df.loc[:,['dpatch_{}'.format(i) for i in range(12)]].min(axis=1)
    ### chosen stuff
    allcols = unflatten(conf['cols'])
    allcols.append(conf['time'])
    df = df.loc[:,allcols]
    tstart = df.iloc[0]['elapsed_time']
    df['elapsed_time'] -= tstart
    return df, vid, pixelpos, meta, conf

def get_pixel_positions(df, meta):
    N = len(df.index)
    pixelpos = {}
    scale, x0, y0 = meta['arena']['scale'], meta['arena']['x'], meta['arena']['y']
    pixelpos['head_x_px'] = (scale*df['head_x'] + x0)
    pixelpos['head_y_px']  = (-scale*df['head_y'] + y0)
    pixelpos['etho'] = df['etho'].astype('int32')
    pixelpos['body_x_px'] = (scale*df['body_x'] + x0)
    pixelpos['body_y_px'] = (-scale*df['body_y'] + y0)
    return pd.DataFrame(pixelpos)

def main():
    ### CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-conf', action="store", dest="conf", type=str)
    parser.add_argument('-ses', action="store", dest="session", type=int)
    parser.add_argument('-sf', action="store", dest="startfr", type=int)
    parser.add_argument('-ef', action="store", dest="endfr", type=int)
    parser.add_argument('folder')
    parser.add_argument('backup')
    parser.add_argument('db')

    ## get all data
    df, video, pixelpos, meta, conf = get_data(parser)
    print(df.head(5))

    ### initialize animation
    f, image, lines, conf, datarrays, play_line = init_animation(df, video, pixelpos, config=conf, figsize=(13,6), meta=meta) ##(10.6666666,6)

    ### save animation to file
    _file = op.join(conf['folders'][-1], "{}_{}.mp4".format(conf['outfile'], conf['session']))
    print('Saving animation to:', _file)
    run_animation(_file, f, image, video, pixelpos, lines, datarrays, play_line, config=conf)


if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
