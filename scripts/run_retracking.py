import argparse
import subprocess, os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pytrack_analysis import Multibench
from pytrack_analysis.dataio import VideoRawData
from pytrack_analysis.image_processing import ShowOverlay, WriteOverlay, PixelDiff, retrack
from pytrack_analysis.profile import get_profile, get_scriptname, show_profile
import pytrack_analysis.preprocessing as prp
import pytrack_analysis.plot as plot
from pytrack_analysis.yamlio import write_yaml
from scipy import signal
from scipy.signal import hilbert

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def remove_mistrack(x, y, ma, mi, thr=100.*0.0333, forced=False):
    xnew, ynew = x.copy(), y.copy()
    dx, dy = np.append(0, np.diff(x)), np.append(0, np.diff(y))
    displ = np.sqrt(dx**2 + dy**2)
    area = np.multiply(ma,mi)
    xnew[area > 10] = np.nan
    ynew[area > 10] = np.nan
    xnew[area < 2] = np.nan
    ynew[area < 2] = np.nan
    print(displ)
    ides = np.where(displ > thr)[0]
    print(ides)
    """
    for jj, each in enumerate(ides):
        if jj == 0:
            print(each)
            if len(ides) > 1:
                xnew[ides[jj]:ides[jj+1]] = np.nan
                ynew[ides[jj]:ides[jj+1]] = np.nan
            else:
                xnew[ides[jj]:] = np.nan
                ynew[ides[jj]:] = np.nan
        if jj < len(ides)-1:
            print(jj, np.mean(ma[ides[jj]:ides[jj+1]])*np.mean(mi[ides[jj]:ides[jj+1]]), ma[each]*mi[each])
            if forced or np.mean(ma[ides[jj]:ides[jj+1]])*np.mean(mi[ides[jj]:ides[jj+1]]) > 10 or np.mean(ma[ides[jj]:ides[jj+1]])*np.mean(mi[ides[jj]:ides[jj+1]]) < 2:
                xnew[ides[jj]:ides[jj+1]] = np.nan
                ynew[ides[jj]:ides[jj+1]] = np.nan
    """
    ma[np.isnan(xnew)] = np.mean(ma)
    mi[np.isnan(xnew)] = np.mean(mi)
    nans, xind = nan_helper(xnew)
    xnew[nans]= np.interp(xind(nans), xind(~nans), xnew[~nans])
    nans, yind = nan_helper(ynew)
    ynew[nans]= np.interp(yind(nans), yind(~nans), ynew[~nans])
    return xnew, ynew, ma, mi



### TODO: move this to signal processing module
def gaussian_filter(_X, _len=16, _sigma=1.6):
    norm = np.sqrt(2*np.pi)*_sigma ### Scipy's gaussian window is not normalized
    window = signal.gaussian(_len+1, std=_sigma)/norm
    convo = np.convolve(_X, window, "same")
    ## eliminate boundary effects
    convo[:_len] = _X[:_len]
    convo[-_len:] = _X[-_len:]
    return convo

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', metavar='basedir', type=str, help='directory where your data files are')
    parser.add_argument('--option', action='store', type=str)
    parser.add_argument('--overwrite', action='store_true')
    BASEDIR = parser.parse_args().basedir
    OVERWRITE = parser.parse_args().overwrite
    if parser.parse_args().option is None:
        OPTION = 'all'
    else:
        OPTION = parser.parse_args().option
    return BASEDIR, OPTION, OVERWRITE

def main():
    BASEDIR, OPTION, OVERWRITE = get_args()
    ### Define raw data structure
    colnames = ['datetime', 'elapsed_time', 'frame_dt', 'body_x',   'body_y',   'angle',    'major',    'minor']
    if not op.isdir(op.join(BASEDIR, 'pytrack_res')):
        os.mkdir(op.join(BASEDIR, 'pytrack_res'))
    RESULT = op.join(BASEDIR, 'pytrack_res')
    if not op.isdir(op.join(RESULT,'retrack')):
        os.mkdir(op.join(RESULT,'retrack'))
    raw_data = VideoRawData(BASEDIR, VERBOSE=(OPTION == 'registration'))
    if OPTION == 'registration':
        return 1
    ### go through all session
    for iv, video in enumerate(raw_data.videos):
        if iv > 0:
            continue
        Nflies = 4
        print('\n{}: {}'.format(iv, video.name))
        ### arena + food spots
        video.load_arena()
        ### trajectory data
        video.load_data()
        ### rename columns
        video.data.reindex(colnames)
        ### data to timestart
        video.data.to_timestart(video.timestart)

        ### RETRACKING TEST
        _bodyfile = op.join(RESULT,'retrack','body_{}.csv'.format(video.timestr))
        _headfile = op.join(RESULT,'retrack','head_{}.csv'.format(video.timestr))
        if not op.isfile(_bodyfile):
            #retrack = Retracking(video.fullpath, start_frame=video.data.first_frame)
            #body, head, tail = retrack.run(video.data.nframes, show=True)
            body, head, tail, area = retrack(video.fullpath, 10000, start_frame=video.data.first_frame, show=True)
            ### save data
            bodydf = pd.DataFrame({ 'x0': body[:, 0, 0], 'y0': body[:, 1, 0],
                                    'x1': body[:, 0, 1], 'y1': body[:, 1, 1],
                                    'x2': body[:, 0, 2], 'y2': body[:, 1, 2],
                                    'x3': body[:, 0, 3], 'y3': body[:, 1, 3],
                    })
            headdf = pd.DataFrame({ 'x0': head[:, 0, 0], 'y0': head[:, 1, 0],
                                    'x1': head[:, 0, 1], 'y1': head[:, 1, 1],
                                    'x2': head[:, 0, 2], 'y2': head[:, 1, 2],
                                    'x3': head[:, 0, 3], 'y3': head[:, 1, 3],
                    })
            bodydf.index = bodydf.index + video.data.first_frame
            headdf.index = headdf.index + video.data.first_frame
            #bodydf.to_csv(_bodyfile, index_label='frame')
            #headdf.to_csv(_headfile, index_label='frame')
            for fly in range(4):
                print('fly {}:'.format(fly), bodydf['x{}'.format(fly)].isnull().sum(), bodydf['y{}'.format(fly)].isnull().sum())
                print(np.amax(area[:,fly]), np.amin(area[:,fly]))
        else:
            bodydf = pd.read_csv(_bodyfile, index_col='frame')
            headdf = pd.read_csv(_headfile, index_col='frame')

        ### plot trajectory
        for i in range(Nflies):
            df = video.data.dfs[i]
            plotfile = op.join(RESULT,'plots','{}_{:03d}.png'.format(raw_data.experiment['ID'], i+iv*4))
            f, ax = plt.subplots(figsize=(10,10))
            ax = plot.arena(video.arena[i], video.spots[i], ax=ax)
            x, y, major, minor = np.array(df['body_x']), np.array(df['body_y']), np.array(df['major']), np.array(df['minor'])
            x -= video.arena[i]['x']
            y -= video.arena[i]['y']
            x /= video.arena[i]['scale']
            y /= -video.arena[i]['scale']
            #ax.plot(x, y, c='#595959', zorder=1, lw=.5, alpha=0.5)
            #xnew, ynew, major, minor = remove_mistrack(x, y, major, minor)
            #xnew, ynew, major, minor = remove_mistrack(xnew, ynew, major, minor, thr=300.*0.0333, forced=True)
            xnew = np.array(bodydf['x{}'.format(i)])
            ynew = np.array(bodydf['y{}'.format(i)])
            xnew -= video.arena[i]['x']
            ynew -= video.arena[i]['y']
            xnew /= video.arena[i]['scale']
            ynew /= -video.arena[i]['scale']
            starts = 0
            ends = starts+108100
            ax.plot(x[starts:ends], y[starts:ends], '-', c='#00e0ff', lw=1, alpha=0.25)
            ax.plot(xnew[starts:ends], ynew[starts:ends], '-', c='#ff00ff', lw=1, alpha=0.25)
            #ax.scatter(x, y, c=displ, s=5, cmap=plt.get_cmap('YlOrRd'), alpha=0.9, edgecolors='none', linewidths=0)
            f.savefig(plotfile, dpi=300)
        ###
        video.unload_data()

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False, SLIM=True)
    test(main)
    del test
