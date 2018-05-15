import argparse
import subprocess, os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pytrack_analysis import Multibench
from pytrack_analysis.dataio import VideoRawData
from pytrack_analysis.profile import get_profile, get_scriptname, show_profile
from pytrack_analysis.image_processing import WriteOverlay, PixelDiff
import pytrack_analysis.preprocessing as prp

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
    raw_data = VideoRawData(BASEDIR)
    if OPTION == 'registration':
        return 1
    ### go through all session
    for iv, video in enumerate(raw_data.videos):
        print('{}: {}'.format(iv, video.name))
        ### arena + food spots
        video.load_arena()
        ### trajectory data
        video.load_data()
        ### rename columns
        video.data.reindex(colnames)
        ### data to timestart
        video.data.to_timestart(video.timestart)
        ### calculate displacements
        """
        video.data.dfs[i]['displacements'] = dr
        win = 30
        threshold = 4.*video.data.dfs[i]['displacements'].rolling(window=30, center=True).median()/.65
        threshold = threshold.fillna(method='bfill')
        threshold = threshold.fillna(method='ffill')
        threshold = 400
        jumps = np.array(dr>threshold)
        jumps = np.convolve(jumps,[1,1,1], mode='same')
        video.data.dfs[i]['jumps'] = jumps
        print(np.amax(angle))
        """
        x, y, tx, ty = [], [], [], []
        for i in range(4):
            bx, by = video.data.dfs[i]['body_x'], video.data.dfs[i]['body_y']
            m = video.data.dfs[i]['major']
            angle = video.data.dfs[i]['angle']
            x.append(bx+0.5*m*np.cos(angle))
            y.append(by+0.5*m*np.sin(angle))
            tx.append(bx-0.5*m*np.cos(angle))
            ty.append(by-0.5*m*np.sin(angle))
            dt = video.data.dfs[i]['frame_dt']
            dx, dy = np.append(0, np.diff(video.data.dfs[i]['body_x'])), np.append(0, np.diff(-video.data.dfs[i]['body_y']))
            dx, dy = np.divide(dx, dt), np.divide(dy, dt)
            theta = np.arctan2(dy, dx)
            dr = np.sqrt(dx*dx+dy*dy)/video.arena[i]['scale']
            alignment = np.cos(angle-theta)
            alignment[dr < 5] = np.nan
            video.data.dfs[i]['align'] = alignment
            video.data.dfs[i]['speed'] = dr
            window_len = 36
            video.data.dfs[i]['smspeed'] = prp.gaussian_filter_np(video.data.dfs[i][['speed']], _len=window_len, _sigma=window_len/10)

        if not op.isdir(op.join(RESULT,'post_tracking')):
            os.mkdir(op.join(RESULT,'post_tracking'))
        _ofile = op.join(RESULT,'post_tracking','pixeldiff_{}.csv'.format(video.timestr))
        if op.isfile(_ofile):
            df = pd.read_csv(_ofile, index_col='frame')
        else:
            pxdiff = PixelDiff(video.fullpath, start_frame=video.data.first_frame)
            px, tpx = pxdiff.run((x,y), (tx,ty), 108000, show=False)
            pxd_data = pd.DataFrame({   'headpx_fly1': px[:,0], 'tailpx_fly1': tpx[:,0],
                                        'headpx_fly2': px[:,1], 'tailpx_fly2': tpx[:,1],
                                        'headpx_fly3': px[:,2], 'tailpx_fly3': tpx[:,2],
                                        'headpx_fly4': px[:,3], 'tailpx_fly4': tpx[:,3],})
            pxd_data.to_csv(_ofile, index_label='frame')

        for interval in range(10):
            f, axes = plt.subplots(4, figsize=(10,8))
            for i, ax in enumerate(axes):
                index = 'smspeed'
                ax.plot(video.data.dfs[i][index], 'r-')
                ax.plot(5.*video.data.dfs[i]['align'], 'b-')
                ax.set_xlim([interval*10800, (interval+1)*10800])
            plt.tight_layout()
            plt.savefig(op.join(RESULT,'plots','speed_{:03d}_{:02d}.png'.format(iv, interval)))
            f, axes = plt.subplots(4, figsize=(10,8))
            for i, ax in enumerate(axes):
                fly = 'fly{}'.format(i+1)
                hi = 'headpx_{}'.format(fly)
                ti = 'tailpx_{}'.format(fly)
                ax.plot(df[hi].rolling(window=333).mean(), 'g-')
                ax.plot(df[ti].rolling(window=333).mean(), 'm-')
                ax.set_xlim([interval*10800, (interval+1)*10800])
            #plt.plot(px[:1800], 'm-')
            #plt.plot(tpx[:1800], 'g-')
            #plt.plot(100*jumps[:1800], 'r-')
            #plt.plot(video.data.dfs[i]['jumps']*200, 'r.')
            plt.tight_layout()
            plt.savefig(op.join(RESULT,'plots','pixeldiff_{:03d}_{:02d}.png'.format(iv, interval)))

        #video.data.center_to_arena(video.arenas)
        ### fly/experiment metadata
        #for fly_idx, fly_data in enumerate(raw_data.get_data()):

        ###
        video.unload_data()

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False, SLIM=True)
    test(main)
    del test
