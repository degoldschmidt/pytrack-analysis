import argparse
import subprocess, os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pytrack_analysis import Multibench
from pytrack_analysis.dataio import VideoRawData
from pytrack_analysis.profile import get_profile, get_scriptname, show_profile
from pytrack_analysis.image_processing import ShowOverlay, PixelDiff
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
    for iv, video in enumerate(raw_data.videos[:1]):
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

        x, y, tx, ty, bx, by = [], [], [], [], [], []
        for i in range(4):
            bx.append(video.data.dfs[i]['body_x'])
            by.append(video.data.dfs[i]['body_y'])
            m = video.data.dfs[i]['major']
            angle = video.data.dfs[i]['angle']
            x.append(bx[-1]+0.5*m*np.cos(angle))
            y.append(by[-1]+0.5*m*np.sin(angle))
            tx.append(bx[-1]-0.5*m*np.cos(angle))
            ty.append(by[-1]-0.5*m*np.sin(angle))
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

        pxdiff = ShowOverlay(video.fullpath, start_frame=video.data.first_frame)
        px, tpx = pxdiff.run((x,y), (tx,ty), (bx,by), 10*1800, show=True)

        ###
        video.unload_data()

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False, SLIM=True)
    test(main)
    del test
