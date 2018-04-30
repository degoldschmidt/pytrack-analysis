import argparse
import subprocess, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pytrack_analysis import Multibench
from pytrack_analysis.dataio import VideoRawData
from pytrack_analysis.profile import get_profile, get_scriptname, show_profile
from pytrack_analysis.image_processing import WriteOverlay, PixelDiff

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
    raw_data = VideoRawData(BASEDIR)
    ### go through all session
    for iv, video in enumerate(raw_data.videos):
        if iv == 0:
            i=0
            ### arena + food spots
            video.load_arena()
            ### trajectory data
            video.load_data()
            ### rename columns
            video.data.reindex(colnames)
            ### data to timestart
            video.data.to_timestart(video.timestart)
            ### calculate displacements
            dt = video.data.dfs[i]['frame_dt']
            dx, dy = np.append(0, np.diff(video.data.dfs[i]['body_x'])), np.append(0, np.diff(video.data.dfs[i]['body_y']))
            dx, dy = np.divide(dx, dt), np.divide(dy, dt)
            dr = np.sqrt(dx*dx + dy*dy)
            video.data.dfs[i]['displacements'] = dr
            win = 30
            threshold = 4.*video.data.dfs[i]['displacements'].rolling(window=30, center=True).median()/.65
            threshold = threshold.fillna(method='bfill')
            threshold = threshold.fillna(method='ffill')
            threshold = 400
            jumps = np.array(dr>threshold)
            jumps = np.convolve(jumps,[1,1,1], mode='same')
            video.data.dfs[i]['jumps'] = jumps
            angle = video.data.dfs[i]['angle']
            print(np.amax(angle))
            bx, by = video.data.dfs[i]['body_x'], video.data.dfs[i]['body_y']
            m = video.data.dfs[i]['major']
            x, y = bx+0.5*m*np.cos(angle), by+0.5*m*np.sin(angle)
            tx, ty = bx-0.5*m*np.cos(angle), by-0.5*m*np.sin(angle)
            x0, y0, w = video.arena[i]['x']-video.arena[i]['outer'], video.arena[i]['y']-video.arena[i]['outer'], 2*video.arena[i]['outer']
            print(x0, y0, w)
            #ow = WriteOverlay(video.fullpath, start_frame=video.data.first_frame, view=(x0, y0, w, w), outfile='jumps_{}.avi'.format(video.timestr))
            #ow.run((x,y), jumps, 150)##2*1800)
            pxdiff = PixelDiff(video.fullpath, start_frame=video.data.first_frame)
            px, tpx = pxdiff.run((x,y), (tx,ty), 108000, show=False)
            pxd_data = pd.DataFrame({'headpx': px, 'tailpx': tpx})
            pxd_data.to_csv(os.path.join(BASEDIR,'processed','pixeldiff.csv'), index_label='frame')
            plt.figure(figsize=(10,1.5))
            plt.plot(px[:1800], 'm-')
            plt.plot(tpx[:1800], 'g-')
            plt.plot(100*jumps[:1800], 'r-')
            #plt.plot(video.data.dfs[i]['jumps']*200, 'r.')
            plt.tight_layout()
            plt.savefig(os.path.join(BASEDIR,'plots','pixeldiff.png'))

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
