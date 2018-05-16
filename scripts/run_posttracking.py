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
import pytrack_analysis.plot as plot
from pytrack_analysis.yamlio import write_yaml

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
            mistracked = np.sum(dr > 50)
            video.data.dfs[i].loc[video.data.dfs[i].angle > np.pi, ['angle']] -= 2.*np.pi
            angle = video.data.dfs[i]['angle']
            window_len = 36

        if not op.isdir(op.join(RESULT,'post_tracking')):
            os.mkdir(op.join(RESULT,'post_tracking'))
        _ofile = op.join(RESULT,'post_tracking','pixeldiff_{}.csv'.format(video.timestr))
        if op.isfile(_ofile):
            pxd_data = pd.read_csv(_ofile, index_col='frame')
        else:
            pxdiff = PixelDiff(video.fullpath, start_frame=video.data.first_frame)
            px, tpx = pxdiff.run((x,y), (tx,ty), 108000, show=False)
            pxd_data = pd.DataFrame({   'headpx_fly1': px[:,0], 'tailpx_fly1': tpx[:,0],
                                        'headpx_fly2': px[:,1], 'tailpx_fly2': tpx[:,1],
                                        'headpx_fly3': px[:,2], 'tailpx_fly3': tpx[:,2],
                                        'headpx_fly4': px[:,3], 'tailpx_fly4': tpx[:,3],})
            pxd_data.to_csv(_ofile, index_label='frame')

        pixels = [(np.array(pxd_data['headpx_fly{}'.format(each)]), np.array(pxd_data['tailpx_fly{}'.format(each)])) for each in range(1,5)]

        pxdiff = ShowOverlay(video.fullpath, start_frame=video.data.first_frame)
        flip = pxdiff.run((x,y), (tx,ty), (bx,by), pixels, 108000, show=False)

        labels = ['topleft', 'topright', 'bottomleft', 'bottomright']
        for i in range(4):
            outdf = video.data.dfs[i].iloc[:108000]
            outdf.loc[:, 'flipped'] = flip[:,i]
            outdf.query('flipped == 1').loc[:, 'angle'] += np.pi
            outdf.loc[outdf.angle > np.pi, ['angle']] -= 2.*np.pi
            print('Arena:', video.arena[i]['x'], video.arena[i]['y'])
            outdf.loc[:, 'body_x'] -= video.arena[i]['x']
            outdf.loc[:, 'body_y'] -= video.arena[i]['y']
            outdf.loc[:, 'body_x'] /= video.arena[i]['scale']
            outdf.loc[:, 'body_y'] /= -video.arena[i]['scale']
            outdf.loc[:, 'major'] /= video.arena[i]['scale']
            outdf.loc[:, 'minor'] /= video.arena[i]['scale']
            print('x: ', np.amax(outdf['body_x']), np.amin(outdf['body_x']))
            print('y: ', np.amax(outdf['body_y']), np.amin(outdf['body_y']))
            print('major/minor: ', np.mean(outdf['major']), np.mean(outdf['minor']))
            outdf = outdf[['datetime', 'elapsed_time', 'frame_dt', 'body_x', 'body_y', 'angle', 'major', 'minor', 'flipped']]
            outfile = op.join(RESULT,'post_tracking','{}_{:03d}.csv'.format(raw_data.experiment['ID'], i+iv*4))
            print('saving to ', outfile)
            outdf.to_csv(outfile, index_label='frame')

            ### metadata
            meta = {}
            meta['arena'] = video.arena[i]
            meta['arena']['layout'] = '6-6 radial'
            meta['arena']['name'] = labels[i]
            meta['condition'] = ' '.join([v[i] for k,v in raw_data.experiment['Conditions'][video.name].items()])
            meta['datafile'] = outfile
            meta['datetime'] = video.time
            meta['flags'] = {}
            meta['flags']['mistracked_frames'] = int(mistracked)
            meta['fly'] = {}
            meta['fly']['mating'] = raw_data.experiment['Constants']['mating']
            meta['fly']['metabolic'] = raw_data.experiment['Constants']['metabolic']
            meta['fly']['sex'] = raw_data.experiment['Constants']['sex']
            meta['fly']['genotype'] = raw_data.experiment['Conditions'][video.name]['genotype'][i]
            meta['fly']['genetic manipulation'] = raw_data.experiment['Conditions'][video.name]['genetic manipulation'][i]
            meta['food_spots'] = video.spots[i]
            meta['setup'] = {}
            meta['setup']['humidity'] = raw_data.experiment['Constants']['humidity']
            meta['setup']['light'] = raw_data.experiment['Constants']['light']
            meta['setup']['n_per_arena'] = raw_data.experiment['Constants']['n_per_arena']
            meta['setup']['room'] = raw_data.experiment['Constants']['room']
            meta['setup']['temperature'] = raw_data.experiment['Constants']['temperature']
            meta['video'] = {}
            meta['video']['dir'] = video.dir
            meta['video']['file'] = video.fullpath
            meta['video']['first_frame'] = int(outdf.index[0])
            meta['video']['start_time'] = video.timestart
            yamlfile = op.join(RESULT,'post_tracking','{}_{:03d}.yaml'.format(raw_data.experiment['ID'], i+iv*4))
            write_yaml(yamlfile, meta)

            ### plot trajectory
            plotfile = op.join(RESULT,'plots','{}_{:03d}.png'.format(raw_data.experiment['ID'], i+iv*4))
            f, ax = plt.subplots(figsize=(4,4))
            ax = plot.arena(video.arena[i], video.spots[i], ax=ax)
            ax.plot(outdf['body_x'], outdf['body_y'], c='#424242', zorder=1, lw=1, alpha=0.5)
            f.savefig(plotfile, dpi=300)

        ###
        video.unload_data()

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False, SLIM=True)
    test(main)
    del test
