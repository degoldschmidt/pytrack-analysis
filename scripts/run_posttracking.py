import argparse
import subprocess, os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pytrack_analysis import Multibench
from pytrack_analysis.dataio import VideoRawData
from pytrack_analysis.profile import get_profile, get_scriptname, show_profile
from pytrack_analysis.image_processing import ShowOverlay, WriteOverlay, PixelDiff
import pytrack_analysis.preprocessing as prp
import pytrack_analysis.plot as plot
from pytrack_analysis.yamlio import write_yaml
from scipy import signal
from scipy.signal import hilbert

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
    if not op.isdir(op.join(RESULT,'post_tracking')):
        os.mkdir(op.join(RESULT,'post_tracking'))
    if not op.isdir(op.join(RESULT,'pixeldiff')):
        os.mkdir(op.join(RESULT,'pixeldiff'))
    if not op.isdir(op.join(RESULT,'jumps')):
        os.mkdir(op.join(RESULT,'jumps'))
    raw_data = VideoRawData(BASEDIR, VERBOSE=(OPTION == 'registration'))
    if OPTION == 'registration':
        return 1
    ### go through all session
    for iv, video in enumerate(raw_data.videos):
        #if iv > 0:
        #    continue
        print('\n{}: {}'.format(iv, video.name))
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
        jumps, dr, dddr, thr, flipped = [], [], [], [], []
        wo = WriteOverlay(video.fullpath, outfolder=op.join(RESULT,'jumps'))

        ### plotting speed, major/minor, decision points etc
        f, axes = plt.subplots(12, figsize=(6,10)) ### TODO
        for i in range(4):
            """
            Extract some kinematics
            """
            ff = int(video.data.dfs[i].index[0])
            lf = int(video.data.dfs[i].index[-1])
            st = 0
            en = min(lf-ff, 108100)
            xpos = video.data.dfs[i]['body_x'].interpolate().fillna(method='ffill').fillna(method='bfill')
            ypos = video.data.dfs[i]['body_y'].interpolate().fillna(method='ffill').fillna(method='bfill')
            m = video.data.dfs[i]['major'].interpolate().fillna(method='ffill').fillna(method='bfill')
            angle = video.data.dfs[i]['angle'].interpolate().fillna(method='ffill').fillna(method='bfill')
            x.append(xpos+0.5*m*np.cos(angle))
            y.append(ypos+0.5*m*np.sin(angle))
            tx.append(xpos-0.5*m*np.cos(angle))
            ty.append(ypos-0.5*m*np.sin(angle))
            bx.append(xpos)
            by.append(ypos)

        """
        PixelDiff Algorithm
        """
        _ofile = op.join(RESULT,'pixeldiff','pixeldiff_{}.csv'.format(video.timestr))
        if op.isfile(_ofile):
            pxd_data = pd.read_csv(_ofile, index_col='frame')
        else:
            pxdiff = PixelDiff(video.fullpath, start_frame=video.data.first_frame)
            px, tpx = pxdiff.run((x,y), (tx,ty), en, show=False)
            pxd_data = pd.DataFrame({   'headpx_fly1': px[:,0], 'tailpx_fly1': tpx[:,0],
                                        'headpx_fly2': px[:,1], 'tailpx_fly2': tpx[:,1],
                                        'headpx_fly3': px[:,2], 'tailpx_fly3': tpx[:,2],
                                        'headpx_fly4': px[:,3], 'tailpx_fly4': tpx[:,3],})
            pxd_data.to_csv(_ofile, index_label='frame')

        for i in range(4):
            ff = int(video.data.dfs[i].index[0])
            lf = int(video.data.dfs[i].index[-1])
            st = 0
            en = min(lf-ff, 108100)
            xpos = video.data.dfs[i]['body_x'].interpolate().fillna(method='ffill').fillna(method='bfill')
            ypos = video.data.dfs[i]['body_y'].interpolate().fillna(method='ffill').fillna(method='bfill')
            m = video.data.dfs[i]['major'].interpolate().fillna(method='ffill').fillna(method='bfill')
            angle = video.data.dfs[i]['angle'].interpolate().fillna(method='ffill').fillna(method='bfill')
            mi = video.data.dfs[i]['minor'].interpolate().fillna(method='ffill').fillna(method='bfill')
            if np.any(np.isnan(xpos)) or np.any(np.isnan(ypos)) or np.any(np.isnan(m)) or np.any(np.isnan(angle)):
                print(np.any(np.isnan(xpos)), np.any(np.isnan(ypos)), np.any(np.isnan(m)), np.any(np.isnan(angle)))

            dt = video.data.dfs[i]['frame_dt']
            dx, dy = np.append(0, np.diff(xpos)), np.append(0, np.diff(-ypos))
            dx, dy = np.divide(dx, dt), np.divide(dy, dt)
            theta = np.arctan2(dy, dx)

            ### pixel data from pixeldiff
            hpx = np.array(pxd_data['headpx_fly{}'.format(i+1)])
            wlen = 36
            hpx = gaussian_filter(hpx, _len=wlen, _sigma=0.1*wlen)
            tpx = np.array(pxd_data['tailpx_fly{}'.format(i+1)])
            tpx = gaussian_filter(tpx, _len=wlen, _sigma=0.1*wlen)

            """
            diff of diff of displacements (spikes are more pronounced)
            """
            dr.append(np.sqrt(dx*dx+dy*dy)/float(video.arena[i]['scale']))
            ddr = np.append(0, np.diff(dr[-1]))
            dddr.append(np.append(0, np.diff(ddr)))
            #wlen = 36
            #dr_sm = gaussian_filter(np.array(dr), _len=wlen, _sigma=0.1*wlen)
            wlen = 120
            dddr_sm = gaussian_filter(np.array(np.abs(dddr[-1])), _len=wlen, _sigma=0.5*wlen)
            """
            Thresholding
            """
            threshold = 10.*dddr_sm
            low, high = 10., 30.
            threshold[threshold<low] = low
            threshold[threshold>high] = high
            thr.append(threshold)
            #### TODO
            jumps.append(np.array(np.array(dddr[-1])[st:en] > threshold[st:en]))
            mistrack_inds = np.where(np.array(dddr[-1])[st:en] > threshold[st:en])[0]

            """
            Rolling mean of pixeldiff for flips (window = 10 secs)
            """
            pxthr = np.array(tpx[st:en+1] < hpx[st:en+1])
            pxavg = np.zeros(pxthr.shape)
            for frm in range(pxavg.shape[0]):
                e = frm + 300
                if e >= pxavg.shape[0]:
                    e = pxavg.shape[0]-1
                if frm == e:
                    pxavg[frm] = pxthr[frm]
                else:
                    pxavg[frm] = np.mean(pxthr[frm:e])

            ### plot
            axes[3*i].plot(dddr[-1][st:en], 'k-', lw=0.5)
            axes[3*i].plot(threshold[st:en], '--', color='#fa6800', lw=0.5)
            axes[3*i].plot(mistrack_inds, 50.*np.ones(len(mistrack_inds)), 'o', color='#d80073', markersize=2)
            axes[3*i].set_ylim([-5,55])
            axes[3*i].set_yticks(np.arange(0,60,25))

            ### plot 2nd
            axes[3*i+1].plot(hpx[st:en], '-', color='#fa0078', lw=0.5)
            axes[3*i+1].plot(tpx[st:en], '-', color='#00fa64', lw=0.5)
            axes[3*i+1].plot(100.*pxthr, '--', color='#6e6e6e', lw=0.5)
            axes[3*i+1].plot(100.*pxavg, '-', color='#000000', lw=0.5)
            axes[3*i+1].set_ylim([0,255])

            axes[3*i+2].plot(m[st:en]/video.arena[i]['scale'], '-', color='#ff2f2f', lw=0.5)
            axes[3*i+2].plot(mi[st:en]/video.arena[i]['scale'], '-', color='#008dff', lw=0.5)
            axes[3*i+2].plot((m[st:en]*mi[st:en])/video.arena[i]['scale'], '--', color='#6f6f6f', lw=0.5)
            axes[3*i+2].set_ylim([-1,6])
            axes[3*i+2].set_yticks(np.arange(0,7,2))
            ####

            view = (video.arena[i]['x']-260, video.arena[i]['y']-260, 520, 520)
            sf, ef = st+ff, en+ff
            total_dur = int((video.data.dfs[i].loc[lf,'elapsed_time'] - video.data.dfs[i].loc[ff,'elapsed_time'])/60.)
            secs = int(round(video.data.dfs[i].loc[lf,'elapsed_time'] - video.data.dfs[i].loc[ff,'elapsed_time']))%60
            if OPTION == 'jump_detection':
                print("fly {}:\tstart@ {} ({} >= {}) total: {}:{:02d} mins ({} frames)".format(i+1, ff, video.data.dfs[i].loc[ff,'datetime'], video.timestart, total_dur, secs, en-st))
            thrs = np.array(np.array(dddr[i])[st:en+1] > threshold[st:en+1])
            flip = np.zeros(thrs.shape)
            flipped.append(flip)
            thr_ix = np.append(np.append(0, np.where(thrs)[0]), len(flip)+ff)
            if OPTION == 'jump_detection':
                print('found {} detection points (start, jumps, mistracking, etc.).'.format(len(thr_ix)-1))
            count = 0
            if len(thr_ix) > 0:
                for jj,ji in enumerate(thr_ix[:-1]):
                    fromfr = thr_ix[jj] + ff
                    tofr = thr_ix[jj+1] + ff - 1
                    flip[thr_ix[jj]:thr_ix[jj+1]] = np.mean(pxthr[thr_ix[jj]:thr_ix[jj+1]])>0.5
                    if flip[thr_ix[jj]] == 1:
                        x[i].loc[fromfr:tofr], tx[i].loc[fromfr:tofr] = tx[i].loc[fromfr:tofr], x[i].loc[fromfr:tofr]
                        y[i].loc[fromfr:tofr], ty[i].loc[fromfr:tofr] = ty[i].loc[fromfr:tofr], y[i].loc[fromfr:tofr]
                    clip_st, clip_en = fromfr-60, fromfr+60
                    if clip_st < int(video.data.dfs[i].index[0]):
                        clip_st = int(video.data.dfs[i].index[0])
                    if clip_en > int(video.data.dfs[i].index[-1]):
                        clip_en = int(video.data.dfs[i].index[-1])
                    if clip_en - clip_st < 30:
                        continue
                    count += 1
                    _ofile = op.join(RESULT,'jumps','{}'.format(video.name[:-4]), 'fly{}_{:06d}.avi'.format(i+1, fromfr))
                    if not op.isfile(_ofile):
                        wo.run((bx[i].loc[clip_st:clip_en], by[i].loc[clip_st:clip_en]), (x[i].loc[clip_st:clip_en], y[i].loc[clip_st:clip_en]), clip_st, clip_en, fromfr, view, i, bool=[thr, flip])
            video.data.dfs[i].loc[:, 'head_x'] = x[i]
            video.data.dfs[i].loc[:, 'head_y'] = y[i]
            if OPTION == 'jump_detection':
                print('wrote {} videos.'.format(count))
            mistracked = np.sum(dr[-1] > 80)
            window_len = 36
        if not op.isdir(op.join(RESULT,'plots')):
            os.mkdir(op.join(RESULT,'plots'))
        if not op.isdir(op.join(RESULT,'plots', 'posttracking')):
            os.mkdir(op.join(RESULT,'plots', 'posttracking'))
        f.savefig(op.join(RESULT,'plots', 'posttracking','speed_{}.png'.format(video.timestr)), dpi=600)
        print()
        if OPTION == 'jump_detection':
            continue

        labels = ['topleft', 'topright', 'bottomleft', 'bottomright']
        for i in range(4):
            outdf = video.data.dfs[i].loc[sf:ef]
            outdf.loc[:, 'flipped'] = flipped[i]
            print(jumps[i].shape)
            outdf.loc[:, 'jumps'] = jumps[i]
            outdf.loc[:, 'dr'] = dr[i]
            outdf.loc[:, 'dddr'] = dddr[i]
            outdf.loc[:, 'threshold'] = thr[i]
            dx, dy = outdf['head_x'] - outdf['body_x'], outdf['body_y'] - outdf['head_y']
            outdf.loc[:, 'angle'] = np.arctan2(dy, dx)
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
            meta['fly']['temperature'] = raw_data.experiment['Conditions'][video.name]['temperature'][i]
            #meta['fly']['genetic manipulation'] = raw_data.experiment['Conditions'][video.name]['genetic manipulation'][i] === Kir
            meta['food_spots'] = video.spots[i]
            meta['setup'] = {}
            meta['setup']['humidity'] = raw_data.experiment['Constants']['humidity']
            meta['setup']['light'] = raw_data.experiment['Constants']['light']
            meta['setup']['n_per_arena'] = raw_data.experiment['Constants']['n_per_arena']
            meta['setup']['room'] = raw_data.experiment['Constants']['room']
            meta['setup']['temperature'] = raw_data.experiment['Conditions'][video.name]['temperature'][i] # raw_data.experiment['Constants']['temperature']
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
            x, y, jumps = np.array(outdf['body_x']), np.array(outdf['body_y']), np.array(outdf['jumps'])
            for i in np.arange(len(x)-2):
                if jumps[i+1]:
                    color = '#ff0000'
                else:
                    color = '#b1b1b1'
                ax.plot(x[i:i+2], y[i:i+2], c=color, zorder=1, lw=.5, alpha=0.5)
            f.savefig(plotfile, dpi=300)

        ###
        video.unload_data()

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False, SLIM=True)
    test(main)
    del test
