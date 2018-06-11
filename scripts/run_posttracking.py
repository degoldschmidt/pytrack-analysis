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
    #print(displ)
    ides = np.where(displ > thr)[0]
    #print(ides)
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
        if iv > 28:
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
        ### calculate displacements

        x, y, tx, ty, bx, by = [], [], [], [], [], []
        jumps, dr, dddr, thr, flipped = [], [], [], [], []
        wo = WriteOverlay(video.fullpath, outfolder=op.join(RESULT,'jumps'))

        ### plotting speed, major/minor, decision points etc
        f, axes = plt.subplots(12, figsize=(6,10)) ### TODO
        print('extract trajectories...')
        for i in range(Nflies):
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
        print('pixeldiff...')
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

        print('head detection...')
        for i in range(Nflies):
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
            pxthr = np.array(tpx[st:en] < hpx[st:en])
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
            thrs = np.array(np.array(dddr[i])[st:en] > threshold[st:en])
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
            mistracked = np.sum(dr[-1] > 100)
            print('Mistracked frames:', mistracked)
            window_len = 36
        if not op.isdir(op.join(RESULT,'plots')):
            os.mkdir(op.join(RESULT,'plots'))
        if not op.isdir(op.join(RESULT,'plots', 'posttracking')):
            os.mkdir(op.join(RESULT,'plots', 'posttracking'))
        f.savefig(op.join(RESULT,'plots', 'posttracking','speed_{}.png'.format(video.timestr)), dpi=600)
        if OPTION == 'jump_detection':
            continue

        labels = ['topleft', 'topright', 'bottomleft', 'bottomright']
        print('pack data...')
        for i in range(Nflies):
            df = video.data.dfs[i].loc[sf:ef-1]
            df.is_copy = False
            df.loc[:, ('flipped')] = np.array(flipped[i])
            df.loc[:, 'jumps'] = jumps[i]
            df.loc[:, 'dr'] = dr[i][st:en]
            df.loc[:, 'dddr'] = dddr[i][st:en]
            df.loc[:, 'threshold'] = thr[i][st:en]
            dx, dy = df['head_x'] - df['body_x'], df['body_y'] - df['head_y']
            df.loc[:, 'angle'] = np.arctan2(dy, dx)
            df.loc[:, 'body_x'] -= video.arena[i]['x']
            df.loc[:, 'body_y'] -= video.arena[i]['y']
            df.loc[:, 'body_x'] /= video.arena[i]['scale']
            df.loc[:, 'body_y'] /= -video.arena[i]['scale']
            df.loc[:, 'major'] /= video.arena[i]['scale']
            df.loc[:, 'minor'] /= video.arena[i]['scale']
            print('x: ', np.amax(df['body_x']), np.amin(df['body_x']))
            print('y: ', np.amax(df['body_y']), np.amin(df['body_y']))
            print('major/minor: ', np.mean(df['major']), np.mean(df['minor']))
            outdf = df[['datetime', 'elapsed_time', 'frame_dt', 'body_x', 'body_y', 'angle', 'major', 'minor', 'flipped']]
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
            f, ax = plt.subplots(figsize=(10,10))
            ax = plot.arena(video.arena[i], video.spots[i], ax=ax)
            x, y, jumps, major, minor = np.array(df['body_x']), np.array(df['body_y']), np.array(df['jumps']), np.array(df['major']), np.array(df['minor'])
            #ax.plot(x, y, c='#595959', zorder=1, lw=.5, alpha=0.5)
            xnew, ynew, major, minor = remove_mistrack(x, y, major, minor)
            xnew, ynew, major, minor = remove_mistrack(xnew, ynew, major, minor, thr=300.*0.0333, forced=True)
            ends = 108100
            ax.plot(x[0], y[0], '.', c='#00ff4f', alpha=0.75, zorder=10)
            ax.plot(x[ends-1], y[ends-1], '.', c='#ff3d00', alpha=0.75, zorder=10)
            #ax.plot(x[:ends], y[:ends], '-', c='#00e0ff', lw=1, alpha=0.5)
            ax.plot(xnew[:ends], ynew[:ends], '-', c='#ff00ff', lw=1, alpha=0.5)
            color = jumps
            color[jumps==1] = '#ff0000'
            color[jumps==0] = '#b1b1b1'
            #ax.scatter(x, y, c=displ, s=5, cmap=plt.get_cmap('YlOrRd'), alpha=0.9, edgecolors='none', linewidths=0)
            f.savefig(plotfile, dpi=300)

        ###
        video.unload_data()

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False, SLIM=True)
    test(main)
    del test
