from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.plot as plot
from pytrack_analysis import Multibench
import matplotlib.pyplot as plt
from pytrack_analysis.yamlio import read_yaml
from pytrack_analysis.array import rle
import seaborn as sns
import numpy as np
import pandas as pd
import os
import os.path as op
from scipy.stats import ranksums
import argparse
import textwrap

defaults = {
                'dpi': 900,
                'fig_width': 5,
                'fig_height': 2.5,
                'n_rows': 1,
                'n_cols': 2,
                'outextension': 'png',
                'outfile': 'out',
}
def parse_yaml(_file):
    out = read_yaml(_file)
    for k,v in defaults.items():
        if k not in out.keys():
            out[k] = v
    return out

def main():
    """
    --- general parameters
     *
    """
    ### CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', metavar='basedir', type=str, help='directory where your data files are')
    #parser.add_argument('infile', metavar='infile', type=str, help='yaml file for plotting')
    #parser.add_argument('--suffix', type=str)
    _base = parser.parse_args().basedir
    _result = op.join(_base, 'pytrack_res')

    rawfolder = op.join(_result, 'post_tracking')
    experiment = [_file for _file in os.listdir(rawfolder) if _file.endswith('csv') and not _file.startswith('.') and _file[:-3]+'yaml' in os.listdir(rawfolder)][0][:4]
    sessions = [_file[:-4] for _file in os.listdir(rawfolder) if experiment in _file and _file.endswith('csv') and not _file.startswith('.') and _file[:-3]+'yaml' in os.listdir(rawfolder)]
    statdf = pd.DataFrame(data={'duration': [], 'condition': [], 'temperature': [], 'substrate': []})

    for i_ses, ses in enumerate(sessions):
        try:
            f, ax = plt.subplots(figsize=(10, 10))
            ### getting data
            yamlfile = op.join(rawfolder, ses+'.yaml')
            meta = read_yaml(yamlfile)
            dfs = []
            cols = [['elapsed_time', 'frame_dt', 'head_x', 'head_y', 'body_x', 'body_y'], ['etho']]
            for i, module in enumerate(['kinematics', 'classifier']):
                infolder = op.join(_result, module)
                _file = "{}_{}.csv".format(ses, module)
                dfs.append(pd.read_csv(op.join(infolder, _file), index_col='frame').loc[:,cols[i]])
            df = pd.concat(dfs, axis=1)

            ax = plot.arena(meta["arena"], meta["food_spots"], ax=ax)
            x, y, tx, ty, etho, dt = np.array(df['head_x']), np.array(df['head_y']), np.array(df['body_x']), np.array(df['body_y']), np.array(df['etho']), np.array(df['frame_dt'])
            ends = 108100
            x, y, tx, ty, etho, dt = x[:ends], y[:ends], tx[:ends], ty[:ends], etho[:ends], dt[:ends]
            ax.plot(x, y, '.', c='#747474', alpha=0.5, ms=3)
            l, p, v = rle(etho)

            print(i_ses, meta['condition'])
            for eachl, eachp, eachv in zip(l, p, v):
                dur = np.sum(dt[eachp:eachp+eachl+1])
                if eachv == 4:
                    sub = 'yeast'
                if eachv == 5:
                    sub = 'sucrose'
                if eachv == 4 or eachv == 5:
                    statdf = statdf.append({'duration': dur, 'condition': meta['condition'], 'temperature': meta['setup']['temperature'], 'substrate': sub}, ignore_index=True)
                    if dur > 0.5:
                        print('{}: {} s (len {} @{})'.format(sub, dur, eachl, eachp))

            ax.plot(x[etho==4], y[etho==4], '.', c='#ffc632', alpha=0.5, ms=4)
            ax.plot(x[etho==5], y[etho==5], '.', c='#1faeff', alpha=0.5, ms=4)
            ax.plot(tx[etho==4], ty[etho==4], '.', c='#af7e00', alpha=0.5, ms=8)
            ax.plot(tx[etho==5], ty[etho==5], '.', c='#0068a2', alpha=0.5, ms=8)
            #ax.plot(tx[:ends], ty[:ends], '.', c='#be0000', alpha=0.5, ms=3)
            ### saving files
            plt.tight_layout()
            ax.set_title(meta['condition'], loc='left')
            _file = os.path.join(_result, 'plots', "head", '{}_head'.format(ses))
            plt.savefig(_file+'.png', dpi=600)
        except FileNotFoundError:
            pass
    print(statdf)
    statdf.to_csv(op.join(_result, 'stats_dur.csv'), index_label='index')
    return 1

def replot():
    """
    --- general parameters
     *
    """
    ### CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', metavar='basedir', type=str, help='directory where your data files are')
    #parser.add_argument('infile', metavar='infile', type=str, help='yaml file for plotting')
    #parser.add_argument('--suffix', type=str)
    _base = parser.parse_args().basedir
    _result = op.join(_base, 'pytrack_res')

    statdf = pd.read_csv(op.join(_result, 'stats_dur.csv'), index_col='index')##.query('duration > 0.75')
    f, ax = plt.subplots(figsize=(6, 2))
    ax = sns.stripplot(x="duration", y="substrate", data=statdf, jitter=True, alpha=0.5, size=1, ax=ax)

    ax.set_xscale('log')

    sns.despine(ax=ax, left=True, trim=True)
    ax.set_ylabel('')
    ax.set_xlabel('duration [s]')
    plt.tight_layout()
    _file = os.path.join(_result, 'plots', 'micromov_durations')
    plt.savefig(_file+'.png', dpi=900)
    return 1



if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(replot)
    del test
