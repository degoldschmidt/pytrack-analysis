from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.plot as plot
from pytrack_analysis import Multibench
import matplotlib.pyplot as plt
from pytrack_analysis.yamlio import read_yaml
import seaborn as sns
import numpy as np
import pandas as pd
import os
import os.path as op
from scipy.stats import ranksums
import argparse

def plot_swarm(data, x, y, sub, conds):
    f, ax = plt.subplots(figsize=(5,2.5))
    rdata = data.query('substrate == "{}"'.format(sub)).dropna()
    querystr = ''
    astr = ' or '
    for condition in conds:
        querystr += 'genotype == "{}"'.format(condition)
        querystr += astr
    rdata = rdata.query(querystr[:-len(astr)])
    ax = sns.boxplot(x=x, y=y, data=rdata, hue='temperature', order=conds)
    return ax

def main():
    """
    --- general parameters
     *
    """
    ### CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', metavar='basedir', type=str, help='directory where your data files are')
    parser.add_argument('--exp', action='store', type=str)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('-c', nargs='+', type=str)
    parser.add_argument('-suf', type=str)
    BASEDIR = parser.parse_args().basedir
    OVERWRITE = parser.parse_args().force
    EXP = parser.parse_args().exp

    rawfolder = op.join(BASEDIR, 'pytrack_res', 'post_tracking')
    outfolder = op.join(BASEDIR, 'pytrack_res', 'kinematics')
    sessions = [_file for _file in os.listdir(rawfolder) if EXP in _file and _file.endswith('csv') and not _file.startswith('.') and _file[:-3]+'yaml' in os.listdir(rawfolder)]
    print(sorted(sessions))
    n_ses = len(sessions)

    conds = ["IR76b", "R57F03", "R67E03", "empty"]
    if parser.parse_args().c is not None:
        conds = parser.parse_args().c
    #colormap = {'SAA': "#424242", 'AA': "#5788e7", 'S': "#999999", 'O': "#B7B7B7"}
    #mypal = {condition: colormap[condition]  for condition in conds}

    _in, _in2, _out = 'classifier', 'segments', 'plots'
    infolder = os.path.join(BASEDIR, 'pytrack_res', _in)
    in2folder = os.path.join(BASEDIR, 'pytrack_res', _in2)
    outfolder = os.path.join(BASEDIR, 'pytrack_res', _out)
    outdf = {'session': [], 'temperature': [], 'genotype': [], 'substrate': [], 'mean_duration': []}
    _outfile = 'visits_mean_duration'
    hook_file = os.path.join(outfolder, "{}.csv".format(_outfile))
    if os.path.isfile(hook_file) and not OVERWRITE:
        print('Found data hook')
        outdf = pd.read_csv(hook_file, index_col='id')
    else:
        print('Compute data')
        for i_ses, each in enumerate(sorted(sessions)):
            ### Loading data
            print(each)
            try:
                yamlfile = op.join(rawfolder, each[:-3]+'yaml')
                meta = read_yaml(yamlfile)
                csv_file = os.path.join(infolder, '{}_{}.csv'.format(each[:-4], _in))
                csv_file2 = os.path.join(in2folder, '{}_{}.csv'.format(each[:-4], _in2+'_visit'))
                ethodf = pd.read_csv(csv_file, index_col='frame')
                segmdf = pd.read_csv(csv_file2, index_col='segment')

                for j, sub in enumerate(['yeast', 'sucrose']):
                    only_mm = segmdf.query("state == {}".format(j+1))
                    meandur = np.nanmean(only_mm['duration'])
                    if np.isnan(meandur):
                        meandur = 0.
                    print(sub,':',meandur)
                    outdf['session'].append(each[:-4])
                    outdf['temperature'].append(meta['condition'].split(' ')[0])
                    outdf['genotype'].append(meta['condition'].split(' ')[1])
                    outdf['substrate'].append(sub)
                    outdf['mean_duration'].append(meandur/60.)
            except FileNotFoundError:
                pass #print(csv_file+ ' not found!')
        outdf = pd.DataFrame(outdf)
        outdf.to_csv(hook_file, index_label='id')
    print(outdf)

    #### Plotting
    f, ax = plt.subplots(figsize=(3,2.5))

    # swarmbox
    ymax = [50, 50]#[50, 5] ### Vero: 50, 10
    yt = [5, 5]
    annos = [(14,.1), (18.5,0.1)] ### 55 -> 48 sucrose deprived
    for j, sub in enumerate(['yeast', 'sucrose']):
        f, ax = plt.subplots(figsize=(5,2.5))
        mypal = sns.color_palette("coolwarm", 2)
        rdata = outdf.query('substrate == "{}"'.format(sub)).dropna()
        ax = sns.boxplot(x='genotype', y='mean_duration', hue='temperature', data=rdata, order=conds, hue_order=['22ºC', '30ºC'], palette=mypal, saturation=1.,width=.5, ax=ax, boxprops=dict(lw=0.0), showfliers=False)
        ax = sns.swarmplot(x='genotype', y='mean_duration', hue='temperature', data=rdata, order=conds, hue_order=['22ºC', '30ºC'], dodge=True, color='#666666', size=2, ax=ax)
        sns.despine(ax=ax, bottom=True, trim=True)
        ax.set_xlabel('')
        ax.set_ylabel('Mean duration\nof {}\nvisits [min]'.format(sub))

        ### saving files
        plt.tight_layout()
        _file = os.path.join(outfolder, "{}_{}".format(_outfile, sub))
        if parser.parse_args().suf is not None:
            _file += '_'+parser.parse_args().suf
        #plt.savefig(_file+'.pdf', dpi=300)
        plt.savefig(_file+'.png', dpi=300)
        plt.cla()

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
