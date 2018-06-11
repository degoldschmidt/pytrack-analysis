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
    print(sessions)

    for ses in sessions[:1]:
        dfs = []
        for module in ['kinematics', 'classifier']:
            infolder = op.join(_result, module)
            _file = "{}_{}.csv".format(ses, module)
            dfs.append(pd.read_csv(op.join(infolder, _file), index_col='frame'))
        df = pd.concat(dfs)
        print(df.head(10))
    return 1

    for _file in _files:
        try:
            kinedf = pd.read_csv(_file, index_col='frame')
            ethodf = pd.read_csv(_file, index_col='frame')
        except FileNotFoundError:
            print('ERROR: File {} not found.'.format(_datafile))

    #### Plotting
    nrow = _prop['n_rows']
    ncol = _prop['n_cols']
    f, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(_prop['fig_width'], _prop['fig_height']))
    # swarmbox
    for j, ax in enumerate(f.axes):
        ### PANELS and PANEL DATA
        panelvar, this_panel = _prop['panel'], _prop['panel_order'][j]
        rdata = outdf.query('{} == "{}"'.format(panelvar, this_panel)).dropna()
        ### COLORS FOR HUE
        mypal = sns.color_palette(_prop['pal'], _prop['n_hues'])
        ### SWARMBOX VARIABLES
        _x, _y, _hue, _order, _hue_order = _prop['x'], _prop['y'], _prop['hue'], _prop['order'], _prop['hue_order']
        ### PLOTTING
        ax = plot.swarmbox(x=_x, y=_y, hue=_hue, data=rdata, order=_order, hue_order=_hue_order, palette=mypal, ax=ax)
        xlab = textwrap.fill(_prop['xlabel'][j], 14, break_long_words=False)
        ylab = textwrap.fill(_prop['ylabel'][j], 14, break_long_words=False)
        title = textwrap.fill(_prop['suptitle'][j], 36, break_long_words=False)
        ax.set_title(title, fontsize=6, fontweight='bold', loc='left')
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

    ### saving files
    plt.tight_layout()
    _file = os.path.join(_base, "{}_swarmbox".format(_prop['outfile']))
    if _suf is not None:
        _file+"_{}.{}".format(_suf, _prop['outextension'])
    plt.savefig(_file+'.png', dpi=_prop['dpi'])

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
