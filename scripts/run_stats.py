import os, argparse
import os.path as op
from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Stats
from pytrack_analysis import Multibench
from pytrack_analysis.yamlio import read_yaml

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.width', 260)
pd.set_option('precision', 4)
import tkinter as tk
RUN_STATS = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', metavar='basedir', type=str, help='directory where your data files are')
    parser.add_argument('--exp', action='store', type=str)
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
    """
    --- general parameters
     * thisscript: scriptname
     * experiment: experiment id (4 letters)
     * profile:    profile for given experiment, user and scriptname (from profile file)
     * db:         database from file
     * _sessions:  which sessions to process
     * n_ses:      number of sessions
     * stats:      list for stats
    """
    rawfolder = op.join(BASEDIR, 'pytrack_res', 'post_tracking')
    infolder = op.join(BASEDIR, 'pytrack_res', 'classifier')
    outfolder = op.join(BASEDIR, 'pytrack_res', 'segments')
    if not op.isdir(outfolder):
        os.mkdir(outfolder)
    experiment = [_file for _file in os.listdir(rawfolder) if _file.endswith('csv') and not _file.startswith('.') and _file[:-3]+'yaml' in os.listdir(rawfolder)][0][:4]
    sessions = [_file for _file in os.listdir(rawfolder) if experiment in _file and _file.endswith('csv') and not _file.startswith('.') and _file[:-3]+'yaml' in os.listdir(rawfolder)]
    print(sorted(sessions))
    n_ses = len(sessions)
    stats = []
    _in, _out = 'classifier', 'segments'


    ### GO THROUGH SESSIONS
    for i_ses, each in enumerate(sorted(sessions)):
        ### Loading data
        try:
            csv_file = os.path.join(infolder,  each[:-4]+'_'+_in+'.csv')
            df = pd.read_csv(csv_file, index_col='frame')
            yamlfile = op.join(rawfolder, each[:-3]+'yaml')
            meta = read_yaml(yamlfile)
            segm = Segments(df, meta)
            dfs = segm.run(save_as=outfolder, ret=True)
        except FileNotFoundError:
            print(csv_file+ ' not found!')
    for each in dfs.keys():
        print(each)
        print(dfs[each].head(15))
        print()
    ### delete objects

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
