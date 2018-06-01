import os, argparse
import os.path as op
from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Classifier
from pytrack_analysis import Multibench
from pytrack_analysis.yamlio import read_yaml

import warnings
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
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
    infolder = op.join(BASEDIR, 'pytrack_res', 'kinematics')
    outfolder = op.join(BASEDIR, 'pytrack_res', 'classifier')
    if not op.isdir(outfolder):
        os.mkdir(outfolder)
    experiment = [_file for _file in os.listdir(rawfolder) if _file.endswith('csv') and not _file.startswith('.') and _file[:-3]+'yaml' in os.listdir(rawfolder)][0][:4]
    sessions = [_file for _file in os.listdir(rawfolder) if experiment in _file and _file.endswith('csv') and not _file.startswith('.') and _file[:-3]+'yaml' in os.listdir(rawfolder)]
    print(sorted(sessions))
    n_ses = len(sessions)
    stats = []
    _in, _out = 'kinematics', 'classifier'

    ### GO THROUGH SESSIONS
    for i_ses, each in enumerate(sorted(sessions)):
        csv_file = os.path.join(infolder,  each[:-4]+'_'+_in+'.csv')
        df = pd.read_csv(csv_file, index_col='frame')
        yamlfile = op.join(rawfolder, each[:-3]+'yaml')
        meta = read_yaml(yamlfile)
        ## total micromoves
        nancheck = df['sm_head_speed'].isnull().values.any()
        if not (meta['flags']['mistracked_frames'] > 30 or meta['condition'] =='NA' or nancheck or len(meta['food_spots']) == 0):
            print('fly: {}, mistracked: {}, food spots: {}'.format(i_ses, meta['flags']['mistracked_frames'], len(meta['food_spots'])))
            classify = Classifier(df, meta)
            odf = classify.run(save_as=outfolder, ret=True)
    #print(odf.iloc[1924:1926])

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
