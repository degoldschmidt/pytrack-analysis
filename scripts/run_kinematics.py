import os
import os.path as op
import argparse
from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
from pytrack_analysis import Kinematics
from pytrack_analysis import Multibench
from pytrack_analysis.yamlio import read_yaml

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.width', 260)
pd.set_option('precision', 4)

RUN_STATS = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', metavar='basedir', type=str, help='directory where your data files are')
    parser.add_argument('--exp', action='store', type=str)
    parser.add_argument('--option', action='store', type=str)
    parser.add_argument('--overwrite', action='store_true')
    BASEDIR = parser.parse_args().basedir
    OVERWRITE = parser.parse_args().overwrite
    EXP = parser.parse_args().exp
    if parser.parse_args().option is None:
        OPTION = 'all'
    else:
        OPTION = parser.parse_args().option
    return BASEDIR, OPTION, OVERWRITE, EXP

def main():
    BASEDIR, OPTION, OVERWRITE, EXP = get_args()
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
    experiment = EXP
    infolder = op.join(BASEDIR, 'pytrack_res', 'post_tracking')
    outfolder = op.join(BASEDIR, 'pytrack_res', 'kinematics')
    sessions = [_file for _file in os.listdir(infolder) if EXP in _file and _file.endswith('csv') and not _file.startswith('.') and _file[:-3]+'yaml' in os.listdir(infolder)]
    print(sessions)
    n_ses = len(sessions)
    stats = []

    ### GO THROUGH SESSIONS
    for i_ses, each in enumerate(sessions):
        datafile = op.join(infolder, each)
        yamlfile = op.join(infolder, each[:-3]+'yaml')
        df = pd.read_csv(datafile, index_col='frame')
        m = np.array(df['major'])/2.
        a = np.array(df['angle'])
        bx, by = np.array(df['body_x']), np.array(df['body_y'])
        df['head_x'] = bx+np.multiply(m,np.cos(a))
        df['head_y'] = by+np.multiply(m,np.sin(a))
        df['tail_x'] = bx-np.multiply(m,np.cos(a))
        df['tail_y'] = by-np.multiply(m,np.sin(a))
        meta = read_yaml(yamlfile)
        kine = Kinematics(df, meta)
        ### run kinematics
        outdf = kine.run(save_as=outfolder, ret=True)
        ### get stats and append to list
        if RUN_STATS: stats.append(kine.stats())

    ### save stats
    statdf = pd.concat(stats, ignore_index=True)
    print(statdf)
    statfile = os.path.join(outfolder, experiment+'_kinematics_stats.csv')
    statdf.to_csv(statfile, index=False)
    ### delete objects
    del profile

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
