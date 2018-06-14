import os, argparse
import os.path as op
from pytrack_analysis.profile import get_profile
from pytrack_analysis.database import Experiment
import pytrack_analysis.preprocessing as prp
#from pytrack_analysis import Stats
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
    infolder = op.join(BASEDIR, 'pytrack_res', 'segments')
    outfolder = op.join(BASEDIR, 'pytrack_res', 'stats')
    if not op.isdir(outfolder):
        os.mkdir(outfolder)
    experiment = [_file for _file in os.listdir(rawfolder) if _file.endswith('csv') and not _file.startswith('.') and _file[:-3]+'yaml' in os.listdir(rawfolder)][0][:4]
    sessions = [_file for _file in os.listdir(rawfolder) if experiment in _file and _file.endswith('csv') and not _file.startswith('.') and _file[:-3]+'yaml' in os.listdir(rawfolder)]
    print(sorted(sessions))
    n_ses = len(sessions)
    stats = []
    states = {  'etho': {4: 'yeast micromovements', 5: 'sucrose micromovements'},
                'visit': {1: 'yeast visit', 2: 'sucrose visit'},
                'encounter': {1: 'yeast encounter', 2: 'sucrose encounter'},
        }
    outdfs = {key: pd.DataFrame({'session':[],'condition':[], 'substrate': [], 'total':[],'mean':[],'median':[],'frequency':[],'number':[]}) for key in states.keys()}

    ### GO THROUGH SESSIONS
    for i_ses, each in enumerate(sorted(sessions)):
        ### Loading data
        session = each[:-4]
        try:
            yamlfile = op.join(rawfolder, session+'.yaml')
            meta = read_yaml(yamlfile)
            meta['session'] = session
            for _in in states.keys():
                csv_file = op.join(infolder,  '{}_{}_{}.csv'.format(session, op.basename(infolder), _in))
                in_df = pd.read_csv(csv_file, index_col='segment')
                for state, val in states[_in].items():
                    df = in_df.query('state == {}'.format(state)).query('duration > 0.6')
                    outdfs[_in] = outdfs[_in].append({  'session': session,
                                                        'condition': meta['condition'],
                                                        'substrate': val.split(' ')[0],
                                                        'total': df['duration'].sum()/60.,
                                                        'mean': df['duration'].mean()/60.,
                                                        'median': df['duration'].median()/60.,
                                                        'frequency': 60.*len(df.index)/in_df.query('state != {}'.format(state))['duration'].sum(),
                                                        'number': len(df.index)}, ignore_index=True)
                    outdfs[_in]['genotype'], outdfs[_in]['temperature'] = outdfs[_in]['condition'].str.split(' ', 1).str
        except FileNotFoundError:
            print(csv_file+ ' not found!')
    for _in in states.keys():
        outdfs[_in].to_csv(op.join(outfolder, _in+'_stats.csv'), index_label='id')
        #print(_in,':')
        #print(outdfs[_in])
    ### delete objects

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False)
    test(main)
    del test
