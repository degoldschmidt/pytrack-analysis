import argparse
import subprocess, os
import numpy as np
import pandas as pd

from pytrack_analysis import Multibench
from pytrack_analysis.dataio import VideoRawData
from pytrack_analysis.profile import get_profile, get_scriptname, show_profile

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

def main(_basedir, _option, _overwrite):
    experiment = 'DIFF' ### TODO
    profile = get_profile(experiment)
    basedir = profile.set_folder(_basedir)

    ### Define raw data structure
    colnames = ['datetime', 'elapsed_time', 'frame_dt', 'body_x',   'body_y',   'angle',    'major',    'minor']
    raw_data = VideoRawData(experiment, basedir)
    ### go through all session
    for i, video in enumerate(raw_data.videos):

        ### arena + food spots
        #video.load_arena()
        ### trajectory data
        video.load_data()
        #video.data.reindex(colnames)
        #video.data.center_to_arena(video.arenas)
        ### fly/experiment metadata
        #for fly_idx, fly_data in enumerate(raw_data.get_data()):

        ###
        video.unload_data()
    del profile

if __name__ == '__main__':
    BASEDIR, OPTION, OVERWRITE = get_args()
    main(BASEDIR, OPTION, OVERWRITE)
