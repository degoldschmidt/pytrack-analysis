import os
import os.path as op
import warnings
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pytrack_analysis.cli import colorprint, flprint, prn, query_yn, bc
from pytrack_analysis.arena import get_geom
from pytrack_analysis.food_spots import get_food
from pytrack_analysis.experiment import parse_setup, parse_time, add, remove, register, show
from pytrack_analysis.geometry import get_angle, get_distance, rot, detect_geometry
from pytrack_analysis.yamlio import read_yaml

SETUPNAMES = {  'cam01': 'Adler',
                'cam02': 'Baer',
                'cam03': 'Chameleon',
                'cam04': 'Dachs',
                'cam05': 'Elefant',
            }

class Data(object):
    def __init__(self, _files):
        self.files = _files
        self.dfs = []
        self.centered, self.flipped_y, self.scaled = False, False, False

    def center_to_arena(self, arenas):
        ### center around arena center
        if not self.centered:
            for i, df in enumerate(self.dfs):
                for col in df.columns:
                    if '_x' in col:
                        df[col] -= arenas[i].x
                    if  '_y' in col:
                        df[col] -= arenas[i].y
        self.centered = True

    def get(self, i):
        return self.dfs[i]

    def load(self):
        for _file in self.files:
            self.dfs.append(pd.read_csv(_file, sep="\s+"))

    def reindex(self, cols):
        for data in self.dfs:
            data.columns = cols

class Video(object):
    def __init__(self, filename, dirname):
        self.name = filename
        self.dir = dirname
        self.fullpath = op.join(dirname, filename)
        self.files = {}
        self.time, self.timestr = parse_time(filename)
        self.required = {'fly': 4, 'timestart': 1, 'arena': 1}
        """
        self.setup = parse_setup(filename)
        self.setup += ' ({})'.format(SETUPNAMES[self.setup])
        self.timestart = parse_timestart(op.join(dirname, self.files['timestart'][0]))
        self.data = Data(self.files['data'])
        self.arenas = None
        """

    def __str__(self):
        full_str = 'Video: {}\nRecorded: {}\nSession start: {}\nSetup: {}\nFiles:\n'.format(self.name, self.time, self.timestart, self.setup)
        for k,v in self.files.items():
            full_str += '\t{}:\n'.format(k)
            for each_v in v:
                full_str += '\t\t- {}\n'.format(each_v)
        return full_str

    def load_arena(self):
        pass

    def load_files(self, key):
        self.files[key] = [op.join(self.dir, eachfile) for eachfile in os.listdir(self.dir) if key in eachfile and self.timestr in eachfile]
        if key == 'arena' and len(self.files[key]) == 0:
            colorprint('no file found: starting automatic arena geometry detection', color='warning')
            self.geometry = detect_geometry(self.fullpath, self.timestr)
            self.files[key] = [op.join(self.dir, eachfile) for eachfile in os.listdir(self.dir) if key in eachfile and self.timestr in eachfile]
        if len(self.files[key]) == self.required[key]:
            return True
        else:
            return False

    def get_data(self):
        return [v for v in self.data.df]


    def load_data(self):
        self.data.load()

    def run_posttracking(self):
        pass



    def unload_data(self):
        del self.data

"""
Returns datetime for session start (DATAIO)
"""
def parse_timestart(filename):
    from datetime import datetime
    with open(filename, 'rt', errors='replace') as f:
        data = f.read()
    return datetime.strptime(data.split(' ')[1][:-14], '%Y-%m-%dT%H:%M:%S')

""" ### v0.1
Returns list of video objects of all raw data files
"""
def parse_videos(basedir):
    all_file_tstamp = [_file[6:] for _file in os.listdir(basedir) if _file.endswith('avi')]
    sorted_ts = sorted(all_file_tstamp)
    print(sorted_ts)
    filelist_sorted = []
    for ts in sorted_ts:
        for _file in os.listdir(basedir):
            if ts in _file:
                filelist_sorted.append(_file)
    return [Video(each_avi, basedir) for each_avi in [_file for _file in filelist_sorted if _file.endswith('avi')]]

"""
Returns translated data for given session start (PROCESSING)
"""
def translate_to(data, start, time=''):
    mask = (data[time] > start)
    data = data.loc[mask]
    return data, data.index[0]

class VideoRawData(object):
    def __init__(self, basedir, columns=None, units=None, noVideo=False, VERBOSE=False):
        ### Load videos
        prn(__name__)
        self.dir = basedir
        flprint("Loading raw data videos...")
        self.videos = parse_videos(basedir)
        self.nvids = len(self.videos)
        flprint("found {} sessions...".format(self.nvids))
        colorprint("done.", color='success')
        ### Register experiment
        self.init_experiment()
        for video in self.videos:
            ### print video name
            prn(__name__)
            print('Video:\t{}'.format(video.name))
            ### load fly data
            prn(__name__)
            self.init_files(video, 'raw fly data files', 'fly')
            ### load timestart file
            prn(__name__)
            self.init_files(video, 'timestart file', 'timestart')
            ### load arena geometry file
            prn(__name__)
            self.init_files(video, 'arena file', 'arena')
            print()

    def init_experiment(self):
        exps_files = [_file for _file in os.listdir(self.dir) if _file.endswith('yaml') and _file.startswith('pytrack_exp')]
        prn(__name__)
        print("Found {} pytrack experiment file(s)...".format(len(exps_files)))
        if len(exps_files) == 0:
            if query_yn('Do you want to register experiment (NO will exit the script)', default='yes'):
                self.experiment = register(self.videos)
            else:
                sys.exit(0)
        else:
            for i, exp in enumerate(exps_files):
                if query_yn('Found pytrack experiment yaml file {} - Do you want to use it?'.format(exp), default='yes'):
                    self.experiment = read_yaml(op.join(self.dir, exp))
                    show(self.experiment)
                    break
                elif i == len(exps_files)-1:
                    self.experiment = register(self.videos)
        if self.nvids > len(self.experiment['Videos']):
            if query_yn('Found'+bc.FAIL+ ' {} '.format(self.nvids-len(self.experiment['Videos'])) +bc.ENDC+'more videos than registered - Do you want to add them to the registry?', default='yes'):
                add(self.videos, self.experiment)
        if self.nvids < len(self.experiment['Videos']):
            if query_yn('Found'+bc.FAIL+ ' {} '.format(len(self.experiment['Videos'])-self.nvids) +bc.ENDC+'less videos than registered - Do you want to remove them to the registry?', default='yes'):
                remove(self.videos, self.experiment)

    def init_files(self, video, title, key):
        flprint("Loading {}...".format(title))
        if video.load_files(key):
            flprint("found {} file(s)...".format(len(video.files[key])))
            colorprint("done.", color='success')
        else:
            colorprint("ERROR: found invalid number of raw fly data files ({} instead of {}).".format(len(video.files[key]), video.required[key]), color='error')
            sys.exit(0)
