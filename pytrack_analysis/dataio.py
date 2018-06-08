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

    def destruct(self):
        del self.dfs

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
            n = len(data.columns)-len(cols)
            for i in range(n):
                cols.append('unknown_'+str(i))
            data.columns = cols

    def to_timestart(self, timestart):
        for i, df in enumerate(self.dfs):
            self.dfs[i]['datetime'] = pd.to_datetime(df['datetime'])
            if timestart - self.dfs[i]['datetime'].iloc[0] > pd.Timedelta(minutes=60):
                self.dfs[i]['datetime'] += pd.Timedelta(hours=1)
            mask = (df['datetime'] > timestart)
            self.dfs[i] = df.loc[mask]
            self.first_frame = self.dfs[i].index[0]
            self.nframes = self.dfs[i].index[-1] - self.first_frame

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
        for _file in self.files['arena']:
            _dict = read_yaml(_file)
        self.arena, self.spots = [{}, {}, {}, {}], [{}, {}, {}, {}]
        for k in _dict.keys():
            if _dict[k]['arena']['name'] == 'topleft':
                self.arena[0] = _dict[k]['arena']
                self.spots[0] = _dict[k]['food_spots']
            elif _dict[k]['arena']['name'] == 'topright':
                self.arena[1] = _dict[k]['arena']
                self.spots[1] = _dict[k]['food_spots']
            elif _dict[k]['arena']['name'] == 'bottomleft':
                self.arena[2] = _dict[k]['arena']
                self.spots[2] = _dict[k]['food_spots']
            elif _dict[k]['arena']['name'] == 'bottomright':
                self.arena[3] = _dict[k]['arena']
                self.spots[3] = _dict[k]['food_spots']
        for i in self.arena:
            print(i['name'], i['x'], i['y'])


    def load_files(self, key):
        onlyIm = False
        self.files[key] = [op.join(self.dir, eachfile) for eachfile in sorted(os.listdir(self.dir)) if key in eachfile and self.timestr in eachfile and not eachfile.startswith('.')]
        if key == 'arena':
            self.files[key] = [op.join(self.dir, 'pytrack_res', 'arena', eachfile) for eachfile in sorted(os.listdir(op.join(self.dir, 'pytrack_res', 'arena'))) if key in eachfile and self.timestr in eachfile and not eachfile.startswith('.')]
            if len(self.files[key]) == 0:
                colorprint('no file found: starting automatic arena geometry detection', color='warning')
                self.geometry = detect_geometry(self.fullpath, self.timestr, onlyIm=onlyIm)
                if onlyIm:
                    return True
                self.files[key] = [op.join(self.dir, 'pytrack_res', 'arena', eachfile) for eachfile in sorted(os.listdir(op.join(self.dir, 'pytrack_res', 'arena'))) if key in eachfile and self.timestr in eachfile and not eachfile.startswith('.')]
        if key == 'timestart' and len(self.files[key]) == 1:
            self.timestart = parse_timestart(op.join(self.dir, self.files['timestart'][0]))
        if len(self.files[key]) == self.required[key]:
            return True
        else:
            return False

    def get_data(self, i=None):
        if i is None:
            return [v for v in self.data.dfs]
        else:
            return self.data.dfs[i]


    def load_data(self):
        self.data = Data(self.files['fly'])
        self.data.load()

    def run_posttracking(self):
        pass

    def unload_data(self):
        self.data.destruct()

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
    def __init__(self, basedir, columns=None, units=None, noVideo=False, VERBOSE=True):
        ### Load videos
        self.VERBOSE = VERBOSE
        prn(__name__)
        self.dir = basedir
        flprint("Loading raw data videos...")
        self.videos = parse_videos(basedir)
        self.nvids = len(self.videos)
        flprint("found {} videos...".format(self.nvids))
        colorprint("done.", color='success')
        ### Register experiment
        self.init_experiment()
        for video in self.videos:
            ### print video name
            if self.VERBOSE:
                prn(__name__)
                print('Video:\t{}'.format(video.name))
            ### load fly data
            self.init_files(video, 'raw fly data files', 'fly')
            ### load timestart file
            self.init_files(video, 'timestart file', 'timestart')
            ### load arena geometry file
            self.init_files(video, 'arena file', 'arena')

    def init_experiment(self):
        exps_files = [_file for _file in os.listdir(op.join(self.dir, 'pytrack_res')) if _file.endswith('yaml') and _file.startswith('pytrack_exp')]
        if self.VERBOSE:
            prn(__name__)
            print("Found {} pytrack experiment file(s)...".format(len(exps_files)))
        if len(exps_files) == 0:
            if query_yn('Do you want to register experiment (NO will exit the script)', default='yes'):
                self.experiment = register(self.videos)
            else:
                sys.exit(0)
        else:
            if self.VERBOSE:
                for i, exp in enumerate(exps_files):
                    if query_yn('Found pytrack experiment yaml file {} - Do you want to use it?'.format(exp), default='yes'):
                        self.experiment = read_yaml(op.join(self.dir, 'pytrack_res', exp))
                        show(self.experiment)
                        break
                    elif i == len(exps_files)-1:
                        self.experiment = register(self.videos)
            else:
                self.experiment = read_yaml(op.join(self.dir, 'pytrack_res', exps_files[0]))
        if self.nvids > len(self.experiment['Videos']):
            if query_yn('Found'+bc.FAIL+ ' {} '.format(self.nvids-len(self.experiment['Videos'])) +bc.ENDC+'more videos than registered - Do you want to add them to the registry?', default='yes'):
                add(self.videos, self.experiment)
        if self.nvids < len(self.experiment['Videos']):
            if query_yn('Found'+bc.FAIL+ ' {} '.format(len(self.experiment['Videos'])-self.nvids) +bc.ENDC+'less videos than registered - Do you want to remove them to the registry?', default='yes'):
                remove(self.videos, self.experiment)

    def init_files(self, video, title, key):
        if self.VERBOSE:
            prn(__name__)
            flprint("Loading {}...".format(title))
        if video.load_files(key):
            if self.VERBOSE:
                flprint("found {} file(s)...".format(len(video.files[key])))
                colorprint("done.", color='success')
        else:
            colorprint("ERROR: found invalid number of {} files ({} instead of {}).".format(key, len(video.files[key]), video.required[key]), color='error')
            sys.exit(0)
