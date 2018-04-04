import os
import numpy as np
import pandas as pd
from pytrack_analysis.cli import colorprint, flprint, prn
from pytrack_analysis.arena import get_geom
from pytrack_analysis.food_spots import get_food
from pytrack_analysis.geometry import get_angle, get_distance, rot
import warnings

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
                for each_col in each_df.columns:
                    if '_x' in each_col:
                        each_df[each_col] -= arenas[ix].x
                    if  '_y' in each_col:
                        each_df[each_col] -= arenas[ix].y
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
        self.files = parse_file(filename, dirname)
        self.time, self.timestr = parse_time(filename)
        self.setup = parse_setup(filename)
        self.setup += ' ({})'.format(SETUPNAMES[self.setup])
        self.timestart = parse_timestart(os.path.join(dirname, self.files['timestart'][0]))
        self.data = Data(self.files['data'])
        self.arenas = None

    def __str__(self):
        full_str = 'Video: {}\nRecorded: {}\nSession start: {}\nSetup: {}\nFiles:\n'.format(self.name, self.time, self.timestart, self.setup)
        for k,v in self.files.items():
            full_str += '\t{}:\n'.format(k)
            for each_v in v:
                full_str += '\t\t- {}\n'.format(each_v)
        return full_str

    def load_arena(self):
        pass

    def get_data(self):
        return [v for v in self.data.df]

    def load_data(self):
        self.data.load()

    def run_posttracking(self):
        pass

    def unload_data(self):
        del self.data


"""
Returns list of directories in given path d with full path (DATAIO)
"""
def flistdir(d):
    return [os.path.join(d, f) for f in os.listdir(d) if '.txt' in f]

def get_conditions(folder):
    import yaml
    fileName = os.path.join(folder, 'conditions.yaml')
    with open(fileName, 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

"""
Returns dictionary of all files for a given video file
"""
def parse_file(filename, basedir):
    dtstamp, timestampstr = parse_time(filename)
    file_dict = {
                    "data" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "fly" in eachfile and timestampstr in eachfile],
                    "food" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "food" in eachfile and timestampstr in eachfile],
                    "geometry" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "geometry" in eachfile and timestampstr in eachfile],
                    "timestart" : [os.path.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "timestart" in eachfile and timestampstr in eachfile],
                }
    return file_dict

"""
Returns list of video objects of all raw data files
"""
def parse_files(basedir):
    return [Video(each_avi, basedir) for each_avi in [_file for _file in os.listdir(basedir) if _file.endswith('avi')]]


"""
Returns datetime for session start (DATAIO)
"""
def parse_timestart(filename):
    from datetime import datetime
    with open(filename, 'rt', errors='replace') as f:
        data = f.read()
    return datetime.strptime(data.split(' ')[1][:-14], '%Y-%m-%dT%H:%M:%S')

"""
Returns setup from video file
"""
def parse_setup(video):
    setup = video.split('.')[0].split('_')[0]
    return setup

"""
Returns timestamp from video file
"""
def parse_time(video):
    from datetime import datetime
    timestampstr = video.split('.')[0][-19:]
    dtstamp = datetime.strptime(timestampstr, "%Y-%m-%dT%H_%M_%S")
    return dtstamp, timestampstr[:-3]

"""
Returns translated data for given session start (PROCESSING)
"""
def translate_to(data, start, time=''):
    mask = (data[time] > start)
    data = data.loc[mask]
    return data, data.index[0]

class VideoRawData(object):
    def __init__(self, experiment, basedir, columns=None, units=None, noVideo=False, VERBOSE=False):
        prn(__name__)
        flprint("Loading raw data folders and file structure...")
        self.experiment = experiment
        ### get timestamp and all files from session folder
        self.videos = parse_files(basedir)
        self.files = [_video.files for _video in self.videos]
        self.dtime = [_video.time for _video in self.videos]
        self.timestr = [_video.timestr for _video in self.videos]
        self.nvids = len(self.videos)
        flprint("found {} sessions...".format(self.nvids))
        if VERBOSE:
            print('\n')
            for i, video in enumerate(self.videos):
                print('[{}]'.format(i))
                print(video)
        ### conditions
        #self.allconditions = get_conditions(_folders['manual'])
        colorprint("done.", color='success')

    def get_data(self, fly=None):
        if fly is None:
            return self.raw_data
        else:
            return self.raw_data[fly]

    def get_session(self, _id):
        prn(__name__)
        self.timestamp = self.dtime[_id]
        self.sessiontimestr = self.timestr[_id]
        self.starttime = get_session_start(self.allfiles[_id]['timestart'])
        print("starting post-tracking analysis for session {}/{} ({})...".format(_id+1, self.nvids, self.timestamp))
        self.video_file = self.allfiles[_id]['video']

        ### load raw data and define columns/units
        self.raw_data = get_data(self.videos[_id]['data'])
        ### raw data is in pixel space (i.e., uncentered, unflipped, and unscaled)
        self.centered, self.flipped_y, self.scaled = False, False, False
        ### load the four data files
        for each_df in self.raw_fly_data:
            # renaming columns with standard header
            each_df.columns = self.columns
            if "Datetime" in self.units:
                # datetime strings to datetime objects
                each_df['datetime'] =  pd.to_datetime(each_df['datetime'])
        ### check whether dataframes are of same dimensions
        lens = [len(each_df) for each_df in self.raw_data]
        minl, maxl = np.amin(lens), np.amax(lens)
        for ix, each_df in enumerate(self.raw_data):
            each_df = each_df.iloc[:minl]
            ### move to start position
            self.raw_data[ix], self.first_frame = translate_to(each_df, self.starttime, time='datetime')
        self.last_frame = minl - 1
        ### getting metadata for each arena
        self.labels = {'topleft': 0, 'topright': 1, 'bottomleft': 2, 'bottomright': 3}
        ### arenas
        self.arenas = get_geom(self.allfiles[_id]['geometry'], self.labels.keys())
        ### food spots
        self.food_spots = get_food(self.allfiles[_id]['food'], self.arenas)
        ### conditions for session
        self.condition = self.allconditions['metabolic'][self.sessiontimestr]

    def center(self):
        ### center around arena center
        if not self.centered:
            for ix, each_df in enumerate(self.raw_data):
                for each_col in each_df.columns:
                    if '_x' in each_col:
                        each_df[each_col] -= self.arenas[ix].x
                    if  '_y' in each_col:
                        each_df[each_col] -= self.arenas[ix].y
        self.centered = True

    def flip_y(self):
        if not self.flipped_y:
            for ix, each_df in enumerate(self.raw_data):
                for each_spot in self.arenas[ix].spots:
                    each_spot.ry *= -1
                for jx, each_col in enumerate(each_df.columns):
                    if '_y' in each_col:
                        self.raw_data[ix][each_col] *= -1
        self.flipped = True

    def print_conditions(self, *args):
        for _conds, _val in self.allconditions.items():
            if type(_val) is not dict:
                print(_conds, ':', _val)
            else:
                print(_conds, ':')
                if len(args) == 0:
                    for _k, _v in _val.items():
                        print("\t",_k, ':', _v)
                for arg in args:
                    print("\t", list(_val.keys())[arg], ':', list(_val.values())[arg])

    def set_scale(self, _which, _value, unit=None):
        if _which == 'fix_scale':
            outval = _value
            if unit == 'px':
                outval = 1/_value
            if not self.scaled:
                self.arenas.set_scale(outval)
        else:
            if _which == 'diameter':
                outval = _value/2
            if unit == 'cm':
                outval *= 10
            elif unit == 'm':
                outval *= 1000
            if not self.scaled:
                self.arenas.set_rscale(outval)
        if not self.scaled:
            for each_df in self.raw_data:
                for each in [c for c in each_df.columns if '_x' in c or '_y' in c or 'or' in c]:
                    each_df[each] *= 1/_value
        self.scaled = True


    def show(self):
        for i,each in enumerate(self.raw_data[0].columns):
            print('{}: {} [{}]'.format(i, each, self.units[i]))
