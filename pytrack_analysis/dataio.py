import os
import numpy as np
import pandas as pd
from pytrack_analysis.cli import colorprint, flprint, prn
from pytrack_analysis.arena import get_geom
from pytrack_analysis.food_spots import get_food
from pytrack_analysis.geometry import get_angle, get_distance, rot
import warnings

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
Returns list of raw data for filenames (DATAIO)
"""
def get_data(filenames):
    prn(__name__)
    flprint("Loading raw data...")
    data = []
    for each in filenames:
        ## load data
        data.append(pd.read_csv(each, sep="\s+"))
    colorprint("done.", color='success')
    return data

"""
Returns dictionary of all raw data files
"""
def get_files(raw, video_folder, noVideo=False):
    dtstamps = []
    timestr = []
    file_list = []
    for each_session in os.listdir(raw):
        session_folder = os.path.join(raw, each_session)
        if os.path.isdir(session_folder):
            dtstamp, timestampstr = get_time(session_folder)
            dtstamps.append(dtstamp)
            timestr.append(timestampstr)
            if noVideo:
                    file_dict = {
                                    "data" : [os.path.join(session_folder, eachfile) for eachfile in os.listdir(session_folder) if "fly" in eachfile and timestampstr in eachfile],
                                    "food" : [os.path.join(session_folder, eachfile) for eachfile in os.listdir(session_folder) if "food" in eachfile and timestampstr in eachfile],
                                    "geometry" : [os.path.join(session_folder, eachfile) for eachfile in os.listdir(session_folder) if "geometry" in eachfile and timestampstr in eachfile][0],
                                    "timestart" : [os.path.join(session_folder, eachfile) for eachfile in os.listdir(session_folder) if "timestart" in eachfile and timestampstr in eachfile][0],
                                }
            else:
                    file_dict = {
                                    "data" : [os.path.join(session_folder, eachfile) for eachfile in os.listdir(session_folder) if "fly" in eachfile and timestampstr in eachfile],
                                    "food" : [os.path.join(session_folder, eachfile) for eachfile in os.listdir(session_folder) if "food" in eachfile and timestampstr in eachfile],
                                    "geometry" : [os.path.join(session_folder, eachfile) for eachfile in os.listdir(session_folder) if "geometry" in eachfile and timestampstr in eachfile][0],
                                    "timestart" : [os.path.join(session_folder, eachfile) for eachfile in os.listdir(session_folder) if "timestart" in eachfile and timestampstr in eachfile][0],
                                    "video" : [os.path.join(video_folder, eachfile) for eachfile in os.listdir(video_folder) if timestampstr in eachfile][0],
                                }
            file_list.append(file_dict)
    flprint("found {} sessions...".format(len(file_list)))
    return file_list, dtstamps, timestr, len(file_list)

"""
Returns frame dimensions as tuple (height, width, channels) (DATAIO)
"""
def get_frame_dims(filename):
    warnings.filterwarnings("ignore")
    import skvideo.io
    videogen = skvideo.io.vreader(filename)
    for frame in videogen:
        dims = frame.shape
        break
    warnings.filterwarnings("default")
    return dims

"""
Returns session numbers as list based on arguments
"""
def get_session_list(N, *args):
    start = 0
    end = N
    nott = []
    if len(args) > 0:
        if type(args[0]) is str:
            start = int(args[0])
    if len(args) > 1:
        if type(args[1]) is str:
            end = int(args[1])
    if len(args) > 2:
        if type(args[2]) is str:
            nott = [int(each) for each in args[2].split(',')]
    outlist = [i for i in range(start,end)]
    for each in nott:
        try:
            outlist.remove(each)
        except ValueError:
            pass
    if len(args) > 3:
        if type(args[3]) is str:
            outlist = [int(each) for each in args[3].split(',')]
    return outlist

"""
Returns datetime for session start (DATAIO)
"""
def get_session_start(filename):
    from datetime import datetime
    filestart = np.loadtxt(filename, dtype=bytes).astype(str)
    return datetime.strptime(filestart[1][:19], '%Y-%m-%dT%H:%M:%S')

"""
Returns timestamp from session folder
"""
def get_time(session):
    from datetime import datetime
    for each in os.listdir(session):
        if "timestart" in each:
            any_file = each
    timestampstr = any_file.split('.')[0][-19:]
    dtstamp = datetime.strptime(timestampstr, "%Y-%m-%dT%H_%M_%S")
    return dtstamp, timestampstr[:-3]

"""
Returns translated data for given session start (PROCESSING)
"""
def translate_to(data, start, time=''):
    mask = (data[time] > start)
    data = data.loc[mask]
    return data, data.index[0]

class RawData(object):
    def __init__(self, _exp_id, _folders, columns=None, units=None, noVideo=False):
        prn(__name__)
        flprint("Loading raw data folders and file structure...")
        ### get timestamp and all files from session folder
        self.allfiles, self.dtime, self.timestr, self.nvids = get_files(_folders['raw'], _folders['videos'], noVideo=noVideo)
        ### conditions
        self.allconditions = get_conditions(_folders['manual'])
        ### data columns
        self.columns = columns
        ### data units
        self.units = units
        ### check whether valid dims
        assert len(self.columns) == len(self.units), 'Error: dimension of given columns is unequal to given units'
        ### noVideo option
        self.noVideo = noVideo
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
        print("starting post-tracking analysis for session {}/{} ({})...".format(_id, self.nvids, self.timestamp))
        if self.noVideo:
            prn(__name__)
            colorprint("Warning: no video!", color='warning')
        else:
            self.video_file = self.allfiles[_id]['video']

        ### load raw data and define columns/units
        self.raw_data = get_data(self.allfiles[_id]['data'])
        ### load the four data files
        for each_df in self.raw_data:
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
        ### center around arena center
        for ix, each_df in enumerate(self.raw_data):
            each_df['body_x'] = each_df['body_x']  - self.arenas[ix].x
            each_df['body_y'] = each_df['body_y']  - self.arenas[ix].y

    def flip_y(self):
        for ix, each_df in enumerate(self.raw_data):
            for each_spot in self.arenas[ix].spots:
                each_spot.ry *= -1
            for jx, each_col in enumerate(each_df.columns):
                if '_y' in each_col:
                    self.raw_data[ix][each_col] *= -1


    def get(self, _index):
        out = None
        if _index in self.labels.keys():
            out = copy.deepcopy(self)
        return out

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
            self.arenas.set_scale(outval)
        else:
            if _which == 'diameter':
                outval = _value/2
            if unit == 'cm':
                outval *= 10
            elif unit == 'm':
                outval *= 1000
            self.arenas.set_rscale(outval)

    def show(self):
        for i,each in enumerate(self.raw_data[0].columns):
            print('{}: {} [{}]'.format(i, each, self.units[i]))
