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
def get_files(raw, session, video_folder, noVideo=False):
    session_folder = os.path.join(raw, "{:02d}".format(session))
    if os.path.isdir(session_folder):
        print("\nStart post-tracking analysis for video session: {:02d}".format(session))
        dtstamp, timestampstr = get_time(session_folder)
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
        prn(__name__)
        print("Timestamp:", dtstamp.strftime("%A, %d. %B %Y %H:%M"))
        return file_dict, dtstamp, timestampstr
    else:
        print("Session {:02d} not found. {}".format(session, session_folder))

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
Returns metadata from session folder
"""
def get_meta(allfiles, dtstamp, conditions):
    meta = {}

    ### Food spots
    food_data = get_food(allfiles['food'])
    food_data = validate_food(food_data, geom_data)
    food_dict = {}
    labels = ['topleft', 'topright', 'bottomleft', 'bottomright']
    for kx, each_arena in enumerate(food_data):
        food_dict[labels[kx]] = {}
        for ix, each_spot in enumerate(each_arena):
            food_dict[labels[kx]][str(ix)] = {}
            labs = ['x', 'y', 'substrate']
            substrate = ['10% yeast', '20 mM sucrose']
            for jx, each_pt in enumerate(each_spot):
                if jx == 2:
                    if int(each_pt) < 2:
                        food_dict[labels[kx]][str(ix)][labs[jx]] = substrate[0]
                    if int(each_pt) == 2:
                        food_dict[labels[kx]][str(ix)][labs[jx]] = substrate[1]
                else:
                    food_dict[labels[kx]][str(ix)][labs[jx]] = float(each_pt)
    meta['food_spots'] = food_dict


    ### video stuff
    dims = get_frame_dims(allfiles["video"])
    meta['frame_height'] = dims[0]
    meta['frame_width'] = dims[1]
    meta['frame_channels'] = dims[2]
    meta['session_start'] = get_session_start(allfiles["timestart"])
    meta['video'] = os.path.basename(allfiles["video"])
    meta['video_start'] = dtstamp

    ### get conditions files
    meta["files"] = flistdir(conditions)
    meta["conditions"] =  [os.path.basename(each).split('.')[0] for each in meta["files"]]
    meta["variables"] = []
    for ix, each in enumerate(meta["files"]):
        with open(each, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                meta["variables"].append(meta["conditions"][ix])
            else:
                meta[meta["conditions"][ix]] = lines[0]
    return meta

"""
Returns session numbers as list based on arguments
"""
def get_session_list(N, *args):
    start = 1
    end = N + 1
    nott = []
    if len(args) > 0:
        if type(args[0]) is str:
            start = int(args[0])
    if len(args) > 1:
        if type(args[1]) is str:
            end = int(args[1]) + 1
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
    def __init__(self, _exp_id, _session_id, _folders, columns=None, units=None, noVideo=False):
        ### get timestamp and all files from session folder
        self.allfiles, self.dtime, self.timestr = get_files(_folders['raw'], _session_id, _folders['videos'], noVideo=noVideo)
        self.starttime = get_session_start(self.allfiles['timestart'])
        if noVideo:
            prn(__name__)
            colorprint("Warning: no video!", color='warning')

        ### define video
        if not noVideo:
            self.video_file = self.allfiles['video']

        ### load raw data and define columns/units
        self.raw_data = get_data(self.allfiles['data'])
        self.data_units = None
        assert len(columns) == len(units), 'Error: dimension of given columns is unequal to given units'
        self.data_units = units
        for each_df in self.raw_data:
            # renaming columns with standard header
            each_df.columns = columns
            if "Datetime" in units:
                # datetime strings to datetime objects
                each_df['datetime'] =  pd.to_datetime(each_df['datetime'])
        ### check whether dataframes are of same dimensions
        lens = []
        for each_df in self.raw_data:
            lens.append(len(each_df))
        minlen = np.amin(lens)
        maxlen = np.amax(lens)
        for i, each_df in enumerate(self.raw_data):
            each_df = each_df.iloc[:minlen]

        ### move to start position
        for ix, each_df in enumerate(self.raw_data):
            self.raw_data[ix], self.first_frame = translate_to(each_df, self.starttime, time='datetime')

        ### getting metadata for each arena
        self.labels = {'topleft': 0, 'topright': 1, 'bottomleft': 2, 'bottomright': 3}
        ### arenas
        self.arenas = get_geom(self.allfiles['geometry'], self.labels.keys())
        ### food spots
        self.food_spots = get_food(self.allfiles['food'], self.arenas)

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

        for ix, each_df in enumerate(self.raw_data):
            scale = self.arenas[ix].pxmm
            for jx, each_col in enumerate(each_df.columns):
                if self.data_units[jx] == 'px':
                    self.raw_data[ix][each_col] *= 1/scale
        for i, each in enumerate(self.data_units):
            if each == 'px':
                self.data_units[i] = 'mm'

    def show(self):
        for i,each in enumerate(self.raw_data[0].columns):
            print('{}: {} [{}]'.format(i, each, self.data_units[i]))
