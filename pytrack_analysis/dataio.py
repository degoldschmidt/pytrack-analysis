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
def get_files(raw, session, video_folder):
    session_folder = os.path.join(raw, "{:02d}".format(session))
    if os.path.isdir(session_folder):
        print("\nStart post-tracking analysis for video session: {:02d}".format(session))
        dtstamp, timestampstr = get_time(session_folder)
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
Returns number of frame skips, big frame skips (more than 10 frames) and maximum skipped time between frames
"""
def get_frame_skips(datal, dt='frame_dt', println=False, printmore=False):
    frameskips = np.array(datal[0].loc[:,dt])
    max_skip = np.amax(frameskips)
    max_skip_arg = frameskips.argmax()
    for odata in datal:
        oframeskips = np.array(odata.loc[:,dt])
        if np.any(oframeskips != frameskips):
            prn(__name__)
            colorprint('WARNING: not same frameskips', color='warning')
    total = frameskips.shape[0]
    strict_skips = np.sum(frameskips > (1/30)+(1/30))
    easy_skips = np.sum(frameskips > (1/30)+(1/3))
    if println:
        if 100*strict_skips/total < 0.1:
            prn(__name__)
            print('detected frameskips: {:} ({:3.3f}% of all frames)'.format(strict_skips, 100*strict_skips/total))
        else:
            prn(__name__)
            flprint('detected frameskips: ')
            colorprint('{:} ({:3.3f}% of all frames)'.format(strict_skips, 100*strict_skips/total))
    if printmore:
        prn(__name__)
        print('skips of more than 1 frames (>{:1.3f} s): {:} ({:3.3f}% of all frames)'.format((1/30)+(1/30), strict_skips, 100*strict_skips/total))
        prn(__name__)
        print('skips of more than 10 frames (>{:1.3f} s): {:} ({:3.3f}% of all frames)'.format((1/30)+(1/3), easy_skips, 100*easy_skips/total))
    return {"Strict frameskips": strict_skips,"Long frameskips": easy_skips, "Max frameskip duration":  max_skip, "Max frameskip index": max_skip_arg}

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

class RawData(object):
    def __init__(self, _exp_id, _session_id, _folders):
        ### get timestamp and all files from session folder
        self.allfiles, self.dtime, self.timestr = get_files(_folders['raw'], _session_id, _folders['videos'])

        ### load raw data
        self.raw_data = get_data(self.allfiles['data'])
        self.data_units = None

        ### check whether dataframes are of same dimensions
        lens = []
        for each_df in self.raw_data:
            lens.append(len(each_df))
        minlen = np.amin(lens)
        maxlen = np.amax(lens)
        for i, each_df in enumerate(self.raw_data):
            each_df = each_df.iloc[:minlen]

        ### getting metadata for each arena
        self.labels = ['topleft', 'topright', 'bottomleft', 'bottomright']
        ### arenas
        self.arenas = get_geom(self.allfiles['geometry'], self.labels)
        ### food spots
        food_spots = get_food(self.allfiles['food'], self.arenas)

        """
        meta['datadir'] = os.path.join(os.path.dirname(_folders['raw']), "{:02d}".format(each_session))
        meta['experiment'] = _exp_id
        meta['num_frames'] = get_num_frames(raw_data)
        for each_condition in meta['variables']:
            for each_file in meta["files"]:
                if each_condition in each_file:
                    conditions = pd.read_csv(each_file, sep='\t', index_col='ix')
        """
        ### detect arena geometry
        #arena = get_arena_geometry()

        ### detect food spots
        #food_spots = get_food_spots()

    def analyze_frameskips(self, dt=None):
        if dt is None:
            self.skips = get_frame_skips(self.raw_data, println=True)
        else:
            self.skips = get_frame_skips(self.raw_data, dt=dt, println=True)

    def define(self, columns=None, units=None):
        assert len(columns) == len(units), 'Error: dimension of given columns is unequal to given units.'
        self.data_units = units
        for each_df in self.raw_data:
            # renaming columns with standard header
            each_df.columns = columns
            if "Datetime" in units:
                # datetime strings to datetime objects
                each_df[units == "Datetime"] =  pd.to_datetime(each_df['datetime'])

    def set_scale(self, _which, _value, unit=None):
        if _which == 'diameter':
            outval = _value/2
        elif _which == 'radius':
            outval = _value
        if unit == 'cm':
            outval *= 10
        elif unit == 'm':
            outval *= 1000
        self.arenas.set_scale(outval)
