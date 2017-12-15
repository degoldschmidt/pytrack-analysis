import os
import numpy as np
import pandas as pd
from pytrack_analysis.cli import colorprint, flprint

NAMESPACE = "[data_io]\t"

def prn():
    colorprint(NAMESPACE, color='namespace', sln=True)

"""
Returns list of directories in given path d with full path (DATAIO)
"""
def flistdir(d):
    return [os.path.join(d, f) for f in os.listdir(d) if '.txt' in f]

"""
Returns list of raw data for filenames (DATAIO)
"""
def get_data(filenames):
    prn()
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
                        #"video" : [os.path.join(video_folder, eachfile) for eachfile in os.listdir(video_folder) if timestampstr in eachfile][0],
                    }
        prn()
        print("Timestamp:", dtstamp.strftime("%A, %d. %B %Y %H:%M"))
        return file_dict, dtstamp, timestampstr
    else:
        print("Session {:02d} not found. {}".format(session, session_folder))

"""
Returns number of frame skips, big frame skips (more than 10 frames) and maximum skipped time between frames
"""
def get_frame_skips(datal, dt='frame_dt', println=False, printmore=False):
    frameskips = datal[0][dt]
    max_skip = np.amax(frameskips)
    max_skip_arg = frameskips.idxmax()
    for odata in datal:
        oframeskips = odata[dt]
        if np.any(oframeskips != frameskips):
            prn()
            colorprint('WARNING: not same frameskips', color='warning')
    total = frameskips.shape[0]
    strict_skips = np.sum(frameskips > (1/30)+(1/30))
    easy_skips = np.sum(frameskips > (1/30)+(1/3))
    if println:
        if 100*strict_skips/total < 0.1:
            prn()
            print('detected frameskips: {:} ({:3.3f}% of all frames)'.format(strict_skips, 100*strict_skips/total))
        else:
            prn()
            flprint('detected frameskips: ')
            colorprint('{:} ({:3.3f}% of all frames)'.format(strict_skips, 100*strict_skips/total))
    if printmore:
        prn()
        print('skips of more than 1 frames (>{:1.3f} s): {:} ({:3.3f}% of all frames)'.format((1/30)+(1/30), strict_skips, 100*strict_skips/total))
        prn()
        print('skips of more than 10 frames (>{:1.3f} s): {:} ({:3.3f}% of all frames)'.format((1/30)+(1/3), easy_skips, 100*easy_skips/total))
    return {"Strict frameskips": strict_skips,"Long frameskips": easy_skips, "Max frameskip duration":  max_skip, "Max frameskip index": max_skip_arg}

"""
Returns session numbers as list based on arguments
"""
def get_session_list(N, *args):
    start = 1
    end = N
    nott = []
    if len(args) > 0:
        if type(args[0]) is str:
            start = int(args[0]) + 1
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
    def __init__(self, _i, _folders):
        ### get timestamp and all files from session folder
        self.allfiles, self.dtime, self.timestr = get_files(_folders['raw'], _i, _folders['videos'])

        ### load raw data
        self.raw_data = get_data(self.allfiles['data'])
        self.data_units = None

        ### check whether dataframes are of same dimensions
        lens = []
        for each_df in self.raw_data:
            lens.append(len(each_df))
        minlen = np.amin(lens)
        maxlen = np.amax(lens)
        print(minlen, maxlen)
        self.rawer = []
        for i, each_df in enumerate(self.raw_data):
            print(minlen)
            self.rawer.append(each_df.loc[:minlen, each_df.columns])

        for each_df in self.rawer:
            print(len(each_df))
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
