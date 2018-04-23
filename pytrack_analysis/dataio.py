import os
import os.path as op
import numpy as np
import pandas as pd
from datetime import datetime
from pytrack_analysis.cli import colorprint, flprint, prn, query_yn, bc
from pytrack_analysis.arena import get_geom
from pytrack_analysis.food_spots import get_food
from pytrack_analysis.geometry import get_angle, get_distance, rot, detect_geometry
import warnings
import io, yaml
import sys

SUFFIX_EXP = 'pytrack_exp'

SETUPNAMES = {  'cam01': 'Adler',
                'cam02': 'Baer',
                'cam03': 'Chameleon',
                'cam04': 'Dachs',
                'cam05': 'Elefant',
            }


def read_yaml(_file):
    """ Returns a dict of a YAML-readable file '_file'. Returns None, if file is empty. """
    with open(_file, 'r') as stream:
        out = yaml.load(stream)
    return out

def write_yaml(_file, _dict):
    """ Writes a given dictionary '_dict' into file '_file' in YAML format. Uses UTF8 encoding and no default flow style. """
    with io.open(_file, 'w+', encoding='utf8') as outfile:
        yaml.dump(_dict, outfile, default_flow_style=False, allow_unicode=True)

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
        self.files = {}
        self.time, self.timestr = parse_time(filename)
        self.filekeys = {'data': 'fly', 'timestart': 'timestart'}
        self.required = {'data': 4, 'timestart': 1}
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
        self.files[key] = [op.join(self.dir, eachfile) for eachfile in os.listdir(self.dir) if self.filekeys[key] in eachfile and self.timestr in eachfile]
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
Returns list of directories in given path d with full path (DATAIO)
"""
def flistdir(d):
    return [op.join(d, f) for f in os.listdir(d) if '.txt' in f]

"""
Returns dictionary of all files for a given video file
"""
def parse_file(filename, basedir):
    dtstamp, timestampstr = parse_time(filename)
    file_dict = {
                    "data" : [op.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "fly" in eachfile and timestampstr in eachfile],
                    "food" : [op.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "food" in eachfile and timestampstr in eachfile],
                    "geometry" : [op.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "geometry" in eachfile and timestampstr in eachfile],
                    "conditions" : [op.join(basedir, eachfile) for eachfile in os.listdir(basedir) if eachfile.endswith('yaml') and timestampstr in eachfile],
                    "timestart" : [op.join(basedir, eachfile) for eachfile in os.listdir(basedir) if "timestart" in eachfile and timestampstr in eachfile],
                }
    return file_dict


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
Returns list of video objects of all raw data files
"""
def parse_videos(basedir):
    return [Video(each_avi, basedir) for each_avi in [_file for _file in sorted(os.listdir(basedir)) if _file.endswith('avi')]]

""" ### v0.1
Returns dict of experiment details
"""
def register(videos):
    outdict = {}
    listtimes = []
    for video in videos:
        videotime, _ = parse_time(video.name)
        listtimes.append(videotime)
        listtimes = sorted(listtimes)
    outdict['Title'] = input('Title of the experiment: ')
    ID = ''
    while len(ID) != 4:
        ID = input('Four-letter ID: ')
    outdict['ID'] = ID.upper()
    outdict['Start date'] = datetime.strftime(listtimes[0], '%Y-%m-%d')
    outdict['End date'] = datetime.strftime(listtimes[-1], '%Y-%m-%d')
    outdict['Number of videos'] = len(videos)
    outdict['Videos'] = [video.name for video in videos]

    outdict['Constants'] = set_constants(outdict['Title'])
    outdict['Variables'] = set_variables(outdict['Title'])
    outdict['Conditions'] = {}
    for video in videos:
        outdict['Conditions'][video.name] = set_conditions(video, variables=outdict['Variables'])

    uniques = []
    for video in videos:
        conds = outdict['Conditions'][video.name]
        condslist = [v for k, v in conds.items()]
        for i in range(4):
            fly_cond = [el[i] for el in condslist]
            if fly_cond not in uniques:
                uniques.append(fly_cond)
    outdict['Number of conditions'] = len(uniques)
    show(outdict)

    outfile = SUFFIX_EXP+ID+'.yaml'
    if query_yn('Confirm and save experiment yaml file {}?'.format(outfile), default='yes'):
        write_yaml(op.join(video.dir, outfile), outdict)
        return outdict
    else:
        return register(videos)

def show(_dict):
    print(bc.BOLD+'\nExperiment summary:\n'+bc.ENDC)
    for k, v in _dict.items():
        if type(v) is dict:
            print("{}:".format(k))
            for k2, v2 in v.items():
                print("\t{}:\t{}".format(k2, v2))
        else:
            print("{}:\t{}".format(k, v))
    print()

def set_conditions(video, variables=None):
    writing = True
    condition_dict = {}
    k = None
    print('\nEnter'+bc.OKBLUE+' conditions '+bc.ENDC+'for'+bc.OKBLUE+' {}'.format(video.name)+bc.ENDC)
    for k in variables.keys():
        v = input('Value for key'+bc.BOLD+' {} '.format(k)+bc.ENDC+'(multiple values are separated by whitespace): '.format(k))
        v = v.split(' ')
        lv = len(v)
        for i in range(4-lv):   ### if less then 4 conditions are given, last condition will be repeated
            v.append(v[-1])
        for each in v:
            if each not in variables[k]:
                print('Warning: {} not found in possible values for {}.'. format(each, k))
                return set_conditions(video, variables=variables)
        condition_dict[k] = v
        k = None
    return condition_dict

""" ### v0.1
Sets constants of experiment
"""
def set_constants(exptitle):
    writing = True
    constants_dict = {}
    k = None
    print('\nEnter'+bc.OKBLUE+' constants '+bc.ENDC+'for'+bc.OKBLUE+' {}'.format(exptitle)+bc.ENDC)
    while writing:
        if k == None:
            k = input('\nPlease type constants key ("enter to quit"): ')
        if k == '':
            writing = False
        else:
            v = input('Value for key'+bc.BOLD+' {}'.format(k)+bc.ENDC+': '.format(k))
            constants_dict[k] = v
            k = None
    return constants_dict

""" ### v0.1
Sets variables of experiment
"""
def set_variables(exptitle):
    writing = True
    variables_dict = {}
    k = None
    print('\nEnter'+bc.OKBLUE+' variables '+bc.ENDC+'for'+bc.OKBLUE+' {}'.format(exptitle)+bc.ENDC)
    while writing:
        if k == None:
            k = input('\nPlease type variables key ("enter to quit"): ')
        if k == '':
            writing = False
        else:
            v = input('Possible values for key'+bc.BOLD+' {} '.format(k)+bc.ENDC+'(multiple values are separated by whitespace): '.format(k))
            v = v.split(' ')
            variables_dict[k] = v
            k = None
    return variables_dict

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
            print('Video:\t{}'.format(video.name))

            ### load fly data
            self.init_files(video, 'raw fly data files', 'data')
            ### load timestart file
            self.init_files(video, 'timestart file', 'timestart')


        ### Load geometry
        """
        flprint("Loading raw fly data files...")
        for video in self.videos:
            video.load_fly_files()
        """



        """
        self.experiment = experiment
        self.dir = basedir
        self.manual_dir = op.join(basedir, 'manual')
        if not op.isdir(self.manual_dir):
            os.mkdir(self.manual_dir)
        if not op.isfile(op.join(self.manual_dir, 'constants.yaml')):
            self.set_constants()
        else:
            self.constants = read_yaml(op.join(self.manual_dir, 'constants.yaml'))
        if not op.isfile(op.join(self.manual_dir, 'variables.yaml')):
            self.set_variables()
        else:
            self.variables = read_yaml(op.join(self.manual_dir, 'variables.yaml'))
        ### get timestamp and all files from session folder
        self.videos = parse_files(basedir, self.variables)
        self.files = [_video.files for _video in self.videos]
        self.dtime = [_video.time for _video in self.videos]
        self.timestr = [_video.timestr for _video in self.videos]
        if VERBOSE:
            print('\n')
            for i, video in enumerate(self.videos):
                print('[{}]'.format(i))
                print(video)

        """

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
                    break
                elif i == len(exps_files)-1:
                    self.experiment = register(self.videos)
        show(self.experiment)

    def init_files(self, video, title, key):
        flprint("Loading {}...".format(title))
        if video.load_files(key):
            flprint("found {} file(s)...".format(len(video.files[key])))
            colorprint("done.", color='success')
        else:
            colorprint("ERROR: found invalid number of raw fly data files ({} instead of {}).".format(len(video.files[key]), video.required[key]), color='error')

    def get_data(self, fly=None):
        if fly is None:
            return self.raw_data
        else:
            return self.raw_data[fly]


    ### not used
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
