import os, sys, time
import numpy as np
import pandas as pd
import yaml, json
from collections.abc import Mapping
from asciitree import draw_tree
from ._globals import *
from pytrack_analysis.cli import colorprint, flprint, prn

"""
DataFileFormats IO (TODO: move to fileio.py)
"""

def json2dict(_file):
    """Convert JSON to dict"""
    with open(_file) as json_data:
        d = json.load(json_data)
    return d

def load_yaml(_file):
    """Returns file structure and timestamps of given database file as dictionary"""
    out = []
    with open(_file,'r') as input_file:
        results = yaml2dict(input_file)
        for value in results:
            out.append(value)
    return out[0], out[1]

def yaml2dict(val):
    """Convert YAML to dict"""
    try:
        return yaml.safe_load_all(val)
    except yaml.YAMLError as exc:
        return exc

"""
Data Integrity Tests
"""

def get_created(filepath):
    """
    Returns created timestamp for given filepath

    args:
    * filepath [str] : full path of given file
    """
    try:
        return time.ctime(os.path.getctime(filepath))
    except OSError:
        return 0

def get_modified(filepath):
    """
    Returns last modified timestamp for given filepath

    args:
    * filepath [str] : full path of given file
    """
    try:
        return time.ctime(os.path.getmtime(filepath))
    except OSError:
        return 0

def check_base(_dict, _dir):
    """
    Checks and returns boolean, whether database is in given directory
    """
    key = list(_dict.keys())[0]
    return (key in os.listdir(_dir) )

def check_meta(_dict, _dir):
    """
    Checks and returns boolean, whether all meta-data files are in given directory
    """
    flag = 0
    for key,val in _dict.items():
        for session in val:
            yamlfile = session[:-3]+"yaml"
            if not yamlfile in os.listdir(_dir):
                flag = 1
    return (flag == 0)

def check_data(_dict, _dir):
    """
    Checks and returns boolean, whether all data files are in given directory
    """
    flag = 0
    for key, val in _dict.items():
        for session in val:
            if not session in os.listdir(_dir):
                flag = 1
    return (flag == 0)

def check_time(_tstamps, _dir):
    """
    Checks and returns boolean, whether all files have valid timestamps
    """
    flag = 0
    for filename, times in _tstamps.items():
        mod = times["modified"]
        cre = times["created"]
        if not cre == get_created(os.path.join(_dir, filename)) and mod == get_modified(os.path.join(_dir, filename)):
            flag = 1
    return (flag == 0)

# test(os.path.dirname(_filename), dictstruct, timestamps)
def test(_dir, _dict, _tstamps, _VERBOSE=False):
    """
    Returns flags of several data integrity tests for given file structure from database file
    """
    """
    if _VERBOSE:
        sys.stdout = sys.__stdout__
    else:
        sys.stdout = open(os.devnull, 'w')
    """
    if _VERBOSE:
        print("STARTING DATA INTEGRITY TEST...")
        print("-------------------------------")
        print("CHECKING DATABASE...\t\t\t", end='')
    basefl = check_base(_dict, _dir)
    if _VERBOSE:
        print("[O.K.]" if basefl else "[FAILED]")
        print("CHECKING METAFILES...\t\t\t", end='')
    metafl = check_meta(_dict, _dir)
    if _VERBOSE:
        print("[O.K.]" if metafl else "[FAILED]")
        print("CHECKING DATAFILES...\t\t\t", end='')
    datafl = check_data(_dict, _dir)
    if _VERBOSE:
        print("[O.K.]" if datafl else "[FAILED]")
        print("CHECKING TIMESTAMPS...\t\t\t", end='')
    timefl = check_time(_tstamps, _dir)
    if _VERBOSE:
        print("[O.K.]" if timefl else "[FAILED]")
        print("\n[DONE]\n***\n")
    return [basefl, metafl, datafl, timefl]


"""
GraphDict Implementation
"""

class Node(object):
    """ Node object for GraphDict """
    def __init__(self, name, children):
        self.name = name
        self.children = children

    def __str__(self):
        return self.name


class GraphDict(Mapping):
    """
    GraphDict class takes a dict to represent it as an ASCII graph using the asciitree module.

    Attributes:
    *    _dict: dictionary in a graph structure (dict in dict...)

    Keywords:
    *    maxshow: maximum length of children shown (default: 10)
    """
    def __init__(self, _dict, maxshow=10):
        self._storage = _dict
        self.max = maxshow

    def __getitem__(self, key):
        return self._storage[key]

    def __iter__(self):
        return iter(self._storage)    # ``ghost`` is invisible

    def __len__(self):
        return len(self._storage)

    def __str__(self):
        net = Node("Nothing here", [])
        for k,v in self._storage.items(): ## go through experiments values (dicts)
            experiments = []
            sessions = []
            for i, sess in enumerate(v):
                if i < self.max-2:
                    sessions.append(Node(sess, []))
                if i == self.max-2:
                    sessions.append(Node("...", []))
                if i == len(v)-1:
                    sessions.append(Node(sess, []))
            experiments.append(Node(k.split('.')[0]+" ({:} sessions)".format(len(v)), sessions))
            net = Node(k, experiments)
        return draw_tree(net)

    def __getattr__(self, name):
        return self[name]


"""
Database Implementation
"""

class Experiment(object):
    """
    Experiment class: contains list of  for experiments
    """
    def __init__(self, _filename):
        dictstruct, timestamps = self.load_db(_filename)
        test(os.path.dirname(_filename), dictstruct, timestamps)
        self.struct = GraphDict(dictstruct)
        self.dir = os.path.dirname(_filename)
        self.name = os.path.basename(_filename)[:4]
        self.filename = os.path.basename(_filename)
        self.active = None

        ### set up sessions
        self.sessions = []
        prn(__name__)
        flprint('Loading files from {:}...'.format(_filename))
        for session in dictstruct[self.filename]:
            mfile = os.path.join(self.dir, session[:-3]+'yaml')
            self.sessions.append(Session(self.dir, session, mfile))
        flprint('found {} sessions in database...'.format(len(self.sessions)))
        colorprint("done.", color='success')

    def conditions(self):
        return np.unique([session.load_meta()['fly']['metabolic'] for session in self.sessions])


    def load_data(self, _id):
        self.active = _id
        return self.session(_id).load()

    def load_db(self, _file):
        filestruct, timestamps = load_yaml(_file)
        return filestruct, timestamps

    def session(self, arg):
        name = "{}_{:03d}".format(self.name, arg)
        for ses in self.sessions:
            if name == ses.name:
                return ses
        return None

    def __str__(self):
        return str(self.struct)

class Session(object):
    """
    Session class creates an object that hold meta-data of single session and has the functionality to load data into pd.DataFrame
    """
    def __init__(self, _dir, _file, _mfile):
        self.dir = _dir
        self.file = _file
        self.mfile = _mfile
        self.exp = _file[:4]
        self.name = _file.split('.')[0]

        #new_list = []
        #for k, v in self.metadata["food_spots"].items():
            #new_list.append(v)
        #self.metadata["food_spots"] = new_list

    def __str__(self):
        return self.name +" <class '"+ self.__class__.__name__+"'>"

    def add_data(self, title, data, descr=""):
        if title in self.datdescr.keys() and not self.in_pynb:
            if not query_yn("Data \'{:}\' does seem to exist in the dataframe. Do you want to overwrite the data?".format(title)):
                return None
        if self.data is None:
            self.data = data
            if len(data.columns) == 1:
                self.data.rename(inplace=True, columns = {data.columns[0] : title})
        else:
            try:
                self.data = pd.concat([self.data, data], axis=1)
                if isinstance(data, pd.Series):
                    data = data.to_frame()
                if len(data.columns) == 1:
                    self.data.rename(inplace=True, columns = {data.columns[0] : title})
            except:
                print(title)
                print("Given data does not fit format of stored dataframe. Maybe data belongs to experiment or database.")
        self.datdescr[title] = descr

    def show_data(self):
        print()
        print("=====")
        print(self.name, '[SESSION]')
        if self.data is None:
            print("EMPTY data")
            print("-----")
        else:
            print("-----")
            print("Found {} columns:".format(len(self.data.columns)) )
            print("...\n" + str(self.data.count()))
            print("-----")

    def keys(self):
        str = ""
        lkeys = self.metadata.keys()
        for i, k in enumerate(lkeys):
            str += "({:})\t{:}\n".format(i,k,self.metadata[k])
        str += "\n"
        return str

    def patches(self):
        out = []
        for i, pos in enumerate(self.dict["PatchPositions"]):
            out.append({})
            out[-1]["position"] = [xy * self.dict["px2mm"] for xy in pos]
            out[-1]["substrate"] = self.dict["SubstrateType"][i]
            out[-1]["radius"] = self.dict["patch_radius"] * self.dict["px2mm"]
        return out


    def load(self, VERBOSE=True):
        try:
            if VERBOSE:
                prn(__name__)
                flprint('Loading session data & metadata for {}...'.format(self.name))
            with open(self.mfile) as f:
                meta_data = yaml.safe_load(f)
        except FileNotFoundError:
            colorprint("[ERROR]: session metadata file not found.", color='error')
        try:
            csv_file = os.path.join(self.dir, self.file)
            data = pd.read_csv(csv_file, index_col='frame')
            if VERBOSE:
                colorprint("done.", color='success')
        except FileNotFoundError:
            colorprint("[ERROR]: session data file not found.", color='error')
        return data, meta_data

    def load_meta(self, VERBOSE=False):
        try:
            if VERBOSE:
                prn(__name__)
                flprint('Loading session data & metadata for {}...'.format(self.name))
            with open(self.mfile) as f:
                meta_data = yaml.safe_load(f)
        except FileNotFoundError:
            colorprint("[ERROR]: session metadata file not found.", color='error')
        return meta_data

    def meta(self):
        return self.dict

    def preview(self, subsampling=50):
        filedir = os.path.dirname(self.file)
        filename = filedir + os.sep + self.name +".csv"
        data = pd.read_csv(filename, sep="\t", escapechar="#")
        data = data.rename(columns = {" body_x":'body_x'})    ### TODO: fix this in data conversion
        return data[::subsampling]


    def nice(self):
        str = """
=================================
Meta-data for session {:}
=================================\n\n
""".format(self.name)
        lkeys = self.metadata.keys()
        for i, k in enumerate(lkeys):
            str += "({:})\t{:}:\n\t\t\t{:}\n\n".format(i,k,self.metadata[k])
        str += "\n"
        return str

    def __str__(self):
        return self.nice()
