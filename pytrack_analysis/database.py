import os, sys, time
import numpy as np
import pandas as pd
import yaml, json
from collections.abc import Mapping
from asciitree import draw_tree
from ._globals import *

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
def test(_dir, _dict, _tstamps, _VERBOSE=True):
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
    if _VERBOSE: print("[O.K.]" if timefl else "[FAILED]")
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
            for ke, exp in v.items(): ## go through sessions values
                sessions = []
                for i, sess in enumerate(exp):
                    if i < self.max-2:
                        sessions.append(Node(sess, []))
                    if i == self.max-2:
                        sessions.append(Node("...", []))
                    if i == len(exp)-1:
                        sessions.append(Node(sess, []))
                experiments.append(Node(ke, sessions))
                experiments.append(Node(" = {:} sessions".format(len(exp)), []))
            net = Node(k, experiments)
        return draw_tree(net)

    def __getattr__(self, name):
        return self[name]


"""
Database Implementation
"""

class Database(object):
    """
    Database class: contains meta-data for experiment database
    """
    def __init__(self, _filename, in_pynb=False):
        dictstruct, timestamps = self.load_db(_filename)
        test(os.path.dirname(_filename), dictstruct, timestamps)
        self.struct = GraphDict(dictstruct)
        self.dir = os.path.dirname(_filename)
        self.name = os.path.basename(_filename)

        ### set up sessions
        self.sessions = []
        for session in dictstruct[self.name]:
            mfile = os.path.join(self.dir, session[:-3]+'yaml')
            self.sessions.append(Session(self.dir, session, mfile, in_pynb=in_pynb))

    def show_data(self):
        for exp in self.experiments:
            exp.show_data()

    def find(self, eqs):
        """
        Function evaluates eqs to find given key-value match and returns session name
        """
        for alleq in eqs:
            key = eqs.split("=")[0]
            val = eqs.split("=")[1]
            lstr = []
            for ses in self.select():
                if ses.dict[key] == val:
                    lstr.append(ses.name)
        return lstr

    def count(self, genotype, mating, metabolic):
        """
        DEPRECATED
        """
        out = []
        for gene in genotype:
            for mate in mating:
                for metab in metabolic:
                    igene = self.experiment("CANS").name2int("Genotype", gene)
                    imate = self.experiment("CANS").name2int("Mating", mate)
                    imetab = self.experiment("CANS").name2int("Metabolic", metab)
                    out.append(self.counts[igene, imate, imetab])
        return (i for i in out)

    def last_select(self, arg):
        if arg in self.last:
            return self.last[arg][0]
        else:
            return None

    def load_db(self, _file):
        filestruct, timestamps = load_yaml(_file)
        return filestruct, timestamps

    def session(self, arg):
        name = "{}_{:03d}".format(self.name[:4], arg)
        print(name)
        for ses in self.sessions:
            if name == ses.name:
                return ses
        return None

    def select(self, **kwargs):
        # TODO: use **kwargs
        outlist = []
        self.last = self.all_conds.copy()
        for key, value in kwargs.items():
            self.last[key] = value


        for exp in self.experiments:
            for ses in exp.sessions:
                this_gen = exp.int2name("Genotype", ses.Genotype)
                this_mate = exp.int2name("Mating", ses.Mating)
                this_metab = exp.int2name("Metabolic", ses.Metabolic)
                if (this_gen in genotype) or len(genotype) == 0:
                    if (this_mate in mating) or len(mating) == 0:
                        if (this_metab in metabolic) or len(metabolic) == 0:
                            outlist.append(ses)
        return outlist

    def __str__(self):
        return str(self.struct)

class Session(object):
    """
    Session class creates an object that hold meta-data of single session and has the functionality to load data into pd.DataFrame or np.ndarray
    """
    def __init__(self, _dir, _file, _mfile, in_pynb=False):
        self.dir = _dir
        self.file = _file
        self.exp = _file[:4]
        self.name = _file.split('.')[0]
        self.in_pynb = in_pynb
        self.data = None    # this is supposed to be a pandas dataframe
        self.datdescr = {}

        with open(_mfile) as f:
            self.metadata = yaml.safe_load(f)

        new_list = []
        for k, v in self.metadata["food_spots"].items():
            new_list.append(v)
        self.metadata["food_spots"] = new_list

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


    def load(self, load_as="pd"):
        meta_data = self.metadata
        csv_file = os.path.join(self.dir, self.file)
        if load_as == "pd":
            data = pd.read_csv(csv_file, sep="\t", index_col=0)
        elif load_as == "np":
            data = np.loadtxt(csv_file)
        else:
            print("[ERROR]: session not found.")                  ### TODO
        return data, meta_data

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
