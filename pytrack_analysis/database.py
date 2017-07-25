import os
import numpy as np
import pandas as pd
from . import graphdict
from . import data_integrity
import yaml
import json

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

class Database(object):
    def __init__(self, _filename):
        dictstruct, timestamps = self.load_db(_filename)
        data_integrity.test(os.path.dirname(_filename), dictstruct, timestamps)
        self.struct = graphdict.GraphDict(dictstruct)
        self.dir = os.path.dirname(_filename)
        self.name = os.path.basename(_filename).split(".")[0]
        basename = os.path.basename(_filename)

        ### set up experiments
        self.experiments = []
        for key in dictstruct[basename].keys():
            jfile = os.path.join(self.dir, key)
            self.experiments.append(Experiment(jfile))

    def experiment(self, identifier):
        """
        identifier: int or string identifying the experiment
        """
        if identifier == "":
            endstr = ""
            for ses in self.sessions:
                endstr += str(ses)+"\n"
            return endstr
        elif type(identifier) is int:
            return self.experiments[identifier]
        elif type(identifier) is str:
            for exp in self.experiments:
                if exp.name == identifier:
                    return exp
        return "[ERROR]: experiment not found."

    def find(self, eqs):
        for alleq in eqs:
            key = eqs.split("=")[0]
            val = eqs.split("=")[1]
            lstr = []
            for ses in self.sessions():
                if ses.dict[key] == val:
                    lstr.append(ses.name)
        return lstr

    def load_db(self, _file):
        filestruct, timestamps = load_yaml(_file)
        return filestruct, timestamps

    def sessions(self):
        outlist = []
        for exp in self.experiments:
            for ses in exp.sessions:
                outlist.append(ses)
        return outlist

    def __str__(self):
        return str(self.struct)


class Experiment(object):
    def __init__(self, _file):
        self.dict = json2dict(_file)
        self.file = _file
        self.name = _file.split(os.sep)[-1].split(".")[0]

        ### set up sessions inside experiment
        self.sessions = []
        for key, val in self.dict.items():
            if self.name in key:
                self.sessions.append(Session(val, _file, key))
        #print(self.sessions[-1])

    def __getattr__(self, name):
        return self.dict[name]

    def __str__(self):
        return self.name +" <class '"+ self.__class__.__name__+"'>"

    def session(self, identifier):
        """
        identifier: int or string identifying the experiment
        """
        if identifier == "":
            endstr = ""
            for ses in self.sessions:
                endstr += str(ses)+"\n"
            return endstr
        if type(identifier) is int:
            return self.sessions[identifier]
        elif type(identifier) is str:
            for ses in self.sessions:
                if ses.name == identifier:
                    return ses
            for ses in self.sessions:
                if identifier in ses.name:
                    return ses
        return "[ERROR]: session not found."


class Session(object):
    """
    Session class creates an object that hold meta-data of single session and has the functionality to load data into pd.DataFrame or np.ndarray
    """
    def __init__(self, _dict, _file, _key):
        self.dict = _dict
        self.file = _file
        self.name = _key

    def __getattr__(self, name):
        return self.dict[name]

    def __str__(self):
        return self.name +" <class '"+ self.__class__.__name__+"'>"

    def keys(self):
        str = ""
        lkeys = self.dict.keys()
        for i, k in enumerate(lkeys):
            str += "({:})\t{:}\n".format(i,k,self.dict[k])
        str += "\n"
        return str

    def load(self, load_as="pd"):
        meta_data = self
        filedir = os.path.dirname(self.file)
        filename = filedir + os.sep + self.name +".csv"
        if load_as == "pd":
            data = pd.read_csv(filename, sep="\t", escapechar="#")
            data = data.rename(columns = {" body_x":'body_x'})    ### TODO: fix this in data conversion
        elif load_as == "np":
            data = np.loadtxt(filename)
        else:
            print("[ERROR]: session not found.")                  ### TODO
        return data, meta_data

    def nice(self):
        str = """
=================================
Meta-data for session {:}
=================================\n\n
""".format(self.name)
        lkeys = self.dict.keys()
        for i, k in enumerate(lkeys):
            str += "({:})\t{:}:\n\t\t\t{:}\n\n".format(i,k,self.dict[k])
        str += "\n"
        return str
