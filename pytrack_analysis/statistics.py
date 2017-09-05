import os, sys
import numpy as np
import pandas as pd
import logging
import yaml
import os.path as osp
import subprocess as sub
import sys
import traceback
import inspect, itertools
from functools import wraps
from ._globals import *
from pkg_resources import get_distribution
__version__ = get_distribution('pytrack_analysis').version


"""
TODO: Get these global definitions out
"""
###
# GLOBAL CONSTANTS (based on OS)
###
PROFILE, NAME, OS = get_globals()

def get_log_path(_file):
    with open(_file, 'r') as stream:
        profile = yaml.load(stream)
    try:
        return profile[profile['active']]['systems'][NAME]['log']
    except KeyError:
        return profile[profile['active']]['systems'][NAME.lower()]['log']

def get_log(_module, _func, _logfile):
    """
    The main entry point of the logging
    """
    logger = logging.getLogger(_module.__class__.__name__+"."+_func)
    logger.setLevel(logging.DEBUG)

    # create the logging file handler
    if not os.path.exists(_logfile):
        print("created file:"+_logfile)
        with open(_logfile, 'w+') as f:
            f.close()
    fh = logging.FileHandler(_logfile)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add handler to logger object
    logger.addHandler(fh)
    return logger


def logged_f(_logfile):
    def wrapper(func):
        @wraps(func)
        def func_wrapper(*args, **kwargs):
            logger = get_log(args[0], func.__name__, _logfile)
            if func.__name__ == "__init__":
                logger.info("Initializing: "+ args[0].__class__.__name__+" (version: "+args[0].vcommit+")")
            else:
                logger.info("calling: "+func.__name__)
            # if you want names and values as a dictionary:
            if len(args) > 0:
                args_name = inspect.getargspec(func)[0]
                args_dict = dict(zip(args_name, [type(arg) for arg in args]))
                logger.info("takes arg: "+str(args_dict))
            if len(args) == 0:
                logger.info("takes arg: "+str(None))

            if len(kwargs) > 0:
                kwargs_name = inspect.getargspec(func)[2]
                kwargs_dict = dict(zip(kwargs_name, type(kwargs)))
                logger.info("takes kwarg: "+str(kwargs_dict))
            if len(kwargs) == 0:
                logger.info("takes kwarg: "+str(None))
            out = func(*args, **kwargs)
            logger.info("returns: "+str(type(out)))
            for handler in logger.handlers:
                handler.close()
            return out
        return func_wrapper
    return wrapper

LOG_PATH = get_log_path(PROFILE)

def get_path(outstr):
    print(outstr+"\t"+LOG_PATH)

def get_func():
    out = traceback.extract_stack(None, 2)[0][2]
    return out


"""
Statistics class: loads centroid data and metadata >> processes and returns kinematic data
"""
class Statistics(object):

    #@Logger.logged
    def __init__(self, _db):
        """
        Initializes the class. Setting up internal variables for input data; setting up logging.
        """
        ## overrides path-to-file and hash of last file-modified commit (version)
        self.filepath = os.path.realpath(__file__)
        self.vcommit = __version__
        ##
        self.db = _db
        ## logging
        logger = get_log(self, get_func(), LOG_PATH)
        logger.info( "initialized Kinematics pipeline (version: "+str(self)+")")

    @logged_f(LOG_PATH)
    def rle(self, inarray):
            """ run length encoding. Partial credit to R rle function.
                Multi datatype arrays catered for including non Numpy
                returns: tuple (runlengths, startpositions, values) """
            ia = np.array(inarray)                  # force numpy
            n = len(ia)
            if n == 0:
                return (None, None, None)
            else:
                y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
                i = np.append(np.where(y), n - 1)   # must include last element posi
                z = np.diff(np.append(-1, i))       # run lengths
                p = np.cumsum(np.append(0, z))[:-1] # positions
                return(z, p, ia[i])

    @logged_f(LOG_PATH)
    def segments(self, _X):
        """
        Returns sequence information (duration of sequences, total duration and cumulative length) from state vectors
        TESTED
        """
        outdict = {
                    'session': [],
                    'genotype': [],
                    'mating': [],
                    'metabolic': [],
                    'state': [],
                    'length [s]': [],
                    'total_length [s]': [],
                    'total_length [min]': [],
                    'cumulative_length [s]': [],
                    'cumulative_length [min]': [],
                    'frame_index': [],
        }
        for col in _X.columns:
            session = col # session name is column name
            genotype = self.db.session(col).dict['Genotype'] # read out genotype of session from database
            mating = self.db.session(col).dict['Mating'] # read out mating state of session from database
            metabolic = self.db.session(col).dict['Metabolic'] # read out metabolic state of session from database
            dt = 1/self.db.session(col).dict['framerate'] # read out framerate of session from database

            lengths, pos, behavior_class = self.rle(_X[col])
            sums = []
            cums = np.zeros(len(lengths), dtype=np.int)
            unique_behaviors = np.sort(np.unique(behavior_class))
            sums_check = np.zeros(len(unique_behaviors), dtype=np.int)
            for i, each_class in enumerate(unique_behaviors): ## find unique behavioral class values
                sums.append(np.sum(lengths[behavior_class == each_class]))
                cums[behavior_class == each_class] = np.cumsum(lengths[behavior_class == each_class])

                ## checking whether total duration is calculated correctly
                for entry in _X[col]:
                    if entry == each_class:
                        sums_check[i] += 1
                assert (sums[i] == sums_check[i]),"Total duration is not calculated correctly! {} != {}".format(sums[i], sums_check[i])

            for index, each_len in enumerate(lengths):
                this_behavior = int(behavior_class[index])
                outdict['session'].append(session)
                outdict['genotype'].append(genotype)
                outdict['mating'].append(mating)
                outdict['metabolic'].append(metabolic)
                outdict['state'].append(this_behavior)
                outdict['length [s]'].append(each_len*dt)
                outdict['total_length [s]'].append(sums[this_behavior]*dt)
                outdict['total_length [min]'].append(sums[this_behavior]*dt/60.)
                outdict['cumulative_length [s]'].append(cums[index]*dt)
                outdict['cumulative_length [min]'].append(cums[index]*dt/60.)
                outdict['frame_index'].append(pos[index])
        return pd.DataFrame(outdict)
