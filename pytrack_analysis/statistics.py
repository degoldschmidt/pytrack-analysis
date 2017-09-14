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
    def frequency(self, _df, _value, _each, _off=0):
        outdict = {
                    'session': [],
                    'genotype': [],
                    'mating': [],
                    'metabolic': [],
                    'rate [1/s]': [],
                    'rate [1/min]': [],
        }
        if len(_each) > 0:
            for each in _df.drop_duplicates(_each)[_each]:
                only_this = _df.query('session == "{:}" & state == {:}'.format(each, _value)) # & state == "{:}"; _value
                num_encounters = len(only_this.index)
                time_spent_outside = np.sum(_df.query('session == "{:}" & state == {:}'.format(each, _off))['length [s]'])
                outdict['session'].append(each)
                geno = _df.drop_duplicates(_each).query('session == "{:}"'.format(each))['genotype'].iloc[0]
                outdict['genotype'].append(geno)
                mate = _df.drop_duplicates(_each).query('session == "{:}"'.format(each))['mating'].iloc[0]
                outdict['mating'].append(mate)
                metab = _df.drop_duplicates(_each).query('session == "{:}"'.format(each))['metabolic'].iloc[0]
                outdict['metabolic'].append(metab)
                outdict['rate [1/s]'].append(num_encounters/time_spent_outside)
                outdict['rate [1/min]'].append(num_encounters/(time_spent_outside/60.))
        return pd.DataFrame(outdict)

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
        Returns segmental information (duration of sequences, total duration and cumulative length) of a given input discrete-valued time series
        TESTED
        """
        outdict = {
                    'session': [],
                    'genotype': [],
                    'mating': [],
                    'metabolic': [],
                    'state': [],
                    'length [s]': [],
                    'num_segments': [],
                    'total_length [s]': [],
                    'total_length [min]': [],
                    'mean_length [s]': [],
                    'mean_length [min]': [],
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

            lengths, pos, state = self.rle(_X[col])
            sums = []
            means = []
            cums = np.zeros(len(lengths), dtype=np.int)
            unique_states = np.sort(np.unique(state))
            sums_check = np.zeros(len(unique_states), dtype=np.int)
            for i, each_class in enumerate(unique_states): ## find unique state values
                sums.append(np.sum(lengths[state == each_class]))
                cums[state == each_class] = np.cumsum(lengths[state == each_class])
                means.append(np.mean(lengths[state == each_class]))

                ## checking whether total duration is calculated correctly
                for entry in _X[col]:
                    if entry == each_class:
                        sums_check[i] += 1
                assert (sums[i] == sums_check[i]),"Total duration is not calculated correctly! {} != {}".format(sums[i], sums_check[i])
            for index, each_len in enumerate(lengths):
                this_state = int(state[index])

                if session == "CANS_001":
                    print("state=", this_state,"#segm:", np.sum(state == this_state))
                outdict['session'].append(session)
                outdict['genotype'].append(genotype)
                outdict['mating'].append(mating)
                outdict['metabolic'].append(metabolic)
                outdict['state'].append(this_state)
                outdict['length [s]'].append(each_len*dt)
                outdict['num_segments'].append(np.sum(state == this_state))
                outdict['total_length [s]'].append(sums[this_state]*dt)
                outdict['total_length [min]'].append(sums[this_state]*dt/60.)
                outdict['mean_length [s]'].append(means[this_state]*dt)
                outdict['mean_length [min]'].append(means[this_state]*dt/60.)
                outdict['cumulative_length [s]'].append(cums[index]*dt)
                outdict['cumulative_length [min]'].append(cums[index]*dt/60.)
                outdict['frame_index'].append(pos[index])
        return pd.DataFrame(outdict)

    @logged_f(LOG_PATH)
    def visit_ratio(self, _encounters, _visits):
        outdict = {
                    'session': [],
                    'genotype': [],
                    'mating': [],
                    'metabolic': [],
                    'state': [],
                    'ratio': [],
        }
        all_sessions = _encounters.drop_duplicates('session')['session']
        for each_session in all_sessions:
            unique_states = np.sort(np.unique(_encounters.query("session == '{}'".format(each_session))['state']))
            if each_session == 'CANS_001':
                print(unique_states)
            session_ratio = 0.0
        return pd.DataFrame(outdict)
