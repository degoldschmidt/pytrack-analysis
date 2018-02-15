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
Statistics class: loads centroid data and metadata >> processes and returns kinematic data
"""
class Statistics(object):

    def __init__(self, _db):
        """
        Initializes the class. Setting up internal variables for input data; setting up logging.
        """
        ## overrides path-to-file and hash of last file-modified commit (version)
        self.filepath = os.path.realpath(__file__)
        self.vcommit = __version__
        ##
        self.db = _db

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
                    'length': [],
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
            unique_states = np.arange(np.max(np.unique(state))+1)               ### state indices are arranged as integers (some states might not be in given session)
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
                outdict['session'].append(session)
                outdict['genotype'].append(genotype)
                outdict['mating'].append(mating)
                outdict['metabolic'].append(metabolic)
                outdict['state'].append(this_state)
                outdict['length'].append(each_len)
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

    def visit_ratio(self, _encounter_segments, _encounter_vectors, _visit_segments, _visit_vectors):
        outdict = {
                    'session': [],
                    'genotype': [],
                    'mating': [],
                    'metabolic': [],
                    'state': [],
                    'ratio': [],
        }
        all_sessions = _encounter_segments.drop_duplicates('session')['session']
        for each_session in all_sessions:
            _encounter_vector = _encounter_vectors[each_session]
            _visit_vector = _visit_vectors[each_session]
            unique_states = np.sort(np.unique( _encounter_segments.query("session == '{}'".format(each_session))['state'] ))
            session_ratio = [0, 0] ## yeast, sucrose
            for ix, state in enumerate(unique_states[1:]):
                outdict['session'].append(each_session)
                outdict['genotype'].append( _encounter_segments.query("session == '{}'".format(each_session))['genotype'].iloc[0] )
                outdict['mating'].append( _encounter_segments.query("session == '{}'".format(each_session))['mating'].iloc[0] )
                outdict['metabolic'].append( _encounter_segments.query("session == '{}'".format(each_session))['metabolic'].iloc[0] )
                outdict['state'].append(state)
                #rand_segm = np.random.choice(_encounter_segments.query("state == {}".format(state)).index) TODO for testing
                #print(rand_segm) TODO for testing
                for index, row in _encounter_segments.query("session == '{}' & state == {}".format(each_session, state)).iterrows():
                    start = row['frame_index']
                    end = row['frame_index']+row['length']
                    #print("index:{:04d}\tstart:{:06d}\tend:{:06d}".format(index, start, end))
                    enc = np.array(_encounter_vector[start:end])
                    vis = np.array(_visit_vector[start:end])
                    if not np.all(enc == state):
                        print("Session:", each_session)
                        print("This should not happen:", enc, state)
                    #if index == rand_segm: TODO for testing
                    #    print(state, vis, state in vis) TODO for testing
                    if np.any(enc == vis):
                        session_ratio[ix] += 1
                num_encs = _encounter_segments.query("session == '{}' & state == {}".format(each_session, state))['num_segments'].iloc[0]
                #print("count = {},\tnums = {}".format(session_ratio[ix], num_encs)) TODO for testing
                outdict['ratio'].append(session_ratio[ix]/num_encs)
        return pd.DataFrame(outdict)
