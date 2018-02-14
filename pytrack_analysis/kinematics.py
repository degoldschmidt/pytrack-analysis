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
from pytrack_analysis import Node
from pytrack_analysis.cli import colorprint, flprint, prn
import pytrack_analysis.preprocessing as prp
from pkg_resources import get_distribution
__version__ = get_distribution('pytrack_analysis').version

"""
Kinematics class: loads centroid data and metadata >> processes and returns kinematic data
"""
class Kinematics(Node):

    def __init__(self, _df, _meta, body=('body_x', 'body_y'), head=('head_x', 'head_y'), dt='frame_dt', angle='angle', ma='major', mi='minor'):
        """
        Initializes the class. Setting up internal variables for input data; setting up logging.
        """
        Node.__init__(self, _df, _meta)
        ### data check
        self.keys = [body[0], body[1], head[0], head[1], dt, angle, ma, mi]
        self.statcols = ['session', 'day', 'daytime', 'condition', 'position', 'head_speed', 'body_speed', 'distance', 'min_dpatch', 'dcenter', 'abs_turn_rate', 'major', 'minor', 'mistracks']
        assert (all([(key in _df.keys()) for key in self.keys])), '[ERROR] Some keys not found in dataframe.'

    def get_angle(self, _data_origin, _data_tip):
        """
        Returns angular heading for given origin and tip positions.
        """
        xb, yb = np.array(_data_origin.iloc[:,0]), np.array(_data_origin.iloc[:,1])
        xh, yh = np.array(_data_tip.iloc[:,0]), np.array(_data_tip.iloc[:,1])
        dx, dy = xh-xb, yh-yb
        angle = np.arctan2(dy,dx)
        angle = np.degrees(angle)
        return angle

    def get_angular_speed(self, _data, _dt):
        """
        Returns angular turning rate for given time series of angles.
        """
        speed = np.append(0, np.diff(_data))
        speed[speed>180] -= 360.  ## correction for circularity
        speed[speed<-180] += 360.  ## correction for circularity
        speed = np.divide(speed, _dt) ## divide by time increments
        return speed

    def get_distance(self, _data):
        dist_sq = np.square(_data.iloc[:,0]) + np.square(_data.iloc[:,1])
        return np.sqrt(dist_sq)

    def get_distance_to_patch(self, _data, _patch):
        xp, yp = _patch["x"], _patch["y"]
        dist_sq = np.square(_data.iloc[:, 0] - xp) + np.square(_data.iloc[:, 1] - yp)
        return np.sqrt(dist_sq)

    def get_forward_speed(self, _X):
        pass

    def get_linear_speed(self, _data, _dt):
        x, y = _data.columns[0], _data.columns[1]
        ### take differences between frames for displacement and divide by dt
        xdiff = np.divide(np.append(0, np.diff(_data[x])), _dt)
        ydiff = np.divide(np.append(0, np.diff(_data[y])), _dt)
        ### linear speed is the squareroot of squared displacements in x and y (Pythagoras' theorem)
        speed =  np.sqrt(np.square(xdiff) + np.square(ydiff))
        return speed

    def get_sideward_speed(self, _X):
        pass

    def hist(x, bins=None):
        hist, _ = np.histogram(self.outdf[x], bins=bins)  # arguments are passed to np.histogram
        hist = hist/np.sum(hist)  # normalize
        return hist

    def run(self, save_as=None, ret=False, _VERBOSE=True):
        """
        returns kinematic data from running kinematics analysis for a session
        """
        ### data from file
        # positions body and head
        bx, by = self.keys[0], self.keys[1]
        body_pos = self.df[[bx, by]]
        hx, hy = self.keys[2], self.keys[3]
        head_pos = self.df[[hx, hy]]
        # get frame duration from framerate
        #dt = 0.0333
        frame_dt = self.df[self.keys[4]]
        # orientation
        angle = self.df[self.keys[5]]
        # major and minor lengths
        makey, mikey = self.keys[6], self.keys[7]
        major = self.df[makey]
        minor = self.df[mikey]
        ###

        ### this prints out header
        if _VERBOSE:
            prn(__name__)
            flprint("{0:8s} (condition: {1:3s})...".format(self.session_name, str(self.meta['fly']['metabolic'])))
        ###

        ### This are the steps for kinematic analysis
        ## STEP 1: NaN removal + interpolation
        body_pos, head_pos = prp.interpolate(body_pos, head_pos)
        ## STEP 2: Gaussian filtering
        window_len = 10 # now: 10/0.333 s #### before used (15/0.5 s)
        sigma = window_len/10.
        body_pos, head_pos = prp.gaussian_filter(body_pos, head_pos, _len=window_len, _sigma=sigma)
        ## STEP 3: Distance from patch
        distances = pd.DataFrame({}, index=body_pos.index)
        for ix, each_spot in enumerate(self.meta['food_spots']):
            distances['dpatch_'+str(ix)] = self.get_distance_to_patch(head_pos, each_spot)
        distances['min_dpatch'] = np.amin(distances, axis=1)
        distances['dcenter'] = self.get_distance(head_pos)
        ## STEP 4: Linear Speed
        speed = pd.DataFrame({}, index=body_pos.index)
        speed['displacements'] = self.get_linear_speed(body_pos, 1)
        speed['head_speed'] = self.get_linear_speed(head_pos, frame_dt)
        speed['body_speed'] = self.get_linear_speed(body_pos, frame_dt)
        ## STEP 5: Smoothing speed
        window_len = 36 # now: 36/1.2 s #### before used (60/2 s)
        speed['sm_head_speed'] = prp.gaussian_filter_np(speed[['head_speed']], _len=window_len, _sigma=window_len/10)
        speed['sm_body_speed'] = prp.gaussian_filter_np(speed[['body_speed']], _len=window_len, _sigma=window_len/10)
        window_len = 72 # now: 72/2.4 s #### before used (120/4 s)
        speed['smm_head_speed'] = prp.gaussian_filter_np(speed[['sm_head_speed']], _len=window_len, _sigma=window_len/10)
        speed['smm_body_speed'] = prp.gaussian_filter_np(speed[['sm_body_speed']], _len=window_len, _sigma=window_len/10)
        ## STEP 6: Angular Heading & Speed
        angular = pd.DataFrame({}, index=body_pos.index)
        angular['new_angle'] = self.get_angle(body_pos, head_pos)
        angular['old_angle'] = np.degrees(angle)
        angular['angular_speed'] = self.get_angular_speed(angular['new_angle'], frame_dt)
        window_len = 36 # now: 36/1.2 s #### before used (60/2 s)
        angular['sm_angular_speed'] = prp.gaussian_filter_np(angular[['angular_speed']], _len=window_len, _sigma=window_len/10)
        ### DONE

        ### Prepare output to DataFrame or file
        ### rounding data
        frame_dt, body_pos, head_pos, distances, speed, angular = frame_dt.round(6), body_pos.round(4), head_pos.round(4), distances.round(4), speed.round(4), angular.round(3)
        listdfs = [frame_dt, body_pos, head_pos, distances, speed, angular]
        self.outdf = pd.concat(listdfs, axis=1)
        if _VERBOSE: colorprint('done.', color='success')
        if save_as is not None:
            outfile = os.path.join(save_as, self.session_name+'_'+self.name+'.csv')
            self.outdf.to_csv(outfile, index_label='frame')
        if ret or save_as is None:
            return self.outdf


    def stats(self):
        data = []
        data.append(self.session_name)
        data.append(self.meta['datetime'].date())
        data.append(self.meta['datetime'].hour)
        data.append(self.meta['condition'])
        data.append(self.meta['arena']['name'])
        data.append(self.outdf['smm_head_speed'].mean())
        data.append(self.outdf['smm_body_speed'].mean())
        data.append(np.cumsum(np.array(self.outdf['displacements']))[-1])
        data.append(self.outdf['min_dpatch'].mean())
        data.append(self.outdf['dcenter'].mean())
        data.append(np.abs(self.outdf['angular_speed']).mean())
        data.append(self.df['major'].mean())
        data.append(self.df['minor'].mean())
        data.append(self.meta['flags']['mistracked_frames'])
        statsdict = {}
        for i, each_col in enumerate(self.statcols):
            statsdict[each_col] = [data[i]]
        statdf = pd.DataFrame(statsdict)
        statdf = statdf.reindex(columns=['session', 'day', 'daytime', 'condition', 'position', 'head_speed', 'body_speed', 'distance', 'min_dpatch', 'dcenter', 'abs_turn_rate', 'major', 'minor', 'mistracks'])
        return statdf
