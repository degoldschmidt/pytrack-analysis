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
from pkg_resources import get_distribution
__version__ = get_distribution('pytrack_analysis').version

"""
Kinematics class: loads centroid data and metadata >> processes and returns kinematic data
"""
class Kinematics(Node):

    def __init__(self, _df, _meta):
        """
        Initializes the class. Setting up internal variables for input data; setting up logging.
        """
        Node.__init__(self, _df, _meta)

    def angular_speed(self, _X, _dt):
        angle = np.array(_X)
        speed = np.append(0, np.diff(angle))
        speed[speed>180] -= 360.  ## correction for circularity
        speed[speed<-180] += 360.  ## correction for circularity
        #speed = np.divide(speed, _dt)
        return speed/_dt

    def distance(self, _X, _Y):
        x1, y1 = np.array(_X[_X.columns[0]]), np.array(_X[_X.columns[1]])
        x2, y2 = np.array(_Y[_Y.columns[0]]), np.array(_Y[_Y.columns[1]])
        dist_sq = np.square(x1 - x2) + np.square(y1 - y2)
        dist = np.sqrt(dist_sq)
        dist[dist==np.nan] = -1 # NaNs to -1
        df = pd.DataFrame({'distance': dist})
        return df

    def distance_to_patch(self, _X, _patch, ix):
        xfly, yfly = np.array(_X["head_x"]), np.array(_X["head_y"])
        xp, yp = _patch["x"], _patch["y"]
        dist_sq = np.square(xfly - xp) + np.square(yfly - yp)
        return np.sqrt(dist_sq)

    def classify_behavior(self, _kinedata, _meta):
        ## 1) smoothed head: 2 mm/s speed threshold walking/nonwalking
        ## 2) body speed, angular speed: sharp turn
        ## 3) gaussian filtered smooth head (120 frames): < 0.2 mm/s
        ## 4) rest of frames >> micromovement
        """
        {       0: "resting",
                1: "micromovement",
                2: "walking",
                3: "sharp turn",
                4: "yeast micromovement",
                5: "sucrose micromovement"}
        """
        head_pos = _kinedata[['head_x', 'head_y']]
        speed = np.array(_kinedata["smooth_head_speed"])
        bspeed = np.array(_kinedata["smooth_body_speed"])
        smoother = np.array(_kinedata["smoother_head_speed"])
        turn = np.array(_kinedata["angular_speed"])
        all_spots = ['distance_patch_'+str(ix) for ix in range(12)]
        aps = np.array(_kinedata[all_spots]).T
        amin = np.amin(aps, axis=0) # all patches minimum distance
        imin = np.argmin(aps, axis=0) # number of patch with minimum distance

        ethogram = np.zeros(speed.shape, dtype=np.int) - 1 ## non-walking/-classified
        ethogram[speed > 2] = 2      ## walking
        ethogram[speed > 20] = 6      ## jumps or mistracking

        mask = (ethogram == 2) & (bspeed < 4) & (np.abs(turn) >= 125.)
        ethogram[mask] = 3           ## sharp turn

        ethogram[smoother <= 0.2] = 0 # new resting

        ethogram[ethogram == -1] = 1 # new micromovement

        ethogram = self.two_pixel_rule(ethogram, head_pos, join=[1])

        visits = np.zeros(ethogram.shape)
        encounters = np.zeros(ethogram.shape)
        encounter_index = np.zeros(ethogram.shape, dtype=np.int) - 1

        substrate_dict = {'10% yeast':1, '20 mM sucrose': 2}
        substrates = np.array([substrate_dict[each_spot['substrate']] for each_spot in _meta['food_spots']])
        visit_mask = (amin <= 2.5) & (ethogram == 1)    # distance < 2.5 mm and Micromovement
        visits[visit_mask] = substrates[imin[visit_mask]]
        encounters[amin <= 3] = substrates[imin[amin <= 3]]
        encounter_index[amin <= 3] = imin[amin <= 3]

        for i in range(1, amin.shape[0]):
            if encounter_index[i-1] >= 0:
                if visits[i-1] > 0:
                    if aps[encounter_index[i-1], i] <= 5. and visits[i] == 0:
                        visits[i] = visits[i-1]
                if encounters[i-1] > 0:
                    if aps[encounter_index[i-1], i] <= 5. and encounters[i] == 0:
                        encounters[i] = encounters[i-1]
                        encounter_index[i] = encounter_index[i-1]
        visits = self.two_pixel_rule(visits, head_pos, join=[1,2])
        encounters = self.two_pixel_rule(encounters, head_pos, join=[1,2])

        mask_yeast = (ethogram == 1) & (visits == 1) & (amin <= 2.5)    # yeast
        mask_sucrose = (ethogram == 1) & (visits == 2) & (amin <= 2.5)  # sucrose
        ethogram[mask_yeast] = 4     ## yeast micromovement
        ethogram[mask_sucrose] = 5   ## sucrose micromovement

        return  ethogram, visits, encounters, encounter_index

    def forward_speed(self, _X):
        pass

    def heading_angle(self, _X):
        """
        Returns angular heading for given body and head positions.

        args:
        - _X [pd.DataFrame] : contains body and head positions.
        return:
        - df [pd.DataFrame] : heading angle (column title: 'heading')
        """
        xb, yb = np.array(_X["body_x"]), np.array(_X["body_y"])
        xh, yh = np.array(_X["head_x"]), np.array(_X["head_y"])
        dx, dy = xh-xb, yh-yb
        angle = np.arctan2(dy,dx)
        angle = np.degrees(angle)
        return angle

    def linear_speed(self, _X, _Y, _dt):
        xfly, yfly = np.array(_X), np.array(_Y)
        xdiff = np.append(0, np.diff(xfly))
        ydiff = np.append(0, np.diff(yfly))
        xdiff = np.divide(xdiff, _dt)
        ydiff = np.divide(ydiff, _dt)
        speed =  np.sqrt( np.square(xdiff) + np.square(ydiff) )
        return speed

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

    def run(self, _session, _VERBOSE=False, _ALL=False):
        """
        returns ethogram and visits from running kinematics analysis for given session
        """
        ethos, visits = [], []
        # this prints out header
        if _VERBOSE and self.print_header:
            self.print_header = False
            header = "{0:10s}   {1:10s}".format("Session", "Metabolic")
            print(header, flush=True)
            autolen = len(header)
            print("-"*autolen)
        # import preprocessing functions
        import pytrack_analysis.preprocessing as prep
        # load session (takes either session string or session object)
        if type(_session) is str:
            this_session = self.db.session(_session)
        else:
            this_session = _session
        # load raw data and meta data
        raw_data, meta_data = this_session.load()
        #print("Raw data: {} rows x {} cols".format(len(raw_data.index), len(raw_data.columns)))
        # this prints out details of each session
        if _VERBOSE:
            print("{0:10s}   {1:10s}".format(this_session.name, str(meta_data['metabolic'])), flush=True)


        # get frame duration from framerate
        dt = 0.0333### np.array(raw_data['frame_dt'])
        ## STEP 1: NaN removal + interpolation + px-to-mm conversion
        raw_data[['body_x','body_y', 'head_x', 'head_y']] = prep.interpolate(raw_data[['body_x','body_y', 'head_x', 'head_y']])
        pxmm = 1./meta_data['px_per_mm']
        raw_data[['body_x','body_y', 'head_x', 'head_y', 'major', 'minor']] = prep.to_mm(raw_data[['body_x','body_y', 'head_x', 'head_y', 'major', 'minor']], pxmm)


        ## STEP 2: Gaussian filtering
        window_len = 16 # = 0.32 s
        raw_data[['body_x','body_y', 'head_x', 'head_y']] = prep.gaussian_filter(raw_data[['body_x','body_y', 'head_x', 'head_y']], _len=window_len, _sigma=window_len/10)

        kinematic_data = raw_data
        kinematic_data.index = raw_data.index
        kinematic_data['frame_dt'] = dt
        kinematic_data['major'] = np.array(raw_data['major'])
        kinematic_data['minor'] = np.array(raw_data['minor'])

        ## STEP 3: Distance from patch
        for ix, each_spot in enumerate(meta_data['food_spots']):
            each_spot['x'] *= pxmm
            each_spot['y'] *= pxmm
            kinematic_data['distance_patch_'+str(ix)] = self.distance_to_patch(kinematic_data[['head_x', 'head_y']], each_spot, ix)

        ## STEP 4: Linear Speed
        kinematic_data['head_speed'] = self.linear_speed(kinematic_data['head_x'], kinematic_data['head_y'], dt)
        kinematic_data['body_speed'] = self.linear_speed(kinematic_data['body_x'], kinematic_data['body_y'], dt)

        window_len = 60 # = 1.2 s
        kinematic_data['smooth_head_speed'] = prep.gaussian_filter_np(kinematic_data['head_speed'], _len=window_len, _sigma=window_len/10)
        kinematic_data['smooth_body_speed'] = prep.gaussian_filter_np(kinematic_data['body_speed'], _len=window_len, _sigma=window_len/10)
        window_len = 120 # = 1.2 s
        kinematic_data['smoother_head_speed'] = prep.gaussian_filter_np(kinematic_data['smooth_head_speed'], _len=window_len, _sigma=window_len/10)
        kinematic_data['smoother_body_speed'] = prep.gaussian_filter_np(kinematic_data['smooth_body_speed'], _len=window_len, _sigma=window_len/10)

        ## STEP 5: Angular Heading & Speed
        kinematic_data['angle'] = self.heading_angle(kinematic_data[['body_x', 'body_y', 'head_x', 'head_y']])
        kinematic_data['angular_speed'] = self.angular_speed(kinematic_data['angle'], dt)

        ## STEP 6: Ethogram classification, encounters & visits
        meta_data["etho_class"] = {    0: "resting",
                                       1: "micromovement",
                                       2: "walking",
                                       3: "sharp turn",
                                       4: "yeast micromovement",
                                       5: "sucrose micromovement",
                                       6: "Jumps/NA"}
        etho_vector, visits, encounters, encounter_index = self.classify_behavior(kinematic_data, meta_data)
        kinematic_data['etho'] = etho_vector
        kinematic_data['visit'] = visits
        kinematic_data['encounter'] = encounters
        kinematic_data['encounter_index'] = encounter_index
        return kinematic_data

    def run_many(self, _group, _VERBOSE=False, output=""):
        if _VERBOSE: print()
        self.print_header = _VERBOSE # this is needed to print header for multi run
        etho_data = {} # dict for DataFrame
        visit_data = {} # dict for DataFrame
        encounter_data = {} # dict for DataFrame
        etho_data['id'] = []
        etho_data['metabolic'] = []
        etho_data['yeast'] = []
        etho_data['sucrose'] = []
        outfile = os.path.join(output, 'etho_lens.csv')
        for session in _group:
            #etho, visits, encounters, encounter_index =
            data = self.run(session.name, _VERBOSE=_VERBOSE) # run session with print out
            etho = np.array(data['etho'])
            N = [ np.sum(etho==each) for each in range(7)]
            etho_data['id'].append(session.name)
            etho_data['metabolic'].append(session.metadata['metabolic'])
            etho_data['yeast'].append(N[4])
            etho_data['sucrose'].append(N[5])
            print(session.name, session.metadata['metabolic'], N[4], N[5])
        ethoDF = pd.DataFrame(etho_data)
        ethoDF.to_csv(outfile, index=False, sep='\t', encoding='utf-8')
        return data

    def two_pixel_rule(self, _dts, _pos, join=[]):
        _pos = np.array(_pos)
        for j in join:
            segm_len, segm_pos, segm_val = self.rle(_dts) #lengths, pos, behavior_class = self.rle(_X[col])
            for (length, start, val) in zip(segm_len, segm_pos, segm_val):
                if start == 0 or start+length == len(_dts):
                    continue
                if val not in join and _dts[start-1] == j and _dts[start+length] == j:
                    dist_vector = _pos[start:start+length,:] - _pos[start,:].transpose()
                    lens_vector = np.linalg.norm(dist_vector, axis=1)
                    if np.all(lens_vector <= 2*0.15539): ## length <= 2 * 0.15539
                        _dts[start:start+length] = j
        if len(join) == 0:
            print("Give values to join segments.")
            return _dts
        else:
            return _dts

    def sideward_speed(self, _X):
        pass
