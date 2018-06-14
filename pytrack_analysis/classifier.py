import os, sys
import numpy as np
import pandas as pd

from pytrack_analysis import Node
from pytrack_analysis.array import rle
from pytrack_analysis.cli import colorprint, flprint, prn

"""
Classifier class: loads kinematics data and metadata >> processes and returns classification data
"""
class Classifier(Node):

    def __init__(self, _df, _meta, head=('head_x', 'head_y'), h_speed='sm_head_speed', b_speed='sm_body_speed', sm_speed='smm_head_speed', turn='angular_speed', dpatch='dpatch', time='elapsed_time', dt='frame_dt'):
        """
        Initializes the class. Setting up internal variables for input data; setting up logging.
        """
        Node.__init__(self, _df, _meta)
        ### data check
        self.spots = _meta['food_spots']
        self.keys = [head[0], head[1], h_speed, b_speed, sm_speed, turn, time, dt]
        self.all_spots = [dpatch+'_'+str(ix) for ix in range(len(_meta['food_spots']))]
        self.statcols = ['session', 'day', 'daytime', 'condition', 'position', 'head_speed', 'body_speed', 'distance', 'min_dpatch', 'dcenter', 'abs_turn_rate', 'major', 'minor', 'mistracks']
        assert (all([(key in _df.keys()) for key in self.keys])), '[ERROR] Some keys not found in dataframe.'
        assert (all([(key in _df.keys()) for key in self.all_spots])), '[ERROR] Some keys not found in dataframe.'


    def get_etho(self):
        amin = np.amin(self.aps, axis=1) # all patches minimum distance
        imin = np.argmin(self.aps, axis=1) # number of patch with minimum distance
        smin = np.zeros(imin.shape)
        for i in range(len(self.spots)):
            if self.spots[i]['substr'] == 'yeast':
                sub = 1
            elif self.spots[i]['substr'] == 'sucrose':
                sub = 2
            smin[imin==i] = sub
        substrate_dict = {'yeast':1, 'sucrose': 2}
        substrates = np.array([substrate_dict[each_spot['substr']] for each_spot in self.meta['food_spots']])

        ethogram = np.zeros(self.hspeed.shape, dtype=np.int) - 1 ## non-walking/-classified
        ethogram[self.hspeed > 2] = 2      ## walking
        ethogram[self.hspeed > 20] = 6      ## jumps or mistracking
        mask = (ethogram == 2) & (self.bspeed < 6) & (np.abs(self.turns) >= 125.)
        ethogram[mask] = 3           ## sharp turn
        ethogram[self.smoother <= 0.2] = 0 # new resting
        ethogram[ethogram == -1] = 1 # new micromovement
        ethogram = self.two_pixel_rule(ethogram, self.head_pos, join=[1])


        mask_yeast = (ethogram == 1) & (amin <= 2.5) & (smin == 1)  # yeast
        mask_sucrose = (ethogram == 1) & (amin <= 2.5) & (smin == 2)   # sucrose
        ethogram[mask_yeast] = 4     ## yeast micromovement
        ethogram[mask_sucrose] = 5   ## sucrose micromovement

        # visits & encounters
        visits = np.zeros(ethogram.shape, dtype=np.int)
        visit_index = np.zeros(ethogram.shape, dtype=np.int) - 1
        encounters = np.zeros(ethogram.shape, dtype=np.int)
        encounter_index = np.zeros(ethogram.shape, dtype=np.int) - 1


        #visit_mask = (ethogram == 4)      # distance < 2.5 mm and Micromovement
        visits[ethogram == 4] = 1
        visits[ethogram == 5] = 2
        visit_index[visits == 1] = imin[visits == 1]
        visit_index[visits == 2] = imin[visits == 2]

        encounters[amin <= 3] = substrates[imin[amin <= 3]]
        encounter_index[amin <= 3] = imin[amin <= 3]

        ### continuity for up to 5 mm
        start = -1
        end = -1
        current = -1
        for i in range(1, amin.shape[0]):
            if encounter_index[i-1] >= 0:
                if visits[i-1] > 0 and visits[i] == 0:
                    start = i
                    current = imin[i-1]
                if start != -1:
                    if self.aps[i, current] <= 5.:
                        end = i
                    if visits[i-1] == 0 and visits[i] > 0:
                        visits[start:end] = visits[i]
                        visit_index[start:end] = visit_index[i]
                        start = -1
                    if self.aps[i, current] > 5.:
                        start = -1
                if encounters[i-1] > 0:
                    if self.aps[i, encounter_index[i-1]] <= 5. and encounters[i] == 0:
                        encounters[i] = encounters[i-1]
                        encounter_index[i] = encounter_index[i-1]

        visits = self.two_pixel_rule(visits, self.head_pos, join=[1,2])
        encounters = self.two_pixel_rule(encounters, self.head_pos, join=[1,2])

        if np.any(visit_index[visits > 0] == -1): print('wtf')

        return ethogram, visits, visit_index, encounters, encounter_index

    def run(self, save_as=None, ret=False, VERBOSE=True):
        ## 1) smoothed head: 2 mm/s speed threshold walking/nonwalking
        ## 2) body speed, angular speed: sharp turn
        ## 3) gaussian filtered smooth head (120 frames): < 0.2 mm/s
        ## 4) rest of frames >> micromovement
        """
        {      -1: "unclassified",
                0: "resting",
                1: "micromovement",
                2: "walking",
                3: "sharp turn",
                4: "yeast micromovement",
                5: "sucrose micromovement",
                6: "jump/mistrack"}

        {
                0: "no encounter"
                1: "yeast encounter"
                2: "sucrose encounter"
        }
        """
        ### this prints out header
        if VERBOSE:
            prn(__name__)
            flprint("{0:8s} (condition: {1:3s})...".format(self.session_name, str(self.meta['fly']['metabolic'])))
        ###
        ### data from file
        # get head positions
        hx, hy = self.keys[0], self.keys[1]
        self.head_pos = self.df[[hx, hy]]
        # get head speed, body speed, and smoother head speed
        self.hspeed, self.bspeed, self.smoother = self.df[self.keys[2]], self.df[self.keys[3]], self.df[self.keys[4]]
        # get angular speed
        self.turns = self.df[self.keys[5]]
        ###
        self.aps = np.array(self.df[self.all_spots])


        outdf = pd.DataFrame({}, index=self.head_pos.index)
        outdf[self.keys[-1]] = self.df[self.keys[-1]]
        outdf[self.keys[-2]] = self.df[self.keys[-2]]
        outdf['etho'], outdf['visit'], outdf['visit_index'], outdf['encounter'], outdf['encounter_index'] = self.get_etho()
        if VERBOSE: colorprint('done.', color='success')
        if save_as is not None:
            outfile = os.path.join(save_as, self.session_name+'_'+self.name+'.csv')
            outdf.to_csv(outfile, index_label='frame')
        if ret or save_as is None:
            return outdf

    def two_pixel_rule(self, _dts, _pos, join=[]):
        _pos = np.array(_pos)
        for j in join:
            segm_len, segm_pos, segm_val = rle(_dts) #lengths, pos, behavior_class, NONE = self.rle(_X[col])
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
