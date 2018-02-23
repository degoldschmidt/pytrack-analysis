import os, sys
import numpy as np
import pandas as pd

from pytrack_analysis import Node
from pytrack_analysis.array import rle
from pytrack_analysis.cli import colorprint, flprint, prn

"""
Segments analyis class: loads centroid data and metadata >> processes and returns segments data
"""
class Segments(Node):
    def __init__(self, _df, _meta, ethogram='etho', visits='visit', visit_spot='visit_index', encounters='encounter', encounter_spot='encounter_index', dt='frame_dt'):
        """
        Initializes the class. Setting up internal variables for input data; setting up logging.
        """
        Node.__init__(self, _df, _meta)
        ### data check
        self.keys = [ethogram, visits, visit_spot, encounters, encounter_spot]
        self.dt = np.array(self.df[dt])
        assert (all([(key in _df.keys()) for key in self.keys])), '[ERROR] Some keys not found in dataframe.'

    def run(self, save_as=None, ret=False, VERBOSE=True):
        """
        index || state || start_pos || length
        """
        ### this prints out header
        if VERBOSE:
            prn(__name__)
            flprint("{0:8s} (condition: {1:3s})...".format(self.session_name, str(self.meta['fly']['metabolic'])))
        dict_ret = {}
        for k in self.keys:
            outdf = pd.DataFrame({})
            l, p, s, rl = rle(self.df[k], dt=self.dt)
            outdf['state'] = s
            outdf['position'] = p
            outdf['arraylen'] = l
            outdf['duration'] = rl
            if save_as is not None:
                outfile = os.path.join(save_as, '{}_{}_{}.csv'.format(self.session_name, self.name, k))
                outdf.to_csv(outfile, index_label='segment')
            if ret or save_as is None:
                dict_ret[k] = outdf
        if VERBOSE: colorprint('done.', color='success')
        if ret or save_as is None:
            return dict_ret
