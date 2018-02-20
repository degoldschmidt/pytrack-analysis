import numpy as np
import pandas as pd

from pytrack_analysis import Node
from pytrack_analysis.array import rle
from pytrack_analysis.cli import colorprint, flprint, prn

"""
Segmantal analyis class: loads centroid data and metadata >> processes and returns segments data
"""
class Segmental(Node):
    def __init__(self, _df, _meta, ethogram='etho', visits='visit', encounters='encounter', encounter_spot='encounter_index'):
        """
        Initializes the class. Setting up internal variables for input data; setting up logging.
        """
        Node.__init__(self, _df, _meta)
        ### data check
        self.keys = [ethogram, visits, encounters, encounter_spot]
        assert (all([(key in _df.keys()) for key in self.keys])), '[ERROR] Some keys not found in dataframe.'

    def run(self, save_as=None, ret=False, VERBOSE=False):
        """
        index || state || start_pos || length
        """
        list_ret = []
        for k in self.keys:
            outdf = pd.DataFrame({})
            l, p, s = rle(self.df[k])
            outdf['states'] = s
            outdf['position'] = p
            outdf['lengths'] = l
            outdf.to_csv()
            if save_as is not None:
                outfile = os.path.join(save_as, self.session_name+'_'+k+'_'+self.name+'_'+k+'.csv')
                outdf.to_csv(outfile, index_label='frame')
            if ret or save_as is None:
                list_ret.append(outdf)
        if ret or save_as is None:
            return list_ret
