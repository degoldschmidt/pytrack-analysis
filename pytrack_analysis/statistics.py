import os, sys
import numpy as np
import pandas as pd

from pytrack_analysis import Node
from pytrack_analysis.array import rle
from pytrack_analysis.cli import colorprint, flprint, prn

"""
Statistics analyis class: loads segments data and metadata >> processes and returns Statistics data
"""
class Statistics(Node):
    def __init__(self, ethogram='etho', visits='visit', encounters='encounter'):
        """
        Initializes the class. Setting up internal variables for input data; setting up logging.
        """
        Node.__init__(self)
        ### data check
        self.keys = [ethogram, visits, encounters]
        ### multiple dataframes
        self.df_dict = {key: pd.DataFrame({'session':[],'condition':[],'total':[],'mean':[],'median':[],'cumul':[],'frequency':[],'number':[]}) for key in self.keys}

    def run(self, dfs, meta, out=None, ret=False, VERBOSE=True):
        """
        index || state || start_pos || length
        """
        ### this prints out header
        if VERBOSE:
            prn(__name__)
            flprint("{0:8s} (condition: {1:3s})...".format(self.session_name, str(self.meta['condition'])))
        for k in self.keys:
            self.df_dict.append({'session': meta['session'],'condition': meta['condition'],'total':dfs[k].query(),'mean':[],'median':[],'cumul':[],'frequency':[],'number':[]}, ignore_index=True)
        if VERBOSE: colorprint('done.', color='success')
        if ret or save_as is None:
            return dict_ret
