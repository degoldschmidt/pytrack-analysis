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
Node class: generic class for processing data
"""
class Node(object):

    def __init__(self, _df, _meta):
        """
        Initializes the class. Setting up internal variables for input data; setting up logging.
        """
        ## overrides path-to-file and hash of last file-modified commit (version)
        self.vcommit = __version__
        self.print_header = True

        ## reference to session (not a copy!!!)
        self.df = _df
        self.meta = _meta
        self.session_name = self.meta['datafile'].replace('\\', '/').split('/')[-1].split('.')[0]
        self.name = (self.__class__.__name__).lower()

    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return str(self.name) +" @ "+ self.vcommit
