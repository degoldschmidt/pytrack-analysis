# -*- coding: utf-8 -*-
# author: degoldschmidt
# date: 25/8/2017
from __future__ import absolute_import

__url__             = "https://pypi.python.org/pypi/pytrack-analysis"
__version__         = "0.0.3"
__license__         = "GPLv3"
__author__          = "Dennis Goldschmidt"
__author_email__    = "dennis.goldschmidt@neuro.fchampalimaud.org"
__all__ = ["benchmark", "database", "kinematics", "logger", "preprocessing", "profile", "settings", "statistics"]

from .benchmark import multibench
from .kinematics import Kinematics
from .logger import Logger
from .statistics import Statistics
