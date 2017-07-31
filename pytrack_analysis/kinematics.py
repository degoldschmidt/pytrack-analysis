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

###
# GLOBAL CONSTANTS (based on OS)
###
PROFILE, NAME, OS = get_globals()

def get_log_path(_file):
    with open(_file, 'r') as stream:
        profile = yaml.load(stream)
    return profile[profile['active']]['systems'][NAME]['log']

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
Kinematics class: loads centroid data and metadata >> processes and returns kinematic data
"""
class Kinematics(object):

    #@Logger.logged
    def __init__(self, _data, _metadata):
        """
        Initializes the class. Setting up internal variables for input data; setting up logging.
        """
        ## overrides path-to-file and hash of last file-modified commit (version)
        self.filepath = os.path.realpath(__file__)
        self.vcommit = __version__
        self.dt = 1/_metadata["framerate"]

        ## logging
        logger = get_log(self, get_func(), LOG_PATH)
        logger.info( "initialized Kinematics pipeline (version: "+str(self)+")")

    @logged_f(LOG_PATH)
    def angular_speed(self, _X, _meta):
        angle = np.array(_X["heading"])
        speed = np.diff(angle)
        speed[speed>180] -= 360.  ## correction for circularity
        speed[speed<-180] += 360.  ## correction for circularity
        speed *= _meta.dict["framerate"]
        df = pd.DataFrame({"speed": np.append(0,speed)})
        return df

    @logged_f(LOG_PATH)
    def distance(self, _X, _Y):
        x1, y1 = np.array(_X[_X.columns[0]]), np.array(_X[_X.columns[1]])
        x2, y2 = np.array(_Y[_Y.columns[0]]), np.array(_Y[_Y.columns[1]])
        dist_sq = np.square(x1 - x2) + np.square(y1 - y2)
        dist = np.sqrt(dist_sq)
        dist[dist==np.nan] = -1 # NaNs to -1
        df = pd.DataFrame({'distance': dist})
        return df

    @logged_f(LOG_PATH)
    def distance_to_patch(self, _X, _meta):
        xfly, yfly = np.array(_X["head_x"]), np.array(_X["head_y"])
        dist = {}
        for ip, patch in enumerate(_meta.patches()):
            xp, yp = patch["position"][0], patch["position"][1]
            dist_sq = np.square(xfly - xp) + np.square(yfly - yp)
            key = "dist_patch_"+str(ip) # column header
            dist[key] = np.sqrt(dist_sq)
            #dist[key][dist[key]==np.nan] = -1 # NaNs to -1
        df = pd.DataFrame(dist)
        return df

    @logged_f(LOG_PATH)
    def ethogram(self, _X, _Y, _Z, _meta):
        ## 1) smoothed head: 2 mm/s speed threshold walking/nonwalking
        ## 2) body speed, angular speed: sharp turn
        ## 3) gaussian filtered smooth head (120 frames): < 0.2 mm/s
        ## 4) rest of frames >> micromovement

        speed = np.array(_X["head"])
        bspeed = np.array(_X["body"])
        smoother = np.array(_X["smoother_head"])
        turn = np.array(_Y["speed"])
        Neach = int(len(_Z.columns)/2) ## number of only yeast/sucrose patches
        yps = np.zeros((Neach, speed.shape[0])) ## yeast patches distances
        sps = np.zeros((Neach, speed.shape[0])) ## sucrose patches distances
        yc = 0 # counter
        sc = 0 # counter
        for i,col in enumerate(_Z.columns):
            idc = int(col.split("_")[2])
            if _meta.dict["SubstrateType"][idc] == 1:
                yps[yc,:] = np.array(_Z[col])
                yc += 1
            if _meta.dict["SubstrateType"][idc] == 2:
                sps[sc,:] = np.array(_Z[col])
                sc += 1
        ymin = np.amin(yps, axis=0) # yeast minimum distance
        smin = np.amin(sps, axis=0) # sucrose minimum distance

        out = np.zeros(speed.shape) - 1
        print(out)
        #out[speed <= 0.2] = 0   ## resting
        #out[speed > 0.2] = 1    ## micromovement
        out[speed > 2] = 2      ## walking

        mask = (out == 2) & (bspeed < 4) & (np.abs(turn) >= 125.)
        out[mask] = 3           ## sharp turn

        out[smoother <= 0.2] = 0 # new resting

        out[out == -1] = 1 # new micromovement

        visits = np.zeros(out.shape)
        for i in range(ymin.shape[0]):
            if ymin[i] <= 2.5:
                visits[i] = 1
            if smin[i] <= 2.5:
                visits[i] = 2

            if visits[i-1] == 1 and ymin[i] <= 5.0:
                visits[i] = 1
            if visits[i-1] == 2 and smin[i] <= 5.0:
                visits[i] = 2

        mask_yeast = (out == 1) & (visits == 1)
        mask_sucrose = (out == 1) & (visits == 2)
        out[mask_yeast] = 4     ## yeast micromovement
        out[mask_sucrose] = 5   ## sucrose micromovement

        return pd.DataFrame({"etho": out}), pd.DataFrame({"visits": visits})

    #@logged(TODO)
    def forward_speed(self, _X):
        pass

    @logged_f(LOG_PATH)
    def head_angle(self, _X):
        xb, yb = np.array(_X["body_x"]), np.array(_X["body_y"])
        xh, yh = np.array(_X["head_x"]), np.array(_X["head_y"])
        dx, dy = xh-xb, yh-yb
        angle = np.arctan2(dy,dx)
        angle = np.degrees(angle)

        df = pd.DataFrame({"heading": angle})
        return df

    @logged_f(LOG_PATH)
    def linear_speed(self, _X, _meta):
        xfly, yfly = np.array(_X[_X.columns[0]]), np.array(_X[_X.columns[1]])
        xdiff = np.diff(xfly)
        ydiff = np.diff(yfly)
        speed = np.sqrt( np.square(xdiff) + np.square(ydiff) ) * _meta.dict["framerate"]
        df = pd.DataFrame({"speed": np.append(0,speed)})
        return df

    #@logged(TODO)
    def sideward_speed(self, _X):
        pass


    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.vcommit
## ** FUNC: distance_from_patch ** (Inputs: fly pos [tuple], patch_id [int] >> look-up from meta OR patch_pos [tuple])

## ** FUNC: linear_speed ** (Inputs: old fly pos [tuple], new fly pos [tuple], px2mm, framerate)

## ** FUNC: angular_speed ** (Inputs: old fly pos [tuple], new fly pos [tuple], px2mm, framerate)

## ** FUNC: detect_jumps **

## ** FUNC: clear_jumps **
