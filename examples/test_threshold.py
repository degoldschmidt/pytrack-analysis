import cv2
import numpy as np
from matplotlib import pyplot as plt
import platform
import os, subprocess

from pytrack_analysis.cli import detect_geometry

if platform.system() == 'Darwin':
    _dir = '/Users/degoldschmidt/Desktop/tracking_test_data/new_data'
else:
    _dir = '/media/degoldschmidt/DATA_BACKUP/data/tracking/sample'
video_file = "cam01_2018-04-14T16_10_40.avi" #"cam02_2018-04-14T16_12_09.avi"#"cam01_2017-11-24T08_26_19.avi"#"cam01_2018-04-14T16_10_40.avi"#"cam01_2017-11-24T08_26_19.avi"

detect_geometry(os.path.join(_dir, video_file))
