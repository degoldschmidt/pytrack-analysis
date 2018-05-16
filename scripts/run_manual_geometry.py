import argparse
import subprocess, os, sys
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pytrack_analysis import Multibench
from pytrack_analysis.dataio import VideoRawData
from pytrack_analysis.profile import get_profile, get_scriptname, show_profile
import pytrack_analysis.preprocessing as prp
from pytrack_analysis.geometry import manual_geometry
from pytrack_analysis.experiment import parse_time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', metavar='video', type=str, help='video file')
    VIDEO = parser.parse_args().video
    return VIDEO

def main():
    VIDEO = get_args()
    time, timestr = parse_time(VIDEO)

    manual_geometry(VIDEO, timestr)

if __name__ == '__main__':
    # runs as benchmark test
    test = Multibench("", SILENT=False, SLIM=True)
    test(main)
    del test
