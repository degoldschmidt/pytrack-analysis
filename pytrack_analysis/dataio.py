import os
import pandas as pd
from pytrack_analysis.cli import colorprint

"""
Returns list of directories in given path d with full path (DATAIO)
"""
def flistdir(d):
    return [os.path.join(d, f) for f in os.listdir(d) if '.txt' in f]

"""
Returns list of raw data for filenames (DATAIO)
"""
def get_data(filenames, columns=['datetime', 'elapsed_time', 'frame_dt', 'body_x', 'body_y', 'angle', 'major', 'minor'], units=['Datetime', 's', 's', 'px', 'px', 'rad', 'px', 'px']):
    print("[data_io]\tLoading raw data...", flush=True, end="")
    renaming = columns
    data_units = {}
    try:
        for i, each_col in enumerate(renaming):
            data_units[each_col] = units[i]
    except IndexError:
        print('Not enough units given for columns.')
    data = []
    for each in filenames:
        ## load data
        data.append(pd.read_csv(each, sep="\s+", skiprows=1))
        # renaming columns with standard header
        data[-1].columns = renaming
        # datetime strings to datetime objects
        data[-1]['datetime'] =  pd.to_datetime(data[-1]['datetime'])
    colorprint("done.", color='success')
    return data, units
