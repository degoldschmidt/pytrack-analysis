from scipy import signal
import numpy as np
import pandas as pd
import time

def interpolate(_data):
    return _data.interpolate()

def to_mm(_data, px2mm):
    return _data * px2mm

def gaussian_filter(_df, _len=16, _sigma=1.6):
    cols = np.empty((len(_df.index), len(_df.columns)))
    cols.fill(np.nan)
    header = []
    for column in _df:
        header.append(column)
        cols[:,len(header)-1] = gaussian_filtered(_df[column], _len=_len, _sigma=_sigma)
    return pd.DataFrame(cols, columns=header)

def gaussian_filtered(_X, _len=16, _sigma=1.6):
    norm = np.sqrt(2*np.pi)*_sigma ### Scipy's gaussian window is not normalized
    window = signal.gaussian(_len+1, std=_sigma)/norm
    return np.convolve(_X, window, "same")
