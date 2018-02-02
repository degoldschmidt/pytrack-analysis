from scipy import signal
import numpy as np
import pandas as pd
import time

def interpolate(*args):
    if len(args) > 1:
        return [arg.interpolate() for arg in args]
    elif len(args) == 1:
        return args[0].interpolate()
    else:
        return None

def to_mm(_data, px2mm):
    return _data * px2mm

def gaussian_filter(*args, _len=16, _sigma=1.6):
    if len(args) > 1:
        return [gaussian_filtered(arg, _len=_len, _sigma=_sigma) for arg in args]
    elif len(args) == 1:
        return gaussian_filtered(args[0], _len=_len, _sigma=_sigma)
    else:
        return None

def gaussian_filter_np(_X, _len=16, _sigma=1.6):
    return gaussian_filtered(_X, _len=_len, _sigma=_sigma)

def gaussian_filtered(_X, _len=16, _sigma=1.6):
    norm = np.sqrt(2*np.pi)*_sigma ### Scipy's gaussian window is not normalized
    window = signal.gaussian(_len+1, std=_sigma)/norm
    outdf = pd.DataFrame({}, index=_X.index)
    for col in _X.columns:
        convo = np.convolve(_X[col], window, "same")
        ## eliminate boundary effects
        convo[:_len] = _X[col].iloc[:_len]
        convo[-_len:] = _X[col].iloc[-_len:]
        outdf[col] = convo
    return outdf
